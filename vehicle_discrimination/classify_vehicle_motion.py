#!/usr/bin/env python3
"""Post-hoc moving vs parked vehicle reclassifier.

Aligned with the annotation protocol: decide from position (travel lane vs
curb), not imagined speed. High box displacement across neighboring frames
only *boosts* moving (a clearly translating car is moving); low motion does
not force parked, because queued/stopped-in-lane cars are still class 1.

Usage (reclassify prediction JSON from a finetune run):
  python classify_vehicle_motion.py \\
    --pred-json .../finetuned_test.json \\
    --gt-json .../data/test.json \\
    --out .../finetuned_test_reclassified.json

Usage (CSV post-process one participant folder or a root of CSVs):
  python classify_vehicle_motion.py --csv-root .../GroundingDINO_finetuned_full_frame_output \\
    --conditions I1_Day --participants PARTICIPANT_ID.d --inplace
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from vehicle_metrics import VEHICLE, area, canon, inter, iou

MOVING = "moving vehicle"
PARKED = "parked vehicle"
MOVING_CODE = "1_Moving Vehicle"
PARKED_CODE = "2_Parked Vehicle"
SIDEWALK_KEYS = ("sidewalk", "3_sidewalk")
CROSSWALK_KEYS = ("crosswalk", "5_crosswalk")

# Displacement (pixels) above which we boost moving. Tuned for ~1920px frames.
MOTION_BOOST_PX = 12.0
NEIGHBOR_RADIUS = 3


def _is_sidewalk(label: str) -> bool:
    t = canon(label)
    return "sidewalk" in t


def _is_crosswalk(label: str) -> bool:
    t = canon(label)
    return "crosswalk" in t


def _is_vehicle(label: str) -> bool:
    return canon(label) in VEHICLE


def box_features(box, width, height, context_boxes):
    """Feature vector for one vehicle box given same-frame context boxes.

    context_boxes: list of (label, box)
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    wn, hn = w / max(width, 1), h / max(height, 1)
    edge = min(cx, 1.0 - cx)
    aspect = w / h
    bottom = y2 / max(height, 1)

    sidewalk_overlap = 0.0
    crosswalk_overlap = 0.0
    for lab, cb in context_boxes:
        ov = inter(box, cb) / area(box)
        if _is_sidewalk(lab):
            sidewalk_overlap = max(sidewalk_overlap, ov)
        if _is_crosswalk(lab):
            crosswalk_overlap = max(crosswalk_overlap, ov)

    return np.array(
        [cx, cy, wn, hn, edge, aspect, bottom, sidewalk_overlap, crosswalk_overlap],
        dtype=np.float64,
    )


def fit_logistic(X, y, l2=1.0, steps=800, lr=0.2):
    """Tiny L2 logistic regression (no sklearn dependency). y in {0,1}, 1=moving."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std
    # add bias
    Xb = np.concatenate([Xn, np.ones((len(Xn), 1))], axis=1)
    w = np.zeros(Xb.shape[1], dtype=np.float64)
    for _ in range(steps):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / len(y)
        grad[:-1] += l2 * w[:-1] / len(y)
        w -= lr * grad
    return {"w": w, "mean": mean, "std": std}


def predict_proba(model, X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    Xn = (X - model["mean"]) / model["std"]
    Xb = np.concatenate([Xn, np.ones((len(Xn), 1))], axis=1)
    z = Xb @ model["w"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_from_gt(train_items):
    X, y = [], []
    for item in train_items:
        w, h = float(item["width"]), float(item["height"])
        context = [(b["label"], b["box"]) for b in item["boxes"]]
        for b in item["boxes"]:
            lab = canon(b["label"])
            if lab not in VEHICLE:
                continue
            X.append(box_features(b["box"], w, h, context))
            y.append(1.0 if lab == MOVING else 0.0)
    if not X:
        raise SystemExit("No vehicle boxes in training set")
    return fit_logistic(X, y)


def parse_native_index(name: str):
    m = re.search(r"_frame_(\d+)", name)
    return int(m.group(1)) if m else None


def load_csv_detections(csv_path: Path):
    """Return list of {image_name, width?, detections, native_index}."""
    rows = []
    with csv_path.open(newline="") as f:
        for rec in csv.DictReader(f):
            cats = json.loads(rec.get("detected_categories") or "[]")
            boxes = json.loads(rec.get("detection_boxes") or "[]")
            scores = json.loads(rec.get("detection_confidences") or "[]")
            phrases = json.loads(rec.get("detected_phrases") or "[]")
            dets = []
            for i, (c, b) in enumerate(zip(cats, boxes)):
                dets.append(
                    {
                        "category": c,
                        "box": [float(v) for v in b],
                        "score": float(scores[i]) if i < len(scores) else 0.0,
                        "phrase": phrases[i] if i < len(phrases) else "",
                    }
                )
            rows.append(
                {
                    "image_name": rec["image_name"],
                    "native_index": parse_native_index(rec["image_name"]),
                    "detections": dets,
                    "record": rec,
                }
            )
    return rows


def neighbor_displacement(box, native_idx, by_index, radius=NEIGHBOR_RADIUS):
    """Max center displacement to best-IoU vehicle in neighboring frames."""
    if native_idx is None:
        return None
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    best = None
    for di in range(1, radius + 1):
        for nb in (native_idx - di, native_idx + di):
            for other in by_index.get(nb, []):
                if not _is_vehicle(other["category"]):
                    continue
                if iou(box, other["box"]) < 0.1 and inter(box, other["box"]) <= 0:
                    # still allow center distance if same-ish location
                    pass
                ocx = 0.5 * (other["box"][0] + other["box"][2])
                ocy = 0.5 * (other["box"][1] + other["box"][3])
                dist = math.hypot(cx - ocx, cy - ocy)
                # prefer overlapping / nearby boxes
                ov = iou(box, other["box"])
                score = dist - 50.0 * ov
                if best is None or score < best[0]:
                    best = (score, dist)
    return None if best is None else best[1]


def classify_box(box, width, height, context_boxes, model, displacement_px=None):
    feats = box_features(box, width, height, context_boxes)
    p_moving = float(predict_proba(model, feats)[0])
    # High motion boosts moving; low motion does not force parked.
    if displacement_px is not None and displacement_px >= MOTION_BOOST_PX:
        p_moving = max(p_moving, 0.65 + min(0.30, (displacement_px - MOTION_BOOST_PX) / 80.0))
    label = MOVING if p_moving >= 0.5 else PARKED
    return label, p_moving


def code_for(label: str, original_category: str) -> str:
    """Preserve coded category style when present."""
    if label == MOVING:
        return MOVING_CODE if ("_" in str(original_category) or original_category[:1].isdigit()) else MOVING
    return PARKED_CODE if ("_" in str(original_category) or original_category[:1].isdigit()) else PARKED


def phrase_for(label: str) -> str:
    return "vehicle in travel lane" if label == MOVING else "parked vehicle at curb"


def reclassify_detections(detections, width, height, model, displacement_by_i=None):
    context = [(d["category"], d["box"]) for d in detections]
    out = []
    for i, d in enumerate(detections):
        d2 = dict(d)
        if _is_vehicle(d["category"]):
            disp = None if displacement_by_i is None else displacement_by_i.get(i)
            new_lab, p = classify_box(d["box"], width, height, context, model, disp)
            d2["category"] = code_for(new_lab, d["category"])
            d2["phrase"] = phrase_for(new_lab)
            d2["vehicle_p_moving"] = p
            d2["vehicle_prev_category"] = d["category"]
        out.append(d2)
    return out


def reclassify_pred_json(pred_path: Path, gt_items, model, out_path: Path):
    preds = json.loads(pred_path.read_text())
    gt_by_id = {it["frame_id"]: it for it in gt_items}
    out = []
    for rec in preds:
        item = gt_by_id.get(rec["frame_id"])
        w = float(item["width"]) if item else 1920.0
        h = float(item["height"]) if item else 1080.0
        dets = reclassify_detections(rec.get("detections") or [], w, h, model)
        # also reclassify candidates if present
        cands = rec.get("candidates")
        new_cands = None
        if cands:
            new_cands = reclassify_detections(cands, w, h, model)
        row = dict(rec)
        row["detections"] = dets
        if new_cands is not None:
            row["candidates"] = new_cands
        out.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    return out


def rewrite_csv_row(rec, detections):
    cats, boxes, scores, phrases = [], [], [], []
    for d in detections:
        cats.append(d["category"])
        boxes.append(d["box"])
        scores.append(d.get("score", 0.0))
        phrases.append(d.get("phrase") or "")
    rec = dict(rec)
    rec["detected_categories"] = json.dumps(cats)
    rec["detection_boxes"] = json.dumps(boxes)
    rec["detection_confidences"] = json.dumps(scores)
    rec["detected_phrases"] = json.dumps(phrases)
    # refresh aggregates
    counts = defaultdict(int)
    for c in cats:
        counts[c] += 1
    rec["categories_present"] = json.dumps(sorted(counts.keys()))
    rec["category_counts"] = json.dumps(dict(counts))
    rec["total_categories_detected"] = str(len(counts))
    rec["total_objects_detected"] = str(len(cats))
    return rec


def process_csv(csv_path: Path, model, inplace=False, frames_root=None):
    rows = load_csv_detections(csv_path)
    by_index = defaultdict(list)
    for r in rows:
        if r["native_index"] is not None:
            for d in r["detections"]:
                if _is_vehicle(d["category"]):
                    by_index[r["native_index"]].append(d)

    # infer size from first box extents / default
    width, height = 1920.0, 1080.0
    out_records = []
    fieldnames = None
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        original = list(reader)

    for r, rec in zip(rows, original):
        disp_map = {}
        for i, d in enumerate(r["detections"]):
            if _is_vehicle(d["category"]):
                disp_map[i] = neighbor_displacement(d["box"], r["native_index"], by_index)
        dets = reclassify_detections(r["detections"], width, height, model, disp_map)
        out_records.append(rewrite_csv_row(rec, dets))

    dest = csv_path if inplace else csv_path.with_name(csv_path.stem + "_vehicle_reclass.csv")
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_records)
    return dest


def save_model(model, path: Path):
    path.write_text(
        json.dumps(
            {
                "w": model["w"].tolist(),
                "mean": model["mean"].tolist(),
                "std": model["std"].tolist(),
            }
        )
    )


def load_model(path: Path):
    d = json.loads(path.read_text())
    return {
        "w": np.array(d["w"], dtype=np.float64),
        "mean": np.array(d["mean"], dtype=np.float64),
        "std": np.array(d["std"], dtype=np.float64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, default=Path(__file__).resolve().parent.parent / "human_validation/finetune/data/train.json")
    parser.add_argument("--model-out", type=Path, default=Path(__file__).resolve().parent / "vehicle_classifier.json")
    parser.add_argument("--model-in", type=Path, default=None)
    parser.add_argument("--pred-json", type=Path, default=None)
    parser.add_argument("--gt-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--csv-root", type=Path, default=None)
    parser.add_argument("--conditions", nargs="*", default=[])
    parser.add_argument("--participants", nargs="*", default=[])
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()

    if args.model_in and args.model_in.exists():
        model = load_model(args.model_in)
    else:
        train_items = json.loads(args.train_json.read_text())
        model = train_from_gt(train_items)
        save_model(model, args.model_out)
        print("Wrote classifier:", args.model_out)

    if args.pred_json:
        gt_items = json.loads(args.gt_json.read_text()) if args.gt_json else []
        out = args.out or args.pred_json.with_name(args.pred_json.stem + "_reclassified.json")
        reclassify_pred_json(args.pred_json, gt_items, model, out)
        print("Wrote:", out)

    if args.csv_root:
        conds = set(args.conditions) if args.conditions else None
        parts = set(args.participants) if args.participants else None
        n = 0
        for csv_path in sorted(args.csv_root.rglob("groundingdino_scene_analysis_full_frames.csv")):
            cond = csv_path.parent.parent.name
            part = csv_path.parent.name
            if conds is not None and cond not in conds:
                continue
            if parts is not None and part not in parts:
                continue
            dest = process_csv(csv_path, model, inplace=args.inplace)
            print("CSV:", dest)
            n += 1
        print("Processed", n, "CSVs")


if __name__ == "__main__":
    main()
