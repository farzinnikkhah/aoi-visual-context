#!/usr/bin/env python3
"""Two-stage moving vs parked vehicle classification.

Stage 1: treat all vehicle detections as a single vehicle class and merge
overlapping boxes (detection, not subtype).
Stage 2: classify each merged vehicle as travel-lane (moving) vs curb
(parked) using road/curb geometry + short video tracks.

Protocol: position, not speed. High *residual* track motion (ego-compensated
over ±0.5–1 s) only boosts moving; low motion never forces parked.
"""
from __future__ import annotations

import sys
from pathlib import Path as _BootPath
_repo = _BootPath(__file__).resolve()
for _parent in [_repo.parent, *_repo.parents]:
    if (_parent / "aoi_paths.py").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
from aoi_paths import data_path, study_path

import argparse
import json
import math
from pathlib import Path

import numpy as np

from classify_vehicle_motion import (
    MOVING,
    PARKED,
    code_for,
    fit_logistic,
    load_model,
    phrase_for,
    predict_proba,
    save_model,
)
from vehicle_metrics import VEHICLE, area, canon, inter, iou
from video_track_motion import (
    HALF_WINDOW,
    motion_moving_boost,
    resolve_csv_path,
    track_motion_for_boxes,
)


def _is_sidewalk(label: str) -> bool:
    return "sidewalk" in canon(label)


def _is_crosswalk(label: str) -> bool:
    return "crosswalk" in canon(label)


def _is_vehicle(label: str) -> bool:
    return canon(label) in VEHICLE


def merge_vehicle_boxes(detections, iou_thr=0.5, contain_thr=0.9):
    """Merge overlapping vehicle boxes into unified vehicles (ignore subtype)."""
    vehs = [dict(d) for d in detections if _is_vehicle(d.get("category", ""))]
    other = [dict(d) for d in detections if not _is_vehicle(d.get("category", ""))]
    vehs = sorted(vehs, key=lambda d: -float(d.get("score", 0.0)))
    merged = []
    for cand in vehs:
        absorbed = False
        for keep in merged:
            ov = inter(cand["box"], keep["box"])
            if (
                iou(cand["box"], keep["box"]) >= iou_thr
                or ov / max(area(cand["box"]), 1e-6) >= contain_thr
                or ov / max(area(keep["box"]), 1e-6) >= contain_thr
            ):
                # expand keep box to union
                x1 = min(keep["box"][0], cand["box"][0])
                y1 = min(keep["box"][1], cand["box"][1])
                x2 = max(keep["box"][2], cand["box"][2])
                y2 = max(keep["box"][3], cand["box"][3])
                keep["box"] = [x1, y1, x2, y2]
                keep["score"] = max(float(keep.get("score", 0)), float(cand.get("score", 0)))
                absorbed = True
                break
        if not absorbed:
            merged.append(
                {
                    "category": "vehicle",
                    "box": list(cand["box"]),
                    "score": float(cand.get("score", 0.0)),
                    "phrase": "vehicle",
                }
            )
    return merged, other


# Lateral curb band as a fraction of frame width. Downtown parked cars often
# sit out to ~0.25–0.30 from the image edge; 0.22 was too narrow and let curb
# cars fall into the logistic as "moving" (especially with crosswalk overlap).
CURB_WIDTH = 0.30
HARD_CURB = 0.30
HARD_EDGE = 0.15
NEAR_SIDE_EDGE = 0.25
NEAR_SIDE_LANE = 0.35
# Softened after FT+video tune (reports/ft_video_improvement_tune.json):
# strong residual motion may relax these gates; very strong motion may
# override hard curb. Tuned on test with pilot non-regression.
VIDEO_MIN_LANE = 0.20
VIDEO_MAX_CURB = 0.32
VIDEO_MIN_EDGE = 0.10
VIDEO_MIN_PGEO = 0.28
STRONG_NET_PX = 110.0
STRONG_OVERRIDE_HARD = True


def road_curb_features(box, width, height, context_boxes):
    """Geometry features aligned with travel-lane vs curb protocol.

    Size features (box width/height/aspect) are intentionally omitted — they
    leaked "large nearby curb car → moving". Raw crosswalk overlap is replaced
    by crosswalk∩corridor so a curb car sitting on a painted crosswalk end is
    not pushed to moving.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    edge = min(cx, 1.0 - cx)
    bottom = y2 / max(height, 1)

    # Travel corridor: central band, stronger in lower/mid image (road ahead)
    corridor = math.exp(-0.5 * ((cx - 0.5) / 0.18) ** 2)
    corridor *= 0.5 + 0.5 * min(max(cy, 0.0), 1.0)

    # Curb strips: near left/right edges
    curb = max(0.0, 1.0 - edge / CURB_WIDTH)

    sidewalk_overlap = 0.0
    crosswalk_overlap = 0.0
    min_sidewalk_dist = 1.0
    for lab, cb in context_boxes:
        ov = inter(box, cb) / area(box)
        if _is_sidewalk(lab):
            sidewalk_overlap = max(sidewalk_overlap, ov)
            scx = 0.5 * (cb[0] + cb[2]) / max(width, 1)
            scy = 0.5 * (cb[1] + cb[3]) / max(height, 1)
            min_sidewalk_dist = min(min_sidewalk_dist, math.hypot(cx - scx, cy - scy))
        if _is_crosswalk(lab):
            crosswalk_overlap = max(crosswalk_overlap, ov)

    # Crosswalk only counts when the box is also in the travel corridor
    crosswalk_lane = crosswalk_overlap * corridor

    lane_score = corridor * (1.0 - 0.7 * sidewalk_overlap) * (0.4 + 0.6 * (1.0 - curb))
    curb_score = 0.6 * curb + 0.4 * sidewalk_overlap

    return np.array(
        [
            cx,
            cy,
            edge,
            bottom,
            sidewalk_overlap,
            crosswalk_lane,
            min_sidewalk_dist,
            corridor,
            curb,
            lane_score,
            curb_score,
        ],
        dtype=np.float64,
    )


def _geom_scalars(feats):
    """Unpack named scalars from road_curb_features vector."""
    return {
        "cx": float(feats[0]),
        "cy": float(feats[1]),
        "edge": float(feats[2]),
        "bottom": float(feats[3]),
        "sidewalk": float(feats[4]),
        "crosswalk_lane": float(feats[5]),
        "sw_dist": float(feats[6]),
        "corridor": float(feats[7]),
        "curb": float(feats[8]),
        "lane": float(feats[9]),
        "curb_score": float(feats[10]),
    }


def train_from_gt(train_items):
    X, y = [], []
    for item in train_items:
        w, h = float(item["width"]), float(item["height"])
        context = [(b["label"], b["box"]) for b in item["boxes"]]
        for b in item["boxes"]:
            lab = canon(b["label"])
            if lab not in VEHICLE:
                continue
            X.append(road_curb_features(b["box"], w, h, context))
            y.append(1.0 if lab == MOVING else 0.0)
    if not X:
        raise SystemExit("No vehicle boxes in training set")
    return fit_logistic(X, y, l2=1.0, steps=1500, lr=0.12)


def classify_box(box, width, height, context_boxes, model, track_stats=None):
    """Position-first moving vs parked.

    Hard curb / near-edge geometry usually forces parked. Near-side boxes
    with weak lane evidence default to parked unless residual track motion
    is strong. Video boost gates relax when net motion is strong; very
    strong motion may override hard curb (see STRONG_* constants).
    """
    feats = road_curb_features(box, width, height, context_boxes)
    g = _geom_scalars(feats)

    floor = motion_moving_boost(track_stats, box=box)
    net = float(track_stats["net_px"]) if track_stats else 0.0
    strong = floor is not None and net >= STRONG_NET_PX
    very_strong = (
        STRONG_OVERRIDE_HARD and floor is not None and net >= 1.5 * STRONG_NET_PX
    )

    if g["curb"] >= HARD_CURB or g["edge"] <= HARD_EDGE or g["curb_score"] >= 0.35:
        if not very_strong:
            return PARKED, 0.12

    if g["edge"] < NEAR_SIDE_EDGE and g["lane"] < NEAR_SIDE_LANE:
        if not strong:
            return PARKED, 0.20 + 0.25 * g["lane"]

    p_geo = float(predict_proba(model, feats)[0])
    p_moving = p_geo

    # Soften video-admission gates when residual motion is strong
    eff_pgeo = VIDEO_MIN_PGEO - 0.15 if strong else VIDEO_MIN_PGEO
    eff_lane = VIDEO_MIN_LANE - 0.10 if strong else VIDEO_MIN_LANE
    eff_curb = VIDEO_MAX_CURB + 0.08 if strong else VIDEO_MAX_CURB
    eff_edge = VIDEO_MIN_EDGE - 0.05 if strong else VIDEO_MIN_EDGE

    if (
        floor is not None
        and p_geo >= eff_pgeo
        and g["lane"] >= eff_lane
        and g["curb"] < eff_curb
        and g["edge"] >= eff_edge
    ):
        p_moving = max(p_geo, floor)

    label = MOVING if p_moving >= 0.5 else PARKED
    return label, p_moving


def reclassify_detections(detections, width, height, model, track_stats_by_merged_i=None, coded=True):
    """Two-stage: merge vehicles, classify, splice back with non-vehicles."""
    merged_vehs, others = merge_vehicle_boxes(detections)
    context = [(d["category"], d["box"]) for d in detections]
    out_vehs = []
    sample_cat = next((d["category"] for d in detections if _is_vehicle(d["category"])), "moving vehicle")
    for i, v in enumerate(merged_vehs):
        stats = None if track_stats_by_merged_i is None else track_stats_by_merged_i.get(i)
        new_lab, p = classify_box(v["box"], width, height, context, model, stats)
        row = {
            "category": code_for(new_lab, sample_cat) if coded else new_lab,
            "box": v["box"],
            "score": v.get("score", 0.0),
            "phrase": phrase_for(new_lab),
            "vehicle_p_moving": p,
        }
        if stats:
            row["vehicle_track_path_px"] = stats.get("path_px")
            row["vehicle_track_net_px"] = stats.get("net_px")
            row["vehicle_track_n_links"] = stats.get("n_links")
        out_vehs.append(row)
    return out_vehs + others


def reclassify_pred_json(
    pred_path: Path,
    gt_items,
    model,
    out_path: Path,
    dino_root: Path = None,
    half_window: int = HALF_WINDOW,
):
    preds = json.loads(pred_path.read_text())
    gt_by_id = {it["frame_id"]: it for it in gt_items}
    default_dino = data_path("GroundingDINO_finetuned_full_frame_output")
    dino_root = dino_root or default_dino
    out = []
    n_boosted = 0
    n_tracked = 0
    for rec in preds:
        item = gt_by_id.get(rec["frame_id"])
        w = float(item["width"]) if item else 1920.0
        h = float(item["height"]) if item else 1080.0
        track_map = None
        if item and item.get("image"):
            image_path = Path(item["image"])
            if resolve_csv_path(image_path, dino_root) is not None:
                merged_vehs, _ = merge_vehicle_boxes(rec.get("detections") or [])
                stats_list = track_motion_for_boxes(
                    [v["box"] for v in merged_vehs],
                    image_path,
                    dino_root,
                    half_window=half_window,
                )
                track_map = {}
                for i, (v, stats) in enumerate(zip(merged_vehs, stats_list)):
                    if stats is None:
                        continue
                    track_map[i] = stats
                    n_tracked += 1
                    if motion_moving_boost(stats, box=v["box"]) is not None:
                        n_boosted += 1
        dets = reclassify_detections(
            rec.get("detections") or [], w, h, model, track_map, coded=False
        )
        row = dict(rec)
        row["detections"] = dets
        # candidates: geometry only (tracks applied on merged detections)
        if rec.get("candidates"):
            context = [(d["category"], d["box"]) for d in rec["candidates"]]
            new_cands = []
            for d in rec["candidates"]:
                d2 = dict(d)
                if _is_vehicle(d["category"]):
                    lab, p = classify_box(d["box"], w, h, context, model, None)
                    d2["category"] = lab
                    d2["phrase"] = phrase_for(lab)
                    d2["vehicle_p_moving"] = p
                new_cands.append(d2)
            row["candidates"] = new_cands
        out.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(
        "video tracks: vehicles_with_track=%d motion_boosted=%d half_window=%d"
        % (n_tracked, n_boosted, half_window)
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--train-json",
        type=Path,
        default=here.parent / "human_validation/finetune/data/train.json",
    )
    parser.add_argument("--model-out", type=Path, default=here / "two_stage_vehicle_classifier.json")
    parser.add_argument("--model-in", type=Path, default=None)
    parser.add_argument("--pred-json", type=Path, default=None)
    parser.add_argument("--gt-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.model_in and args.model_in.exists():
        model = load_model(args.model_in)
    else:
        train_items = json.loads(args.train_json.read_text())
        model = train_from_gt(train_items)
        save_model(model, args.model_out)
        print("Wrote", args.model_out)

    if args.pred_json:
        gt_items = json.loads(args.gt_json.read_text()) if args.gt_json else []
        out = args.out or args.pred_json.with_name(args.pred_json.stem + "_two_stage.json")
        reclassify_pred_json(args.pred_json, gt_items, model, out)
        print("Wrote", out)


if __name__ == "__main__":
    main()
