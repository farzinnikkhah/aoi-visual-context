#!/usr/bin/env python3
"""Score zero-shot and fine-tuned boxes against the human labels.

Match rule: same class, and intersection over either box > 0.5 or IoU > 0.5.
Also scores the saved production pipeline on the same held-out frames.

Usage: python3 score_run.py runs/<timestamp>
"""
import sys
from pathlib import Path as _BootPath
_repo = _BootPath(__file__).resolve()
for _parent in [_repo.parent, *_repo.parents]:
    if (_parent / "aoi_paths.py").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
from aoi_paths import data_path, study_path


import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DINO = data_path("GroundingDINO_full_frame_output")
MAIN = data_path("human_validation/main")
PILOT = data_path("human_validation/pilot")
CLASSES = json.loads((DATA / "classes.json").read_text())


def canon(label):
    text = " ".join(str(label).replace("_", " ").split()).lower()
    parts = text.split(" ", 1)
    if parts and parts[0].isdigit() and len(parts) == 2:
        text = parts[1]
    return {"building": "buildings"}.get(text, text)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def is_match(a, b):
    overlap = inter(a, b)
    aa, bb = area(a), area(b)
    union = aa + bb - overlap
    over_a = overlap / aa if aa else 0.0
    over_b = overlap / bb if bb else 0.0
    iou = overlap / union if union else 0.0
    return over_a > 0.5 or over_b > 0.5 or iou > 0.5


def prf(mt, mn, ht, hn):
    prec = mt / float(mn) if mn else 0.0
    rec = ht / float(hn) if hn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def score(items, predictions):
    per = defaultdict(lambda: [0, 0, 0, 0])
    for item in items:
        gt = [(b["label"], b["box"]) for b in item["boxes"]]
        pred = [(canon(d["category"]), d["box"]) for d in predictions.get(item["frame_id"], [])]
        gt_hit = [False] * len(gt)
        pred_hit = [False] * len(pred)
        for i, (gl, gb) in enumerate(gt):
            for j, (pl, pb) in enumerate(pred):
                if gl == pl and is_match(gb, pb):
                    gt_hit[i] = True
                    pred_hit[j] = True
        for lab in set([g[0] for g in gt] + [p[0] for p in pred]):
            gi = [k for k, (label, _) in enumerate(gt) if label == lab]
            pj = [k for k, (label, _) in enumerate(pred) if label == lab]
            per[lab][0] += len(gi)
            per[lab][1] += len(pj)
            per[lab][2] += sum(gt_hit[k] for k in gi)
            per[lab][3] += sum(pred_hit[k] for k in pj)
    return per


def saved_pipeline(items, manifest_path):
    rows = {r["frame_id"]: r for r in csv.DictReader(manifest_path.open())}
    out = {}
    for item in items:
        row = rows[item["frame_id"]]
        csv_path = DINO / row["stratum"] / row["participant"] / "groundingdino_scene_analysis_full_frames.csv"
        image_name = Path(row["source_image"]).name
        with csv_path.open(newline="") as handle:
            for record in csv.DictReader(handle):
                if record["image_name"] != image_name:
                    continue
                cats = json.loads(record["detected_categories"] or "[]")
                boxes = json.loads(record["detection_boxes"] or "[]")
                out[item["frame_id"]] = [{"category": c, "box": [float(v) for v in b]} for c, b in zip(cats, boxes)]
                break
    return out


def table(title, per):
    print("== %s" % title)
    totals = [0, 0, 0, 0]
    for lab in CLASSES:
        hn, mn, ht, mt = per[lab]
        for i, v in enumerate((hn, mn, ht, mt)):
            totals[i] += v
        prec, rec, f1 = prf(mt, mn, ht, hn)
        print("%-30s human %4d matched %4d  model %4d matched %4d  P %5.1f R %5.1f F1 %5.1f" % (
            lab, hn, ht, mn, mt, 100 * prec, 100 * rec, 100 * f1))
    prec, rec, f1 = prf(totals[3], totals[1], totals[2], totals[0])
    print("%-30s human %4d matched %4d  model %4d matched %4d  P %5.1f R %5.1f F1 %5.1f" % (
        "ALL", totals[0], totals[2], totals[1], totals[3], 100 * prec, 100 * rec, 100 * f1))
    return totals


def load_pred(path):
    return {r["frame_id"]: r["detections"] for r in json.loads(path.read_text())}


def iou(a, b):
    overlap = inter(a, b)
    union = area(a) + area(b) - overlap
    return overlap / union if union else 0.0


def rebuild(path, thresh):
    """Apply a different score cutoff to the saved candidates, then NMS and same-class merge."""
    out = {}
    for record in json.loads(path.read_text()):
        kept = sorted((c for c in record.get("candidates", []) if c["score"] >= thresh), key=lambda c: -c["score"])
        final = []
        for cand in kept:
            if any(iou(cand["box"], f["box"]) >= 0.5 for f in final):
                continue
            final.append(cand)
        merged = []
        for cand in final:
            absorbed = False
            for f in merged:
                if f["category"] != cand["category"]:
                    continue
                overlap = inter(cand["box"], f["box"])
                if iou(cand["box"], f["box"]) >= 0.70 or overlap / max(area(cand["box"]), 1e-6) >= 0.90:
                    absorbed = True
                    break
            if not absorbed:
                merged.append(cand)
        out[record["frame_id"]] = merged
    return out


def main():
    run = Path(sys.argv[1])
    test = json.loads((DATA / "test.json").read_text())
    pilot = json.loads((DATA / "pilot.json").read_text())
    rows = {}
    rows["saved pipeline, test"] = score(test, saved_pipeline(test, MAIN / "manifest.csv"))
    rows["zero-shot same prompt, test"] = score(test, load_pred(run / "zero_shot_test.json"))
    rows["fine-tuned, test"] = score(test, load_pred(run / "finetuned_test.json"))
    rows["saved pipeline, pilot"] = score(pilot, saved_pipeline(pilot, PILOT / "manifest.csv"))
    rows["zero-shot same prompt, pilot"] = score(pilot, load_pred(run / "zero_shot_pilot.json"))
    rows["fine-tuned, pilot"] = score(pilot, load_pred(run / "finetuned_pilot.json"))
    sample = json.loads((run / "finetuned_test.json").read_text())
    if sample and "candidates" in sample[0]:
        for thresh in (0.20, 0.25):
            rows["fine-tuned @%.2f, test" % thresh] = score(test, rebuild(run / "finetuned_test.json", thresh))
            rows["fine-tuned @%.2f, pilot" % thresh] = score(pilot, rebuild(run / "finetuned_pilot.json", thresh))
    summary = {}
    for title, per in rows.items():
        totals = table(title, per)
        prec, rec, f1 = prf(totals[3], totals[1], totals[2], totals[0])
        summary[title] = {"P": round(100 * prec, 1), "R": round(100 * rec, 1), "F1": round(100 * f1, 1)}
        print()
    print(json.dumps(summary, indent=1))
    out = run / "scores.json"
    out.write_text(json.dumps({k: dict(v) for k, v in rows.items()}, indent=1))


if __name__ == "__main__":
    main()
