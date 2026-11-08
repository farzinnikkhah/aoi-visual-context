#!/usr/bin/env python3
"""Draw priority-NMS comparisons and write the per-class table."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from make_pilot_comparisons import (
    PILOT,
    canon,
    compose,
    load_human,
    panel,
    legend_strip,
)

SUMMARY = PILOT / "priority_nms" / "summary.json"
OUT = PILOT / "comparisons_priority"
CLASSES = [
    "moving vehicle",
    "parked vehicle",
    "sidewalk pavement",
    "traffic signs",
    "crosswalk",
    "crossing signal",
    "general light features",
    "pedestrians",
    "cyclists",
    "storefront",
    "buildings",
    "art and banners",
    "natural elements",
    "miscellaneous infrastructure",
]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def is_match(a, b):
    overlap = inter(a, b)
    aa, bb = area(a), area(b)
    union = aa + bb - overlap
    over_a = overlap / aa if aa else 0.0
    over_b = overlap / bb if bb else 0.0
    iou = overlap / union if union else 0.0
    return over_a > 0.5 or over_b > 0.5 or iou > 0.5


def prf(pred_hit, pred_n, gt_hit, gt_n):
    prec = pred_hit / float(pred_n) if pred_n else 0.0
    rec = gt_hit / float(gt_n) if gt_n else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main():
    summary = json.loads(SUMMARY.read_text())
    by_id = {row["frame_id"]: row for row in summary}
    manifest = list(csv.DictReader((PILOT / "manifest.csv").open()))
    OUT.mkdir(parents=True, exist_ok=True)

    per = defaultdict(lambda: [0, 0, 0, 0])
    names = []
    for index, row in enumerate(manifest, start=1):
        human = load_human(row["frame_id"])
        dets = by_id[row["frame_id"]]["detections"]
        model = [(canon(item["category"]), item["box"], item["score"]) for item in dets]
        image = compose(
            row["frame_id"],
            row["stratum"],
            row["export_image"],
            human,
            model,
            right_title="GroundingDINO priority",
            match_text="Priority: crossing signal, art, misc. infrastructure",
        )
        # Replace the IoU header note by redrawing is unnecessary; compose already
        # reports class-aware IoU matches. The saved image is still the priority boxes.
        name = "%02d_%s.jpg" % (index, row["frame_id"])
        image.save(OUT / name, quality=85, optimize=True)
        names.append(name)

        gt = [(label, box) for label, box, _ in human]
        pred = [(label, box) for label, box, _ in model]
        gt_hit = [False] * len(gt)
        pred_hit = [False] * len(pred)
        for i, (gl, gb) in enumerate(gt):
            for j, (pl, pb) in enumerate(pred):
                if gl == pl and is_match(gb, pb):
                    gt_hit[i] = True
                    pred_hit[j] = True
        labels = set([g[0] for g in gt] + [p[0] for p in pred])
        for lab in labels:
            gi = [k for k, (label, _) in enumerate(gt) if label == lab]
            pj = [k for k, (label, _) in enumerate(pred) if label == lab]
            per[lab][0] += len(gi)
            per[lab][1] += len(pj)
            per[lab][2] += sum(gt_hit[k] for k in gi)
            per[lab][3] += sum(pred_hit[k] for k in pj)

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Priority comparisons</title>",
        "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}",
        "h1{font-size:18px;padding:16px 20px;margin:0}img{width:100%;height:auto;display:block;margin:0 0 28px}</style>",
        "</head><body><h1>Human on the left. GroundingDINO with priority for crossing signal, art and banners, and miscellaneous infrastructure on the right.</h1>",
    ]
    for name in names:
        html.append("<img src='%s' alt='%s'>" % (name, name))
    html.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(html))

    table_path = PILOT / "priority_nms" / "per_class.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "human", "human_matched", "model", "model_matched", "precision", "recall", "f1"])
        order = CLASSES + [lab for lab in per if lab not in CLASSES]
        totals = [0, 0, 0, 0]
        for lab in order:
            hn, mn, ht, mt = per[lab]
            if hn == 0 and mn == 0:
                continue
            totals[0] += hn
            totals[1] += mn
            totals[2] += ht
            totals[3] += mt
            prec, rec, f1 = prf(mt, mn, ht, hn)
            writer.writerow([lab, hn, ht, mn, mt, "%.4f" % prec, "%.4f" % rec, "%.4f" % f1])
            print("%-32s human %3d matched %3d   model %3d matched %3d   P %5.1f R %5.1f F1 %5.1f" % (
                lab, hn, ht, mn, mt, 100 * prec, 100 * rec, 100 * f1))
        prec, rec, f1 = prf(totals[3], totals[1], totals[2], totals[0])
        writer.writerow(["all", totals[0], totals[2], totals[1], totals[3], "%.4f" % prec, "%.4f" % rec, "%.4f" % f1])
        print("ALL human %d matched %d model %d matched %d P %.1f R %.1f F1 %.1f" % (
            totals[0], totals[2], totals[1], totals[3], 100 * prec, 100 * rec, 100 * f1))
    print("images", OUT)
    print("table", table_path)


if __name__ == "__main__":
    main()
