#!/usr/bin/env python3
"""Moving vs parked vehicle metrics and pairwise confusion."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

VEHICLE = ("moving vehicle", "parked vehicle")


def canon(label: str) -> str:
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


def is_match(a, b) -> bool:
    overlap = inter(a, b)
    aa, bb = area(a), area(b)
    union = aa + bb - overlap
    over_a = overlap / aa if aa else 0.0
    over_b = overlap / bb if bb else 0.0
    iou = overlap / union if union else 0.0
    return over_a > 0.5 or over_b > 0.5 or iou > 0.5


def iou(a, b) -> float:
    overlap = inter(a, b)
    union = area(a) + area(b) - overlap
    return overlap / union if union else 0.0


def prf(matched_pred, n_pred, matched_gt, n_gt):
    prec = matched_pred / float(n_pred) if n_pred else 0.0
    rec = matched_gt / float(n_gt) if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def score_vehicle(items, predictions):
    """Per-class same-class P/R plus pairwise vehicle confusion.

    Returns:
      per: label -> [n_gt, n_pred, gt_matched_same_class, pred_matched_same_class]
      confusion: {(gt_lab, pred_lab): count} for GT vehicle boxes matched to any vehicle pred
      unmatched_gt / unmatched_pred: GT/pred vehicle boxes with no same-class match
    """
    per = defaultdict(lambda: [0, 0, 0, 0])
    confusion = defaultdict(int)
    unmatched_gt = defaultdict(int)
    unmatched_pred = defaultdict(int)

    for item in items:
        gt = [(canon(b["label"]), b["box"]) for b in item["boxes"] if canon(b["label"]) in VEHICLE]
        pred = [
            (canon(d["category"]), d["box"])
            for d in predictions.get(item["frame_id"], [])
            if canon(d.get("category", "")) in VEHICLE
        ]

        # Same-class matching for P/R/F1
        gt_same = [False] * len(gt)
        pred_same = [False] * len(pred)
        for i, (gl, gb) in enumerate(gt):
            for j, (pl, pb) in enumerate(pred):
                if pred_same[j] or gl != pl:
                    continue
                if is_match(gb, pb):
                    gt_same[i] = True
                    pred_same[j] = True
                    break

        # Cross-class vehicle matching for pairwise confusion (greedy best IoU)
        pred_taken = [False] * len(pred)
        for i, (gl, gb) in enumerate(gt):
            best_j, best_score = -1, -1.0
            for j, (pl, pb) in enumerate(pred):
                if pred_taken[j] or not is_match(gb, pb):
                    continue
                score = iou(gb, pb) + (1.0 if gl == pl else 0.0)
                if score > best_score:
                    best_j, best_score = j, score
            if best_j >= 0:
                pred_taken[best_j] = True
                confusion[(gl, pred[best_j][0])] += 1

        for lab in VEHICLE:
            gi = [k for k, (label, _) in enumerate(gt) if label == lab]
            pj = [k for k, (label, _) in enumerate(pred) if label == lab]
            per[lab][0] += len(gi)
            per[lab][1] += len(pj)
            per[lab][2] += sum(gt_same[k] for k in gi)
            per[lab][3] += sum(pred_same[k] for k in pj)
            unmatched_gt[lab] += sum(1 for k in gi if not gt_same[k])
            unmatched_pred[lab] += sum(1 for k in pj if not pred_same[k])

    return per, confusion, unmatched_gt, unmatched_pred


def format_report(title, per, confusion, unmatched_gt, unmatched_pred) -> str:
    lines = ["== %s" % title]
    for lab in VEHICLE:
        hn, mn, ht, mt = per[lab]
        prec, rec, f1 = prf(mt, mn, ht, hn)
        lines.append(
            "%-16s human %3d matched %3d  model %3d matched %3d  P %5.1f R %5.1f F1 %5.1f"
            % (lab, hn, ht, mn, mt, 100 * prec, 100 * rec, 100 * f1)
        )
    lines.append("pairwise (GT -> pred) among matched vehicle boxes:")
    for gl in VEHICLE:
        for pl in VEHICLE:
            lines.append("  %s -> %s: %d" % (gl, pl, confusion.get((gl, pl), 0)))
    lines.append(
        "unmatched GT moving=%d parked=%d | unmatched pred moving=%d parked=%d"
        % (
            unmatched_gt["moving vehicle"],
            unmatched_gt["parked vehicle"],
            unmatched_pred["moving vehicle"],
            unmatched_pred["parked vehicle"],
        )
    )
    return "\n".join(lines)


def load_pred_json(path: Path):
    return {r["frame_id"]: r["detections"] for r in json.loads(path.read_text())}


def preds_from_records(records):
    """records: list of {frame_id, detections}."""
    return {r["frame_id"]: r["detections"] for r in records}
