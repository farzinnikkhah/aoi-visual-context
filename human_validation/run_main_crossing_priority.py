#!/usr/bin/env python3
"""Rerun GroundingDINO on the 160 main frames with crossing-signal priority.

A crossing-signal box wins an overlap even when its score is lower.
Other classes still use ordinary score order. Miscellaneous infrastructure
is not prioritized, because those detections are mostly poles.
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
import types
from collections import Counter
from pathlib import Path

stub = types.ModuleType("transformers.dependency_versions_check")
stub.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = stub

import torch
from PIL import Image
import transformers.utils.hub as hf_hub
from transformers import AutoProcessor, GroundingDinoForObjectDetection

# This model has no chat-template folder. A newer transformers build treats that 404 as fatal.
hf_hub.list_repo_templates = lambda *args, **kwargs: []

from box_merge import detections_from_grounding_lists, grounding_lists_from_detections, merge_same_category_overlaps
from run_full_frame_groundingdino import (
    CAT_THRESH,
    DEFAULT_BOX_THRESH,
    MODEL_ID,
    NMS_IOU_THRESH,
    category_to_phrases,
    make_prompt,
)

MAIN = data_path("human_validation/main")
OUT = MAIN / "priority_nms"
PRIORITY = {"6_Crossing Signal"}


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def nms_keep(boxes, scores, cats, priority=False):
    order = list(range(len(boxes)))
    if priority:
        order.sort(key=lambda i: (cats[i] in PRIORITY, scores[i]), reverse=True)
    else:
        order.sort(key=lambda i: scores[i], reverse=True)
    suppressed = set()
    keep = []
    wins = []
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j == i or j in suppressed:
                continue
            overlap = iou(boxes[i], boxes[j])
            if overlap < NMS_IOU_THRESH:
                continue
            suppressed.add(j)
            if priority and cats[i] in PRIORITY and cats[j] not in PRIORITY:
                wins.append(
                    {
                        "kept_category": cats[i],
                        "kept_score": round(float(scores[i]), 3),
                        "dropped_category": cats[j],
                        "dropped_score": round(float(scores[j]), 3),
                        "iou": round(overlap, 3),
                    }
                )
    return keep, wins


def apply_merge(boxes, scores, phrases, cats):
    detections = detections_from_grounding_lists(boxes, scores, phrases, cats)
    detections = merge_same_category_overlaps(detections, iou_threshold=0.70, containment_threshold=0.90)
    return grounding_lists_from_detections(detections)


def select(items, indexes):
    return [items[i] for i in indexes]


def count_cat(cats, name):
    return sum(cat == name for cat in cats)


def main():
    rows = list(csv.DictReader((MAIN / "manifest.csv").open()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "frames:", len(rows), flush=True)
    if device != "cuda":
        raise SystemExit("A GPU is required for this rerun.")

    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID, local_files_only=True).to(device)
    model.eval()
    prompts = [(cat, make_prompt(phrases)) for cat, phrases in category_to_phrases.items()]

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    signal = "6_Crossing Signal"
    for index, row in enumerate(rows, start=1):
        image = Image.open(row["source_image"]).convert("RGB")
        boxes, scores, phrases, cats = [], [], [], []
        for cat_name, prompt in prompts:
            threshold = CAT_THRESH.get(cat_name, DEFAULT_BOX_THRESH)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            target = torch.tensor([image.size[::-1]], device=device)
            with torch.no_grad():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=target,
            )[0]
            keep = result["scores"] >= threshold
            if int(keep.sum().item()) == 0:
                continue
            label_key = "text_labels" if "text_labels" in result else "labels"
            kept_labels = [label for label, flag in zip(result[label_key], keep.tolist()) if flag]
            boxes.extend(result["boxes"][keep].tolist())
            scores.extend(result["scores"][keep].tolist())
            phrases.extend(kept_labels)
            cats.extend([cat_name] * len(kept_labels))
            del inputs, outputs, result

        standard_idx, _ = nms_keep(boxes, scores, cats, priority=False)
        priority_idx, wins = nms_keep(boxes, scores, cats, priority=True)
        std_boxes, std_scores, std_phrases, std_cats = apply_merge(
            select(boxes, standard_idx),
            select(scores, standard_idx),
            select(phrases, standard_idx),
            select(cats, standard_idx),
        )
        pri_boxes, pri_scores, pri_phrases, pri_cats = apply_merge(
            select(boxes, priority_idx),
            select(scores, priority_idx),
            select(phrases, priority_idx),
            select(cats, priority_idx),
        )
        record = {
            "frame_id": row["frame_id"],
            "raw_crossing_signal": count_cat(cats, signal),
            "standard_crossing_signal": count_cat(std_cats, signal),
            "priority_crossing_signal": count_cat(pri_cats, signal),
            "wins": wins,
            "detections": [
                {
                    "category": cat,
                    "phrase": phrase,
                    "score": round(float(score), 3),
                    "box": [round(float(v), 1) for v in box],
                }
                for box, score, phrase, cat in zip(pri_boxes, pri_scores, pri_phrases, pri_cats)
            ],
        }
        summary.append(record)
        print(
            "%03d %s raw=%d score=%d priority=%d wins=%d"
            % (
                index,
                row["frame_id"],
                record["raw_crossing_signal"],
                record["standard_crossing_signal"],
                record["priority_crossing_signal"],
                len(wins),
            ),
            flush=True,
        )
        image.close()

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        "TOTAL crossing signal raw=%d score_nms=%d priority=%d wins=%d"
        % (
            sum(r["raw_crossing_signal"] for r in summary),
            sum(r["standard_crossing_signal"] for r in summary),
            sum(r["priority_crossing_signal"] for r in summary),
            sum(len(r["wins"]) for r in summary),
        )
    )
    dropped = Counter(win["dropped_category"] for record in summary for win in record["wins"])
    print("Dropped by a crossing signal", dict(dropped))
    print("Wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
