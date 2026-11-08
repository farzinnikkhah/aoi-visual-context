#!/usr/bin/env python3
"""Rerun GroundingDINO on the 24 pilot frames and compare score-NMS with priority-NMS.

Priority classes win an overlap even when their confidence is lower:
crossing signal, art and banners, miscellaneous infrastructure.
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

# The environment's huggingface-hub is newer than this transformers build expects.
stub = types.ModuleType("transformers.dependency_versions_check")
stub.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = stub

import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from box_merge import detections_from_grounding_lists, grounding_lists_from_detections, merge_same_category_overlaps
from run_full_frame_groundingdino import (
    CAT_THRESH,
    DEFAULT_BOX_THRESH,
    MODEL_ID,
    NMS_IOU_THRESH,
    category_to_phrases,
    make_prompt,
)

PILOT = data_path("human_validation/pilot")
OUT = PILOT / "priority_nms"
PRIORITY = {
    "6_Crossing Signal",
    "12_Art and Banners",
    "14_Miscellaneous Infrastructure",
}


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
                        "kept_score": scores[i],
                        "dropped_category": cats[j],
                        "dropped_score": scores[j],
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


def count_priority(cats):
    counts = Counter(cat for cat in cats if cat in PRIORITY)
    return {name: counts.get(name, 0) for name in sorted(PRIORITY)}


def main():
    rows = list(csv.DictReader((PILOT / "manifest.csv").open()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)
    if device != "cuda":
        raise SystemExit("A GPU is required for this rerun.")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()
    prompts = [(cat, make_prompt(phrases)) for cat, phrases in category_to_phrases.items()]

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(row["source_image"])
        image = Image.open(image_path).convert("RGB")
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
            "raw_priority": count_priority(cats),
            "standard_priority": count_priority(std_cats),
            "priority_nms_priority": count_priority(pri_cats),
            "standard_total": len(std_cats),
            "priority_total": len(pri_cats),
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
            "%02d %s raw=%s standard=%s priority=%s wins=%d"
            % (
                index,
                row["frame_id"],
                record["raw_priority"],
                record["standard_priority"],
                record["priority_nms_priority"],
                len(wins),
            ),
            flush=True,
        )

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    raw = Counter()
    standard = Counter()
    priority = Counter()
    for record in summary:
        raw.update(record["raw_priority"])
        standard.update(record["standard_priority"])
        priority.update(record["priority_nms_priority"])
    print("TOTAL raw before NMS", dict(raw))
    print("TOTAL score NMS", dict(standard))
    print("TOTAL priority NMS", dict(priority))
    print("Wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
