#!/usr/bin/env python3
"""Run the fine-tuned GroundingDINO on the full-frame corpus.

Uses the single-prompt 14-class setup from training, with a lower score
cutoff for storefront (0.20) and 0.30 for every other class. Writes the
same CSV schema as run_full_frame_groundingdino.py into a new output root
so the original results stay untouched.

Example:
  python -u run_full_frame_finetuned.py --model-dir /path/to/checkpoint
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
import sys
import types
from pathlib import Path

stub = types.ModuleType("transformers.dependency_versions_check")
stub.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = stub

import transformers.utils.hub as hf_hub

hf_hub.list_repo_templates = lambda *args, **kwargs: []

from box_merge import (
    detections_from_grounding_lists,
    grounding_lists_from_detections,
    merge_same_category_overlaps,
)
from run_full_frame_groundingdino import (
    ROOT,
    MERGE_CONTAINMENT_THRESH,
    MERGE_IOU_THRESH,
    NMS_IOU_THRESH,
    batched,
    iter_condition_dirs,
    iter_participant_dirs,
    parse_frame_timestamp_us,
    parse_native_frame_index,
    parse_native_frame_timestamp_us,
    safe_json,
    safety_critical_classes,
)

OUT_ROOT = data_path("GroundingDINO_finetuned_full_frame_output")
MODEL_DIR = data_path("human_validation/finetune/runs/20261102_143436/best")
_shipped_classes = Path(__file__).resolve().parent / "human_validation/finetune/data/classes.json"
_data_classes = data_path("human_validation/finetune/data/classes.json")
CLASSES_JSON = _data_classes if _data_classes.exists() else _shipped_classes

CLASS_TO_CODE = {
    "moving vehicle": "1_Moving Vehicle",
    "parked vehicle": "2_Parked Vehicle",
    "sidewalk pavement": "3_Sidewalk Pavement",
    "traffic signs": "4_Traffic signs",
    "crosswalk": "5_Crosswalk",
    "crossing signal": "6_Crossing Signal",
    "general light features": "7_General Light Features",
    "pedestrians": "8_Pedestrians",
    "cyclists": "9_Cyclists",
    "storefront": "10_Storefront",
    "buildings": "11_Buildings",
    "art and banners": "12_Art and Banners",
    "natural elements": "13_Natural Elements",
    "miscellaneous infrastructure": "14_Miscellaneous Infrastructure",
}

# Position-based phrases (travel lane vs curb), matching ANNOTATION_PROTOCOL.md.
PHRASES = {
    "moving vehicle": "vehicle in travel lane",
    "parked vehicle": "parked vehicle at curb",
    "sidewalk pavement": "sidewalk",
    "traffic signs": "traffic sign",
    "crosswalk": "crosswalk",
    "crossing signal": "pedestrian signal",
    "general light features": "street light",
    "pedestrians": "pedestrian",
    "cyclists": "cyclist",
    "storefront": "storefront",
    "buildings": "building",
    "art and banners": "banner",
    "natural elements": "tree",
    "miscellaneous infrastructure": "street furniture",
}

DEFAULT_THRESH = 0.30
CLASS_THRESH = {"storefront": 0.20}
CANDIDATE_THRESH = 0.10


def build_prompt(classes):
    return ". ".join(PHRASES[c] for c in classes) + "."


def detect_batch(model, processor, images, classes, prompt, device):
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps

    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(device)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)
    logits = outputs.logits.float().sigmoid()
    pred_boxes = outputs.pred_boxes.float()
    label_maps = build_label_maps(outputs.logits, inputs.input_ids)
    batch_out = []
    for bi, (w, h) in enumerate(zip(widths, heights)):
        label_map = label_maps[bi].to(logits.device)
        if label_map.shape[0] != len(classes):
            raise RuntimeError(
                "Prompt produced %d label groups, expected %d" % (label_map.shape[0], len(classes))
            )
        class_scores = (logits[bi].unsqueeze(0) * label_map.unsqueeze(1)).max(dim=-1).values.transpose(0, 1)
        best_scores, best_cls = class_scores.max(dim=1)
        cx, cy, bw, bh = pred_boxes[bi].unbind(-1)
        xyxy = torch.stack(
            [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
            dim=-1,
        )
        cand = best_scores >= CANDIDATE_THRESH
        boxes = xyxy[cand].tolist()
        scores = best_scores[cand].tolist()
        cls_idx = best_cls[cand].tolist()
        kept_boxes, kept_scores, kept_phrases, kept_cats = [], [], [], []
        for box, score, ci in zip(boxes, scores, cls_idx):
            name = classes[int(ci)]
            thresh = CLASS_THRESH.get(name, DEFAULT_THRESH)
            if score < thresh:
                continue
            kept_boxes.append([float(v) for v in box])
            kept_scores.append(float(score))
            kept_phrases.append(PHRASES[name])
            kept_cats.append(CLASS_TO_CODE[name])
        batch_out.append((kept_boxes, kept_scores, kept_phrases, kept_cats))
    del inputs, outputs
    return batch_out


def postprocess(boxes, scores, phrases, cats):
    import torch
    from torchvision.ops import nms

    if not boxes:
        return [], [], [], [], [], [], [], []
    t_boxes = torch.tensor(boxes, dtype=torch.float32)
    t_scores = torch.tensor(scores, dtype=torch.float32)
    keep = nms(t_boxes, t_scores, NMS_IOU_THRESH).tolist()
    raw_boxes = [boxes[i] for i in keep]
    raw_scores = [scores[i] for i in keep]
    raw_phrases = [phrases[i] for i in keep]
    raw_cats = [cats[i] for i in keep]
    detections = detections_from_grounding_lists(raw_boxes, raw_scores, raw_phrases, raw_cats)
    detections = merge_same_category_overlaps(
        detections,
        iou_threshold=MERGE_IOU_THRESH,
        containment_threshold=MERGE_CONTAINMENT_THRESH,
    )
    out_boxes, out_scores, out_phrases, out_cats = grounding_lists_from_detections(detections)
    return out_boxes, out_scores, out_phrases, out_cats, raw_boxes, raw_scores, raw_phrases, raw_cats


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--participants", nargs="*", default=None)
    parser.add_argument("--max-participants", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    classes = json.loads(CLASSES_JSON.read_text())
    prompt = build_prompt(classes)
    work_items = []
    for condition_dir in iter_condition_dirs(args.root, args.conditions):
        participant_dirs = iter_participant_dirs(condition_dir, args.participants)
        if args.max_participants is not None:
            participant_dirs = participant_dirs[: args.max_participants]
        for participant_dir in participant_dirs:
            frames_dir = participant_dir / "frames"
            out_csv = (
                args.out_root
                / condition_dir.name
                / participant_dir.name
                / "groundingdino_scene_analysis_full_frames.csv"
            )
            work_items.append((condition_dir, participant_dir, frames_dir, out_csv))

    print("Root:", args.root)
    print("Output:", args.out_root)
    print("Model:", args.model_dir)
    print("Prompt:", prompt)
    print("Storefront thresh:", CLASS_THRESH["storefront"], "default:", DEFAULT_THRESH)
    print("Participant folders:", len(work_items))
    if args.dry_run:
        for condition_dir, participant_dir, frames_dir, out_csv in work_items[:10]:
            n = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png"))) if frames_dir.exists() else 0
            print("%s/%s images=%d -> %s" % (condition_dir.name, participant_dir.name, n, out_csv))
        return

    import pandas as pd
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    if not torch.cuda.is_available():
        raise SystemExit("A GPU is required.")
    device = "cuda"
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model_dir, local_files_only=True).to(device)
    model.eval()

    for condition_dir, participant_dir, frames_dir, out_csv in work_items:
        if not frames_dir.exists():
            print("Missing frames:", frames_dir)
            continue
        image_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
        image_files = [
            p
            for p in image_files
            if not p.name.startswith(".") and "_dup" not in p.stem
        ]
        if args.max_images is not None:
            image_files = image_files[: args.max_images]
        if not image_files:
            continue
        if out_csv.exists() and not args.overwrite:
            print("Skip existing:", out_csv)
            continue

        participant_id = participant_dir.stem
        cname = condition_dir.name.lower()
        time_of_day = "night" if "night" in cname else ("day" if "day" in cname else "unknown")
        rows = []
        desc = "%s/%s" % (condition_dir.name, participant_dir.name)
        pbar = tqdm(total=(len(image_files) + args.batch_size - 1) // args.batch_size, desc=desc, leave=False)
        for start_idx, batch_paths in batched(image_files, args.batch_size):
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            batch_dets = detect_batch(model, processor, images, classes, prompt, device)
            for img_path, (boxes, scores, phrases, cats) in zip(batch_paths, batch_dets):
                (
                    all_boxes,
                    all_scores,
                    all_phrases,
                    all_categories,
                    raw_boxes,
                    raw_scores,
                    raw_phrases,
                    raw_categories,
                ) = postprocess(boxes, scores, phrases, cats)
                counts = {}
                for category in all_categories:
                    counts[category] = counts.get(category, 0) + 1
                safety = [c for c in all_categories if c in safety_critical_classes]
                non_safety = [c for c in all_categories if c not in safety_critical_classes]
                complexity = len(safety) * 2.5 + len(non_safety) * 1.0
                frame_timestamp_us = parse_frame_timestamp_us(img_path.name)
                native_frame_timestamp_us = parse_native_frame_timestamp_us(img_path.name)
                native_frame_index = parse_native_frame_index(img_path.name)
                rows.append(
                    {
                        "image_path": str(img_path),
                        "image_name": img_path.name,
                        "participant_id": participant_id,
                        "time_of_day": time_of_day,
                        "category_folder": condition_dir.name,
                        "participant_folder": participant_dir.name,
                        "frame_timestamp_us": frame_timestamp_us,
                        "frame_time_sec": frame_timestamp_us / 1e6 if frame_timestamp_us is not None else None,
                        "native_frame_timestamp_us": native_frame_timestamp_us,
                        "native_frame_time_sec": (
                            native_frame_timestamp_us / 1e6 if native_frame_timestamp_us is not None else None
                        ),
                        "native_frame_index": native_frame_index,
                        "total_categories_detected": len(set(all_categories)),
                        "total_objects_detected": len(all_categories),
                        "num_safety_critical_categories": len(set(safety)),
                        "num_safety_critical_objects": len(safety),
                        "scene_complexity_score": complexity,
                        "categories_present": safe_json(sorted(set(all_categories))),
                        "category_counts": safe_json(counts),
                        "safety_critical_categories": safe_json(sorted(set(safety))),
                        "detected_phrases": safe_json(all_phrases),
                        "detected_categories": safe_json(all_categories),
                        "detection_boxes": safe_json(all_boxes),
                        "detection_confidences": safe_json(all_scores),
                        "raw_detected_phrases_before_merge": safe_json(raw_phrases),
                        "raw_detected_categories_before_merge": safe_json(raw_categories),
                        "raw_detection_boxes_before_merge": safe_json(raw_boxes),
                        "raw_detection_confidences_before_merge": safe_json(raw_scores),
                    }
                )
            for image in images:
                image.close()
            torch.cuda.empty_cache()
            pbar.update(1)
        pbar.close()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("Saved %s | rows %d" % (out_csv, len(rows)), flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
