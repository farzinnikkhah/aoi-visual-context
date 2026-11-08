#!/usr/bin/env python3
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
import re
import sys
from pathlib import Path

from box_merge import (
    detections_from_grounding_lists,
    grounding_lists_from_detections,
    merge_same_category_overlaps,
)


ROOT = data_path("AOI_full_frame_output")
OUT_ROOT = data_path("GroundingDINO_full_frame_output")
LOCAL_PYTHON_DEPS = Path(__file__).resolve().parent / "python_deps"

MODEL_ID = "IDEA-Research/grounding-dino-base"
DEFAULT_BOX_THRESH = 0.30

USE_NMS = True
NMS_IOU_THRESH = 0.50
MERGE_SAME_CATEGORY_OVERLAPS = True
MERGE_IOU_THRESH = 0.70
MERGE_CONTAINMENT_THRESH = 0.90


category_to_phrases = {
    "1_Moving Vehicle": [
        "moving car",
        "car in motion",
        "driving car",
        "moving vehicle",
        "moving truck",
        "moving bus",
        "moving van",
        "vehicle in motion",
        "driving vehicle",
    ],
    "2_Parked Vehicle": [
        "parked car",
        "parked vehicle",
        "stationary car",
        "parked truck",
        "parked bus",
        "parked van",
        "stopped car",
        "stopped vehicle",
    ],
    "3_Sidewalk Pavement": ["sidewalk", "pavement", "walkway", "footpath", "curb", "curb ramp"],
    "4_Traffic signs": [
        "stop sign",
        "yield sign",
        "speed limit sign",
        "street sign",
        "traffic sign",
        "road sign",
        "signpost",
    ],
    "5_Crosswalk": ["crosswalk", "zebra crossing", "pedestrian crossing", "crosswalk markings"],
    "6_Crossing Signal": [
        "pedestrian signal",
        "walk signal",
        "don't walk sign",
        "crossing signal",
        "crosswalk signal",
    ],
    "7_General Light Features": [
        "street light",
        "streetlamp",
        "lamp post",
        "traffic light",
        "signal light",
        "light pole",
    ],
    "8_Pedestrians": ["person", "pedestrian", "people walking", "man", "woman", "child"],
    "9_Cyclists": ["bicycle", "bike", "cyclist", "person riding a bicycle"],
    "10_Storefront": [
        "storefront",
        "shop front",
        "store window",
        "shop window",
        "store entrance",
        "shop entrance",
        "store sign",
    ],
    "11_Buildings": ["building", "building facade", "house", "apartment building", "wall", "window", "door"],
    "12_Art and Banners": ["banner", "flag", "poster", "mural", "art", "signboard", "advertisement"],
    "13_Natural Elements": ["tree", "plant", "vegetation", "bush", "grass", "flower", "shrub"],
    "14_Miscellaneous Infrastructure": [
        "bench",
        "trash can",
        "fire hydrant",
        "parking meter",
        "utility box",
        "bollard",
        "traffic cone",
        "fence",
        "railing",
        "pole",
        "bus stop",
        "outdoor seating",
        "awning",
    ],
}

CAT_THRESH = {
    "13_Natural Elements": 0.30,
}

safety_critical_classes = {
    "1_Moving Vehicle",
    "2_Parked Vehicle",
    "4_Traffic signs",
    "5_Crosswalk",
    "6_Crossing Signal",
    "9_Cyclists",
}


def safe_json(obj):
    try:
        return json.dumps(obj)
    except Exception:
        return json.dumps(str(obj))


def parse_frame_timestamp_us(filename: str):
    match = re.search(r"ts_(\d+)", filename)
    return int(match.group(1)) if match else None


def parse_native_frame_timestamp_us(filename: str):
    match = re.search(r"frame_ts_(\d+)", filename)
    return int(match.group(1)) if match else None


def parse_native_frame_index(filename: str):
    match = re.search(r"_frame_(\d+)(?:_|\.|$)", filename)
    return int(match.group(1)) if match else None


def make_prompt(phrases):
    return " ".join(f"{phrase}." for phrase in phrases)


def nms_dedup(boxes, scores, phrases, cats, iou_thresh=0.5):
    try:
        import torch
        from torchvision.ops import nms
    except Exception:
        return boxes, scores, phrases, cats

    if not boxes:
        return boxes, scores, phrases, cats

    keep = nms(
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        iou_thresh,
    ).cpu().numpy().tolist()
    return (
        [boxes[i] for i in keep],
        [scores[i] for i in keep],
        [phrases[i] for i in keep],
        [cats[i] for i in keep],
    )


def iter_condition_dirs(root: Path, selected_conditions):
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if selected_conditions:
        selected = set(selected_conditions)
        dirs = [d for d in dirs if d.name in selected]
    return dirs


def iter_participant_dirs(condition_dir: Path, selected_participants):
    dirs = sorted(d for d in condition_dir.iterdir() if d.is_dir() and (d / "frames").is_dir())
    if selected_participants:
        selected = set(selected_participants)
        dirs = [d for d in dirs if d.name in selected or d.stem in selected]
    return dirs


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run GroundingDINO on full-frame AOI extraction output.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--conditions", nargs="*", default=[])
    parser.add_argument("--participants", nargs="*", default=[])
    parser.add_argument("--max-participants", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-nms", action="store_true")
    parser.add_argument("--no-merge-overlaps", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    category_prompts = [(cat, make_prompt(phrases)) for cat, phrases in category_to_phrases.items()]
    work_items = []
    for condition_dir in iter_condition_dirs(args.root, args.conditions):
        participant_dirs = iter_participant_dirs(condition_dir, args.participants)
        if args.max_participants is not None:
            participant_dirs = participant_dirs[: args.max_participants]
        for participant_dir in participant_dirs:
            frames_dir = participant_dir / "frames"
            out_csv = args.out_root / condition_dir.name / participant_dir.name / "groundingdino_scene_analysis_full_frames.csv"
            work_items.append((condition_dir, participant_dir, frames_dir, out_csv))

    print(f"Root: {args.root}")
    print(f"Output root: {args.out_root}")
    print(f"Participant folders queued: {len(work_items)}")
    print(f"Batch size: {args.batch_size}")
    if args.dry_run:
        for condition_dir, participant_dir, frames_dir, out_csv in work_items[:20]:
            n_images = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png"))) if frames_dir.exists() else 0
            print(f"{condition_dir.name}/{participant_dir.name}: images={n_images} -> {out_csv}")
        return

    import pandas as pd
    import torch
    from PIL import Image
    from tqdm import tqdm

    if LOCAL_PYTHON_DEPS.exists():
        sys.path.insert(0, str(LOCAL_PYTHON_DEPS))

    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model_id).to(device)
    model.eval()

    for condition_dir, participant_dir, frames_dir, out_csv in work_items:
        if not frames_dir.exists():
            print(f"  Missing frames: {frames_dir}")
            continue

        image_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
        if args.max_images is not None:
            image_files = image_files[: args.max_images]
        if not image_files:
            print(f"  No images: {frames_dir}")
            continue

        if out_csv.exists() and not args.overwrite:
            print(f"  Already exists, skipping: {out_csv}")
            continue

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        participant_id = participant_dir.stem
        cname = condition_dir.name.lower()
        time_of_day = "night" if "night" in cname else ("day" if "day" in cname else "unknown")

        desc = f"{condition_dir.name}/{participant_dir.name}"
        per_image = [
            {
                "boxes": [],
                "scores": [],
                "phrases": [],
                "categories": [],
            }
            for _ in image_files
        ]

        total_batches = len(category_prompts) * ((len(image_files) + args.batch_size - 1) // args.batch_size)
        pbar = tqdm(total=total_batches, desc=desc, leave=False)

        for cat_name, prompt in category_prompts:
            threshold = CAT_THRESH.get(cat_name, DEFAULT_BOX_THRESH)
            for start_idx, batch_paths in batched(image_files, args.batch_size):
                images = [Image.open(img_path).convert("RGB") for img_path in batch_paths]
                target_sizes = torch.tensor([image.size[::-1] for image in images], device=device)
                text = [prompt] * len(images)

                inputs = processor(images=images, text=text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    target_sizes=target_sizes,
                )

                for offset, result in enumerate(results):
                    keep = result["scores"] >= threshold
                    if keep.sum().item() == 0:
                        continue

                    boxes = result["boxes"][keep].tolist()
                    scores = result["scores"][keep].tolist()
                    label_key = "text_labels" if "text_labels" in result else "labels"
                    phrases = [phrase for phrase, k in zip(result[label_key], keep.tolist()) if k]

                    bucket = per_image[start_idx + offset]
                    bucket["boxes"].extend(boxes)
                    bucket["scores"].extend(scores)
                    bucket["phrases"].extend(phrases)
                    bucket["categories"].extend([cat_name] * len(phrases))

                del inputs, outputs, results, images
                if device == "cuda":
                    torch.cuda.empty_cache()
                pbar.update(1)

        pbar.close()

        rows = []
        for img_path, det in zip(image_files, per_image):
            all_boxes = det["boxes"]
            all_scores = det["scores"]
            all_phrases = det["phrases"]
            all_categories = det["categories"]

            if USE_NMS and not args.no_nms and all_boxes:
                all_boxes, all_scores, all_phrases, all_categories = nms_dedup(
                    all_boxes,
                    all_scores,
                    all_phrases,
                    all_categories,
                    iou_thresh=NMS_IOU_THRESH,
                )

            raw_boxes = list(all_boxes)
            raw_scores = list(all_scores)
            raw_phrases = list(all_phrases)
            raw_categories = list(all_categories)

            if MERGE_SAME_CATEGORY_OVERLAPS and not args.no_merge_overlaps and all_boxes:
                detections = detections_from_grounding_lists(all_boxes, all_scores, all_phrases, all_categories)
                detections = merge_same_category_overlaps(
                    detections,
                    iou_threshold=MERGE_IOU_THRESH,
                    containment_threshold=MERGE_CONTAINMENT_THRESH,
                )
                all_boxes, all_scores, all_phrases, all_categories = grounding_lists_from_detections(detections)

            counts = {}
            for category in all_categories:
                counts[category] = counts.get(category, 0) + 1

            safety = [category for category in all_categories if category in safety_critical_classes]
            non_safety = [category for category in all_categories if category not in safety_critical_classes]
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

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv} | rows: {len(rows)}")

    print("Done.")


if __name__ == "__main__":
    main()
