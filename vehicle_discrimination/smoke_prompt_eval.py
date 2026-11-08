#!/usr/bin/env python3
"""Run current fine-tuned weights with position-aligned vehicle phrases on test/pilot."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

stub = types.ModuleType("transformers.dependency_versions_check")
stub.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = stub

import transformers.utils.hub as hf_hub

hf_hub.list_repo_templates = lambda *args, **kwargs: []

import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps

_repo = Path(__file__).resolve()
for _parent in [_repo.parent, *_repo.parents]:
    if (_parent / "aoi_paths.py").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
from box_merge import detections_from_grounding_lists, grounding_lists_from_detections, merge_same_category_overlaps

HERE = Path(__file__).resolve().parent
FT = HERE.parent / "human_validation" / "finetune"
DATA = FT / "data"
MODEL_DIR = FT / "runs" / "20261019_154219" / "best"
OUT = HERE

CLASSES = json.loads((DATA / "classes.json").read_text())
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
PROMPT = ". ".join(PHRASES[c] for c in CLASSES) + "."
CANDIDATE_THRESH = 0.10
BOX_THRESH = 0.30
NMS_IOU = 0.50


def finalize(candidates, thresh):
    kept = [c for c in candidates if c["score"] >= thresh]
    if not kept:
        return []
    boxes = torch.tensor([c["box"] for c in kept], dtype=torch.float32)
    scores = torch.tensor([c["score"] for c in kept], dtype=torch.float32)
    idx = nms(boxes, scores, NMS_IOU).tolist()
    cats = [kept[i]["category"] for i in idx]
    phrases = [PHRASES[c] for c in cats]
    detections = detections_from_grounding_lists(
        boxes[idx].tolist(), scores[idx].tolist(), phrases, cats
    )
    detections = merge_same_category_overlaps(detections, iou_threshold=0.70, containment_threshold=0.90)
    out_boxes, out_scores, out_phrases, out_cats = grounding_lists_from_detections(detections)
    return [
        {"category": c, "score": round(float(s), 3), "box": [round(float(v), 1) for v in b]}
        for b, s, c in zip(out_boxes, out_scores, out_cats)
    ]


@torch.no_grad()
def detect(model, processor, item, device):
    image = Image.open(item["image"]).convert("RGB")
    w, h = image.size
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**inputs)
    image.close()
    logits = outputs.logits[0].float().sigmoid()
    pred_boxes = outputs.pred_boxes[0].float()
    label_map = build_label_maps(outputs.logits, inputs.input_ids)[0].to(logits.device)
    if label_map.shape[0] != len(CLASSES):
        raise SystemExit("Prompt produced %d label groups, expected %d" % (label_map.shape[0], len(CLASSES)))
    masked = logits.unsqueeze(0) * label_map.unsqueeze(1)
    class_scores = masked.max(dim=-1).values.transpose(0, 1)
    best_scores, best_cls = class_scores.max(dim=1)
    cx, cy, bw, bh = pred_boxes.unbind(-1)
    xyxy = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], dim=-1)
    cand = best_scores >= CANDIDATE_THRESH
    candidates = [
        {"category": CLASSES[int(c)], "score": round(float(s), 4), "box": [round(float(v), 1) for v in b]}
        for b, s, c in zip(xyxy[cand].tolist(), best_scores[cand].tolist(), best_cls[cand].tolist())
    ]
    return finalize(candidates, BOX_THRESH), candidates


def run_split(model, processor, items, device, out_name):
    results = []
    for item in items:
        final, candidates = detect(model, processor, item, device)
        results.append({"frame_id": item["frame_id"], "detections": final, "candidates": candidates})
    path = OUT / out_name
    path.write_text(json.dumps(results, indent=1))
    print("wrote", path, "boxes", sum(len(r["detections"]) for r in results), flush=True)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("A GPU is required.")
    device = "cuda"
    print("prompt:", PROMPT, flush=True)
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = GroundingDinoForObjectDetection.from_pretrained(str(MODEL_DIR), local_files_only=True).to(device)
    model.eval()
    test = json.loads((DATA / "test.json").read_text())
    pilot = json.loads((DATA / "pilot.json").read_text())
    run_split(model, processor, test, device, "prompt_smoke_test.json")
    run_split(model, processor, pilot, device, "prompt_smoke_pilot.json")


if __name__ == "__main__":
    main()
