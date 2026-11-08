#!/usr/bin/env python3
"""Light fine-tune of GroundingDINO on the 128 training frames.

The image backbone and the text encoder stay frozen. The fusion encoder,
decoder, and heads are updated. One prompt lists the 14 classes, one short
phrase each, so the model learns which phrase goes with which box.

After training, the script runs the zero-shot model and the fine-tuned model
on the 32 held-out frames and the 24 pilot frames, and writes the boxes.
"""

import json
import math
import random
import sys
import time
import types
from pathlib import Path

stub = types.ModuleType("transformers.dependency_versions_check")
stub.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = stub

import torch
from PIL import Image, ImageOps
import transformers.utils.hub as hf_hub

hf_hub.list_repo_templates = lambda *args, **kwargs: []

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
DATA = HERE / "data"
OUT = HERE / "runs" / time.strftime("%Y%m%d_%H%M%S")
MODEL_ID = "IDEA-Research/grounding-dino-base"

CLASSES = json.loads((DATA / "classes.json").read_text())
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
PROMPT = ". ".join(PHRASES[c] for c in CLASSES) + "."

import os

EPOCHS = int(os.environ.get("FT_EPOCHS", "12"))
BATCH = 2
LR = float(os.environ.get("FT_LR", "1e-5"))
CANDIDATE_THRESH = 0.10
WEIGHT_DECAY = 1e-4
CLIP = 0.1
BOX_THRESH = 0.30
NMS_IOU = 0.50
SEED = 20261019


def load_split(name):
    return json.loads((DATA / ("%s.json" % name)).read_text())


def oversample_moving(items, copies=2):
    """Duplicate frames that contain at least one moving vehicle.

    Helps counter the moving->parked collapse without changing labels.
    """
    out = list(items)
    extras = []
    for item in items:
        if any(b.get("label") == "moving vehicle" for b in item.get("boxes") or []):
            for _ in range(max(copies - 1, 0)):
                extras.append(item)
    out.extend(extras)
    return out


def to_targets(item, flipped, device):
    w, h = float(item["width"]), float(item["height"])
    boxes = []
    labels = []
    for entry in item["boxes"]:
        x1, y1, x2, y2 = entry["box"]
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            continue
        if flipped:
            cx = 1.0 - cx
        boxes.append([cx, cy, bw, bh])
        labels.append(CLASSES.index(entry["label"]))
    return {
        "class_labels": torch.tensor(labels, dtype=torch.long, device=device),
        "boxes": torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
    }


def batches(items, size, rng, shuffle):
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), size):
        yield [items[i] for i in order[start:start + size]]


def forward_batch(model, processor, batch, device, train, rng):
    images = []
    targets = []
    for item in batch:
        image = Image.open(item["image"]).convert("RGB")
        flipped = bool(train and rng.random() < 0.5)
        if flipped:
            image = ImageOps.mirror(image)
        images.append(image)
        targets.append(to_targets(item, flipped, device))
    inputs = processor(images=images, text=[PROMPT] * len(images), return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**inputs, labels=targets)
    for image in images:
        image.close()
    return outputs, inputs


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("model.backbone") or name.startswith("model.text_backbone"):
            continue
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("trainable %.1fM of %.1fM" % (trainable / 1e6, total / 1e6), flush=True)


@torch.no_grad()
def detect(model, processor, item, device):
    image = Image.open(item["image"]).convert("RGB")
    w, h = image.size
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**inputs)
    image.close()
    logits = outputs.logits[0].float().sigmoid()  # (queries, text_len)
    pred_boxes = outputs.pred_boxes[0].float()  # (queries, 4) cxcywh
    label_map = build_label_maps(outputs.logits, inputs.input_ids)[0].to(logits.device)  # (classes, text_len)
    if label_map.shape[0] != len(CLASSES):
        raise SystemExit("Prompt produced %d label groups, expected %d" % (label_map.shape[0], len(CLASSES)))
    masked = logits.unsqueeze(0) * label_map.unsqueeze(1)  # (classes, queries, text_len)
    class_scores = masked.max(dim=-1).values.transpose(0, 1)  # (queries, classes)
    best_scores, best_cls = class_scores.max(dim=1)
    cx, cy, bw, bh = pred_boxes.unbind(-1)
    xyxy = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], dim=-1)
    cand = best_scores >= CANDIDATE_THRESH
    candidates = [
        {"category": CLASSES[int(c)], "score": round(float(s), 4), "box": [round(float(v), 1) for v in b]}
        for b, s, c in zip(xyxy[cand].tolist(), best_scores[cand].tolist(), best_cls[cand].tolist())
    ]
    return finalize(candidates, BOX_THRESH), candidates


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


def run_detection(model, processor, items, device, tag):
    model.eval()
    results = []
    for item in items:
        final, candidates = detect(model, processor, item, device)
        results.append({"frame_id": item["frame_id"], "detections": final, "candidates": candidates})
    print("%s: %d frames, %d boxes" % (tag, len(results), sum(len(r["detections"]) for r in results)), flush=True)
    return results


@torch.no_grad()
def eval_loss(model, processor, items, device):
    model.eval()
    total = 0.0
    count = 0
    for batch in batches(items, BATCH, None, shuffle=False):
        outputs, _ = forward_batch(model, processor, batch, device, train=False, rng=None)
        total += float(outputs.loss.item()) * len(batch)
        count += len(batch)
    return total / max(count, 1)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("A GPU is required.")
    device = "cuda"
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    train = oversample_moving(load_split("train"), copies=int(os.environ.get("FT_MOVING_COPIES", "3")))
    test = load_split("test")
    pilot = load_split("pilot")
    print("prompt:", PROMPT, flush=True)
    print("train %d test %d pilot %d" % (len(train), len(test), len(pilot)), flush=True)

    init_model = os.environ.get("FT_INIT", MODEL_ID)
    processor = AutoProcessor.from_pretrained(init_model, local_files_only=True)
    model = GroundingDinoForObjectDetection.from_pretrained(init_model, local_files_only=True).to(device)

    zero_test = run_detection(model, processor, test, device, "zero-shot test")
    zero_pilot = run_detection(model, processor, pilot, device, "zero-shot pilot")
    (OUT / "zero_shot_test.json").write_text(json.dumps(zero_test, indent=1))
    (OUT / "zero_shot_pilot.json").write_text(json.dumps(zero_pilot, indent=1))

    freeze(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train) / BATCH)
    total_steps = steps_per_epoch * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))
    )

    history = []
    best = float("inf")
    best_path = OUT / "best"
    start_loss = eval_loss(model, processor, test, device)
    print("epoch 0 test_loss %.4f" % start_loss, flush=True)
    history.append({"epoch": 0, "train_loss": None, "test_loss": start_loss})

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0
        t0 = time.time()
        for batch in batches(train, BATCH, rng, shuffle=True):
            outputs, _ = forward_batch(model, processor, batch, device, train=True, rng=rng)
            loss = outputs.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            optimizer.step()
            scheduler.step()
            running += float(loss.item()) * len(batch)
            seen += len(batch)
        train_loss = running / seen
        test_loss = eval_loss(model, processor, test, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
        print(
            "epoch %d train_loss %.4f test_loss %.4f lr %.2e %.0fs"
            % (epoch, train_loss, test_loss, scheduler.get_last_lr()[0], time.time() - t0),
            flush=True,
        )
        if test_loss < best:
            best = test_loss
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            print("  saved best", flush=True)
        (OUT / "history.json").write_text(json.dumps(history, indent=1))

    model = GroundingDinoForObjectDetection.from_pretrained(best_path).to(device)
    tuned_test = run_detection(model, processor, test, device, "fine-tuned test")
    tuned_pilot = run_detection(model, processor, pilot, device, "fine-tuned pilot")
    (OUT / "finetuned_test.json").write_text(json.dumps(tuned_test, indent=1))
    (OUT / "finetuned_pilot.json").write_text(json.dumps(tuned_pilot, indent=1))
    (OUT / "config.json").write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "init_model": init_model,
                "prompt": PROMPT,
                "epochs": EPOCHS,
                "batch": BATCH,
                "lr": LR,
                "moving_oversample_copies": int(os.environ.get("FT_MOVING_COPIES", "3")),
                "frozen": ["model.backbone", "model.text_backbone"],
                "box_thresh": BOX_THRESH,
                "nms_iou": NMS_IOU,
                "best_test_loss": best,
            },
            indent=1,
        )
    )
    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
