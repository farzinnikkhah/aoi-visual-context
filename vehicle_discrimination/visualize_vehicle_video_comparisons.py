#!/usr/bin/env python3
"""Side-by-side before/after vehicle labels (finetuned vs geometry+video)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OLD_ROOT = HERE.parent / "GroundingDINO_finetuned_full_frame_output"
NEW_ROOT = HERE.parent / "GroundingDINO_finetuned_full_frame_output_vehicle_video"
AOI_ROOT = HERE.parent / "AOI_full_frame_output"
OUT_DIR = HERE / "reports" / "vehicle_video_comparisons"

MOVING_COLOR = "#1b9e77"  # teal
PARKED_COLOR = "#d95f02"  # orange
OTHER_COLOR = "#666666"


def canon_veh(c: str):
    t = str(c).lower().replace("_", " ")
    if "moving" in t and "vehicle" in t:
        return "moving"
    if "parked" in t and "vehicle" in t:
        return "parked"
    return None


def font(size=18):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_row_map(csv_path: Path):
    with csv_path.open(newline="") as f:
        return {r["image_name"]: r for r in csv.DictReader(f)}


def vehicle_dets(rec):
    cats = json.loads(rec.get("detected_categories") or "[]")
    boxes = json.loads(rec.get("detection_boxes") or "[]")
    scores = json.loads(rec.get("detection_confidences") or "[]")
    out = []
    for i, (c, b) in enumerate(zip(cats, boxes)):
        kind = canon_veh(c)
        if not kind:
            continue
        out.append(
            {
                "kind": kind,
                "box": [float(v) for v in b],
                "score": float(scores[i]) if i < len(scores) else 0.0,
                "label": "moving" if kind == "moving" else "parked",
            }
        )
    return out


def draw_panel(image: Image.Image, dets, title: str) -> Image.Image:
    im = image.copy()
    draw = ImageDraw.Draw(im)
    f = font(20)
    f_small = font(16)
    # title bar
    draw.rectangle([0, 0, im.width, 36], fill=(20, 20, 20))
    draw.text((10, 8), title, fill="white", font=f)
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        color = MOVING_COLOR if d["kind"] == "moving" else PARKED_COLOR
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        tag = f"{d['label']} {d['score']:.2f}"
        tx, ty = x0 + 2, max(40, y0 - 22)
        bbox = draw.textbbox((tx, ty), tag, font=f_small)
        draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=color)
        draw.text((tx, ty), tag, fill="white", font=f_small)
    # legend
    draw.rectangle([im.width - 210, im.height - 58, im.width - 8, im.height - 8], fill=(0, 0, 0, 160))
    draw.rectangle([im.width - 200, im.height - 48, im.width - 180, im.height - 28], fill=MOVING_COLOR)
    draw.text((im.width - 172, im.height - 48), "moving", fill="white", font=f_small)
    draw.rectangle([im.width - 200, im.height - 28, im.width - 180, im.height - 12], fill=PARKED_COLOR)
    draw.text((im.width - 172, im.height - 30), "parked", fill="white", font=f_small)
    return im


def find_candidates(n_per_stratum=2, seed=20261103):
    rng = random.Random(seed)
    strata = ["I1_Day", "I1_Night", "I3_Day", "I3_Night"]
    picked = []
    for stratum in strata:
        parts = sorted((NEW_ROOT / stratum).glob("*/groundingdino_scene_analysis_full_frames.csv"))
        if not parts:
            continue
        local = []
        for csv_path in rng.sample(parts, min(4, len(parts))):
            old_map = load_row_map(OLD_ROOT / csv_path.relative_to(NEW_ROOT))
            with csv_path.open(newline="") as f:
                for rec in csv.DictReader(f):
                    old = old_map.get(rec["image_name"])
                    if not old:
                        continue
                    nd = vehicle_dets(rec)
                    od = vehicle_dets(old)
                    if not nd:
                        continue
                    n_mov = sum(d["kind"] == "moving" for d in nd)
                    n_park = sum(d["kind"] == "parked" for d in nd)
                    o_mov = sum(d["kind"] == "moving" for d in od)
                    o_park = sum(d["kind"] == "parked" for d in od)
                    if n_mov >= 1 and n_park >= 1 and (n_mov != o_mov or n_park != o_park):
                        img = Path(rec["image_path"])
                        if not img.is_file():
                            img = AOI_ROOT / stratum / csv_path.parent.name / "frames" / rec["image_name"]
                        if img.is_file():
                            local.append(
                                {
                                    "stratum": stratum,
                                    "part": csv_path.parent.name,
                                    "image": img,
                                    "name": rec["image_name"],
                                    "old": old,
                                    "new": rec,
                                    "o_mov": o_mov,
                                    "o_park": o_park,
                                    "n_mov": n_mov,
                                    "n_park": n_park,
                                }
                            )
        rng.shuffle(local)
        picked.extend(local[:n_per_stratum])
    return picked


def compose(left: Image.Image, right: Image.Image, caption: str) -> Image.Image:
    gap = 12
    w = left.width + right.width + gap
    h = max(left.height, right.height) + 44
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    canvas.paste(left, (0, 44))
    canvas.paste(right, (left.width + gap, 44))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, 44], fill=(30, 30, 30))
    draw.text((12, 12), caption, fill="white", font=font(18))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-per-stratum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    picks = find_candidates(args.n_per_stratum, args.seed)
    manifest = []
    for i, p in enumerate(picks, 1):
        raw = Image.open(p["image"]).convert("RGB")
        ow, oh = raw.size
        max_w = 960
        if ow > max_w:
            nh = int(oh * max_w / ow)
            image = raw.resize((max_w, nh), Image.Resampling.BILINEAR)
            sx, sy = max_w / float(ow), nh / float(oh)
        else:
            image = raw
            sx = sy = 1.0

        def scale_dets(dets):
            out = []
            for d in dets:
                x0, y0, x1, y1 = d["box"]
                out.append({**d, "box": [x0 * sx, y0 * sy, x1 * sx, y1 * sy]})
            return out

        left = draw_panel(
            image,
            scale_dets(vehicle_dets(p["old"])),
            f"Before (finetuned)  moving={p['o_mov']} parked={p['o_park']}",
        )
        right = draw_panel(
            image,
            scale_dets(vehicle_dets(p["new"])),
            f"After (geometry+video)  moving={p['n_mov']} parked={p['n_park']}",
        )
        caption = f"{p['stratum']} / {p['part']}  |  parked→mixed labels via geometry + video tracks"
        combo = compose(left, right, caption)
        out = args.out_dir / f"{i:02d}_{p['stratum']}_{p['part']}.jpg"
        combo.save(out, quality=92)
        manifest.append(
            {
                "file": out.name,
                "stratum": p["stratum"],
                "participant": p["part"],
                "image_name": p["name"],
                "before_moving": p["o_mov"],
                "before_parked": p["o_park"],
                "after_moving": p["n_mov"],
                "after_parked": p["n_park"],
            }
        )
        print("Wrote", out)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Wrote", args.out_dir / "manifest.json")


if __name__ == "__main__":
    main()
