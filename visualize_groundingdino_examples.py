#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#17becf",
    "#bcbd22",
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
]


def parse_json_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def color_for_category(category):
    idx = abs(hash(str(category))) % len(PALETTE)
    return PALETTE[idx]


def draw_label(draw, xy, text, fill):
    x0, y0 = xy
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((x0, y0), text, font=font)
    pad = 3
    bg = (max(0, int(fill[1:3], 16) - 40), max(0, int(fill[3:5], 16) - 40), max(0, int(fill[5:7], 16) - 40))
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=bg,
    )
    draw.text((x0, y0), text, fill="white", font=font)


def annotate_row(row, out_dir: Path, index: int):
    image_path = Path(row["image_path"])
    if not image_path.exists():
        print(f"Missing image: {image_path}")
        return None

    boxes = parse_json_list(row.get("detection_boxes"))
    scores = parse_json_list(row.get("detection_confidences"))
    phrases = parse_json_list(row.get("detected_phrases"))
    categories = parse_json_list(row.get("detected_categories"))

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    n = min(len(boxes), len(scores), len(phrases), len(categories))
    for det_idx in range(n):
        box = boxes[det_idx]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in box]
        category = str(categories[det_idx])
        phrase = str(phrases[det_idx])
        score = float(scores[det_idx])
        color = color_for_category(category)
        width = 4 if category.startswith(("1_", "2_", "4_", "5_", "6_", "9_")) else 3
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        label = f"{category}: {phrase} {score:.2f}"
        draw_label(draw, (x0 + 2, max(0, y0 - 24)), label, color)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{index:03d}_{image_path.stem}_annotated.jpg"
    image.save(out_path, quality=92)
    return out_path


def choose_rows(df, mode, n):
    if mode == "top_objects":
        return df.sort_values("total_objects_detected", ascending=False).head(n)
    if mode == "nonempty":
        return df[df["total_objects_detected"].fillna(0) > 0].head(n)
    return df.head(n)


def main():
    parser = argparse.ArgumentParser(description="Draw GroundingDINO boxes on example full-frame images.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--mode", choices=["top_objects", "nonempty", "first"], default="top_objects")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    rows = choose_rows(df, args.mode, args.n)
    print(f"CSV: {args.csv}")
    print(f"Rows selected: {len(rows)}")
    print(f"Output: {args.out_dir}")

    written = []
    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        out_path = annotate_row(row, args.out_dir, idx)
        if out_path is not None:
            written.append(out_path)
            print(out_path)

    print(f"Done. Wrote {len(written)} annotated images.")


if __name__ == "__main__":
    main()
