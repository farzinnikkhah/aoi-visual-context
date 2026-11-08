#!/usr/bin/env python3
"""Side-by-side human boxes and GroundingDINO boxes for the 24 pilot frames."""
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
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PILOT = data_path("human_validation/pilot")
DINO = data_path("GroundingDINO_full_frame_output")
OUT = PILOT / "comparisons"
IOU_THR = 0.5

FONT = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 22)
FONT_SM = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 16)
FONT_BOLD = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 26)

COLORS = {
    "moving vehicle": "#d62728",
    "parked vehicle": "#ff7f0e",
    "sidewalk pavement": "#c4b000",
    "traffic signs": "#9467bd",
    "crosswalk": "#1f77b4",
    "crossing signal": "#e377c2",
    "general light features": "#17becf",
    "pedestrians": "#2ca02c",
    "cyclists": "#8c564b",
    "storefront": "#e41a1c",
    "buildings": "#7f7f7f",
    "art and banners": "#f781bf",
    "natural elements": "#4daf4a",
    "miscellaneous infrastructure": "#a65628",
}

DISPLAY = {
    "moving vehicle": "Moving vehicle",
    "parked vehicle": "Parked vehicle",
    "sidewalk pavement": "Sidewalk pavement",
    "traffic signs": "Traffic signs",
    "crosswalk": "Crosswalk",
    "crossing signal": "Crossing signal",
    "general light features": "General light features",
    "pedestrians": "Pedestrians",
    "cyclists": "Cyclists",
    "storefront": "Storefront",
    "buildings": "Buildings",
    "art and banners": "Art and banners",
    "natural elements": "Natural elements",
    "miscellaneous infrastructure": "Miscellaneous infrastructure",
}


def canon(label):
    text = " ".join(str(label).replace("_", " ").split()).lower()
    parts = text.split(" ", 1)
    if parts and parts[0].isdigit() and len(parts) == 2:
        text = parts[1]
    if text == "building":
        text = "buildings"
    return text


def display_name(label):
    return DISPLAY.get(label, label)


def color_of(label):
    return COLORS.get(label, "#333333")


def box_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


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


def match_count(gt, pred):
    pairs = []
    for i, (gl, gb) in enumerate(gt):
        for j, (pl, pb) in enumerate(pred):
            if gl != pl:
                continue
            value = iou(gb, pb)
            if value >= IOU_THR:
                pairs.append((value, i, j))
    pairs.sort(reverse=True)
    used_g = set()
    used_p = set()
    for _, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
    return len(used_g)


def load_human(frame_id):
    data = json.loads((PILOT / (frame_id + ".json")).read_text())
    boxes = []
    for shape in data.get("shapes") or []:
        if shape.get("shape_type") != "rectangle":
            continue
        boxes.append((canon(shape["label"]), box_from_points(shape["points"]), None))
    return boxes


def load_model(row):
    csv_path = DINO / row["stratum"] / row["participant"] / "groundingdino_scene_analysis_full_frames.csv"
    image_name = Path(row["source_image"]).name
    with csv_path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            if record["image_name"] != image_name:
                continue
            cats = json.loads(record["detected_categories"] or "[]")
            boxes = json.loads(record["detection_boxes"] or "[]")
            scores = json.loads(record["detection_confidences"] or "[]")
            parsed = []
            for cat, box, score in zip(cats, boxes, scores):
                parsed.append((canon(cat), [float(v) for v in box], float(score)))
            return parsed
    return []


def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for label, box, score in boxes:
        x0, y0, x1, y1 = box
        color = color_of(label)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        text = display_name(label)
        if score is not None:
            text = "%s %.2f" % (text, score)
        text_box = draw.textbbox((0, 0), text, font=FONT_SM)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        tx = max(0, min(x0, width - tw - 8))
        ty = y0 - th - 8
        if ty < 0:
            ty = min(y0 + 2, height - th - 6)
        draw.rectangle([tx, ty, tx + tw + 8, ty + th + 6], fill=color)
        draw.text((tx + 4, ty + 2), text, fill="white", font=FONT_SM)


def panel(image_path, boxes, title, subtitle):
    image = Image.open(image_path).convert("RGB")
    draw_boxes(image, boxes)
    width, height = image.size
    title_h = 46
    canvas = Image.new("RGB", (width, height + title_h), "white")
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, title_h], fill="#102a43")
    draw.text((14, 8), title, fill="white", font=FONT_BOLD)
    sub_box = draw.textbbox((0, 0), subtitle, font=FONT_SM)
    draw.text((width - (sub_box[2] - sub_box[0]) - 14, 14), subtitle, fill="#d9e2ec", font=FONT_SM)
    return canvas


def legend_strip(width, labels):
    height = 42
    strip = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(strip)
    x = 12
    y = 8
    for label in labels:
        name = display_name(label)
        color = color_of(label)
        box = draw.textbbox((0, 0), name, font=FONT_SM)
        tw = box[2] - box[0]
        if x + tw + 28 > width:
            break
        draw.rectangle([x, y + 4, x + 14, y + 18], fill=color)
        draw.text((x + 18, y), name, fill="#243b53", font=FONT_SM)
        x += tw + 36
    return strip


def compose(frame_id, stratum, image_path, human, model, right_title="GroundingDINO", match_text=None):
    left = panel(image_path, human, "Human", "%d boxes" % len(human))
    right = panel(image_path, model, right_title, "%d boxes" % len(model))
    gap = 12
    header_h = 54
    matched = match_count([(a, b) for a, b, _ in human], [(a, b) for a, b, _ in model])
    if match_text is None:
        match_text = "Matched %d at IoU >= 0.5" % matched
    labels = []
    seen = set()
    for label, _, _ in list(human) + list(model):
        if label not in seen:
            seen.add(label)
            labels.append(label)
    legend = legend_strip(left.width + gap + right.width, labels)
    width = left.width + gap + right.width
    height = header_h + left.height + legend.height
    canvas = Image.new("RGB", (width, height), "#e4e7eb")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, header_h], fill="#243b53")
    title = "%s    %s" % (frame_id, stratum.replace("_", " "))
    draw.text((16, 12), title, fill="white", font=FONT)
    summary = match_text
    box = draw.textbbox((0, 0), summary, font=FONT)
    draw.text((width - (box[2] - box[0]) - 16, 14), summary, fill="#d9e2ec", font=FONT)
    canvas.paste(left, (0, header_h))
    canvas.paste(right, (left.width + gap, header_h))
    canvas.paste(legend, (0, header_h + left.height))
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((PILOT / "manifest.csv").open()))
    written = []
    for index, row in enumerate(rows, start=1):
        human = load_human(row["frame_id"])
        model = load_model(row)
        image = compose(row["frame_id"], row["stratum"], row["export_image"], human, model)
        out_path = OUT / ("%02d_%s.jpg" % (index, row["frame_id"]))
        image.save(out_path, quality=85, optimize=True)
        written.append(out_path.name)
        print(out_path.name, image.size)

    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Pilot comparisons</title>",
        "<style>",
        "body{margin:0;background:#111;color:#eee;font-family:sans-serif}",
        "h1{font-size:18px;padding:16px 20px;margin:0}",
        "img{width:100%;height:auto;display:block;margin:0 0 28px;background:#000}",
        "</style></head><body>",
        "<h1>Human annotation on the left, GroundingDINO on the right. Same color means the same class.</h1>",
    ]
    for name in written:
        html.append("<img src='%s' alt='%s'>" % (name, name))
    html.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(html))
    print("Wrote", len(written), "comparisons to", OUT)


if __name__ == "__main__":
    main()
