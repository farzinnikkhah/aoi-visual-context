#!/usr/bin/env python3
"""Turn the 160 main annotations into a training set for GroundingDINO.

Label spellings are mapped onto the 14 protocol names. Four frames from each
of the 8 conditions are held out for testing. The 24 pilot frames are never
used for training.
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
import random
import zipfile
from collections import Counter
from pathlib import Path

MAIN = data_path("human_validation/main")
PILOT = data_path("human_validation/pilot")
OUT = data_path("human_validation/finetune/data")
SEED = 20261019
HOLDOUT_PER_STRATUM = 4

CLASSES = [
    "moving vehicle",
    "parked vehicle",
    "sidewalk pavement",
    "traffic signs",
    "crosswalk",
    "crossing signal",
    "general light features",
    "pedestrians",
    "cyclists",
    "storefront",
    "buildings",
    "art and banners",
    "natural elements",
    "miscellaneous infrastructure",
]
ALIAS = {
    "building": "buildings",
    "sidewalk": "sidewalk pavement",
    "pedestrains": "pedestrians",
    "natural elelments": "natural elements",
}


def canon(label):
    text = " ".join(str(label).replace("_", " ").split()).lower()
    parts = text.split(" ", 1)
    if parts and parts[0].isdigit() and len(parts) == 2:
        text = parts[1]
    return ALIAS.get(text, text)


def read_zip_labels(zpath):
    out = {}
    with zipfile.ZipFile(zpath) as archive:
        for name in archive.namelist():
            if not name.endswith(".json"):
                continue
            data = json.loads(archive.read(name))
            boxes = []
            for shape in data.get("shapes") or []:
                if shape.get("shape_type") != "rectangle":
                    continue
                label = canon(shape["label"])
                if label not in CLASSES:
                    raise SystemExit("Unknown label %r in %s" % (shape["label"], name))
                xs = [p[0] for p in shape["points"]]
                ys = [p[1] for p in shape["points"]]
                boxes.append({"label": label, "box": [min(xs), min(ys), max(xs), max(ys)]})
            out[Path(name).stem] = boxes
    return out


def read_dir_labels(folder):
    out = {}
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text())
        boxes = []
        for shape in data.get("shapes") or []:
            if shape.get("shape_type") != "rectangle":
                continue
            xs = [p[0] for p in shape["points"]]
            ys = [p[1] for p in shape["points"]]
            boxes.append({"label": canon(shape["label"]), "box": [min(xs), min(ys), max(xs), max(ys)]})
        out[path.stem] = boxes
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = {}
    for zpath in ("images-a.zip", "images-b.zip", "main_annotation_pack_20_labeled.zip"):
        labels.update(read_zip_labels(MAIN / zpath))
    rows = list(csv.DictReader((MAIN / "manifest.csv").open()))
    assert len(rows) == 160 and all(r["frame_id"] in labels for r in rows)

    by_stratum = {}
    for row in rows:
        by_stratum.setdefault(row["stratum"], []).append(row)
    rng = random.Random(SEED)
    train, test = [], []
    for stratum in sorted(by_stratum):
        items = sorted(by_stratum[stratum], key=lambda r: r["frame_id"])
        rng.shuffle(items)
        test.extend(items[:HOLDOUT_PER_STRATUM])
        train.extend(items[HOLDOUT_PER_STRATUM:])

    def record(row):
        return {
            "frame_id": row["frame_id"],
            "stratum": row["stratum"],
            "lighting": row["lighting"],
            "image": row["source_image"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "boxes": labels[row["frame_id"]],
        }

    pilot_rows = list(csv.DictReader((PILOT / "manifest.csv").open()))
    pilot_labels = read_dir_labels(PILOT)
    pilot = []
    for row in pilot_rows:
        pilot.append(
            {
                "frame_id": row["frame_id"],
                "stratum": row["stratum"],
                "lighting": row["lighting"],
                "image": row["source_image"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "boxes": pilot_labels[row["frame_id"]],
            }
        )

    splits = {
        "train": [record(r) for r in train],
        "test": [record(r) for r in test],
        "pilot": pilot,
    }
    for name, items in splits.items():
        (OUT / ("%s.json" % name)).write_text(json.dumps(items, indent=1))
        counts = Counter(b["label"] for item in items for b in item["boxes"])
        print(name, "frames", len(items), "boxes", sum(counts.values()))
        for cls in CLASSES:
            print("   %4d  %s" % (counts.get(cls, 0), cls))
    (OUT / "classes.json").write_text(json.dumps(CLASSES, indent=1))


if __name__ == "__main__":
    main()
