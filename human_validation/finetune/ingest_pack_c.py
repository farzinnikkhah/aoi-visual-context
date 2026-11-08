#!/usr/bin/env python3
"""Ingest IlyasHarris pack_c annotations into an augmented train split.

Keeps original train/test/pilot.json untouched. Writes:
  finetune/data/train_plus_pack_c.json
  finetune/data/pack_c.json
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
HV = HERE.parent
DATA = HERE / "data"
ZIP = HV / "IlyasHarris_pack_c_annotations.zip"
MANIFEST = HV / "batch_c" / "manifest.csv"

CLASSES = json.loads((DATA / "classes.json").read_text())
ALIAS = {
    "building": "buildings",
    "sidewalk": "sidewalk pavement",
    "pedestrains": "pedestrians",
    "natural elelments": "natural elements",
}


def canon(label: str) -> str:
    text = " ".join(str(label).replace("_", " ").split()).lower()
    parts = text.split(" ", 1)
    if parts and parts[0].isdigit() and len(parts) == 2:
        text = parts[1]
    return ALIAS.get(text, text)


def read_zip_labels(zpath: Path):
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


def main():
    labels = read_zip_labels(ZIP)
    rows = list(csv.DictReader(MANIFEST.open()))
    by_id = {r["frame_id"]: r for r in rows}

    pack = []
    missing_manifest = []
    missing_labels = [r["frame_id"] for r in rows if r["frame_id"] not in labels]
    for fid, boxes in sorted(labels.items()):
        row = by_id.get(fid)
        if row is None:
            missing_manifest.append(fid)
            continue
        pack.append(
            {
                "frame_id": fid,
                "stratum": row["stratum"],
                "lighting": row["lighting"],
                "image": row["source_image"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "boxes": boxes,
                "source": "pack_c",
            }
        )

    train = json.loads((DATA / "train.json").read_text())
    test_ids = {r["frame_id"] for r in json.loads((DATA / "test.json").read_text())}
    pilot_ids = {r["frame_id"] for r in json.loads((DATA / "pilot.json").read_text())}
    train_ids = {r["frame_id"] for r in train}

    overlap = [r["frame_id"] for r in pack if r["frame_id"] in (train_ids | test_ids | pilot_ids)]
    if overlap:
        raise SystemExit("pack_c overlaps existing splits: %s" % overlap[:10])

    # Verify images exist (warn only)
    missing_img = [r["frame_id"] for r in pack if not Path(r["image"]).is_file()]
    augmented = train + pack

    (DATA / "pack_c.json").write_text(json.dumps(pack, indent=1))
    (DATA / "train_plus_pack_c.json").write_text(json.dumps(augmented, indent=1))

    def veh_counts(items):
        c = Counter(b["label"] for it in items for b in it["boxes"])
        return c.get("moving vehicle", 0), c.get("parked vehicle", 0), len(items)

    print("pack_c frames:", len(pack), "manifest missing labels:", missing_labels)
    print("pack_c not in manifest:", missing_manifest)
    print("pack_c missing images:", len(missing_img), missing_img[:5])
    print("train vehicles (mov, park, frames):", veh_counts(train))
    print("pack_c vehicles (mov, park, frames):", veh_counts(pack))
    print("train+pack_c (mov, park, frames):", veh_counts(augmented))
    print("Wrote", DATA / "pack_c.json")
    print("Wrote", DATA / "train_plus_pack_c.json")


if __name__ == "__main__":
    main()
