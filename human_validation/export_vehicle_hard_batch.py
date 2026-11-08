#!/usr/bin/env python3
"""Export a hard vehicle annotation pack.

Prefer clear frames where the current fine-tuned model predicts BOTH a moving
and a parked vehicle (co-occurrence hard cases). Falls back to frames with
many vehicle boxes. Excludes pilot / main / batch_c frames.

Default: 48 frames (6 per stratum).
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

import csv
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from export_batch_c_clear import blur_scores
from sample_validation_frames import (
    FRAME_INDEX_RE,
    OUT_ROOT,
    ROOT,
    STRATA,
    TS_RE,
    make_frame_id,
    parse_stratum,
    read_manifest,
)

SEED = 20261102
PER_STRATUM = 6
SOFT_CUTOFF = 0.019
MOTION_CUTOFF = 0.52
DINO = data_path("GroundingDINO_finetuned_full_frame_output")
BATCH = OUT_ROOT / "batch_vehicle_hard"
ZIP_PATH = OUT_ROOT / "vehicle_hard_annotation_pack.zip"


def used_sets():
    blocked_frames = set()
    blocked_sources = set()
    for name in ("pilot", "main", "batch_c", "batch_vehicle_hard"):
        man = OUT_ROOT / name / "manifest.csv"
        if not man.exists():
            continue
        for row in read_manifest(man):
            blocked_frames.add(row["frame_id"])
            blocked_sources.add(row.get("source_image") or "")
            blocked_sources.add(row.get("export_image") or "")
    return blocked_frames, blocked_sources


def vehicle_labels(cats):
    moving = parked = 0
    for c in cats:
        t = " ".join(str(c).replace("_", " ").split()).lower()
        parts = t.split(" ", 1)
        if parts and parts[0].isdigit() and len(parts) == 2:
            t = parts[1]
        if t == "moving vehicle":
            moving += 1
        elif t == "parked vehicle":
            parked += 1
    return moving, parked


def candidate_rows():
    """Scan finetuned CSVs for hard vehicle frames."""
    by_stratum = defaultdict(list)
    for csv_path in sorted(DINO.rglob("groundingdino_scene_analysis_full_frames.csv")):
        stratum = csv_path.parent.parent.name
        if stratum not in STRATA:
            continue
        participant = csv_path.parent.name
        with csv_path.open(newline="") as f:
            for rec in csv.DictReader(f):
                try:
                    cats = json.loads(rec.get("detected_categories") or "[]")
                except Exception:
                    continue
                moving, parked = vehicle_labels(cats)
                total = moving + parked
                if total < 2:
                    continue
                both = moving >= 1 and parked >= 1
                image_name = rec["image_name"]
                source = ROOT / stratum / participant / "frames" / image_name
                if not source.is_file():
                    # image_path in CSV may be absolute
                    alt = Path(rec.get("image_path") or "")
                    source = alt if alt.is_file() else source
                if not source.is_file():
                    continue
                idx_m = FRAME_INDEX_RE.search(image_name)
                if not idx_m:
                    continue
                frame_id = make_frame_id(stratum, participant, idx_m.group(1))
                by_stratum[stratum].append(
                    {
                        "stratum": stratum,
                        "participant": participant,
                        "source": source,
                        "frame_id": frame_id,
                        "moving": moving,
                        "parked": parked,
                        "both": both,
                        "total": total,
                        "image_name": image_name,
                    }
                )
    return by_stratum


def draw_batch():
    blocked_frames, blocked_sources = used_sets()
    cands = candidate_rows()
    rows = []
    notes = []
    for stratum in STRATA:
        rng = random.Random("%s:veh_hard:%s" % (SEED, stratum))
        pool = [c for c in cands.get(stratum, []) if c["frame_id"] not in blocked_frames and str(c["source"]) not in blocked_sources]
        # prefer both-class frames, then high vehicle count
        pool.sort(key=lambda c: (0 if c["both"] else 1, -c["total"], c["frame_id"]))
        # diversify participants: take at most 2 per participant first pass
        taken = 0
        per_part = defaultdict(int)
        selected = []
        # shuffle within score bands
        both = [c for c in pool if c["both"]]
        rest = [c for c in pool if not c["both"]]
        rng.shuffle(both)
        rng.shuffle(rest)
        ordered = both + rest
        for c in ordered:
            if taken >= PER_STRATUM:
                break
            if per_part[c["participant"]] >= 2:
                continue
            soft, motion, blurry = blur_scores(c["source"])
            if blurry or soft is None:
                continue
            selected.append((c, soft, motion))
            per_part[c["participant"]] += 1
            taken += 1
            blocked_frames.add(c["frame_id"])
            blocked_sources.add(str(c["source"]))
        # if short, allow more from same participants
        if taken < PER_STRATUM:
            for c in ordered:
                if taken >= PER_STRATUM:
                    break
                if c["frame_id"] in blocked_frames:
                    continue
                soft, motion, blurry = blur_scores(c["source"])
                if blurry or soft is None:
                    continue
                selected.append((c, soft, motion))
                taken += 1
                blocked_frames.add(c["frame_id"])
                blocked_sources.add(str(c["source"]))
        both_n = sum(1 for c, _, _ in selected if c["both"])
        notes.append("%s: %d frames (%d with both vehicle classes)" % (stratum, len(selected), both_n))
        print(notes[-1], flush=True)
        for c, soft, motion in selected:
            index_match = FRAME_INDEX_RE.search(c["source"].name)
            ts_match = TS_RE.search(c["source"].name)
            intersection, lighting, period = parse_stratum(stratum)
            from PIL import Image

            with Image.open(c["source"]) as image:
                width, height = image.size
            rows.append(
                {
                    "frame_id": c["frame_id"],
                    "split": "batch_vehicle_hard",
                    "stratum": stratum,
                    "intersection": intersection,
                    "lighting": lighting,
                    "period": period,
                    "participant": c["participant"],
                    "frame_index": index_match.group(1) if index_match else "",
                    "recording_timestamp_us": ts_match.group(1) if ts_match else "",
                    "width": str(width),
                    "height": str(height),
                    "source_image": str(c["source"]),
                    "model_moving_count": str(c["moving"]),
                    "model_parked_count": str(c["parked"]),
                    "hard_both_classes": "1" if c["both"] else "0",
                    "soft_score": "%.4f" % soft,
                    "motion_score": "%.3f" % motion,
                }
            )
        if len(selected) < PER_STRATUM:
            print("WARNING %s only %d/%d" % (stratum, len(selected), PER_STRATUM), flush=True)
    return rows, notes


def export(rows):
    image_dir = BATCH / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True)
    for row in rows:
        dest = image_dir / ("%s.jpg" % row["frame_id"])
        shutil.copy2(row["source_image"], dest)
        row["export_image"] = str(dest)
    fieldnames = [
        "frame_id",
        "split",
        "stratum",
        "intersection",
        "lighting",
        "period",
        "participant",
        "frame_index",
        "recording_timestamp_us",
        "width",
        "height",
        "source_image",
        "export_image",
        "model_moving_count",
        "model_parked_count",
        "hard_both_classes",
        "soft_score",
        "motion_score",
    ]
    BATCH.mkdir(parents=True, exist_ok=True)
    manifest = BATCH / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["stratum"], item["participant"])):
            writer.writerow(row)
    return image_dir


def write_zip(image_dir, notes):
    lines = [
        "Hard vehicle annotation pack.",
        "",
        "Focus: Moving vehicle vs Parked vehicle.",
        "Use the protocol rule: travel lane (incl. stopped in queue) = moving;",
        "curb / parking space = parked. Decide from position, not imagined speed.",
        "",
        "These frames were sampled because the model often sees BOTH classes",
        "(or many vehicles) in the same view — hard cases for discrimination.",
        "No model boxes are drawn. Annotate independently.",
        "",
    ]
    if notes:
        lines.append("Sampling note:")
        lines.extend("  " + n for n in notes)
        lines.append("")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("README.txt", "\n".join(lines))
        archive.writestr(
            "FOCUS.txt",
            "Priority classes for this pack: Moving vehicle, Parked vehicle.\n"
            "Still annotate all visible AOI classes per the protocol.\n",
        )
        for path in sorted(image_dir.glob("*.jpg")):
            archive.write(path, "images/%s" % path.name)


def main():
    rows, notes = draw_batch()
    image_dir = export(rows)
    write_zip(image_dir, notes)
    print("frames", len(rows))
    print("zip", ZIP_PATH)
    print("manifest", BATCH / "manifest.csv")


if __name__ == "__main__":
    main()
