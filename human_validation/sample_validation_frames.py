#!/usr/bin/env python3
"""Draw a stratified validation sample and export clean frames.

Sampling unit is one frame from one participant within a stratum.
Strata are the eight condition folders:

    I1/I3 x Day/Night x during-crossing/before-crossing

The pilot draw is the calibration set. The main draw excludes every
participant already used in that stratum's pilot, and it excludes the
pilot frames themselves. Images are copied with no model overlays.

Example:
    python sample_validation_frames.py --stage pilot
    python sample_validation_frames.py --stage main
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


import argparse
import csv
import random
import re
import shutil
from pathlib import Path

from PIL import Image


ROOT = data_path("AOI_full_frame_output")
OUT_ROOT = data_path("human_validation")

STRATA = [
    "I1_Day",
    "I1_Day_before",
    "I1_Night",
    "I1_Night_before",
    "I3_Day",
    "I3_Day_before",
    "I3_Night",
    "I3_Night_before",
]

STAGE_N = {"pilot": 3, "main": 20}
FRAME_INDEX_RE = re.compile(r"_frame_(\d+)(?:_|\.|$)")
TS_RE = re.compile(r"_ts_(\d+)_frame_ts_")


def parse_stratum(name):
    period = "before" if name.endswith("_before") else "during"
    base = name[: -len("_before")] if period == "before" else name
    intersection, lighting = base.split("_", 1)
    return intersection, lighting, period


def eligible_frames(frames_dir):
    frames = []
    for path in frames_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if path.name.startswith("."):
            continue
        if "_dup" in path.stem:
            continue
        frames.append(path)
    return sorted(frames)


def participant_dirs(stratum_dir):
    dirs = []
    for path in stratum_dir.iterdir():
        if path.is_dir() and (path / "frames").is_dir() and eligible_frames(path / "frames"):
            dirs.append(path)
    return sorted(dirs, key=lambda p: p.name)


def read_manifest(path):
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def excluded_participants(pilot_rows, stratum):
    return {row["participant"] for row in pilot_rows if row["stratum"] == stratum}


def make_frame_id(stratum, participant, frame_index):
    safe_participant = participant.replace(".", "_")
    return f"{stratum}_{safe_participant}_f{int(frame_index):07d}"


def sample_stratum(stratum, n, rng, blocked_participants):
    stratum_dir = ROOT / stratum
    if not stratum_dir.is_dir():
        raise SystemExit(f"Missing stratum folder: {stratum_dir}")

    candidates = [
        path for path in participant_dirs(stratum_dir) if path.name not in blocked_participants
    ]
    if len(candidates) < n:
        raise SystemExit(
            f"{stratum}: need {n} participants, found {len(candidates)} "
            f"after excluding {len(blocked_participants)} pilot participants"
        )

    chosen = candidates[:]
    rng.shuffle(chosen)
    chosen = chosen[:n]
    intersection, lighting, period = parse_stratum(stratum)
    rows = []
    for participant_dir in chosen:
        frames = eligible_frames(participant_dir / "frames")
        source = frames[rng.randrange(len(frames))]
        index_match = FRAME_INDEX_RE.search(source.name)
        ts_match = TS_RE.search(source.name)
        if not index_match:
            raise SystemExit(f"Cannot parse frame index from {source.name}")
        frame_index = index_match.group(1)
        frame_id = make_frame_id(stratum, participant_dir.name, frame_index)
        with Image.open(source) as image:
            width, height = image.size
        rows.append(
            {
                "frame_id": frame_id,
                "stratum": stratum,
                "intersection": intersection,
                "lighting": lighting,
                "period": period,
                "participant": participant_dir.name,
                "frame_index": frame_index,
                "recording_timestamp_us": ts_match.group(1) if ts_match else "",
                "source_image": str(source),
                "width": str(width),
                "height": str(height),
            }
        )
    return rows


def export_stage(stage, rows, out_dir):
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        dest = image_dir / f"{row['frame_id']}.jpg"
        shutil.copy2(row["source_image"], dest)
        row["export_image"] = str(dest)
        row["split"] = stage

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
    ]
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["stratum"], item["participant"])):
            writer.writerow(row)


def draw(stage, seed, per_stratum):
    n = STAGE_N[stage] if per_stratum is None else per_stratum
    pilot_rows = read_manifest(OUT_ROOT / "pilot" / "manifest.csv")
    if stage == "main" and not pilot_rows:
        raise SystemExit("Draw the pilot first so the main sample can hold those participants out.")

    rows = []
    for stratum in STRATA:
        blocked = excluded_participants(pilot_rows, stratum) if stage == "main" else set()
        rng = random.Random(f"{seed}:{stage}:{stratum}")
        rows.extend(sample_stratum(stratum, n, rng, blocked))

    frame_ids = [row["frame_id"] for row in rows]
    if len(frame_ids) != len(set(frame_ids)):
        raise SystemExit("Duplicate frame ids in the sample")

    out_dir = OUT_ROOT / stage
    export_stage(stage, rows, out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument(
        "--per-stratum",
        type=int,
        default=None,
        help="Override the default count (pilot=3, main=20)",
    )
    args = parser.parse_args()
    out_dir = draw(args.stage, args.seed, args.per_stratum)
    manifest = out_dir / "manifest.csv"
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {}
    for row in rows:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    print(f"Wrote {len(rows)} frames to {out_dir}")
    for stratum in STRATA:
        print(f"  {stratum}: {counts.get(stratum, 0)}")


if __name__ == "__main__":
    main()
