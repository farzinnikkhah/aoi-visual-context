#!/usr/bin/env python3
"""Export the main annotation set, skipping blurry frames.

Same draw as sample_validation_frames.py --stage main:
20 participants per stratum, pilot participants in that stratum held out,
seed 20261006. If the drawn frame is blurry, later frames from that
participant are tried. A participant with no clear frame in the first
30 tries is replaced by the next held-out participant.
"""

import csv
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from sample_validation_frames import (
    FRAME_INDEX_RE,
    OUT_ROOT,
    ROOT,
    STRATA,
    TS_RE,
    eligible_frames,
    excluded_participants,
    make_frame_id,
    parse_stratum,
    participant_dirs,
    read_manifest,
)

SEED = 20261006
PER_STRATUM = 20
MAX_TRIES = 30
SOFT_CUTOFF = 0.019
MOTION_CUTOFF = 0.52
MAIN = OUT_ROOT / "main"
ZIP_PATH = OUT_ROOT / "main_annotation_images.zip"


def blur_scores(path):
    im = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    ims = im[::2, ::2]
    h, w = ims.shape
    ts = 96
    laps = []
    cohs = []
    for y in range(0, h - ts + 1, ts):
        for x in range(0, w - ts + 1, ts):
            t = ims[y:y + ts, x:x + ts]
            gx = t[:, 1:] - t[:, :-1]
            gy = t[1:, :] - t[:-1, :]
            gx, gy = gx[1:, :], gy[:, 1:]
            hh = min(gx.shape[0], gy.shape[0])
            ww = min(gx.shape[1], gy.shape[1])
            gx, gy = gx[:hh, :ww], gy[:hh, :ww]
            jxx = float(np.mean(gx * gx))
            jyy = float(np.mean(gy * gy))
            jxy = float(np.mean(gx * gy))
            tmp = ((jxx - jyy) ** 2 + 4.0 * jxy * jxy) ** 0.5
            lam1 = 0.5 * (jxx + jyy + tmp)
            lam2 = 0.5 * (jxx + jyy - tmp)
            coh = (lam1 - lam2) / (lam1 + lam2) if (lam1 + lam2) > 1e-6 else 0.0
            c = t[1:-1, 1:-1]
            lap = -4.0 * c + t[:-2, 1:-1] + t[2:, 1:-1] + t[1:-1, :-2] + t[1:-1, 2:]
            laps.append(float(lap.var()))
            cohs.append(coh)
    laps = np.asarray(laps)
    cohs = np.asarray(cohs)
    top = np.argsort(laps)[-max(3, len(laps) // 5):]
    motion = float(np.median(cohs[top]))
    cy, cx = h // 2, w // 2
    crop = ims[cy - 256:cy + 256, cx - 256:cx + 256]
    crop = crop - crop.mean()
    freq = np.abs(np.fft.fftshift(np.fft.fft2(crop)))
    yy, xx = np.mgrid[-256:256, -256:256]
    rad = np.sqrt(xx * xx + yy * yy)
    fine = float(freq[(rad > 40) & (rad < 160)].mean())
    broad = float(freq[rad < 15].mean())
    soft = fine / (broad + 1e-6)
    return soft, motion, soft < SOFT_CUTOFF or motion > MOTION_CUTOFF


def first_clear_frame(frames, rng):
    order = frames[:]
    rng.shuffle(order)
    checked = 0
    for source in order:
        checked += 1
        soft, motion, blurry = blur_scores(source)
        if not blurry:
            return source, soft, motion, checked - 1
        if checked >= MAX_TRIES:
            return None, soft, motion, checked
    return None, None, None, checked


def draw_main():
    pilot_rows = read_manifest(OUT_ROOT / "pilot" / "manifest.csv")
    if not pilot_rows:
        raise SystemExit("Pilot manifest is missing.")
    pilot_ids = {row["frame_id"] for row in pilot_rows}
    rows = []
    skipped = []
    for stratum in STRATA:
        blocked = excluded_participants(pilot_rows, stratum)
        rng = random.Random("%s:%s:%s" % (SEED, "main", stratum))
        candidates = [
            path for path in participant_dirs(ROOT / stratum) if path.name not in blocked
        ]
        rng.shuffle(candidates)
        intersection, lighting, period = parse_stratum(stratum)
        taken = 0
        for participant_dir in candidates:
            if taken >= PER_STRATUM:
                break
            frames = eligible_frames(participant_dir / "frames")
            source, soft, motion, n_blurry = first_clear_frame(frames, rng)
            if source is None:
                skipped.append((stratum, participant_dir.name, n_blurry))
                continue
            index_match = FRAME_INDEX_RE.search(source.name)
            ts_match = TS_RE.search(source.name)
            if not index_match:
                raise SystemExit("Cannot parse frame index from %s" % source.name)
            frame_index = index_match.group(1)
            frame_id = make_frame_id(stratum, participant_dir.name, frame_index)
            if frame_id in pilot_ids:
                raise SystemExit("Pilot frame reused: %s" % frame_id)
            with Image.open(source) as image:
                width, height = image.size
            rows.append(
                {
                    "frame_id": frame_id,
                    "split": "main",
                    "stratum": stratum,
                    "intersection": intersection,
                    "lighting": lighting,
                    "period": period,
                    "participant": participant_dir.name,
                    "frame_index": frame_index,
                    "recording_timestamp_us": ts_match.group(1) if ts_match else "",
                    "width": str(width),
                    "height": str(height),
                    "source_image": str(source),
                    "blurry_frames_skipped": str(n_blurry),
                    "soft_score": "%.4f" % soft,
                    "motion_score": "%.3f" % motion,
                }
            )
            taken += 1
            print("%s %d/%d %s" % (stratum, taken, PER_STRATUM, frame_id), flush=True)
        if taken < PER_STRATUM:
            raise SystemExit("%s: only found %d clear frames" % (stratum, taken))
    return rows, skipped


def export(rows):
    image_dir = MAIN / "images"
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
        "blurry_frames_skipped",
        "soft_score",
        "motion_score",
    ]
    manifest = MAIN / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["stratum"], item["participant"])):
            writer.writerow(row)
    return image_dir


def write_zip(image_dir):
    note = image_dir.parent / "README.txt"
    note.write_text(
        "\n".join(
            [
                "Main annotation images",
                "",
                "Use the same rules as the annotation protocol.",
                "There are 160 frames, 20 in each of the 8 conditions.",
                "Blurry frames were left out.",
                "These images have no model boxes.",
                "Annotate independently, and do not look at the other annotator's boxes.",
                "",
            ]
        )
    )
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(note, "README.txt")
        for path in sorted(image_dir.glob("*.jpg")):
            archive.write(path, "images/%s" % path.name)
    note.unlink()


def main():
    rows, skipped = draw_main()
    image_dir = export(rows)
    write_zip(image_dir)
    print("frames", len(rows))
    print("replaced_participants", len(skipped))
    for item in skipped:
        print("  replaced", item)
    print("zip", ZIP_PATH)


if __name__ == "__main__":
    main()
