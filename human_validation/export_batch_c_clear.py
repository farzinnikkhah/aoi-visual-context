#!/usr/bin/env python3
"""Export batch C: 120 clear frames for annotation.

15 frames per stratum. Pilot and main participants are preferred unused.
For I3 night strata, after unused participants are exhausted, a second clear
frame is taken from a previously used participant (never the same frame).
Blur cutoffs match the main set: soft < 0.019 or motion > 0.52.
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
    make_frame_id,
    parse_stratum,
    participant_dirs,
    read_manifest,
)

SEED = 20261020
PER_STRATUM = 15
MAX_TRIES = 30
SOFT_CUTOFF = 0.019
MOTION_CUTOFF = 0.52
BATCH = OUT_ROOT / "batch_c"
ZIP_PATH = OUT_ROOT / "main_annotation_pack_c.zip"


def blur_scores(path):
    im = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    ims = im[::2, ::2]
    h, w = ims.shape
    ts = 96
    laps = []
    cohs = []
    for y in range(0, h - ts + 1, ts):
        for x in range(0, w - ts + 1, ts):
            t = ims[y : y + ts, x : x + ts]
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
    top = np.argsort(laps)[-max(3, len(laps) // 5) :]
    motion = float(np.median(cohs[top]))
    cy, cx = h // 2, w // 2
    crop = ims[cy - 256 : cy + 256, cx - 256 : cx + 256]
    crop = crop - crop.mean()
    freq = np.abs(np.fft.fftshift(np.fft.fft2(crop)))
    yy, xx = np.mgrid[-256:256, -256:256]
    rad = np.sqrt(xx * xx + yy * yy)
    fine = float(freq[(rad > 40) & (rad < 160)].mean())
    broad = float(freq[rad < 15].mean())
    soft = fine / (broad + 1e-6)
    return soft, motion, soft < SOFT_CUTOFF or motion > MOTION_CUTOFF


def first_clear_frame(frames, rng, blocked_sources):
    order = [f for f in frames if str(f) not in blocked_sources]
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


def used_sets():
    pilot = read_manifest(OUT_ROOT / "pilot" / "manifest.csv")
    main = read_manifest(OUT_ROOT / "main" / "manifest.csv")
    blocked_participants = {(r["stratum"], r["participant"]) for r in pilot + main}
    blocked_frames = {r["frame_id"] for r in pilot + main}
    blocked_sources = {r["source_image"] for r in pilot + main}
    return blocked_participants, blocked_frames, blocked_sources


def make_row(stratum, participant_dir, source, soft, motion, n_blurry, reused):
    index_match = FRAME_INDEX_RE.search(source.name)
    ts_match = TS_RE.search(source.name)
    if not index_match:
        raise SystemExit("Cannot parse frame index from %s" % source.name)
    frame_index = index_match.group(1)
    frame_id = make_frame_id(stratum, participant_dir.name, frame_index)
    intersection, lighting, period = parse_stratum(stratum)
    with Image.open(source) as image:
        width, height = image.size
    return {
        "frame_id": frame_id,
        "split": "batch_c",
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
        "reused_participant": "1" if reused else "0",
    }


def draw_batch():
    blocked_participants, blocked_frames, blocked_sources = used_sets()
    rows = []
    notes = []
    for stratum in STRATA:
        rng = random.Random("%s:batch_c:%s" % (SEED, stratum))
        all_dirs = participant_dirs(ROOT / stratum)
        unused = [p for p in all_dirs if (stratum, p.name) not in blocked_participants]
        used = [p for p in all_dirs if (stratum, p.name) in blocked_participants]
        rng.shuffle(unused)
        rng.shuffle(used)
        taken = 0
        for participant_dir, reused in [(p, False) for p in unused] + [(p, True) for p in used]:
            if taken >= PER_STRATUM:
                break
            frames = eligible_frames(participant_dir / "frames")
            source, soft, motion, n_blurry = first_clear_frame(frames, rng, blocked_sources)
            if source is None:
                continue
            row = make_row(stratum, participant_dir, source, soft, motion, n_blurry, reused)
            if row["frame_id"] in blocked_frames:
                continue
            blocked_frames.add(row["frame_id"])
            blocked_sources.add(row["source_image"])
            rows.append(row)
            taken += 1
            print(
                "%s %d/%d %s reused=%s"
                % (stratum, taken, PER_STRATUM, row["frame_id"], row["reused_participant"]),
                flush=True,
            )
        if taken < PER_STRATUM:
            raise SystemExit("%s: only found %d clear frames" % (stratum, taken))
        reused_n = sum(1 for r in rows if r["stratum"] == stratum and r["reused_participant"] == "1")
        if reused_n:
            notes.append("%s reused %d previously annotated participants (new frames)" % (stratum, reused_n))
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
        "blurry_frames_skipped",
        "soft_score",
        "motion_score",
        "reused_participant",
    ]
    manifest = BATCH / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["stratum"], item["participant"])):
            writer.writerow(row)
    return image_dir


def write_zip(image_dir, notes):
    lines = [
        "Main annotation images, pack C (120 frames).",
        "",
        "Use the same rules as the annotation protocol and the later guideline notes.",
        "There are 120 frames, 15 in each of the 8 conditions.",
        "Blurry frames were left out.",
        "These images have no model boxes.",
        "Annotate independently, and do not look at the other annotator's boxes.",
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
        for path in sorted(image_dir.glob("*.jpg")):
            archive.write(path, "images/%s" % path.name)


def main():
    rows, notes = draw_batch()
    image_dir = export(rows)
    write_zip(image_dir, notes)
    print("frames", len(rows))
    for note in notes:
        print("note:", note)
    print("zip", ZIP_PATH)


if __name__ == "__main__":
    main()
