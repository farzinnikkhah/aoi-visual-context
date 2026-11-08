#!/usr/bin/env python3
"""Apply two-stage lane/curb + video-track vehicle labels to corpus CSVs.

Reads GroundingDINO finetuned scene-analysis CSVs, merges overlapping vehicle
boxes, classifies moving vs parked with the current geometry+video pipeline,
and writes updated CSVs (default: parallel output root, originals untouched).

Example (smoke one participant):
  python apply_vehicle_video_to_corpus.py \\
    --csv-root ../GroundingDINO_finetuned_full_frame_output \\
    --out-root ../GroundingDINO_finetuned_full_frame_output_vehicle_video \\
    --conditions I1_Day --participants PARTICIPANT_ID.d

Full corpus:
  python apply_vehicle_video_to_corpus.py \\
    --csv-root ../GroundingDINO_finetuned_full_frame_output \\
    --out-root ../GroundingDINO_finetuned_full_frame_output_vehicle_video

Then remake gaze-merged excels from the new root:
  python ../merge_groundingdino_to_separate_excels.py \\
    --dino-root ../GroundingDINO_finetuned_full_frame_output_vehicle_video \\
    --output-root ../seperate_excels_with_finetuned_vehicle_video/Separate_excels
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from classify_vehicle_motion import code_for, load_model, phrase_for
from two_stage_vehicle import classify_box, merge_vehicle_boxes, _is_vehicle
from video_track_motion import (
    HALF_WINDOW,
    load_participant_index,
    motion_moving_boost,
    parse_native_index,
    track_residual_motion,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CSV_ROOT = HERE.parent / "GroundingDINO_finetuned_full_frame_output"
DEFAULT_OUT_ROOT = HERE.parent / "GroundingDINO_finetuned_full_frame_output_vehicle_video"
DEFAULT_MODEL = HERE / "two_stage_vehicle_classifier.json"

# Keep CSV category coding consistent with production outputs.
SAMPLE_CODED = "1_Moving Vehicle"


def rewrite_row(rec: dict, detections: list) -> dict:
    cats, boxes, scores, phrases = [], [], [], []
    for d in detections:
        cats.append(d["category"])
        boxes.append(d["box"])
        scores.append(float(d.get("score", 0.0)))
        phrases.append(d.get("phrase") or "")
    out = dict(rec)
    out["detected_categories"] = json.dumps(cats)
    out["detection_boxes"] = json.dumps(boxes)
    out["detection_confidences"] = json.dumps(scores)
    out["detected_phrases"] = json.dumps(phrases)
    counts = defaultdict(int)
    for c in cats:
        counts[c] += 1
    out["categories_present"] = json.dumps(sorted(counts.keys()))
    out["category_counts"] = json.dumps(dict(counts))
    out["total_categories_detected"] = str(len(counts))
    out["total_objects_detected"] = str(len(cats))
    # Safety-critical: vehicles, signs, crosswalk, signal, pedestrians, cyclists
    safety_keys = []
    for c in counts:
        t = c.lower().replace("_", " ")
        if any(
            k in t
            for k in (
                "vehicle",
                "traffic sign",
                "crosswalk",
                "crossing signal",
                "pedestrian",
                "cyclist",
            )
        ):
            safety_keys.append(c)
    out["safety_critical_categories"] = json.dumps(sorted(safety_keys))
    out["num_safety_critical_categories"] = str(len(safety_keys))
    out["num_safety_critical_objects"] = str(sum(counts[c] for c in safety_keys))
    return out


def process_one_csv(csv_path: Path, dest: Path, model, half_window: int = HALF_WINDOW):
    by_index = load_participant_index(str(csv_path))
    width, height = 1920.0, 1080.0

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        original = list(reader)

    out_records = []
    n_veh_in = n_veh_out = n_boosted = 0
    for rec in original:
        try:
            cats = json.loads(rec.get("detected_categories") or "[]")
            boxes = json.loads(rec.get("detection_boxes") or "[]")
            scores = json.loads(rec.get("detection_confidences") or "[]")
            phrases = json.loads(rec.get("detected_phrases") or "[]")
        except Exception:
            out_records.append(rec)
            continue

        dets = []
        for i, (c, b) in enumerate(zip(cats, boxes)):
            dets.append(
                {
                    "category": c,
                    "box": [float(v) for v in b],
                    "score": float(scores[i]) if i < len(scores) else 0.0,
                    "phrase": phrases[i] if i < len(phrases) else "",
                }
            )
        n_veh_in += sum(1 for d in dets if _is_vehicle(d["category"]))

        self_idx = parse_native_index(rec.get("image_name") or "")
        merged, others = merge_vehicle_boxes(dets)
        context = [(d["category"], d["box"]) for d in dets]
        track_map = {}
        for i, v in enumerate(merged):
            stats = track_residual_motion(v["box"], self_idx, by_index, half_window)
            if stats is not None:
                # annotate cohort within-frame later; for single-frame we compute after all
                track_map[i] = stats

        # cohort relative fields among this frame's tracked vehicles
        from video_track_motion import annotate_cohort

        stats_list = [track_map.get(i) for i in range(len(merged))]
        stats_list = annotate_cohort(stats_list)
        for i, st in enumerate(stats_list):
            if st is not None:
                track_map[i] = st

        out_vehs = []
        for i, v in enumerate(merged):
            stats = track_map.get(i)
            new_lab, p = classify_box(v["box"], width, height, context, model, stats)
            if stats is not None and motion_moving_boost(stats, box=v["box"]) is not None:
                n_boosted += 1
            out_vehs.append(
                {
                    "category": code_for(new_lab, SAMPLE_CODED),
                    "box": v["box"],
                    "score": float(v.get("score", 0.0)),
                    "phrase": phrase_for(new_lab),
                    "vehicle_p_moving": p,
                }
            )
        n_veh_out += len(out_vehs)
        new_dets = out_vehs + others
        out_records.append(rewrite_row(rec, new_dets))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_records)
    return {
        "rows": len(out_records),
        "vehicles_in": n_veh_in,
        "vehicles_out": n_veh_out,
        "motion_boost_candidates": n_boosted,
        "dest": str(dest),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-root", type=Path, default=DEFAULT_CSV_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                        help="Write corrected CSVs here (recommended). Ignored with --inplace.")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite source CSVs (not recommended; prefer --out-root).")
    parser.add_argument("--model-in", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conditions", nargs="*", default=[])
    parser.add_argument("--participants", nargs="*", default=[])
    parser.add_argument("--half-window", type=int, default=HALF_WINDOW)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N CSVs (0=all).")
    args = parser.parse_args()

    if not args.model_in.is_file():
        raise SystemExit("Missing classifier: %s" % args.model_in)
    model = load_model(args.model_in)

    conds = set(args.conditions) if args.conditions else None
    parts = set(args.participants) if args.participants else None

    csv_paths = sorted(args.csv_root.rglob("groundingdino_scene_analysis_full_frames.csv"))
    selected = []
    for csv_path in csv_paths:
        # .../<stratum>/<participant>/groundingdino_scene_analysis_full_frames.csv
        part = csv_path.parent.name
        stratum = csv_path.parent.parent.name
        if conds is not None and stratum not in conds:
            continue
        if parts is not None and part not in parts:
            continue
        selected.append(csv_path)

    if args.limit > 0:
        selected = selected[: args.limit]

    print("Model:", args.model_in)
    print("Source:", args.csv_root)
    print("Dest:  ", "INPLACE" if args.inplace else args.out_root)
    print("CSVs:  ", len(selected), " half_window=", args.half_window)

    summary = []
    for i, csv_path in enumerate(selected, 1):
        rel = csv_path.relative_to(args.csv_root)
        dest = csv_path if args.inplace else (args.out_root / rel)
        info = process_one_csv(csv_path, dest, model, half_window=args.half_window)
        summary.append({"rel": str(rel), **info})
        print("[%d/%d] %s  veh %d→%d  boost_cand=%d" % (
            i, len(selected), rel, info["vehicles_in"], info["vehicles_out"], info["motion_boost_candidates"]
        ))

    report = (args.out_root if not args.inplace else args.csv_root) / "vehicle_video_apply_summary.json"
    if not args.inplace:
        args.out_root.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"n_csv": len(summary), "files": summary}, indent=2))
    print("Wrote", report)


if __name__ == "__main__":
    main()
