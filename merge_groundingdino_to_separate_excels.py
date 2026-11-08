#!/usr/bin/env python3
import sys
from pathlib import Path as _BootPath
_repo = _BootPath(__file__).resolve()
for _parent in [_repo.parent, *_repo.parents]:
    if (_parent / "aoi_paths.py").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
from aoi_paths import data_path, study_path

from pathlib import Path
import argparse

import pandas as pd


DEFAULT_SEPARATE_ROOT = study_path("seperate_excels/Separate_excels")
DEFAULT_DINO_ROOT = data_path("GroundingDINO_full_frame_output")
DEFAULT_OUTPUT_ROOT = data_path("seperate_excels_with_full_frame_groundingdino/Separate_excels")
DINO_FILENAME = "groundingdino_scene_analysis_full_frames.csv"
TIMESTAMP_COL = "Recording timestamp"


PRIMARY_DINO_COLUMNS = [
    "total_categories_detected",
    "total_objects_detected",
    "num_safety_critical_categories",
    "num_safety_critical_objects",
    "scene_complexity_score",
    "categories_present",
    "category_counts",
    "safety_critical_categories",
    "detected_phrases",
    "detected_categories",
    "detection_boxes",
    "detection_confidences",
    "raw_detected_phrases_before_merge",
    "raw_detected_categories_before_merge",
    "raw_detection_boxes_before_merge",
    "raw_detection_confidences_before_merge",
]

METADATA_RENAMES = {
    "image_path": "groundingdino_image_path",
    "image_name": "groundingdino_image_name",
    "native_frame_timestamp_us": "groundingdino_native_frame_timestamp_us",
    "native_frame_time_sec": "groundingdino_native_frame_time_sec",
    "native_frame_index": "groundingdino_native_frame_index",
}


def replacement_columns():
    return list(METADATA_RENAMES.values()) + PRIMARY_DINO_COLUMNS + ["has_groundingdino_full_frame"]


def participant_folder_from_csv(csv_path: Path) -> str:
    return csv_path.name[:-4] if csv_path.name.lower().endswith(".csv") else csv_path.stem


def read_original_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"Missing {TIMESTAMP_COL!r}: {path}")
    df[TIMESTAMP_COL] = pd.to_numeric(df[TIMESTAMP_COL], errors="coerce").astype("Int64")
    return df


def read_dino_csv(path: Path) -> pd.DataFrame:
    dino = pd.read_csv(path, low_memory=False)
    if "frame_timestamp_us" not in dino.columns:
        raise ValueError(f"Missing 'frame_timestamp_us': {path}")

    dino["frame_timestamp_us"] = pd.to_numeric(dino["frame_timestamp_us"], errors="coerce").astype("Int64")
    dino = dino.dropna(subset=["frame_timestamp_us"]).copy()

    # The early test run left one participant with _dup images. Keep the non-dup row first,
    # then one deterministic row per Tobii timestamp for a clean original-CSV merge.
    dino["_is_dup_image"] = dino.get("image_name", "").astype(str).str.contains(r"_dup\d+", regex=True)
    if "native_frame_index" in dino.columns:
        dino["_native_sort"] = pd.to_numeric(dino["native_frame_index"], errors="coerce")
    else:
        dino["_native_sort"] = 0
    dino = (
        dino.sort_values(["frame_timestamp_us", "_is_dup_image", "_native_sort"])
        .drop_duplicates(subset=["frame_timestamp_us"], keep="first")
        .drop(columns=["_is_dup_image", "_native_sort"])
    )

    keep_cols = ["frame_timestamp_us"]
    keep_cols += [c for c in METADATA_RENAMES if c in dino.columns]
    keep_cols += [c for c in PRIMARY_DINO_COLUMNS if c in dino.columns]
    dino = dino[keep_cols].rename(columns=METADATA_RENAMES)
    dino["has_groundingdino_full_frame"] = True
    return dino


def merge_one(condition: str, csv_path: Path, dino_root: Path, output_root: Path) -> dict:
    participant_folder = participant_folder_from_csv(csv_path)
    dino_path = dino_root / condition / participant_folder / DINO_FILENAME
    original = read_original_csv(csv_path)

    if not dino_path.exists():
        columns_to_replace = [c for c in replacement_columns() if c in original.columns]
        original = original.drop(columns=columns_to_replace)
        for col in replacement_columns():
            original[col] = False if col == "has_groundingdino_full_frame" else pd.NA

        out_dir = output_root / condition
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / csv_path.name
        original.to_csv(out_path, index=False)
        return {
            "condition": condition,
            "participant_folder": participant_folder,
            "status": "copied_missing_dino",
            "original_rows": int(len(original)),
            "output_rows": int(len(original)),
            "dino_unique_timestamps": 0,
            "matched_rows": 0,
            "output_path": str(out_path),
        }

    dino = read_dino_csv(dino_path)

    columns_to_replace = [c for c in replacement_columns() if c in original.columns]
    original = original.drop(columns=columns_to_replace)

    merged = original.merge(
        dino,
        how="left",
        left_on=TIMESTAMP_COL,
        right_on="frame_timestamp_us",
        validate="many_to_one",
    )
    merged = merged.drop(columns=["frame_timestamp_us"])
    merged["has_groundingdino_full_frame"] = merged["has_groundingdino_full_frame"].fillna(False)

    out_dir = output_root / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / csv_path.name
    merged.to_csv(out_path, index=False)

    return {
        "condition": condition,
        "participant_folder": participant_folder,
        "status": "merged",
        "original_rows": int(len(original)),
        "output_rows": int(len(merged)),
        "dino_unique_timestamps": int(len(dino)),
        "matched_rows": int(merged["has_groundingdino_full_frame"].sum()),
        "output_path": str(out_path),
    }


def iter_original_csvs(root: Path, conditions):
    condition_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if conditions:
        wanted = set(conditions)
        condition_dirs = [d for d in condition_dirs if d.name in wanted]
    for condition_dir in condition_dirs:
        for csv_path in sorted(condition_dir.glob("*.csv")):
            yield condition_dir.name, csv_path


def main():
    parser = argparse.ArgumentParser(description="Merge full-frame GroundingDINO results back into separate Tobii CSVs.")
    parser.add_argument("--separate-root", type=Path, default=DEFAULT_SEPARATE_ROOT)
    parser.add_argument("--dino-root", type=Path, default=DEFAULT_DINO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--conditions", nargs="*", default=[])
    parser.add_argument("--summary-name", default="merge_summary.csv")
    args = parser.parse_args()

    summaries = []
    for condition, csv_path in iter_original_csvs(args.separate_root, args.conditions):
        summary = merge_one(condition, csv_path, args.dino_root, args.output_root)
        summaries.append(summary)
        if summary["status"] == "merged":
            print(
                f"{condition}/{summary['participant_folder']}: "
                f"matched {summary['matched_rows']}/{summary['original_rows']} -> {summary['output_path']}"
            )
        else:
            print(f"{condition}/{summary['participant_folder']}: {summary['status']}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root.parent / args.summary_name
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
