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

import argparse
from pathlib import Path

import pandas as pd

from visualize_groundingdino_examples import annotate_row


DEFAULT_DINO_ROOT = data_path("GroundingDINO_full_frame_output")
DEFAULT_OUT_DIR = data_path("GroundingDINO_full_frame_output/random_visual_examples")
CSV_NAME = "groundingdino_scene_analysis_full_frames.csv"


def iter_csvs(root: Path, conditions):
    condition_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if conditions:
        wanted = set(conditions)
        condition_dirs = [d for d in condition_dirs if d.name in wanted]
    for condition_dir in condition_dirs:
        yield from sorted(condition_dir.glob(f"*/*{CSV_NAME}"))


def main():
    parser = argparse.ArgumentParser(description="Draw random GroundingDINO examples across participant CSVs.")
    parser.add_argument("--dino-root", type=Path, default=DEFAULT_DINO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--conditions", nargs="*", default=[])
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--top-pool", type=int, default=2000)
    args = parser.parse_args()

    rows = []
    for csv_path in iter_csvs(args.dino_root, args.conditions):
        df = pd.read_csv(csv_path)
        if "total_objects_detected" in df.columns:
            df = df[df["total_objects_detected"].fillna(0) >= args.min_objects].copy()
            df = df.sort_values("total_objects_detected", ascending=False).head(args.top_pool)
        df["source_dino_csv"] = str(csv_path)
        rows.append(df)

    if not rows:
        raise SystemExit("No rows found.")

    all_rows = pd.concat(rows, ignore_index=True)
    if len(all_rows) == 0:
        raise SystemExit("No rows matched the filters.")

    sample_n = min(args.n, len(all_rows))
    sample = all_rows.sample(n=sample_n, random_state=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rows available: {len(all_rows)}")
    print(f"Rows selected: {sample_n}")
    print(f"Output: {args.out_dir}")

    written = []
    for idx, (_, row) in enumerate(sample.iterrows(), start=1):
        out_path = annotate_row(row, args.out_dir, idx)
        if out_path is not None:
            written.append(out_path)
            print(out_path)

    print(f"Done. Wrote {len(written)} annotated images.")


if __name__ == "__main__":
    main()
