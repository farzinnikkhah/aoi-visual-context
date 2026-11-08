import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_tobii_timeline(tobii_csv: Path, timestamp_col: str) -> pd.DataFrame:
    df = pd.read_csv(tobii_csv)
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column {timestamp_col!r} in {tobii_csv}")

    df = df.copy()
    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    df[timestamp_col] = df[timestamp_col].astype(np.int64)
    return df


def align_frames_to_tobii(
    frame_manifest: Path,
    tobii_csv: Path,
    out_csv: Path,
    frame_time_col: str = "frame_time_sec",
    timestamp_col: str = "Recording timestamp",
    tolerance_sec: float | None = None,
) -> None:
    frames = pd.read_csv(frame_manifest)
    tobii = build_tobii_timeline(tobii_csv, timestamp_col)

    if frame_time_col not in frames.columns:
        raise ValueError(f"Missing frame time column {frame_time_col!r}")

    frames = frames.copy()
    frames[frame_time_col] = pd.to_numeric(frames[frame_time_col], errors="coerce")
    frames = frames.dropna(subset=[frame_time_col]).sort_values(frame_time_col)
    frames["frame_timestamp_us"] = (frames[frame_time_col] * 1_000_000).round().astype(np.int64)

    aligned = pd.merge_asof(
        frames,
        tobii,
        left_on="frame_timestamp_us",
        right_on=timestamp_col,
        direction="nearest",
        suffixes=("", "_tobii"),
    )

    aligned["tobii_time_delta_sec"] = (
        aligned["frame_timestamp_us"] - aligned[timestamp_col]
    ).abs() / 1_000_000.0

    if tolerance_sec is not None:
        aligned["tobii_match_within_tolerance"] = aligned["tobii_time_delta_sec"] <= tolerance_sec
    else:
        aligned["tobii_match_within_tolerance"] = True

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Align extracted video frame manifest rows to nearest Tobii rows."
    )
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--tobii-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--frame-time-col", default="frame_time_sec")
    parser.add_argument("--timestamp-col", default="Recording timestamp")
    parser.add_argument("--tolerance-sec", type=float, default=None)
    args = parser.parse_args()

    align_frames_to_tobii(
        frame_manifest=args.frame_manifest,
        tobii_csv=args.tobii_csv,
        out_csv=args.out_csv,
        frame_time_col=args.frame_time_col,
        timestamp_col=args.timestamp_col,
        tolerance_sec=args.tolerance_sec,
    )


if __name__ == "__main__":
    main()
