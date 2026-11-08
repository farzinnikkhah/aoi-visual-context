#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path as _BootPath
_repo = _BootPath(__file__).resolve()
for _parent in [_repo.parent, *_repo.parents]:
    if (_parent / "aoi_paths.py").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
from aoi_paths import data_path
from pathlib import Path

import pandas as pd

MANUAL_MAP = {
    "1. Moving Vehicle": "1_Moving Vehicle",
    "2. Parked Vehicle": "2_Parked Vehicle",
    "3. Gen Ped TI": "3_Sidewalk Pavement",
    "4. Non-Ped TI": "14_Miscellaneous Infrastructure",
    "5. Non-Lighted Crossing": "5_Crosswalk",
    "6. Lighted Crossing": "6_Crossing Signal",
    "7. General Light Features": "7_General Light Features",
    "8. Pedestrians": "8_Pedestrians",
    "9. Cyclists": "9_Cyclists",
    "11. Storefront": "10_Storefront",
    "13. Other Structures": "11_Buildings",
    "14. Art and Banners": "12_Art and Banners",
    "15. Natural Elements": "13_Natural Elements",
    "16. General Infrastructure": "14_Miscellaneous Infrastructure",
}


def parse_json_list(value):
    try:
        return json.loads(value) if isinstance(value, str) else []
    except Exception:
        return []


def manual_labels(row, manual_cols):
    labels = []
    for col in manual_cols:
        try:
            hit = float(row[col]) == 1
        except Exception:
            hit = False
        if not hit:
            continue
        match = re.search(r" - (.*?)\]$", col)
        if match:
            raw = match.group(1)
            labels.append(MANUAL_MAP.get(raw, raw))
    return labels


def condition_from_path(path, root):
    try:
        return path.relative_to(root).parts[0]
    except Exception:
        return path.parent.name


def main():
    ap = argparse.ArgumentParser(description="Frequency/proportion of a GroundingDINO object within most-attended manual AOIs.")
    ap.add_argument("--root", default=str(data_path("seperate_excels_with_full_frame_groundingdino/Separate_excels")))
    ap.add_argument("--object", required=True, help="GroundingDINO AOI category, e.g., 11_Buildings or 5_Crosswalk")
    ap.add_argument("--top-n", type=int, default=3, help="Top N manual AOI categories treated as most visually attended")
    ap.add_argument("--scope", choices=["overall", "condition"], default="condition", help="Compute top attended AOIs overall or separately by condition")
    ap.add_argument("--out", default=str(data_path("analysis_stats/object_in_attended_areas_stats.csv")))
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("*/*.csv"))
    records = []
    attention_counts_by_scope = defaultdict(Counter)

    for path in files:
        condition = condition_from_path(path, root)
        df = pd.read_csv(path, low_memory=False)
        manual_cols = [c for c in df.columns if c.startswith("AOI hit [WaterStreet_AOI_Snapchat_V2")]
        if not manual_cols or "detected_categories" not in df.columns:
            continue
        for _, row in df.iterrows():
            manuals = manual_labels(row, manual_cols)
            if not manuals:
                continue
            detected = set(parse_json_list(row.get("detected_categories", "[]")))
            has_dino = str(row.get("has_groundingdino_full_frame", "")).lower() in {"true", "1"}
            rec = {
                "condition": condition,
                "participant_file": path.stem,
                "timestamp": row.get("Recording timestamp"),
                "manual_labels": manuals,
                "object_present": args.object in detected,
                "has_groundingdino_full_frame": has_dino,
            }
            records.append(rec)
            for label in manuals:
                attention_counts_by_scope["overall"][label] += 1
                attention_counts_by_scope[condition][label] += 1

    if not records:
        raise SystemExit("No usable records found.")

    if args.scope == "overall":
        top_by_condition = {"overall": [x for x, _ in attention_counts_by_scope["overall"].most_common(args.top_n)]}
    else:
        conditions = sorted({r["condition"] for r in records})
        top_by_condition = {
            c: [x for x, _ in attention_counts_by_scope[c].most_common(args.top_n)]
            for c in conditions
        }

    rows = []
    conditions = sorted({r["condition"] for r in records})
    for condition in conditions:
        top_labels = top_by_condition["overall"] if args.scope == "overall" else top_by_condition[condition]
        subset = [r for r in records if r["condition"] == condition and any(l in top_labels for l in r["manual_labels"])]
        dino_subset = [r for r in subset if r["has_groundingdino_full_frame"]]
        obj_subset = [r for r in dino_subset if r["object_present"]]
        rows.append({
            "condition": condition,
            "object_category": args.object,
            "top_n_attended_aoi": args.top_n,
            "top_attended_aoi_categories": json.dumps(top_labels),
            "manual_fixation_rows_in_top_aoi": len(subset),
            "rows_with_groundingdino_in_top_aoi": len(dino_subset),
            "object_present_rows_in_top_aoi": len(obj_subset),
            "object_present_proportion_among_dino_rows": len(obj_subset) / len(dino_subset) if dino_subset else None,
            "object_present_proportion_among_all_top_aoi_rows": len(obj_subset) / len(subset) if subset else None,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
