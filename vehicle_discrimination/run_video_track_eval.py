#!/usr/bin/env python3
"""A/B: two-stage geometry vs two-stage + human-like video tracks."""

from __future__ import annotations

import json
from pathlib import Path

from two_stage_vehicle import load_model, reclassify_pred_json, train_from_gt, save_model
from vehicle_metrics import format_report, load_pred_json, prf, score_vehicle
from video_track_motion import HALF_WINDOW

HERE = Path(__file__).resolve().parent
FT = HERE.parent / "human_validation" / "finetune"
DATA = FT / "data"
RUN = FT / "runs" / "20261102_143436"
OUT = HERE / "reports"
OUT.mkdir(parents=True, exist_ok=True)


def f1_pair(per):
    out = {}
    for lab, (hn, mn, ht, mt) in per.items():
        _, _, f1 = prf(mt, mn, ht, hn)
        out[lab] = 100.0 * f1
    # overall vehicles
    hn = sum(per[l][0] for l in per)
    mn = sum(per[l][1] for l in per)
    ht = sum(per[l][2] for l in per)
    mt = sum(per[l][3] for l in per)
    _, _, f1 = prf(mt, mn, ht, hn)
    out["overall"] = 100.0 * f1
    return out


def evaluate(name, items, preds, sink):
    per, confusion, ug, up = score_vehicle(items, preds)
    text = format_report(name, per, confusion, ug, up)
    print(text)
    print()
    sink[name] = {
        "per": {k: list(v) for k, v in per.items()},
        "f1": f1_pair(per),
        "confusion": {"%s->%s" % k: v for k, v in confusion.items()},
        "unmatched_gt": dict(ug),
        "unmatched_pred": dict(up),
    }
    return text


def main():
    train = json.loads((DATA / "train.json").read_text())
    test = json.loads((DATA / "test.json").read_text())
    pilot = json.loads((DATA / "pilot.json").read_text())

    model_path = HERE / "two_stage_vehicle_classifier.json"
    if model_path.exists():
        model = load_model(model_path)
    else:
        model = train_from_gt(train)
        save_model(model, model_path)

    summary = {}
    sections = []

    for split_name, items, pred_name in [
        ("test", test, "finetuned_test.json"),
        ("pilot", pilot, "finetuned_pilot.json"),
    ]:
        pred_path = RUN / pred_name
        # baseline: raw new FT
        sections.append(
            evaluate(
                "%s | new finetuned only" % split_name,
                items,
                load_pred_json(pred_path),
                summary,
            )
        )

        # two-stage WITHOUT video tracks (geometry only): temporarily patch by
        # classifying with empty dino root that has no CSVs — cleaner: call with
        # a fake root. Instead re-run with half_window=0 via a dedicated out.
        geo_out = OUT / ("two_stage_geo_%s.json" % split_name)
        # Use half_window=0 → tracks never long enough → geometry only
        reclassify_pred_json(pred_path, items, model, geo_out, half_window=0)
        sections.append(
            evaluate(
                "%s | two-stage geometry only" % split_name,
                items,
                load_pred_json(geo_out),
                summary,
            )
        )

        vid_out = OUT / ("two_stage_video_%s.json" % split_name)
        reclassify_pred_json(
            pred_path, items, model, vid_out, half_window=HALF_WINDOW
        )
        sections.append(
            evaluate(
                "%s | two-stage + video tracks (±%df ~0.6s)" % (split_name, HALF_WINDOW),
                items,
                load_pred_json(vid_out),
                summary,
            )
        )

        # also try ±25 (~1s)
        vid1_out = OUT / ("two_stage_video1s_%s.json" % split_name)
        reclassify_pred_json(pred_path, items, model, vid1_out, half_window=25)
        sections.append(
            evaluate(
                "%s | two-stage + video tracks (±25f ~1s)" % split_name,
                items,
                load_pred_json(vid1_out),
                summary,
            )
        )

    report = OUT / "video_track_ab_report.txt"
    report.write_text("\n\n".join(sections) + "\n")
    (OUT / "video_track_ab_summary.json").write_text(json.dumps(summary, indent=2))

    # manuscript-ish split table
    lines = [
        r"% Auto-generated: video-track vehicle split comparison",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Setup & Moving F1 & Parked F1 & Overall F1 \\",
        r"\hline",
    ]
    order = [
        ("test | new finetuned only", "New FT only"),
        ("test | two-stage geometry only", "Two-stage (geometry)"),
        ("test | two-stage + video tracks (±%df ~0.6s)" % HALF_WINDOW, "Two-stage + tracks 0.6s"),
        ("test | two-stage + video tracks (±25f ~1s)", "Two-stage + tracks 1s"),
    ]
    for key, label in order:
        f1 = summary[key]["f1"]
        lines.append(
            "%s & %.1f & %.1f & %.1f \\\\"
            % (label, f1["moving vehicle"], f1["parked vehicle"], f1["overall"])
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    tex = OUT / "manuscript_video_track_split_tables.tex"
    tex.write_text("\n".join(lines))

    print("Wrote", report)
    print("Wrote", OUT / "video_track_ab_summary.json")
    print("Wrote", tex)


if __name__ == "__main__":
    main()
