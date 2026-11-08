#!/usr/bin/env python3
"""Test pack_c: retrain two-stage on train vs train+pack_c, score with/without video."""

from __future__ import annotations

import json
from pathlib import Path

from two_stage_vehicle import load_model, reclassify_pred_json, save_model, train_from_gt
from vehicle_metrics import VEHICLE, format_report, load_pred_json, prf, score_vehicle
from video_track_motion import HALF_WINDOW

HERE = Path(__file__).resolve().parent
FT = HERE.parent / "human_validation" / "finetune"
DATA = FT / "data"
RUN = FT / "runs" / "20261102_143436"
OUT = HERE / "reports"
OUT.mkdir(parents=True, exist_ok=True)


def f1_map(per):
    out = {}
    for lab in VEHICLE:
        hn, mn, ht, mt = per[lab]
        _, _, f1 = prf(mt, mn, ht, hn)
        out[lab] = 100.0 * f1
    hn = sum(per[l][0] for l in VEHICLE)
    mn = sum(per[l][1] for l in VEHICLE)
    ht = sum(per[l][2] for l in VEHICLE)
    mt = sum(per[l][3] for l in VEHICLE)
    _, _, of1 = prf(mt, mn, ht, hn)
    out["overall"] = 100.0 * of1
    return out


def evaluate(name, items, preds, sink, sections):
    per, confusion, ug, up = score_vehicle(items, preds)
    text = format_report(name, per, confusion, ug, up)
    print(text)
    print()
    sections.append(text)
    sink[name] = {
        "per": {k: list(v) for k, v in per.items()},
        "f1": f1_map(per),
        "confusion": {"%s->%s" % k: v for k, v in confusion.items()},
    }
    return text


def main():
    test = json.loads((DATA / "test.json").read_text())
    pilot = json.loads((DATA / "pilot.json").read_text())
    train = json.loads((DATA / "train.json").read_text())
    train_plus = json.loads((DATA / "train_plus_pack_c.json").read_text())

    model_base = train_from_gt(train)
    save_model(model_base, HERE / "two_stage_vehicle_classifier.json")

    model_pack = train_from_gt(train_plus)
    save_model(model_pack, HERE / "two_stage_vehicle_classifier_pack_c.json")
    print("Trained base on", len(train), "frames; pack_c-augmented on", len(train_plus))

    summary = {}
    sections = []

    for split_name, items, pred_name in [
        ("test", test, "finetuned_test.json"),
        ("pilot", pilot, "finetuned_pilot.json"),
    ]:
        pred_path = RUN / pred_name

        evaluate(
            "%s | FT only" % split_name,
            items,
            load_pred_json(pred_path),
            summary,
            sections,
        )

        for tag, model in [("base", model_base), ("pack_c", model_pack)]:
            geo = OUT / ("packc_%s_geo_%s.json" % (tag, split_name))
            reclassify_pred_json(pred_path, items, model, geo, half_window=0)
            evaluate(
                "%s | two-stage %s geometry" % (split_name, tag),
                items,
                load_pred_json(geo),
                summary,
                sections,
            )

            vid = OUT / ("packc_%s_video_%s.json" % (tag, split_name))
            reclassify_pred_json(
                pred_path, items, model, vid, half_window=HALF_WINDOW
            )
            evaluate(
                "%s | two-stage %s + video 0.6s" % (split_name, tag),
                items,
                load_pred_json(vid),
                summary,
                sections,
            )

    report = OUT / "pack_c_ab_report.txt"
    report.write_text("\n\n".join(sections) + "\n")
    (OUT / "pack_c_ab_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        r"% Auto-generated: pack_c augmentation A/B (held-out test)",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Setup & Moving F1 & Parked F1 & Overall F1 \\",
        r"\hline",
    ]
    order = [
        ("test | FT only", "New FT only"),
        ("test | two-stage base geometry", "Two-stage (train)"),
        ("test | two-stage base + video 0.6s", "Two-stage + video (train)"),
        ("test | two-stage pack_c geometry", "Two-stage (train+pack\\_c)"),
        ("test | two-stage pack_c + video 0.6s", "Two-stage + video (train+pack\\_c)"),
    ]
    for key, label in order:
        f1 = summary[key]["f1"]
        lines.append(
            "%s & %.1f & %.1f & %.1f \\\\"
            % (label, f1["moving vehicle"], f1["parked vehicle"], f1["overall"])
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    tex = OUT / "manuscript_pack_c_split_tables.tex"
    tex.write_text("\n".join(lines))

    print("Wrote", report)
    print("Wrote", tex)
    print("\nTEST summary:")
    for key, label in order:
        f1 = summary[key]["f1"]
        print(
            "  %-40s  mov=%5.1f  park=%5.1f  all=%5.1f"
            % (label, f1["moving vehicle"], f1["parked vehicle"], f1["overall"])
        )


if __name__ == "__main__":
    main()
