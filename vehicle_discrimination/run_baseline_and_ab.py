#!/usr/bin/env python3
"""Baseline + A/B vehicle metrics: current, reclassified, and optional prompt-smoke preds."""

from __future__ import annotations

import json
from pathlib import Path

from classify_vehicle_motion import load_model, reclassify_pred_json, save_model, train_from_gt
from vehicle_metrics import format_report, load_pred_json, score_vehicle

HERE = Path(__file__).resolve().parent
FT = HERE.parent / "human_validation" / "finetune"
DATA = FT / "data"
RUN = FT / "runs" / "20261019_154219"
OUT = HERE / "reports"
OUT.mkdir(parents=True, exist_ok=True)


def evaluate(name, items, preds, sink):
    per, confusion, ug, up = score_vehicle(items, preds)
    text = format_report(name, per, confusion, ug, up)
    print(text)
    print()
    sink[name] = {
        "per": {k: list(v) for k, v in per.items()},
        "confusion": {"%s->%s" % k: v for k, v in confusion.items()},
        "unmatched_gt": dict(ug),
        "unmatched_pred": dict(up),
    }
    return text


def main():
    train = json.loads((DATA / "train.json").read_text())
    test = json.loads((DATA / "test.json").read_text())
    pilot = json.loads((DATA / "pilot.json").read_text())

    model_path = HERE / "vehicle_classifier.json"
    model = train_from_gt(train)
    save_model(model, model_path)

    summary = {}
    sections = []

    for split_name, items, pred_name in [
        ("test", test, "finetuned_test.json"),
        ("pilot", pilot, "finetuned_pilot.json"),
    ]:
        pred_path = RUN / pred_name
        sections.append(
            evaluate(
                "current finetuned, %s" % split_name,
                items,
                load_pred_json(pred_path),
                summary,
            )
        )
        re_path = OUT / (pred_path.stem + "_reclassified.json")
        reclassify_pred_json(pred_path, items, model, re_path)
        sections.append(
            evaluate(
                "finetuned + posthoc classifier, %s" % split_name,
                items,
                load_pred_json(re_path),
                summary,
            )
        )

    # Optional prompt-smoke predictions if present
    for split_name, items in [("test", test), ("pilot", pilot)]:
        smoke = HERE / ("prompt_smoke_%s.json" % split_name)
        if smoke.exists():
            sections.append(
                evaluate("prompt-aligned smoke, %s" % split_name, items, load_pred_json(smoke), summary)
            )
            re_path = OUT / (smoke.stem + "_reclassified.json")
            reclassify_pred_json(smoke, items, model, re_path)
            sections.append(
                evaluate(
                    "prompt-aligned smoke + posthoc, %s" % split_name,
                    items,
                    load_pred_json(re_path),
                    summary,
                )
            )

    report = OUT / "vehicle_ab_report.txt"
    report.write_text("\n\n".join(sections) + "\n")
    (OUT / "vehicle_ab_summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote", report)
    print("Wrote", OUT / "vehicle_ab_summary.json")


if __name__ == "__main__":
    main()
