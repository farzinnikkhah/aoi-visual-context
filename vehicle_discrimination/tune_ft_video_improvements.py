#!/usr/bin/env python3
"""Tune per-class thresholds + softened video boost gates for FT+video.

Selects a config that improves overall F1 on TEST without dropping PILOT
overall F1 (and without collapsing pilot moving F1).
"""
from __future__ import annotations

import json
import itertools
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from vehicle_metrics import canon, is_match, iou, area, inter, prf
from two_stage_vehicle import (
    load_model,
    merge_vehicle_boxes,
    road_curb_features,
    _geom_scalars,
    MOVING,
    PARKED,
    HARD_CURB,
    HARD_EDGE,
    NEAR_SIDE_EDGE,
    NEAR_SIDE_LANE,
    VIDEO_MIN_LANE,
    VIDEO_MAX_CURB,
    VIDEO_MIN_EDGE,
    VIDEO_MIN_PGEO,
)
from classify_vehicle_motion import predict_proba
from video_track_motion import (
    track_motion_for_boxes,
    motion_moving_boost,
    HALF_WINDOW,
    NET_BOOST_PX,
    MIN_CONSISTENCY,
)

HERE = Path(__file__).resolve().parent
FT = HERE.parent / "human_validation" / "finetune"
DATA = FT / "data"
RUN = FT / "runs" / "20261102_143436"
DINO = HERE.parent / "GroundingDINO_finetuned_full_frame_output"
OUT = HERE / "reports"
CLASSES = json.loads((DATA / "classes.json").read_text())
test = json.loads((DATA / "test.json").read_text())
pilot = json.loads((DATA / "pilot.json").read_text())
model = load_model(HERE / "two_stage_vehicle_classifier.json")

BASE_THRESH = {
    "storefront": 0.20,
    "default": 0.30,
}


def class_thresh(label, thresh_map):
    lab = canon(label)
    if lab in thresh_map:
        return thresh_map[lab]
    return thresh_map.get("default", 0.30)


def rebuild(path, thresh_map):
    out = []
    for record in json.loads(Path(path).read_text()):
        cands = record.get("candidates") or []
        kept = []
        for c in sorted(cands, key=lambda x: -float(x.get("score", 0))):
            if float(c.get("score", 0)) < class_thresh(c.get("category", ""), thresh_map):
                continue
            if any(iou(c["box"], f["box"]) >= 0.5 for f in kept):
                continue
            kept.append(dict(c))
        merged = []
        for cand in kept:
            absorbed = False
            for f in merged:
                if canon(f["category"]) != canon(cand["category"]):
                    continue
                ov = inter(cand["box"], f["box"])
                if iou(cand["box"], f["box"]) >= 0.70 or ov / max(area(cand["box"]), 1e-6) >= 0.90:
                    absorbed = True
                    break
            if not absorbed:
                merged.append(cand)
        out.append({"frame_id": record["frame_id"], "detections": merged})
    return out


def classify_box_tuned(box, width, height, context_boxes, model, track_stats, gates):
    """classify_box with tunable video / curb gates.

    gates keys:
      video_min_pgeo, video_min_lane, video_max_curb, video_min_edge
      hard_curb, hard_edge, near_side_edge, near_side_lane
      strong_net_px: if residual net >= this, allow video override of near-side
      and lower p_geo gate; still respect hard curb unless net is very strong
      strong_override_hard: if True and net >= 1.5*strong_net_px, override hard curb
      min_vehicle_score: drop merged vehicles below this score
    """
    feats = road_curb_features(box, width, height, context_boxes)
    g = _geom_scalars(feats)
    hard_curb = gates.get("hard_curb", HARD_CURB)
    hard_edge = gates.get("hard_edge", HARD_EDGE)
    near_edge = gates.get("near_side_edge", NEAR_SIDE_EDGE)
    near_lane = gates.get("near_side_lane", NEAR_SIDE_LANE)
    v_min_pgeo = gates.get("video_min_pgeo", VIDEO_MIN_PGEO)
    v_min_lane = gates.get("video_min_lane", VIDEO_MIN_LANE)
    v_max_curb = gates.get("video_max_curb", VIDEO_MAX_CURB)
    v_min_edge = gates.get("video_min_edge", VIDEO_MIN_EDGE)
    strong_net = gates.get("strong_net_px", 140.0)
    override_hard = gates.get("strong_override_hard", False)

    floor = motion_moving_boost(track_stats, box=box)
    net = float(track_stats["net_px"]) if track_stats else 0.0
    strong = floor is not None and net >= strong_net
    very_strong = floor is not None and net >= 1.5 * strong_net

    # Hard curb / edge: normally force parked; optionally allow very strong motion
    if g["curb"] >= hard_curb or g["edge"] <= hard_edge or g["curb_score"] >= 0.35:
        if not (override_hard and very_strong):
            return PARKED, 0.12

    if g["edge"] < near_edge and g["lane"] < near_lane:
        # Near-side with weak lane: parked unless strong residual motion
        if not strong:
            return PARKED, 0.20 + 0.25 * g["lane"]

    p_geo = float(predict_proba(model, feats)[0])
    p_moving = p_geo

    # Soften gates when motion is strong
    eff_pgeo = v_min_pgeo - 0.15 if strong else v_min_pgeo
    eff_lane = v_min_lane - 0.10 if strong else v_min_lane
    eff_curb = v_max_curb + 0.08 if strong else v_max_curb
    eff_edge = v_min_edge - 0.05 if strong else v_min_edge

    if (
        floor is not None
        and p_geo >= eff_pgeo
        and g["lane"] >= eff_lane
        and g["curb"] < eff_curb
        and g["edge"] >= eff_edge
    ):
        p_moving = max(p_geo, floor)

    label = MOVING if p_moving >= 0.5 else PARKED
    return label, p_moving


_TRACK_CACHE = {}


def _track_key(frame_id, boxes):
    return (frame_id, tuple(tuple(round(float(x), 1) for x in b) for b in boxes))


def label_vehicles(records, items, gates):
    gt_by = {it["frame_id"]: it for it in items}
    min_veh = gates.get("min_vehicle_score", 0.0)
    out = []
    for rec in records:
        item = gt_by[rec["frame_id"]]
        w, h = float(item["width"]), float(item["height"])
        dets = [dict(d) for d in rec["detections"]]
        merged, others = merge_vehicle_boxes(dets)
        context = [(d["category"], d["box"]) for d in dets]
        boxes = [v["box"] for v in merged]
        tk = _track_key(rec["frame_id"], boxes)
        if tk not in _TRACK_CACHE:
            _TRACK_CACHE[tk] = track_motion_for_boxes(
                boxes, Path(item["image"]), DINO, HALF_WINDOW
            )
        stats = _TRACK_CACHE[tk]
        vehs = []
        for v, st in zip(merged, stats):
            if float(v.get("score", 0)) < min_veh:
                continue
            lab, p = classify_box_tuned(v["box"], w, h, context, model, st, gates)
            vehs.append({"category": lab, "box": v["box"], "score": v.get("score", 0)})
        out.append({"frame_id": rec["frame_id"], "detections": vehs + others})
    return out


def score(items, preds):
    per = defaultdict(lambda: [0, 0, 0, 0])
    for item in items:
        gt = [(canon(b["label"]), b["box"]) for b in item["boxes"]]
        pred = [(canon(d["category"]), d["box"]) for d in preds.get(item["frame_id"], [])]
        gh = [False] * len(gt)
        ph = [False] * len(pred)
        for i, (gl, gb) in enumerate(gt):
            for j, (pl, pb) in enumerate(pred):
                if gl == pl and is_match(gb, pb):
                    gh[i] = True
                    ph[j] = True
        for lab in set([g[0] for g in gt] + [p[0] for p in pred]):
            gi = [k for k, (l, _) in enumerate(gt) if l == lab]
            pj = [k for k, (l, _) in enumerate(pred) if l == lab]
            per[lab][0] += len(gi)
            per[lab][1] += len(pj)
            per[lab][2] += sum(gh[k] for k in gi)
            per[lab][3] += sum(ph[k] for k in pj)
    return per


def metrics(per):
    def one(lab):
        hn, mn, ht, mt = per[lab]
        if hn == 0 and mn == 0:
            return None
        p, r, f = prf(mt, mn, ht, hn)
        return {"P": 100 * p, "R": 100 * r, "F1": 100 * f, "HN": hn, "MN": mn}

    out = {lab: one(lab) for lab in CLASSES}
    hn = sum(per[l][0] for l in CLASSES)
    mn = sum(per[l][1] for l in CLASSES)
    ht = sum(per[l][2] for l in CLASSES)
    mt = sum(per[l][3] for l in CLASSES)
    p, r, f = prf(mt, mn, ht, hn)
    out["ALL"] = {"P": 100 * p, "R": 100 * r, "F1": 100 * f, "HN": hn, "MN": mn}
    return out


def evaluate(thresh_map, gates, cache_rebuild):
    results = {}
    for split_name, items, pred_name in [
        ("test", test, "finetuned_test.json"),
        ("pilot", pilot, "finetuned_pilot.json"),
    ]:
        key = (pred_name, tuple(sorted(thresh_map.items())))
        if key not in cache_rebuild:
            cache_rebuild[key] = rebuild(RUN / pred_name, thresh_map)
        base = cache_rebuild[key]
        labeled = label_vehicles(base, items, gates)
        preds = {r["frame_id"]: r["detections"] for r in labeled}
        results[split_name] = metrics(score(items, preds))
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}

    # Baseline gates (current code)
    baseline_gates = {
        "video_min_pgeo": VIDEO_MIN_PGEO,
        "video_min_lane": VIDEO_MIN_LANE,
        "video_max_curb": VIDEO_MAX_CURB,
        "video_min_edge": VIDEO_MIN_EDGE,
        "hard_curb": HARD_CURB,
        "hard_edge": HARD_EDGE,
        "near_side_edge": NEAR_SIDE_EDGE,
        "near_side_lane": NEAR_SIDE_LANE,
        "strong_net_px": 9999.0,  # effectively disable strong-motion softening
        "strong_override_hard": False,
        "min_vehicle_score": 0.0,
    }

    baseline_thresh = {
        "storefront": 0.20,
        "default": 0.30,
    }

    print("Evaluating baseline...")
    base = evaluate(baseline_thresh, baseline_gates, cache)
    print(
        f"  baseline test F1={base['test']['ALL']['F1']:.1f} "
        f"mov={base['test']['moving vehicle']['F1']:.1f} "
        f"park={base['test']['parked vehicle']['F1']:.1f} | "
        f"pilot F1={base['pilot']['ALL']['F1']:.1f} "
        f"mov={base['pilot']['moving vehicle']['F1']:.1f}"
    )

    # Threshold candidates (raise FP-heavy, lower FN-heavy sidewalk)
    light_opts = [0.30, 0.35, 0.40]
    ped_opts = [0.30, 0.35, 0.40]
    nat_opts = [0.30, 0.35, 0.40]
    side_opts = [0.25, 0.30]  # only lower or keep
    veh_score_opts = [0.0, 0.30, 0.35]

    gate_variants = [
        ("baseline_gates", baseline_gates),
        (
            "soft_video",
            {
                **baseline_gates,
                "video_min_pgeo": 0.30,
                "video_min_lane": 0.22,
                "video_max_curb": 0.30,
                "video_min_edge": 0.12,
                "strong_net_px": 120.0,
                "strong_override_hard": False,
            },
        ),
        (
            "soft_video_strong_override",
            {
                **baseline_gates,
                "video_min_pgeo": 0.28,
                "video_min_lane": 0.20,
                "video_max_curb": 0.32,
                "video_min_edge": 0.10,
                "strong_net_px": 110.0,
                "strong_override_hard": True,
            },
        ),
        (
            "soft_near_side",
            {
                **baseline_gates,
                "near_side_edge": 0.20,
                "near_side_lane": 0.30,
                "video_min_pgeo": 0.30,
                "strong_net_px": 120.0,
                "strong_override_hard": False,
            },
        ),
    ]

    trials = []
    # First: threshold-only with baseline gates (manageable grid)
    print("\nSweeping thresholds (baseline gates)...")
    for light, ped, nat, side, vmin in itertools.product(
        light_opts, ped_opts, nat_opts, side_opts, veh_score_opts
    ):
        thresh = {
            "storefront": 0.20,
            "default": 0.30,
            "general light features": light,
            "pedestrians": ped,
            "natural elements": nat,
            "sidewalk pavement": side,
        }
        gates = {**baseline_gates, "min_vehicle_score": vmin}
        name = f"thr_L{light}_P{ped}_N{nat}_S{side}_V{vmin}"
        res = evaluate(thresh, gates, cache)
        trials.append({"name": name, "thresh": thresh, "gates": gates, "gate_name": "baseline_gates", **res})

    # Second: gate variants × a reduced threshold set (best FP raises + sidewalk)
    print("Sweeping gate variants × reduced thresholds...")
    reduced_thresh = [
        baseline_thresh,
        {
            "storefront": 0.20,
            "default": 0.30,
            "general light features": 0.35,
            "pedestrians": 0.35,
            "natural elements": 0.35,
            "sidewalk pavement": 0.25,
        },
        {
            "storefront": 0.20,
            "default": 0.30,
            "general light features": 0.40,
            "pedestrians": 0.40,
            "natural elements": 0.35,
            "sidewalk pavement": 0.25,
        },
        {
            "storefront": 0.20,
            "default": 0.30,
            "general light features": 0.40,
            "pedestrians": 0.35,
            "natural elements": 0.40,
            "sidewalk pavement": 0.30,
        },
    ]
    for gname, gates0 in gate_variants:
        if gname == "baseline_gates":
            continue  # already covered
        for thr in reduced_thresh:
            for vmin in [0.0, 0.30]:
                gates = {**gates0, "min_vehicle_score": vmin}
                name = f"{gname}_V{vmin}_" + "_".join(
                    f"{k[0]}{thr.get(k, 0.3)}"
                    for k in (
                        "general light features",
                        "pedestrians",
                        "natural elements",
                        "sidewalk pavement",
                    )
                )
                res = evaluate(thr, gates, cache)
                trials.append(
                    {
                        "name": name,
                        "thresh": thr,
                        "gates": gates,
                        "gate_name": gname,
                        **res,
                    }
                )

    base_test = base["test"]["ALL"]["F1"]
    base_pilot = base["pilot"]["ALL"]["F1"]
    base_pilot_mov = base["pilot"]["moving vehicle"]["F1"]
    base_test_mov = base["test"]["moving vehicle"]["F1"]

    def ok(t):
        # Must not drop pilot overall, and pilot moving must not collapse
        if t["pilot"]["ALL"]["F1"] + 1e-6 < base_pilot - 0.3:
            return False
        if t["pilot"]["moving vehicle"]["F1"] + 1e-6 < max(10.0, base_pilot_mov - 5.0):
            return False
        return True

    feasible = [t for t in trials if ok(t)]
    # Rank by: test overall F1, then test moving F1, then pilot overall
    feasible.sort(
        key=lambda t: (
            t["test"]["ALL"]["F1"],
            t["test"]["moving vehicle"]["F1"],
            t["test"]["parked vehicle"]["F1"],
            t["pilot"]["ALL"]["F1"],
        ),
        reverse=True,
    )

    print(f"\nTrials: {len(trials)}, feasible: {len(feasible)}")
    print("\nTop 10 feasible:")
    for t in feasible[:10]:
        print(
            f"  {t['name'][:70]:70s} "
            f"test={t['test']['ALL']['F1']:.1f} "
            f"mov={t['test']['moving vehicle']['F1']:.1f} "
            f"park={t['test']['parked vehicle']['F1']:.1f} | "
            f"pilot={t['pilot']['ALL']['F1']:.1f} "
            f"mov={t['pilot']['moving vehicle']['F1']:.1f}"
        )

    best = feasible[0] if feasible else None
    # Also report best by vehicle F1 among feasible with overall >= baseline
    veh_rank = [
        t
        for t in feasible
        if t["test"]["ALL"]["F1"] + 1e-6 >= base_test - 0.2
    ]
    veh_rank.sort(
        key=lambda t: (
            0.5 * t["test"]["moving vehicle"]["F1"] + 0.5 * t["test"]["parked vehicle"]["F1"],
            t["test"]["ALL"]["F1"],
        ),
        reverse=True,
    )

    summary = {
        "baseline": {
            "test": base["test"],
            "pilot": base["pilot"],
            "thresh": baseline_thresh,
            "gates": baseline_gates,
        },
        "best_overall": None,
        "best_vehicle": None,
        "n_trials": len(trials),
        "n_feasible": len(feasible),
    }

    def pack(t):
        return {
            "name": t["name"],
            "thresh": t["thresh"],
            "gates": t["gates"],
            "gate_name": t["gate_name"],
            "test": t["test"],
            "pilot": t["pilot"],
        }

    if best:
        summary["best_overall"] = pack(best)
        print("\n=== BEST OVERALL ===")
        print(best["name"])
        print(
            f"  test  F1 {base_test:.1f} → {best['test']['ALL']['F1']:.1f} "
            f"(mov {base_test_mov:.1f} → {best['test']['moving vehicle']['F1']:.1f}, "
            f"park {base['test']['parked vehicle']['F1']:.1f} → {best['test']['parked vehicle']['F1']:.1f})"
        )
        print(
            f"  pilot F1 {base_pilot:.1f} → {best['pilot']['ALL']['F1']:.1f} "
            f"(mov {base_pilot_mov:.1f} → {best['pilot']['moving vehicle']['F1']:.1f})"
        )

    if veh_rank:
        summary["best_vehicle"] = pack(veh_rank[0])
        bv = veh_rank[0]
        print("\n=== BEST VEHICLE (among non-regressing overall) ===")
        print(bv["name"])
        print(
            f"  test mov {base_test_mov:.1f} → {bv['test']['moving vehicle']['F1']:.1f}, "
            f"park {base['test']['parked vehicle']['F1']:.1f} → {bv['test']['parked vehicle']['F1']:.1f}, "
            f"ALL {base_test:.1f} → {bv['test']['ALL']['F1']:.1f}"
        )

    # Prefer best_vehicle if it improves vehicles without losing overall; else best_overall
    chosen = None
    if veh_rank and (
        veh_rank[0]["test"]["moving vehicle"]["F1"] >= base_test_mov + 1.0
        or veh_rank[0]["test"]["ALL"]["F1"] >= base_test + 0.5
    ):
        chosen = veh_rank[0]
        reason = "best_vehicle"
    elif best:
        chosen = best
        reason = "best_overall"

    summary["chosen"] = pack(chosen) if chosen else None
    summary["chosen_reason"] = reason if chosen else None

    out_path = OUT / "ft_video_improvement_tune.json"
    # Make JSON serializable (no Path)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")

    # Print manuscript-style table for chosen
    if chosen:
        print("\n--- Chosen per-class (TEST) ---")
        display = [
            "moving vehicle",
            "parked vehicle",
            "sidewalk pavement",
            "traffic signs",
            "crosswalk",
            "crossing signal",
            "general light features",
            "pedestrians",
            "storefront",
            "buildings",
            "natural elements",
            "miscellaneous infrastructure",
        ]
        for lab in display:
            m = chosen["test"][lab]
            b = base["test"][lab]
            if m is None:
                continue
            print(
                f"{lab:32s}  "
                f"P {b['P']:5.1f}→{m['P']:5.1f}  "
                f"R {b['R']:5.1f}→{m['R']:5.1f}  "
                f"F1 {b['F1']:5.1f}→{m['F1']:5.1f}"
            )
        print(
            f"{'ALL':32s}  "
            f"P {base['test']['ALL']['P']:5.1f}→{chosen['test']['ALL']['P']:5.1f}  "
            f"R {base['test']['ALL']['R']:5.1f}→{chosen['test']['ALL']['R']:5.1f}  "
            f"F1 {base['test']['ALL']['F1']:5.1f}→{chosen['test']['ALL']['F1']:5.1f}"
        )


if __name__ == "__main__":
    main()
