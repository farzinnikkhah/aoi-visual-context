#!/usr/bin/env python3
"""Short-track vehicle motion from consecutive corpus frames (video-as-humans).

Humans rarely decide moving vs parked from one still. Over ~0.5–1 s they watch
whether a car *drifts relative to the curb / buildings*. Absolute box motion is
misleading under ego-motion (walking camera), so we:

  1. Link vehicle boxes into short tracks across ±HALF_WINDOW frames
  2. Estimate ego translation from matched static scene boxes
  3. Measure residual track motion (vehicle − ego)
  4. High residual → boost *moving* only (queued in-lane cars stay moving by
     position; low motion never forces parked)

Uses GroundingDINO corpus CSVs (1-frame stride ~25 fps). No raw mp4 required.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from vehicle_metrics import canon, iou

# ~25 fps → ±15 ≈ 0.6 s, ±25 ≈ 1.0 s
HALF_WINDOW = 15
MIN_LINKS = 3
LINK_IOU = 0.12
MAX_CENTER_JUMP_PX = 140.0

# Residual *net* motion that signals translation (not parallax jitter).
# Tuned on train tracks: parked net med≈59, moving med≈118; use directed net.
NET_BOOST_PX = 100.0
NET_OVER_HEIGHT = 0.90
# net/path: directed drift vs jittery parallax
MIN_CONSISTENCY = 0.45
# Within-frame: boost only if clearly above the vehicle cohort median
COHORT_NET_MARGIN_PX = 35.0
COHORT_NET_RATIO = 1.55

STATIC_TOKENS = (
    "building",
    "sidewalk",
    "crosswalk",
    "storefront",
    "natural",
    "infrastructure",
    "light feature",
    "traffic sign",
    "traffic light",
)


def is_vehicle_label(label: str) -> bool:
    t = canon(label)
    if t in ("moving vehicle", "parked vehicle", "vehicle"):
        return True
    return "vehicle" in t and ("moving" in t or "parked" in t)


def is_static_label(label: str) -> bool:
    t = canon(label)
    return any(tok in t for tok in STATIC_TOKENS)


def parse_native_index(name: str):
    m = re.search(r"_frame_(\d+)", name or "")
    return int(m.group(1)) if m else None


def center(box):
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def _match_best(box, candidates, iou_thr=LINK_IOU, max_jump=MAX_CENTER_JUMP_PX):
    """Greedy best match: prefer IoU, fall back to near centers."""
    cx, cy = center(box)
    best = None
    for i, other in enumerate(candidates):
        if other is None:
            continue
        ov = iou(box, other["box"])
        ocx, ocy = center(other["box"])
        dist = math.hypot(cx - ocx, cy - ocy)
        if ov < iou_thr and dist > max_jump:
            continue
        # score: higher better
        score = ov * 100.0 - 0.15 * dist
        if best is None or score > best[0]:
            best = (score, i, dist, ov)
    return best


def _median_xy(pairs):
    if not pairs:
        return 0.0, 0.0
    xs = sorted(p[0] for p in pairs)
    ys = sorted(p[1] for p in pairs)
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid], ys[mid]
    return 0.5 * (xs[mid - 1] + xs[mid]), 0.5 * (ys[mid - 1] + ys[mid])


def ego_shift(boxes_a, boxes_b):
    """Median center shift of matched static boxes from a → b."""
    static_a = [b for b in boxes_a if is_static_label(b["category"])]
    static_b = [b for b in boxes_b if is_static_label(b["category"])]
    if not static_a or not static_b:
        return 0.0, 0.0
    used = [False] * len(static_b)
    shifts = []
    for a in sorted(static_a, key=lambda d: -float(d.get("score", 0.0))):
        cand = [None if used[i] else static_b[i] for i in range(len(static_b))]
        hit = _match_best(a["box"], cand, iou_thr=0.2, max_jump=80.0)
        if hit is None:
            continue
        _, j, _, _ = hit
        used[j] = True
        ax, ay = center(a["box"])
        bx, by = center(static_b[j]["box"])
        shifts.append((bx - ax, by - ay))
    return _median_xy(shifts)


@lru_cache(maxsize=64)
def load_participant_index(csv_path_str: str):
    """native_index -> list of {category, box, score}."""
    csv_path = Path(csv_path_str)
    by_index = {}
    if not csv_path.is_file():
        return by_index
    with csv_path.open(newline="") as f:
        for rec in csv.DictReader(f):
            idx = parse_native_index(rec.get("image_name") or "")
            if idx is None:
                continue
            try:
                cats = json.loads(rec.get("detected_categories") or "[]")
                boxes = json.loads(rec.get("detection_boxes") or "[]")
                scores = json.loads(rec.get("detection_confidences") or "[]")
            except Exception:
                continue
            dets = []
            for i, (c, b) in enumerate(zip(cats, boxes)):
                dets.append(
                    {
                        "category": c,
                        "box": [float(v) for v in b],
                        "score": float(scores[i]) if i < len(scores) else 0.0,
                    }
                )
            by_index[idx] = dets
    return by_index


def resolve_csv_path(image_path: Path, dino_root: Path) -> Path | None:
    """.../AOI_full_frame_output/<stratum>/<participant>/frames/<file> → DINO CSV."""
    try:
        frames_dir = image_path.parent
        participant = frames_dir.parent.name
        stratum = frames_dir.parent.parent.name
    except Exception:
        return None
    csv_path = dino_root / stratum / participant / "groundingdino_scene_analysis_full_frames.csv"
    return csv_path if csv_path.is_file() else None


def follow_track(box, self_idx, by_index, direction, half_window=HALF_WINDOW):
    """Walk forward (+1) or backward (-1), linking vehicles; return list of (idx, box)."""
    track = [(self_idx, list(box))]
    cur_box = list(box)
    step = 1 if direction > 0 else -1
    for k in range(1, half_window + 1):
        nb = self_idx + step * k
        dets = by_index.get(nb)
        if not dets:
            # allow one-frame gaps
            continue
        vehs = [d for d in dets if is_vehicle_label(d["category"])]
        if not vehs:
            continue
        hit = _match_best(cur_box, vehs)
        if hit is None:
            break
        _, j, _, _ = hit
        cur_box = list(vehs[j]["box"])
        track.append((nb, cur_box))
    return track


def track_residual_motion(box, self_idx, by_index, half_window=HALF_WINDOW):
    """Build short bidirectional track and return residual motion stats.

    Returns dict:
      path_px, net_px, n_links, mean_speed, ego_path_px
    or None if track too short / missing index.
    """
    if self_idx is None or not by_index:
        return None

    fwd = follow_track(box, self_idx, by_index, +1, half_window)
    bak = follow_track(box, self_idx, by_index, -1, half_window)
    # bak is self + older; reverse older part
    ordered = list(reversed(bak[1:])) + fwd  # chronological
    if len(ordered) < MIN_LINKS + 1:
        return None

    residuals = []
    ego_mags = []
    for (i0, b0), (i1, b1) in zip(ordered, ordered[1:]):
        dets0 = by_index.get(i0, [])
        dets1 = by_index.get(i1, [])
        ex, ey = ego_shift(dets0, dets1)
        c0x, c0y = center(b0)
        c1x, c1y = center(b1)
        rx, ry = (c1x - c0x) - ex, (c1y - c0y) - ey
        residuals.append((rx, ry))
        ego_mags.append(math.hypot(ex, ey))

    if not residuals:
        return None

    path = sum(math.hypot(rx, ry) for rx, ry in residuals)
    net = math.hypot(sum(r[0] for r in residuals), sum(r[1] for r in residuals))
    n = len(residuals)
    return {
        "path_px": path,
        "net_px": net,
        "n_links": n,
        "mean_speed": path / max(n, 1),
        "ego_path_px": sum(ego_mags),
        "span_frames": ordered[-1][0] - ordered[0][0],
    }


def _box_height(box) -> float:
    return max(float(box[3]) - float(box[1]), 1.0)


def annotate_cohort(stats_list):
    """Add cohort-relative fields using median net among tracked vehicles in-frame."""
    nets = [float(s["net_px"]) for s in stats_list if s is not None]
    if not nets:
        return stats_list
    nets_sorted = sorted(nets)
    mid = len(nets_sorted) // 2
    if len(nets_sorted) % 2:
        med = nets_sorted[mid]
    else:
        med = 0.5 * (nets_sorted[mid - 1] + nets_sorted[mid])
    out = []
    for s in stats_list:
        if s is None:
            out.append(None)
            continue
        s2 = dict(s)
        s2["cohort_median_net_px"] = med
        s2["net_minus_cohort"] = float(s["net_px"]) - med
        out.append(s2)
    return out


def motion_moving_boost(stats, box=None) -> float | None:
    """Map track stats → a floor on p_moving, or None if no boost.

    Protocol: only boost moving; never force parked from low motion.
    Uses directed residual net (and net/height), consistency, and optional
    within-frame cohort contrast so parallax on parked cars does not fire.
    """
    if not stats:
        return None
    path = float(stats["path_px"])
    net = float(stats["net_px"])
    cons = net / max(path, 1e-6)
    if cons < MIN_CONSISTENCY:
        return None

    h = _box_height(box) if box is not None else None
    net_h = net / h if h else None

    absolute = net >= NET_BOOST_PX or (net_h is not None and net_h >= NET_OVER_HEIGHT)
    cohort_med = stats.get("cohort_median_net_px")
    relative = False
    if cohort_med is not None and cohort_med > 1.0:
        relative = (net - cohort_med) >= COHORT_NET_MARGIN_PX or net >= COHORT_NET_RATIO * cohort_med
    elif cohort_med is not None:
        relative = net >= NET_BOOST_PX

    # Need absolute evidence; if a cohort exists, also require standing out
    if not absolute:
        return None
    if cohort_med is not None and not relative and net < 1.25 * NET_BOOST_PX:
        return None

    strength = max(
        0.0,
        (net - NET_BOOST_PX) / 100.0,
        ((net_h - NET_OVER_HEIGHT) / 1.2) if net_h is not None else 0.0,
        (cons - MIN_CONSISTENCY) / 0.4,
    )
    # Soft floor in [0.58, 0.92] — enough to flip ambiguous geometry, not all curb cars
    return min(0.92, 0.58 + 0.34 * min(1.0, strength))


def displacement_proxy(stats) -> float | None:
    """Scalar displacement for APIs that still take displacement_px."""
    if not stats:
        return None
    return float(stats["net_px"])


def track_motion_for_boxes(boxes, image_path: Path, dino_root: Path, half_window=HALF_WINDOW):
    """For each box, compute track residual stats. Returns list aligned with boxes."""
    csv_path = resolve_csv_path(image_path, dino_root)
    if csv_path is None:
        return [None] * len(boxes)
    by_index = load_participant_index(str(csv_path))
    self_idx = parse_native_index(image_path.name)
    out = []
    for box in boxes:
        out.append(track_residual_motion(box, self_idx, by_index, half_window))
    return annotate_cohort(out)
