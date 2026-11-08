import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


Box = Tuple[float, float, float, float]


@dataclass
class Detection:
    box: Box
    score: float
    phrase: str
    category: str


def parse_json_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def area(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = [float(v) for v in box]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def intersection(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = intersection(a, b)
    denom = area(a) + area(b) - inter
    return inter / denom if denom > 0 else 0.0


def smaller_containment(a: Sequence[float], b: Sequence[float]) -> float:
    inter = intersection(a, b)
    small = min(area(a), area(b))
    return inter / small if small > 0 else 0.0


def enclosing_box(a: Sequence[float], b: Sequence[float]) -> Box:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    return (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))


def should_merge_same_category(
    a: Detection,
    b: Detection,
    iou_threshold: float = 0.70,
    containment_threshold: float = 0.90,
) -> bool:
    if a.category != b.category:
        return False
    return (
        iou(a.box, b.box) >= iou_threshold
        or smaller_containment(a.box, b.box) >= containment_threshold
    )


def merge_detection_pair(a: Detection, b: Detection) -> Detection:
    # Keep the label/score from the stronger detection but use the enclosing box.
    keep = a if a.score >= b.score else b
    return Detection(
        box=enclosing_box(a.box, b.box),
        score=max(a.score, b.score),
        phrase=keep.phrase,
        category=keep.category,
    )


def merge_same_category_overlaps(
    detections: Iterable[Detection],
    iou_threshold: float = 0.70,
    containment_threshold: float = 0.90,
) -> List[Detection]:
    merged = list(detections)
    changed = True

    while changed:
        changed = False
        out: List[Detection] = []
        used = [False] * len(merged)

        for i, det_i in enumerate(merged):
            if used[i]:
                continue

            current = det_i
            used[i] = True

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                det_j = merged[j]
                if should_merge_same_category(
                    current,
                    det_j,
                    iou_threshold=iou_threshold,
                    containment_threshold=containment_threshold,
                ):
                    current = merge_detection_pair(current, det_j)
                    used[j] = True
                    changed = True

            out.append(current)

        merged = out

    return merged


def detections_from_grounding_lists(
    boxes,
    scores,
    phrases,
    categories,
) -> List[Detection]:
    n = min(len(boxes), len(scores), len(phrases), len(categories))
    detections = []
    for idx in range(n):
        box = boxes[idx]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            detections.append(
                Detection(
                    box=tuple(float(v) for v in box),
                    score=float(scores[idx]),
                    phrase=str(phrases[idx]),
                    category=str(categories[idx]),
                )
            )
        except Exception:
            continue
    return detections


def grounding_lists_from_detections(detections: Iterable[Detection]):
    boxes = [list(det.box) for det in detections]
    scores = [float(det.score) for det in detections]
    phrases = [det.phrase for det in detections]
    categories = [det.category for det in detections]
    return boxes, scores, phrases, categories
