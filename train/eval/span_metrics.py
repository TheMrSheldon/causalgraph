"""
Span-level evaluation metrics for Task 2 (causal event extraction).

Based on:
  https://github.com/touche-webis-de/touche-code/blob/main/clef26/advertisement-detection/docker-evaluator/subtask-2/evaluator.py

API mirrors scikit-learn's scorer convention: individual *_score functions plus
a span_report summary, all taking (y_true, y_pred) as positional arguments.

Span format: (start, end) integer tuples with exclusive end (Python convention).

Single-instance scorers
-----------------------
precision_score(y_true, y_pred)    -> float
recall_score(y_true, y_pred)       -> float
f1_score(y_true, y_pred)           -> float
granularity_score(y_true, y_pred)  -> float
f1_gran_score(y_true, y_pred)      -> float
iou_score(y_true, y_pred)          -> float
span_report(y_true, y_pred)        -> dict[str, float]  (all metrics at once)

Dataset-level aggregation
-------------------------
score_dataset(records)  -> dict[str, float]
    records: list of {"y_true": [...], "y_pred": [...]} dicts
"""
from __future__ import annotations

import math
from typing import TypeVar

Span = tuple[int, int]
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------

def _mean(values: list[T], default: T) -> T:
    return sum(values) / len(values) if values else default


def overlap(a: Span, b: Span) -> float:
    """Portion of interval *a* covered by interval *b* (0.0–1.0)."""
    if a[0] >= a[1]:
        return 0.0
    covered = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return covered / (a[1] - a[0])


def max_span_score(a: Span, bs: list[Span]) -> tuple[float, int]:
    """(best_overlap_of_a_with_any_b, count_of_bs_that_overlap_a)."""
    scores = [overlap(a, b) for b in bs]
    return max(scores, default=0.0), sum(s > 0 for s in scores)


# ---------------------------------------------------------------------------
# Public sklearn-style scorers
# ---------------------------------------------------------------------------

def precision_score(y_true: list[Span], y_pred: list[Span]) -> float:
    """Average best-match overlap of each predicted span against the truth spans."""
    return _mean([max_span_score(p, y_true)[0] for p in y_pred], default=0.0)


def recall_score(y_true: list[Span], y_pred: list[Span]) -> float:
    """Average best-match overlap of each truth span against the predicted spans."""
    return _mean([max_span_score(t, y_pred)[0] for t in y_true], default=0.0)


def f1_score(y_true: list[Span], y_pred: list[Span]) -> float:
    """Harmonic mean of precision_score and recall_score."""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def granularity_score(y_true: list[Span], y_pred: list[Span]) -> float:
    """
    Average number of predicted spans matching each matched truth span (Potthast 2014).
    Values >1 indicate over-segmentation; 1.0 is ideal.
    """
    results = (max_span_score(t, y_pred) for t in y_true)
    return _mean([n for _score, n in results if _score > 0], default=0.0)


def iou_score(y_true: list[Span], y_pred: list[Span]) -> float:
    """Character-set intersection-over-union across all span pairs."""
    set_a = {x for a in y_true for x in range(a[0], a[1])}
    set_b = {x for b in y_pred for x in range(b[0], b[1])}
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def f1_gran_score(y_true: list[Span], y_pred: list[Span]) -> float:
    """F1 penalized by log2(1 + granularity); the primary extraction quality metric."""
    f1 = f1_score(y_true, y_pred)
    gran = granularity_score(y_true, y_pred)
    return f1 / math.log2(1 + gran) if gran > 0 else f1


def span_report(y_true: list[Span], y_pred: list[Span]) -> dict[str, float]:
    """
    All span metrics for a single instance. Analogous to classification_report.

    Returns keys: precision, recall, f1, granularity, f1_gran, iou.
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    gran = granularity_score(y_true, y_pred)
    iou = iou_score(y_true, y_pred)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    f1_gran = f1 / math.log2(1 + gran) if gran > 0 else f1
    return {"precision": p, "recall": r, "f1": f1,
            "granularity": gran, "f1_gran": f1_gran, "iou": iou}


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

def score_dataset(records: list[dict[str, list[Span]]]) -> dict[str, float]:
    """
    Macro-average span_report over multiple instances.

    Each record must have keys ``y_true`` and ``y_pred``.
    """
    if not records:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "granularity": 0.0, "f1_gran": 0.0, "iou": 0.0}
    per_instance = [span_report(r["y_true"], r["y_pred"]) for r in records]
    keys = per_instance[0].keys()
    return {k: _mean([s[k] for s in per_instance], default=0.0) for k in keys}
