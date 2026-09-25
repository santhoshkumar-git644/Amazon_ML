"""Scoring utilities.

``macro_f_beta`` / ``full_report`` implement the competition's own metric
exactly: per-Source-1-entity F_0.5, averaged across entities (macro-average),
with singletons scored 1.0 for a correct empty prediction and 0.0 for any
false merge on them. ``full_report`` additionally reports macro/micro
precision, recall, F1 and F0.5 so you can see the full picture during
validation — only macro F_0.5 is what the leaderboard actually scores you on.
"""
from __future__ import annotations


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def per_entity_precision_recall(predicted: set, truth: set) -> tuple[float, float]:
    """Precision/recall for one Source-1 entity, with the same singleton
    convention the challenge uses for F_0.5: a singleton (empty truth) is a
    perfect (1.0, 1.0) when predicted empty, and (0.0, 0.0) — a pure false
    merge — when anything is predicted for it."""
    if not truth:
        return (1.0, 1.0) if not predicted else (0.0, 0.0)
    if not predicted:
        return (0.0, 0.0)
    tp = len(predicted & truth)
    return tp / len(predicted), tp / len(truth)


def per_entity_f_beta(predicted: set, truth: set, beta: float = 0.5) -> float:
    precision, recall = per_entity_precision_recall(predicted, truth)
    return f_beta(precision, recall, beta)


def macro_f_beta(pred_by_s1: dict, truth_by_s1: dict, all_s1_ids, beta: float = 0.5) -> float:
    """pred_by_s1 / truth_by_s1: {s1_id: set(matched ids)}. all_s1_ids: every S1
    entity that must be scored (missing from a dict = empty set)."""
    scores = []
    for s1 in all_s1_ids:
        pred = pred_by_s1.get(s1, set())
        truth = truth_by_s1.get(s1, set())
        scores.append(per_entity_f_beta(pred, truth, beta))
    return sum(scores) / len(scores) if scores else 0.0


def full_report(pred_by_s1: dict, truth_by_s1: dict, all_s1_ids) -> dict:
    """Full metrics report for a validation split.

    - macro_* : per-entity precision/recall/F1/F0.5, averaged across entities.
      This is what the challenge's own F_0.5 macro-average generalizes to; it
      is the number that matters for the leaderboard (macro F0.5 specifically).
    - micro_* : all entities' true/false positives pooled first, then one
      precision/recall/F1/F0.5 computed globally. This is a pairwise-matching
      quality view — unlike macro, it gives singletons no separate credit for
      an empty prediction (there's nothing to pool for them), so don't expect
      it to match the macro numbers; both are reported for a complete picture.
    """
    precisions, recalls, f1s, f05s = [], [], [], []
    tp_total = fp_total = fn_total = 0
    n_singletons = n_with_matches = 0

    for s1 in all_s1_ids:
        pred = pred_by_s1.get(s1, set())
        truth = truth_by_s1.get(s1, set())
        if truth:
            n_with_matches += 1
        else:
            n_singletons += 1

        p, r = per_entity_precision_recall(pred, truth)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f_beta(p, r, 1.0))
        f05s.append(f_beta(p, r, 0.5))

        tp_total += len(pred & truth)
        fp_total += len(pred - truth)
        fn_total += len(truth - pred)

    n = len(all_s1_ids) or 1
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0

    return {
        "n_entities": len(all_s1_ids),
        "n_singletons": n_singletons,
        "n_with_matches": n_with_matches,
        "macro_precision": sum(precisions) / n,
        "macro_recall": sum(recalls) / n,
        "macro_f1": sum(f1s) / n,
        "macro_f0.5": sum(f05s) / n,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": f_beta(micro_p, micro_r, 1.0),
        "micro_f0.5": f_beta(micro_p, micro_r, 0.5),
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
    }
