"""Macro-averaged F_0.5 scoring, matching the challenge's evaluation formula.

F_0.5 is computed per Source-1 entity (predicted-match-set vs. true-match-set)
and then averaged over every Source-1 entity in the evaluation universe,
including singletons (an entity with no true matches scores 1.0 for an empty
prediction, 0.0 for any non-empty one).
"""
import numpy as np


def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def per_entity_f05(pred_set, true_set):
    if not true_set and not pred_set:
        return 1.0
    if not pred_set:
        return 0.0  # recall 0 -> F_0.5 0, unless true_set also empty (handled above)
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set) if true_set else 0.0
    return f_beta(precision, recall, beta=0.5)


def macro_f05(all_ids, pred_map, true_map):
    """``pred_map``/``true_map``: dict s1_id -> set(other_ids). Missing key == empty set."""
    scores = np.empty(len(all_ids), dtype=np.float64)
    for i, sid in enumerate(all_ids):
        pred = pred_map.get(sid, frozenset())
        true = true_map.get(sid, frozenset())
        scores[i] = per_entity_f05(pred, true)
    return scores.mean()
