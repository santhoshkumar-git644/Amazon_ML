"""Train the pairwise matching model end-to-end:

  train_source{1,2,3}.tsv + train_ground_truth.tsv
    -> normalize -> block (candidate generation) -> compute features
    -> group-split S1 entities into train/val
    -> train LightGBM binary classifier on (features -> is_match)
    -> tune a decision threshold on val to maximize macro F_0.5
    -> save model + threshold + feature list to models/

Usage (from student_resource/):
    python3 code/business_entity_resolution/src/train.py \
        --data-dir dataset/train \
        --model-dir code/business_entity_resolution/models \
        [--max-s1 N]   # optional: subsample S1 entities for a fast dev run. Note this
                        # only shrinks the S1 side -- S2/S3 are always used in full, since
                        # blocking needs the complete pool to search against.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as feat_mod
import io_utils
import metrics
import pipeline

RANDOM_STATE = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_labeled_pairs(s1n, s2n, s3n, gt_df, k_per_source):
    log(f"blocking {len(s1n)} S1 entities against {len(s2n)} S2 / {len(s3n)} S3 records ...")
    t0 = time.time()
    pairs = pipeline.generate_candidates(s1n, s2n, s3n, k_per_source=k_per_source)
    log(f"  -> {len(pairs)} candidate pairs in {time.time() - t0:.1f}s")

    truth_by_s1 = {}
    for row in gt_df.itertuples(index=False):
        ids = row.matched_entity_ids.split(",") if row.matched_entity_ids else []
        truth_by_s1[row.source1_entity_id] = set(ids)

    log("computing features ...")
    t0 = time.time()
    feats = pipeline.compute_features_for_pairs(pairs, s1n, s2n, s3n)
    log(f"  -> features shape {feats.shape} in {time.time() - t0:.1f}s")

    feats["label"] = [
        int(cand in truth_by_s1.get(s1, ())) for s1, cand in zip(feats["s1_id"], feats["cand_id"])
    ]
    return feats, truth_by_s1


def report_blocking_recall(pairs_s1_ids, truth_by_s1, found_pairs_set, all_s1_ids):
    total_true = sum(len(truth_by_s1.get(s1, ())) for s1 in all_s1_ids)
    found_true = 0
    for s1 in all_s1_ids:
        for cand in truth_by_s1.get(s1, ()):
            if (s1, cand) in found_pairs_set:
                found_true += 1
    recall = found_true / total_true if total_true else 1.0
    log(f"BLOCKING RECALL CEILING: {found_true}/{total_true} = {recall:.4f}")
    return recall


def tune_threshold(val_feats: pd.DataFrame, probs: np.ndarray, truth_by_s1: dict,
                    val_s1_ids) -> tuple[float, float]:
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.02):
        pred_by_s1 = {}
        keep = probs >= t
        for s1, cand in zip(val_feats["s1_id"][keep], val_feats["cand_id"][keep]):
            pred_by_s1.setdefault(s1, set()).add(cand)
        f = metrics.macro_f_beta(pred_by_s1, truth_by_s1, val_s1_ids, beta=0.5)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/train")
    ap.add_argument("--model-dir", default="code/business_entity_resolution/models")
    ap.add_argument("--k-per-source", type=int, default=20)
    ap.add_argument("--max-s1", type=int, default=None,
                     help="Subsample this many S1 training entities (dev/debug speed).")
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    log("loading train files ...")
    s1 = io_utils.load_source(os.path.join(args.data_dir, "train_source1.tsv"))
    s2 = io_utils.load_source(os.path.join(args.data_dir, "train_source2.tsv"))
    s3 = io_utils.load_source(os.path.join(args.data_dir, "train_source3.tsv"))
    gt = io_utils.load_ground_truth(os.path.join(args.data_dir, "train_ground_truth.tsv"))
    log(f"  s1={len(s1)} s2={len(s2)} s3={len(s3)} gt={len(gt)}")

    if args.max_s1:
        s1 = s1.sample(n=min(args.max_s1, len(s1)), random_state=RANDOM_STATE).reset_index(drop=True)
        gt = gt[gt["source1_entity_id"].isin(s1["entity_id"])].reset_index(drop=True)
        log(f"  subsampled to {len(s1)} S1 entities for this run")

    log("normalizing ...")
    s1n = pipeline.normalize_source(s1)
    s2n = pipeline.normalize_source(s2)
    s3n = pipeline.normalize_source(s3)

    all_s1_ids = s1n["entity_id"].tolist()
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled = rng.permutation(all_s1_ids)
    n_val = int(len(shuffled) * args.val_frac)
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])
    log(f"train S1={len(train_ids)} val S1={len(val_ids)}")

    feats, truth_by_s1 = build_labeled_pairs(s1n, s2n, s3n, gt, args.k_per_source)

    found_pairs_set = set(zip(feats["s1_id"], feats["cand_id"]))
    report_blocking_recall(feats["s1_id"], truth_by_s1, found_pairs_set, all_s1_ids)

    train_mask = feats["s1_id"].isin(train_ids)
    val_mask = feats["s1_id"].isin(val_ids)
    train_feats = feats[train_mask].reset_index(drop=True)
    val_feats = feats[val_mask].reset_index(drop=True)
    log(f"train pairs={len(train_feats)} (pos={train_feats['label'].sum()}) "
        f"val pairs={len(val_feats)} (pos={val_feats['label'].sum()})")

    X_train = train_feats[feat_mod.FEATURE_COLUMNS]
    y_train = train_feats["label"]
    X_val = val_feats[feat_mod.FEATURE_COLUMNS]
    y_val = val_feats["label"]

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    scale_pos_weight = (n_neg / n_pos) if n_pos else 1.0
    log(f"scale_pos_weight={scale_pos_weight:.2f}")

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": RANDOM_STATE,
    }

    log("training LightGBM ...")
    booster = lgb.train(
        params, train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    log(f"best iteration: {booster.best_iteration}")

    val_probs = booster.predict(X_val, num_iteration=booster.best_iteration)
    best_t, best_f = tune_threshold(val_feats, val_probs, truth_by_s1, val_ids)
    log(f"BEST THRESHOLD={best_t:.2f}  VAL MACRO F0.5={best_f:.4f}")

    # Full entity-level (macro) + pooled (micro) precision/recall/F1/F0.5 report at
    # the tuned threshold. macro_f0.5 is the number that matches how the
    # leaderboard scores matching_results.tsv; the rest is context.
    pred_by_s1 = {}
    keep = val_probs >= best_t
    for s1, cand in zip(val_feats["s1_id"][keep], val_feats["cand_id"][keep]):
        pred_by_s1.setdefault(s1, set()).add(cand)
    report = metrics.full_report(pred_by_s1, truth_by_s1, list(val_ids))
    log("VALIDATION METRICS @ tuned threshold:\n" + json.dumps(report, indent=2))

    # raw pairwise (row-level) precision/recall of the classifier itself, for
    # reference only -- NOT the same as the entity-level macro numbers above.
    pred_labels = (val_probs >= best_t).astype(int)
    tp = int(((pred_labels == 1) & (y_val == 1)).sum())
    fp = int(((pred_labels == 1) & (y_val == 0)).sum())
    fn = int(((pred_labels == 0) & (y_val == 1)).sum())
    pair_precision = tp / (tp + fp) if (tp + fp) else 0.0
    pair_recall = tp / (tp + fn) if (tp + fn) else 0.0
    log(f"pairwise (row-level) precision={pair_precision:.4f} recall={pair_recall:.4f} "
        f"(tp={tp} fp={fp} fn={fn})")

    model_path = os.path.join(args.model_dir, "matcher.txt")
    booster.save_model(model_path)
    meta = {
        "feature_columns": feat_mod.FEATURE_COLUMNS,
        "threshold": float(best_t),
        "val_metrics": report,
        "pairwise_precision": pair_precision,
        "pairwise_recall": pair_recall,
        "k_per_source": args.k_per_source,
        "best_iteration": booster.best_iteration,
    }
    with open(os.path.join(args.model_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"saved model to {model_path} and meta.json")

    importances = pd.Series(
        booster.feature_importance(importance_type="gain"), index=feat_mod.FEATURE_COLUMNS
    ).sort_values(ascending=False)
    log("top feature importances (gain):\n" + importances.head(15).to_string())


if __name__ == "__main__":
    main()
