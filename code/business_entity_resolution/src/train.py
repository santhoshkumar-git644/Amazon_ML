"""Train the pairwise matching model end-to-end, GPU throughout:

  train_source{1,2,3}.tsv + train_ground_truth.tsv
    -> GPU normalize (cuDF) -> GPU block (cuDF merges) -> GPU features (cuDF/cuML)
    -> group-split S1 entities into train/val
    -> train XGBoost binary classifier on GPU (device="cuda") on (features -> is_match)
    -> tune a decision threshold on val to maximize macro F_0.5
    -> save model + threshold + feature list to models/

This is the higher-confidence half of the GPU branch: XGBoost's cuDF
integration and device="cuda" training is a standard, well-documented
RAPIDS pattern. The lower-confidence half is what normalize.py/blocking.py/
features.py do before this file ever runs -- see those files' docstrings.

Usage (from student_resource/):
    python3 code/business_entity_resolution/src/train.py \
        --data-dir dataset/train \
        --model-dir code/business_entity_resolution/models \
        [--max-s1 N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cudf
import cupy as cp
import numpy as np
import xgboost as xgb

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

    # Ground truth bookkeeping (label lookup dict, id set membership) is
    # small-scale Python dict/set work over the S1 count, not the multi-
    # million candidate-pair count -- no GPU benefit, stays on host exactly
    # like the CPU version.
    truth_by_s1 = {}
    for row in gt_df.to_pandas().itertuples(index=False):
        ids = row.matched_entity_ids.split(",") if row.matched_entity_ids else []
        truth_by_s1[row.source1_entity_id] = set(ids)

    log("computing features ...")
    t0 = time.time()
    feats = pipeline.compute_features_for_pairs(pairs, s1n, s2n, s3n)
    log(f"  -> features shape {feats.shape} in {time.time() - t0:.1f}s")

    # Label assignment: pulling s1_id/cand_id to host to check dict
    # membership (truth_by_s1 is a Python dict, not GPU data) -- this one
    # pair of columns round-tripping to host is unavoidable given the label
    # source is the ground-truth file's per-entity match lists, not
    # something expressible as a GPU join without first building a full
    # (s1_id, cand_id) ground-truth pairs table. At candidate-pair scale
    # (tens of millions, not billions) this round trip is not the
    # bottleneck.
    s1_ids_host = feats["s1_id"].to_pandas()
    cand_ids_host = feats["cand_id"].to_pandas()
    labels = [int(c in truth_by_s1.get(s, ())) for s, c in zip(s1_ids_host, cand_ids_host)]
    feats["label"] = cudf.Series(labels, index=feats.index)
    return feats, truth_by_s1


def report_blocking_recall(truth_by_s1, found_pairs_set, all_s1_ids):
    total_true = sum(len(truth_by_s1.get(s1, ())) for s1 in all_s1_ids)
    found_true = 0
    for s1 in all_s1_ids:
        for cand in truth_by_s1.get(s1, ()):
            if (s1, cand) in found_pairs_set:
                found_true += 1
    recall = found_true / total_true if total_true else 1.0
    log(f"BLOCKING RECALL CEILING: {found_true}/{total_true} = {recall:.4f}")
    return recall


def tune_threshold(val_s1_host, val_cand_host, probs: np.ndarray, truth_by_s1: dict,
                    val_s1_ids) -> tuple[float, float]:
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.02):
        pred_by_s1 = {}
        keep = probs >= t
        for s1, cand in zip(val_s1_host[keep], val_cand_host[keep]):
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
    ap.add_argument("--num-boost-round", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    log("loading train files (GPU CSV parse) ...")
    s1 = io_utils.load_source(os.path.join(args.data_dir, "train_source1.tsv"))
    s2 = io_utils.load_source(os.path.join(args.data_dir, "train_source2.tsv"))
    s3 = io_utils.load_source(os.path.join(args.data_dir, "train_source3.tsv"))
    gt = io_utils.load_ground_truth(os.path.join(args.data_dir, "train_ground_truth.tsv"))
    log(f"  s1={len(s1)} s2={len(s2)} s3={len(s3)} gt={len(gt)}")

    if args.max_s1:
        s1 = s1.sample(n=min(args.max_s1, len(s1)), random_state=RANDOM_STATE).reset_index(drop=True)
        gt = gt[gt["source1_entity_id"].isin(s1["entity_id"])].reset_index(drop=True)
        log(f"  subsampled to {len(s1)} S1 entities for this run")

    log("normalizing (GPU) ...")
    t0 = time.time()
    s1n = pipeline.normalize_source(s1)
    s2n = pipeline.normalize_source(s2)
    s3n = pipeline.normalize_source(s3)
    log(f"  -> normalized in {time.time() - t0:.1f}s")

    # Train/val split by S1 id: small-scale (S1 count, not candidate-pair
    # count) shuffling, done on host with plain numpy exactly like the CPU
    # version -- this bookkeeping was never the bottleneck, no reason to
    # move it to GPU.
    all_s1_ids = s1n["entity_id"].to_pandas().tolist()
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled = rng.permutation(all_s1_ids)
    n_val = int(len(shuffled) * args.val_frac)
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])
    log(f"train S1={len(train_ids)} val S1={len(val_ids)}")

    feats, truth_by_s1 = build_labeled_pairs(s1n, s2n, s3n, gt, args.k_per_source)

    found_pairs_set = set(zip(feats["s1_id"].to_pandas(), feats["cand_id"].to_pandas()))
    report_blocking_recall(truth_by_s1, found_pairs_set, all_s1_ids)

    train_mask = feats["s1_id"].isin(train_ids)
    val_mask = feats["s1_id"].isin(val_ids)
    train_feats = feats[train_mask].reset_index(drop=True)
    val_feats = feats[val_mask].reset_index(drop=True)
    log(f"train pairs={len(train_feats)} (pos={int(train_feats['label'].sum())}) "
        f"val pairs={len(val_feats)} (pos={int(val_feats['label'].sum())})")

    X_train = train_feats[feat_mod.FEATURE_COLUMNS]
    y_train = train_feats["label"]
    X_val = val_feats[feat_mod.FEATURE_COLUMNS]
    y_val = val_feats["label"]

    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos else 1.0
    log(f"scale_pos_weight={scale_pos_weight:.2f}")

    # DMatrix accepts cuDF DataFrames/Series directly -- data stays on GPU,
    # no host round trip. feature_names is inferred from the cuDF column
    # names automatically.
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "device": "cuda",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "scale_pos_weight": scale_pos_weight,
        "seed": RANDOM_STATE,
    }

    log("training XGBoost on GPU (device=cuda) ...")
    booster = xgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dval, "validation")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    log(f"best iteration: {booster.best_iteration}")

    val_probs = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    val_probs = cp.asarray(val_probs).get() if hasattr(val_probs, "get") else np.asarray(val_probs)

    val_s1_host = val_feats["s1_id"].to_pandas().values
    val_cand_host = val_feats["cand_id"].to_pandas().values
    best_t, best_f = tune_threshold(val_s1_host, val_cand_host, val_probs, truth_by_s1, list(val_ids))
    log(f"BEST THRESHOLD={best_t:.2f}  VAL MACRO F0.5={best_f:.4f}")

    pred_by_s1 = {}
    keep = val_probs >= best_t
    for s1, cand in zip(val_s1_host[keep], val_cand_host[keep]):
        pred_by_s1.setdefault(s1, set()).add(cand)
    report = metrics.full_report(pred_by_s1, truth_by_s1, list(val_ids))
    log("VALIDATION METRICS @ tuned threshold:\n" + json.dumps(report, indent=2))

    y_val_host = y_val.to_pandas().values
    pred_labels = (val_probs >= best_t).astype(int)
    tp = int(((pred_labels == 1) & (y_val_host == 1)).sum())
    fp = int(((pred_labels == 1) & (y_val_host == 0)).sum())
    fn = int(((pred_labels == 0) & (y_val_host == 1)).sum())
    pair_precision = tp / (tp + fp) if (tp + fp) else 0.0
    pair_recall = tp / (tp + fn) if (tp + fn) else 0.0
    log(f"pairwise (row-level) precision={pair_precision:.4f} recall={pair_recall:.4f} "
        f"(tp={tp} fp={fp} fn={fn})")

    model_path = os.path.join(args.model_dir, "matcher.json")
    booster.save_model(model_path)
    meta = {
        "feature_columns": feat_mod.FEATURE_COLUMNS,
        "threshold": float(best_t),
        "val_metrics": report,
        "pairwise_precision": pair_precision,
        "pairwise_recall": pair_recall,
        "k_per_source": args.k_per_source,
        "best_iteration": int(booster.best_iteration),
    }
    with open(os.path.join(args.model_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"saved model to {model_path} and meta.json")

    importance = booster.get_score(importance_type="gain")
    ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    log("top feature importances (gain):\n" + "\n".join(f"{k}: {v:.1f}" for k, v in ranked[:15]))


if __name__ == "__main__":
    main()
