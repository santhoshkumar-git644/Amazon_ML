"""Train the pair-matching classifier and tune the decision threshold.

Pipeline: normalize (cached) -> block candidates (Source-1 x Source-2 and
Source-1 x Source-3) -> compute pairwise features -> label pairs from
train_ground_truth.tsv -> train a LightGBM classifier -> tune a single
probability threshold on a held-out slice of Source-1 entities to maximize
the macro F_0.5 the challenge actually scores on.

Positions (0..N-1 row indices into the normalized frames), not entity_id
strings, are used everywhere in the hot path — this keeps the multi-ten-
-million-row candidate table numeric and memory-light; entity_id strings are
only re-attached for the handful of rows that make it into an output file.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from prepare import load_normalized, ROOT  # noqa: E402
from blocking import explode_all, generate_candidates  # noqa: E402
from features import compute_features, FEATURE_COLUMNS  # noqa: E402
from scoring import macro_f05  # noqa: E402

WORK = os.path.join(ROOT, "work")
MODEL_PATH = os.path.join(WORK, "model.txt")
META_PATH = os.path.join(WORK, "model_meta.json")
VAL_FRAC = 0.15
SEED = 42


def encode_pairs(s1_pos, other_pos):
    return (s1_pos.astype(np.int64) << np.int64(32)) | other_pos.astype(np.int64)


def build_gt_keys(gt_df, s1_id_to_pos, s2_id_to_pos, s3_id_to_pos):
    """Encoded true-pair keys, split by which "other" source they reference."""
    keys2, keys3 = [], []
    for sid, matches in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if not matches:
            continue
        s1p = s1_id_to_pos.get(sid)
        if s1p is None:
            continue
        base = np.int64(s1p) << np.int64(32)
        for m in matches.split(","):
            if m.startswith("S2-"):
                p = s2_id_to_pos.get(m)
                if p is not None:
                    keys2.append(base | np.int64(p))
            elif m.startswith("S3-"):
                p = s3_id_to_pos.get(m)
                if p is not None:
                    keys3.append(base | np.int64(p))
    k2 = np.array(sorted(keys2), dtype=np.int64)
    k3 = np.array(sorted(keys3), dtype=np.int64)
    return k2, k3


def build_candidate_features(s1_df, other_df, s1_explodes, src_label):
    t0 = time.time()
    cand = generate_candidates(s1_df, other_df, s1_explodes=s1_explodes)
    feat = compute_features(cand, s1_df, other_df)
    feat["src"] = np.int8(src_label)
    print(f"  source{src_label}: {len(feat)} candidate pairs in {time.time() - t0:.1f}s", flush=True)
    return feat


def main():
    os.makedirs(WORK, exist_ok=True)
    print("loading normalized frames (cached)...", flush=True)
    s1 = load_normalized("train_source1")
    s2 = load_normalized("train_source2")
    s3 = load_normalized("train_source3")

    gt_path = os.path.join(ROOT, "dataset", "train", "train_ground_truth.tsv")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    s1_id_to_pos = dict(zip(s1["entity_id"].to_numpy(), range(len(s1))))
    s2_id_to_pos = dict(zip(s2["entity_id"].to_numpy(), range(len(s2))))
    s3_id_to_pos = dict(zip(s3["entity_id"].to_numpy(), range(len(s3))))

    print("blocking + features...", flush=True)
    s1_explodes = explode_all(s1)
    feat2 = build_candidate_features(s1, s2, s1_explodes, 2)
    feat3 = build_candidate_features(s1, s3, s1_explodes, 3)

    print("labeling from ground truth...", flush=True)
    keys2, keys3 = build_gt_keys(gt_df, s1_id_to_pos, s2_id_to_pos, s3_id_to_pos)
    n_true2 = len(keys2)
    n_true3 = len(keys3)

    enc2 = encode_pairs(feat2["s1_pos"].to_numpy(), feat2["other_pos"].to_numpy())
    enc3 = encode_pairs(feat3["s1_pos"].to_numpy(), feat3["other_pos"].to_numpy())
    feat2["label"] = np.isin(enc2, keys2).astype(np.int8)
    feat3["label"] = np.isin(enc3, keys3).astype(np.int8)

    recall2 = feat2["label"].sum() / max(n_true2, 1)
    recall3 = feat3["label"].sum() / max(n_true3, 1)
    print(f"  blocking recall vs Source-2 ground truth: {recall2:.4f} "
          f"({feat2['label'].sum()}/{n_true2})", flush=True)
    print(f"  blocking recall vs Source-3 ground truth: {recall3:.4f} "
          f"({feat3['label'].sum()}/{n_true3})", flush=True)

    pairs = pd.concat([feat2, feat3], ignore_index=True)
    del feat2, feat3

    n_s1 = len(s1)
    train_pos, val_pos = train_test_split(np.arange(n_s1), test_size=VAL_FRAC, random_state=SEED)
    val_pos_set = set(val_pos.tolist())
    is_val = pairs["s1_pos"].isin(val_pos_set).to_numpy()

    X = pairs[FEATURE_COLUMNS]
    y = pairs["label"]

    X_train, y_train = X[~is_val], y[~is_val]
    X_val, y_val = X[is_val], y[is_val]
    print(f"train pairs: {len(X_train)} (pos rate {y_train.mean():.4f}), "
          f"val pairs: {len(X_val)} (pos rate {y_val.mean():.4f})", flush=True)

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        is_unbalance=True,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(30, verbose=True), lgb.log_evaluation(25)],
    )

    val_pairs = pairs[is_val][["s1_pos", "other_pos", "src", "label"]].copy()
    val_pairs["prob"] = model.predict_proba(X_val)[:, 1]

    true_map = {}
    for sid, matches in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        p = s1_id_to_pos.get(sid)
        if p is not None and p in val_pos_set:
            true_map[p] = set(matches.split(",")) if matches else set()
    for p in val_pos_set:
        true_map.setdefault(p, set())

    def preds_at(threshold):
        sub = val_pairs[val_pairs["prob"] >= threshold]
        pred_map = {}
        s2_ids = s2["entity_id"].to_numpy()
        s3_ids = s3["entity_id"].to_numpy()
        for s1p, other_p, src in zip(sub["s1_pos"].to_numpy(), sub["other_pos"].to_numpy(), sub["src"].to_numpy()):
            oid = s2_ids[other_p] if src == 2 else s3_ids[other_p]
            pred_map.setdefault(s1p, set()).add(oid)
        return pred_map

    best_t, best_score = 0.5, -1.0
    all_val_ids = list(val_pos_set)
    for t in np.linspace(0.05, 0.95, 19):
        pred_map = preds_at(t)
        score = macro_f05(all_val_ids, pred_map, true_map)
        print(f"  threshold={t:.2f} val macro F_0.5={score:.4f}", flush=True)
        if score > best_score:
            best_score, best_t = score, float(t)

    # local refine around the best coarse threshold
    for t in np.linspace(max(best_t - 0.045, 0.01), min(best_t + 0.045, 0.99), 19):
        pred_map = preds_at(t)
        score = macro_f05(all_val_ids, pred_map, true_map)
        if score > best_score:
            best_score, best_t = score, float(t)

    print(f"BEST threshold={best_t:.3f} val macro F_0.5={best_score:.4f}", flush=True)

    model.booster_.save_model(MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({
            "threshold": best_t,
            "val_macro_f05": best_score,
            "feature_columns": FEATURE_COLUMNS,
            "blocking_recall_source2": recall2,
            "blocking_recall_source3": recall3,
        }, f, indent=2)
    print(f"saved model to {MODEL_PATH} and metadata to {META_PATH}", flush=True)


if __name__ == "__main__":
    main()
