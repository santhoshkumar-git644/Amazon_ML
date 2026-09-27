"""Score the test set and write output/matching_results.tsv + candidate_pairs.tsv.

Reuses the exact same blocking + feature pipeline as training (see
``train_model.py``), scores every candidate pair with the trained LightGBM
model, and applies the threshold tuned on held-out training data.
``candidate_pairs.tsv`` is the *full* candidate set scored by the model (the
blocking output); ``matching_results.tsv`` is the subset that cleared the
threshold — so matches are guaranteed to be a subset of candidates.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from prepare import load_normalized, ROOT  # noqa: E402
from blocking import explode_all, generate_candidates  # noqa: E402
from features import compute_features, FEATURE_COLUMNS  # noqa: E402

WORK = os.path.join(ROOT, "work")
MODEL_PATH = os.path.join(WORK, "model.txt")
META_PATH = os.path.join(WORK, "model_meta.json")
OUTPUT_DIR = os.path.join(ROOT, "output")


def build_candidate_features(s1_df, other_df, s1_explodes, src_label):
    t0 = time.time()
    cand = generate_candidates(s1_df, other_df, s1_explodes=s1_explodes)
    feat = compute_features(cand, s1_df, other_df)
    feat["src"] = np.int8(src_label)
    feat["other_entity_id"] = other_df["entity_id"].to_numpy()[feat["other_pos"].to_numpy()]
    print(f"  source{src_label}: {len(feat)} candidate pairs in {time.time() - t0:.1f}s", flush=True)
    return feat


def write_id_list_tsv(path, header_id_col, s1_ids, s1_pos_with_rows, id_lists_by_pos):
    """Write one row per Source-1 entity (in file order), comma-joined id list."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_id_col}\n")
        for pos, sid in enumerate(s1_ids):
            ids = id_lists_by_pos.get(pos)
            f.write(f"{sid}\t{','.join(ids) if ids else ''}\n")


def group_ids_by_s1(df):
    """df with columns [s1_pos, other_entity_id] -> dict s1_pos -> list[str]."""
    out = {}
    for s1p, oid in zip(df["s1_pos"].to_numpy(), df["other_entity_id"].to_numpy()):
        out.setdefault(s1p, []).append(oid)
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(META_PATH) as f:
        meta = json.load(f)
    threshold = meta["threshold"]
    print(f"using threshold={threshold}", flush=True)

    booster = lgb.Booster(model_file=MODEL_PATH)

    print("loading normalized test frames (cached)...", flush=True)
    s1 = load_normalized("test_source1")
    s2 = load_normalized("test_source2")
    s3 = load_normalized("test_source3")

    print("blocking + features...", flush=True)
    s1_explodes = explode_all(s1)
    feat2 = build_candidate_features(s1, s2, s1_explodes, 2)
    feat3 = build_candidate_features(s1, s3, s1_explodes, 3)

    print("scoring candidates...", flush=True)
    feat2["prob"] = booster.predict(feat2[FEATURE_COLUMNS])
    feat3["prob"] = booster.predict(feat3[FEATURE_COLUMNS])

    candidates = pd.concat(
        [feat2[["s1_pos", "other_entity_id", "prob"]], feat3[["s1_pos", "other_entity_id", "prob"]]],
        ignore_index=True,
    )
    del feat2, feat3

    matches = candidates[candidates["prob"] >= threshold]

    s1_ids = s1["entity_id"].to_numpy()
    candidate_by_s1 = group_ids_by_s1(candidates)
    match_by_s1 = group_ids_by_s1(matches)

    write_id_list_tsv(
        os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
        "candidate_entity_ids", s1_ids, None, candidate_by_s1,
    )
    write_id_list_tsv(
        os.path.join(OUTPUT_DIR, "matching_results.tsv"),
        "matched_entity_ids", s1_ids, None, match_by_s1,
    )
    n_with_match = sum(1 for v in match_by_s1.values() if v)
    print(f"wrote {len(s1_ids)} rows; {n_with_match} entities got >=1 match "
          f"({len(matches)} matched pairs total, {len(candidates)} candidate pairs total)", flush=True)


if __name__ == "__main__":
    main()
