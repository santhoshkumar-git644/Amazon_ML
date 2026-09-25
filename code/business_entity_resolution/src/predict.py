"""Run the trained pipeline on the test set and write the two submission files:

  output/candidate_pairs.tsv   -- every candidate the blocking stage kept,
                                   i.e. exactly what the model scored
  output/matching_results.tsv  -- candidates the model accepted as matches

Usage (from student_resource/):
    python3 code/business_entity_resolution/src/predict.py \
        --data-dir dataset/test \
        --model-dir code/business_entity_resolution/models \
        --output-dir output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_utils
import pipeline


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_output(path: str, header_col: str, s1_to_ids: dict, all_s1_ids: list) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for s1 in all_s1_ids:
            ids = s1_to_ids.get(s1, [])
            f.write(f"{s1}\t{','.join(ids)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/test")
    ap.add_argument("--model-dir", default="code/business_entity_resolution/models")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--k-per-source", type=int, default=None,
                     help="Override the k used at train time (default: use trained meta.json value).")
    ap.add_argument("--threshold", type=float, default=None,
                     help="Override the tuned decision threshold from meta.json.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.model_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    feature_columns = meta["feature_columns"]
    threshold = args.threshold if args.threshold is not None else meta["threshold"]
    k_per_source = args.k_per_source if args.k_per_source is not None else meta["k_per_source"]
    log(f"using threshold={threshold} k_per_source={k_per_source}")

    booster = lgb.Booster(model_file=os.path.join(args.model_dir, "matcher.txt"))

    log("loading test files ...")
    s1 = io_utils.load_source(os.path.join(args.data_dir, "test_source1.tsv"))
    s2 = io_utils.load_source(os.path.join(args.data_dir, "test_source2.tsv"))
    s3 = io_utils.load_source(os.path.join(args.data_dir, "test_source3.tsv"))
    log(f"  s1={len(s1)} s2={len(s2)} s3={len(s3)}")
    all_s1_ids = s1["entity_id"].tolist()

    log("normalizing ...")
    s1n = pipeline.normalize_source(s1)
    s2n = pipeline.normalize_source(s2)
    s3n = pipeline.normalize_source(s3)

    log("blocking (candidate generation) ...")
    t0 = time.time()
    pairs = pipeline.generate_candidates(s1n, s2n, s3n, k_per_source=k_per_source)
    log(f"  -> {len(pairs)} candidate pairs in {time.time() - t0:.1f}s")

    candidate_by_s1 = pairs.groupby("s1_id")["cand_id"].apply(list).to_dict()
    write_output(os.path.join(args.output_dir, "candidate_pairs.tsv"), "candidate_entity_ids",
                 candidate_by_s1, all_s1_ids)
    log("wrote candidate_pairs.tsv")

    log("computing features ...")
    t0 = time.time()
    feats = pipeline.compute_features_for_pairs(pairs, s1n, s2n, s3n)
    log(f"  -> {feats.shape} in {time.time() - t0:.1f}s")

    log("scoring with model ...")
    probs = booster.predict(feats[feature_columns])
    feats["prob"] = probs

    accepted = feats[feats["prob"] >= threshold]
    matched_by_s1 = accepted.groupby("s1_id")["cand_id"].apply(list).to_dict()
    write_output(os.path.join(args.output_dir, "matching_results.tsv"), "matched_entity_ids",
                 matched_by_s1, all_s1_ids)
    log("wrote matching_results.tsv")

    n_with_match = sum(1 for v in matched_by_s1.values() if v)
    log(f"S1 entities with >=1 predicted match: {n_with_match} / {len(all_s1_ids)}")


if __name__ == "__main__":
    main()
