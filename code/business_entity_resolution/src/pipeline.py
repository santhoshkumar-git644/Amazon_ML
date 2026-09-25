"""End-to-end blocking + feature pipeline shared by train.py and predict.py."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import blocking
import features as feat_mod
import normalize

# Below this row count, splitting into chunks and spinning up a process pool
# costs more than it saves -- just normalize directly.
MIN_ROWS_PER_WORKER_FOR_PARALLEL = 20_000


def _normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Top-level (picklable) worker function for normalize_source's parallel path."""
    return normalize.build_normalized_frame(df)


def normalize_source(df: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
    """Normalize a source dataframe. normalize.build_normalized_frame is pure
    per-row string work (regex/tokenization) with no cross-row dependencies,
    so it parallelizes cleanly across processes -- this is the single biggest
    speedup available for it (there's no GPU-friendly way to do this step;
    it's plain Python string processing, not a numeric/tensor workload).

    n_jobs>1 splits the dataframe into that many chunks and normalizes them
    in parallel worker processes, then concatenates the results back in
    order. n_jobs<=1 (the default) runs single-threaded, unchanged from
    before.
    """
    if n_jobs <= 1 or len(df) < n_jobs * MIN_ROWS_PER_WORKER_FOR_PARALLEL:
        return normalize.build_normalized_frame(df)

    chunks = np.array_split(df, n_jobs)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(_normalize_chunk, chunks))
    return pd.concat(results, ignore_index=True)


def generate_candidates(s1_norm: pd.DataFrame, s2_norm: pd.DataFrame, s3_norm: pd.DataFrame,
                         k_per_source: int = 20) -> pd.DataFrame:
    """Run blocking of S1 against S2 and against S3 separately (so one source
    can't crowd out the other), tag each pair with its source, and concatenate.
    Returns columns: s1_id, cand_id, score, n_blocking_sources, candidate_rank,
    score_gap_to_next, source.
    """
    parts = []
    if len(s2_norm):
        p2 = blocking.build_candidates(s1_norm, s2_norm, k=k_per_source)
        p2["source"] = "S2"
        parts.append(p2)
    if len(s3_norm):
        p3 = blocking.build_candidates(s1_norm, s3_norm, k=k_per_source)
        p3["source"] = "S3"
        parts.append(p3)
    if not parts:
        return pd.DataFrame(columns=["s1_id", "cand_id", "score", "n_blocking_sources",
                                      "candidate_rank", "score_gap_to_next", "source"])
    return pd.concat(parts, ignore_index=True)


def compute_features_for_pairs(pairs: pd.DataFrame, s1_norm: pd.DataFrame,
                                s2_norm: pd.DataFrame, s3_norm: pd.DataFrame) -> pd.DataFrame:
    s1_idx = s1_norm.set_index("entity_id")
    other_idx = pd.concat([s2_norm, s3_norm], ignore_index=True).set_index("entity_id")
    other_idx = other_idx[~other_idx.index.duplicated(keep="first")]
    return feat_mod.compute_features(pairs, s1_idx, other_idx)
