"""End-to-end blocking + feature pipeline shared by train.py and predict.py."""
from __future__ import annotations

import pandas as pd

import blocking
import features as feat_mod
import normalize


def normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    return normalize.build_normalized_frame(df)


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
