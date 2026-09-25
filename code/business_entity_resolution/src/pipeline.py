"""GPU end-to-end blocking + feature pipeline shared by train.py and predict.py.

cuDF equivalent of the CPU pipeline.py. No parallel-process chunking here --
unlike the CPU version's --workers flag (which splits work across CPU
processes), a single cuDF call already dispatches across the whole GPU as
one kernel launch; there's no analogous "more workers" knob at this layer.
UNVERIFIED beyond what normalize.py/blocking.py/features.py's own docstrings
already flag -- this file is pure orchestration, lowest risk of the four.
"""
from __future__ import annotations

import cudf

import blocking
import features as feat_mod
import normalize


def normalize_source(df: cudf.DataFrame) -> cudf.DataFrame:
    return normalize.build_normalized_frame(df)


def generate_candidates(s1_norm: cudf.DataFrame, s2_norm: cudf.DataFrame, s3_norm: cudf.DataFrame,
                         k_per_source: int = 20) -> cudf.DataFrame:
    """Run blocking of S1 against S2 and against S3 separately (so one source
    can't crowd out the other), tag each pair with its source, and concatenate.
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
        return cudf.DataFrame({
            "s1_id": [], "cand_id": [], "score": [], "n_blocking_sources": [],
            "candidate_rank": [], "score_gap_to_next": [], "source": [],
        })
    return cudf.concat(parts, ignore_index=True)


def compute_features_for_pairs(pairs: cudf.DataFrame, s1_norm: cudf.DataFrame,
                                s2_norm: cudf.DataFrame, s3_norm: cudf.DataFrame) -> cudf.DataFrame:
    s1_idx = s1_norm.set_index("entity_id")
    other_idx = cudf.concat([s2_norm, s3_norm], ignore_index=True).set_index("entity_id")
    other_idx = other_idx[~other_idx.index.duplicated(keep="first")]
    return feat_mod.compute_features(pairs, s1_idx, other_idx)
