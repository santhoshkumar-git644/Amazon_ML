"""GPU end-to-end blocking + feature pipeline shared by train.py and predict.py.

cuDF equivalent of the CPU pipeline.py.
"""
from __future__ import annotations

import cudf
import cupy as cp

import blocking
import features as feat_mod
import normalize

# generate_candidates processes S1 in chunks of this size (see its
# docstring) -- 5000 is the largest scale proven to run cleanly end-to-end
# on real cluster data as of this branch's last successful test; may need
# retuning if a larger chunk is later shown to also run cleanly, or a
# smaller one is needed if this size itself turns out not to be safe at
# every point in the pipeline.
S1_CHUNK_SIZE = 5000


def normalize_source(df: cudf.DataFrame) -> cudf.DataFrame:
    return normalize.build_normalized_frame(df)


def _generate_candidates_single_chunk(s1_norm: cudf.DataFrame, s2_norm: cudf.DataFrame,
                                       s3_norm: cudf.DataFrame, k_per_source: int) -> cudf.DataFrame:
    """The original, unchunked logic -- run blocking of one S1 chunk against
    the full S2 and full S3 separately (so one source can't crowd out the
    other), tag each pair with its source, and concatenate."""
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


def generate_candidates(s1_norm: cudf.DataFrame, s2_norm: cudf.DataFrame, s3_norm: cudf.DataFrame,
                         k_per_source: int = 20, s1_chunk_size: int = S1_CHUNK_SIZE) -> cudf.DataFrame:
    """Run blocking of S1 against S2 and against S3, processing S1 in
    bounded-size chunks against the full S2/S3 each time, rather than one
    GPU operation over all of S1 at once.

    Why: every GPU memory crash hit so far (four distinct ones across
    normalize/blocking iteration) has been some intermediate structure
    growing too large as S1's entity count increased -- 200 and 5000
    entities ran cleanly, 50,000 kept finding a new chokepoint each time
    one was fixed (the token/fallback join, the char-ngram rescoring, the
    postal/prefix/soundex join, and now a groupby inside rescoring itself).
    Rather than keep patching individual GPU calls one at a time as new
    ones surface at ever-larger scale, this bounds the S1 side of every
    intermediate structure in the whole blocking pipeline to
    s1_chunk_size at once -- the proven-clean 5000-entity process, just
    run repeatedly. cupy's memory pool is flushed between chunks so one
    chunk's freed memory doesn't count against the next chunk's budget.
    """
    if len(s1_norm) <= s1_chunk_size:
        return _generate_candidates_single_chunk(s1_norm, s2_norm, s3_norm, k_per_source)

    parts = []
    for start in range(0, len(s1_norm), s1_chunk_size):
        chunk = s1_norm.iloc[start:start + s1_chunk_size].reset_index(drop=True)
        parts.append(_generate_candidates_single_chunk(chunk, s2_norm, s3_norm, k_per_source))
        del chunk
        cp.get_default_memory_pool().free_all_blocks()
    return cudf.concat(parts, ignore_index=True)


def compute_features_for_pairs(pairs: cudf.DataFrame, s1_norm: cudf.DataFrame,
                                s2_norm: cudf.DataFrame, s3_norm: cudf.DataFrame) -> cudf.DataFrame:
    s1_idx = s1_norm.set_index("entity_id")
    other_idx = cudf.concat([s2_norm, s3_norm], ignore_index=True).set_index("entity_id")
    other_idx = other_idx[~other_idx.index.duplicated(keep="first")]
    return feat_mod.compute_features(pairs, s1_idx, other_idx)
