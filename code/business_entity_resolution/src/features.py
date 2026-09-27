"""Pairwise similarity features for candidate (Source-1, other) record pairs.

Feature computation is done only over the candidate pairs that blocking
already produced (not the full cross product), and token-set/fuzzy-string
work is restricted to the distinct positions actually referenced by those
pairs — this keeps memory and CPU cost proportional to the candidate set
rather than to the raw source sizes.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from normalize import significant_tokens

FEATURE_COLUMNS = [
    "n_shared_name_tokens",
    "n_shared_addr_tokens",
    "postal_match",
    "prefix_match",
    "country_match",
    "name_jaccard",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "addr_jaccard",
    "addr_fuzz_partial_ratio",
    "name_len_ratio",
]


def _token_sets(name_norm_values):
    """token-set for each distinct string, as a dict value->frozenset(tokens)."""
    uniq = pd.unique(name_norm_values)
    return {v: frozenset(significant_tokens(v)) for v in uniq}


def _jaccard(a_sets, b_sets, a_vals, b_vals):
    out = np.empty(len(a_vals), dtype=np.float64)
    for i, (a, b) in enumerate(zip(a_vals, b_vals)):
        sa, sb = a_sets[a], b_sets[b]
        if not sa and not sb:
            out[i] = 1.0
        elif not sa or not sb:
            out[i] = 0.0
        else:
            out[i] = len(sa & sb) / len(sa | sb)
    return out


def compute_features(cand_df, s1_df, other_df):
    """Join candidate pairs with s1_df/other_df and compute similarity features.

    ``cand_df`` must have columns [s1_pos, other_pos, n_shared_tokens,
    postal_match, prefix_match] (the output of ``blocking.generate_candidates``).
    Returns ``cand_df`` with the extra columns in ``FEATURE_COLUMNS`` appended.
    """
    s1_name = s1_df["name_norm"].to_numpy()[cand_df["s1_pos"].to_numpy()]
    o_name = other_df["name_norm"].to_numpy()[cand_df["other_pos"].to_numpy()]
    s1_addr = s1_df["addr_norm"].to_numpy()[cand_df["s1_pos"].to_numpy()]
    o_addr = other_df["addr_norm"].to_numpy()[cand_df["other_pos"].to_numpy()]
    s1_country = s1_df["country"].to_numpy()[cand_df["s1_pos"].to_numpy()]
    o_country = other_df["country"].to_numpy()[cand_df["other_pos"].to_numpy()]

    out = cand_df.copy()
    out["country_match"] = (s1_country == o_country).astype(np.int8)

    name_sets_s1 = _token_sets(s1_name)
    name_sets_o = _token_sets(o_name)
    out["name_jaccard"] = _jaccard(name_sets_s1, name_sets_o, s1_name, o_name).astype(np.float32)

    addr_sets_s1 = {v: frozenset(v.split(" ")) if v else frozenset() for v in pd.unique(s1_addr)}
    addr_sets_o = {v: frozenset(v.split(" ")) if v else frozenset() for v in pd.unique(o_addr)}
    out["addr_jaccard"] = _jaccard(addr_sets_s1, addr_sets_o, s1_addr, o_addr).astype(np.float32)

    out["name_fuzz_ratio"] = np.array([fuzz.ratio(a, b) for a, b in zip(s1_name, o_name)], dtype=np.float32)
    out["name_token_sort_ratio"] = np.array(
        [fuzz.token_sort_ratio(a, b) for a, b in zip(s1_name, o_name)], dtype=np.float32)
    out["addr_fuzz_partial_ratio"] = np.array(
        [fuzz.partial_ratio(a, b) for a, b in zip(s1_addr, o_addr)], dtype=np.float32)

    len1 = np.array([len(x) for x in s1_name], dtype=np.float32)
    len2 = np.array([len(x) for x in o_name], dtype=np.float32)
    out["name_len_ratio"] = np.minimum(len1, len2) / np.maximum(np.maximum(len1, len2), 1.0)

    return out
