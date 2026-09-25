"""Pairwise feature computation for the entity-matching model.

Operates only on a DataFrame of already-blocked (s1_id, cand_id) pairs, never
on the full cross product. All row-wise similarity metrics use rapidfuzz
(C-accelerated) instead of pure-Python string distance for speed at scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

FEATURE_COLUMNS = [
    "name_jaccard", "name_lev_ratio", "name_jw", "name_token_sort_ratio",
    "name_partial_ratio", "name_len_ratio", "name_acronym_match",
    "name_full_exact", "name_core_exact", "same_script",
    "addr_jaccard", "addr_lev_ratio", "addr_token_sort_ratio",
    "postal_exact_match", "postal_both_present",
    "street_number_match", "street_number_both_present",
    "city_token_overlap",
    "country_match", "country_both_present",
    "blocking_score", "n_blocking_sources", "candidate_rank", "score_gap_to_next",
]


def _token_jaccard(a_tokens, b_tokens) -> float:
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _acronym_match(a_tokens, b_tokens) -> int:
    """1 if one side's tokens' initials spell (approximately) the other side's
    single-token name, e.g. ["ibm"] vs ["international","business","machines"]."""
    if len(a_tokens) == 1 and len(b_tokens) > 1:
        short, long_ = a_tokens[0], b_tokens
    elif len(b_tokens) == 1 and len(a_tokens) > 1:
        short, long_ = b_tokens[0], a_tokens
    else:
        return 0
    initials = "".join(t[0] for t in long_ if t)
    return int(short == initials)


def compute_features(pairs: pd.DataFrame, s1_norm: pd.DataFrame, other_norm: pd.DataFrame
                      ) -> pd.DataFrame:
    """pairs: columns s1_id, cand_id[, score]. s1_norm / other_norm: normalized
    frames indexed by entity_id (output of normalize.build_normalized_frame,
    with entity_id set as index). Returns pairs with feature columns appended.
    """
    s1 = s1_norm.loc[pairs["s1_id"].values].reset_index(drop=True)
    o = other_norm.loc[pairs["cand_id"].values].reset_index(drop=True)

    out = pairs.reset_index(drop=True).copy()

    out["name_jaccard"] = [
        _token_jaccard(a, b) for a, b in zip(s1["name_core_tokens"], o["name_core_tokens"])
    ]
    out["name_lev_ratio"] = [
        Levenshtein.normalized_similarity(a, b) for a, b in zip(s1["name_norm"], o["name_norm"])
    ]
    out["name_jw"] = [
        fuzz.WRatio(a, b) / 100.0 for a, b in zip(s1["name_norm"], o["name_norm"])
    ]
    out["name_token_sort_ratio"] = [
        fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1["name_norm"], o["name_norm"])
    ]
    out["name_partial_ratio"] = [
        fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1["name_norm"], o["name_norm"])
    ]
    len_a = s1["name_norm"].str.len().clip(lower=1)
    len_b = o["name_norm"].str.len().clip(lower=1)
    out["name_len_ratio"] = np.minimum(len_a, len_b) / np.maximum(len_a, len_b)
    out["name_acronym_match"] = [
        _acronym_match(a, b) for a, b in zip(s1["name_core_tokens"], o["name_core_tokens"])
    ]
    out["name_full_exact"] = (s1["name_norm_full"].values == o["name_norm_full"].values).astype(int)
    out["name_core_exact"] = (s1["name_norm"].values == o["name_norm"].values).astype(int)
    out["same_script"] = (s1["name_script"].values == o["name_script"].values).astype(int)

    out["addr_jaccard"] = [
        _token_jaccard(a, b) for a, b in zip(s1["addr_tokens"], o["addr_tokens"])
    ]
    out["addr_lev_ratio"] = [
        Levenshtein.normalized_similarity(a, b) for a, b in zip(s1["addr_norm"], o["addr_norm"])
    ]
    out["addr_token_sort_ratio"] = [
        fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1["addr_norm"], o["addr_norm"])
    ]

    s1_post, o_post = s1["postal_code"].values, o["postal_code"].values
    both_post = (s1_post != "") & (o_post != "")
    out["postal_both_present"] = both_post.astype(int)
    out["postal_exact_match"] = (both_post & (s1_post == o_post)).astype(int)

    s1_sn, o_sn = s1["street_number"].values, o["street_number"].values
    both_sn = (s1_sn != "") & (o_sn != "")
    out["street_number_both_present"] = both_sn.astype(int)
    out["street_number_match"] = (both_sn & (s1_sn == o_sn)).astype(int)

    out["city_token_overlap"] = (
        (s1["city_guess"].values != "") & (s1["city_guess"].values == o["city_guess"].values)
    ).astype(int)

    s1_country, o_country = s1["country"].values, o["country"].values
    both_country = (s1_country != "") & (o_country != "")
    out["country_both_present"] = both_country.astype(int)
    out["country_match"] = (both_country & (s1_country == o_country)).astype(int)

    out["blocking_score"] = out["score"] if "score" in out.columns else out["name_jaccard"]
    out["n_blocking_sources"] = out.get("n_blocking_sources", 1)
    out["candidate_rank"] = out.get("candidate_rank", 0)
    out["score_gap_to_next"] = out.get("score_gap_to_next", 0.0)

    return out
