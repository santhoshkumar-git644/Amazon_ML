"""GPU pairwise feature computation for the entity-matching model.

Rewritten on cuDF/cupy/cuML. UNVERIFIED: written from documented APIs; no
GPU available here to execute it.

Honest limitation, flagged up front rather than glossed over: the CPU
version's edit-distance features (Levenshtein similarity, Jaro-Winkler via
rapidfuzz.fuzz.WRatio, token_sort_ratio, partial_ratio) have **no mature GPU
library equivalent** I can point to with confidence -- rapidfuzz itself is
CPU-only (C-accelerated, but CPU), and there is no widely-used GPU port of
these specific string-distance algorithms. Rather than silently keep those
four features running on CPU inside an otherwise-GPU pipeline (which would
contradict "everything on GPU"), they're replaced here with GPU-native
substitutes:
  - name_lev_ratio, name_jw, name_token_sort_ratio, name_partial_ratio,
    addr_lev_ratio, addr_token_sort_ratio
    -> all replaced by variants of the same char n-gram TF-IDF cosine
       similarity used in blocking.py's GPU rescoring (different n-gram
       windows / word-level vs char-level where it maps reasonably).
This is a real change in what the model is trained on, not a drop-in
equivalent -- flagging this explicitly so it's an informed choice, not a
silently swapped detail.
"""
from __future__ import annotations

import cudf
import cupy as cp
from cuml.feature_extraction.text import TfidfVectorizer

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


def _gpu_pairwise_char_cosine(a: cudf.Series, b: cudf.Series, all_text_for_fit: cudf.Series,
                               ngram_range: tuple[int, int] = (3, 3)) -> cp.ndarray:
    """1:1 row-aligned cosine similarity between a[i] and b[i], char n-grams,
    TF-IDF weighted, fit once over all_text_for_fit -- same GPU pattern as
    blocking.py's _gpu_char_ngram_cosine."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, lowercase=False)
    # See blocking.py's _gpu_char_ngram_cosine: cuML's char-ngram tokenizer
    # breaks on duplicate-heavy input text with a raw cudf reindex
    # ValueError. Fitting the vocabulary doesn't need duplicates anyway.
    vectorizer.fit(all_text_for_fit.unique())
    va = vectorizer.transform(a)
    vb = vectorizer.transform(b)

    def _row_norms(mat):
        sq = mat.multiply(mat)
        return cp.sqrt(cp.asarray(sq.sum(axis=1)).ravel())

    dots = cp.asarray(va.multiply(vb).sum(axis=1)).ravel()
    denom = _row_norms(va) * _row_norms(vb)
    denom[denom == 0] = 1.0
    return dots / denom


def _gpu_word_token_jaccard(a: cudf.Series, b: cudf.Series) -> cp.ndarray:
    """Row-aligned Jaccard on list<string> token columns. cuDF list columns
    support .list.len() and set-like ops are not directly vectorized across
    two *different* list columns per row, so this uses the same
    explode-based approach as normalize.py's stopword filtering: build a
    per-row union/intersection count via exploding both sides tagged with a
    row id, rather than a Python per-row set() loop.
    UNVERIFIED -- this is the same class of uncertain cuDF list-column
    operation flagged in normalize.py; test on a tiny sample first.
    """
    n = len(a)
    row_id = cudf.Series(range(n))
    ea = cudf.DataFrame({"row_id": row_id, "token": a}).explode("token").dropna()
    eb = cudf.DataFrame({"row_id": row_id, "token": b}).explode("token").dropna()

    ea_u = ea.drop_duplicates(["row_id", "token"])
    eb_u = eb.drop_duplicates(["row_id", "token"])

    inter = ea_u.merge(eb_u, on=["row_id", "token"], how="inner").groupby("row_id").size()
    union = cudf.concat([ea_u, eb_u]).drop_duplicates(["row_id", "token"]).groupby("row_id").size()

    inter = inter.reindex(row_id.values).fillna(0)
    union = union.reindex(row_id.values).fillna(0)
    result = (inter / union.where(union != 0, 1)).fillna(0.0)
    return cp.asarray(result.values)


def _gpu_acronym_match(a_tokens: cudf.Series, b_tokens: cudf.Series) -> cp.ndarray:
    """UNVERIFIED. Initials-of-multi-token-name vs single-token-name match,
    vectorized via list column ops (.list.len(), string .str.get(0) applied
    per exploded token then re-joined). Left as a cuDF list/string op chain
    rather than a Python loop to keep it on GPU; the exact chain of list
    accessor methods is the least-standard part of this file, most likely to
    need hand-fixing against the installed cuDF version."""
    a_len = a_tokens.list.len()
    b_len = b_tokens.list.len()
    a_single = (a_len == 1)
    b_single = (b_len == 1)

    row_id = cudf.Series(range(len(a_tokens)))

    def initials_of(tokens_col, row_id):
        ex = cudf.DataFrame({"row_id": row_id, "token": tokens_col}).explode("token").dropna()
        ex["first_char"] = ex["token"].str.slice(0, 1)
        return ex.groupby("row_id")["first_char"].agg(lambda s: s.str.cat())  # UNVERIFIED aggregation

    a_initials = initials_of(a_tokens, row_id).reindex(row_id.values)
    b_initials = initials_of(b_tokens, row_id).reindex(row_id.values)

    a_short = a_tokens.list.get(0).where(a_single, "")
    b_short = b_tokens.list.get(0).where(b_single, "")

    match_a_short = a_single & (~b_single) & (a_short == b_initials.reset_index(drop=True))
    match_b_short = b_single & (~a_single) & (b_short == a_initials.reset_index(drop=True))
    return cp.asarray((match_a_short | match_b_short).fillna(False).values.astype("int32"))


def compute_features(pairs: cudf.DataFrame, s1_norm: cudf.DataFrame, other_norm: cudf.DataFrame
                      ) -> cudf.DataFrame:
    """cuDF equivalent of features.compute_features. pairs: columns s1_id,
    cand_id[, score, ...]. s1_norm / other_norm: GPU normalized frames
    indexed by entity_id. Returns pairs with feature columns appended.
    """
    s1 = s1_norm.loc[pairs["s1_id"]].reset_index(drop=True)
    o = other_norm.loc[pairs["cand_id"]].reset_index(drop=True)
    out = pairs.reset_index(drop=True).copy()

    all_names = cudf.concat([s1["name_norm"], o["name_norm"]]).reset_index(drop=True)
    all_addrs = cudf.concat([s1["addr_norm"], o["addr_norm"]]).reset_index(drop=True)

    out["name_jaccard"] = _gpu_word_token_jaccard(s1["name_core_tokens"], o["name_core_tokens"])
    out["name_lev_ratio"] = _gpu_pairwise_char_cosine(s1["name_norm"], o["name_norm"], all_names, (2, 3))
    out["name_jw"] = _gpu_pairwise_char_cosine(s1["name_norm"], o["name_norm"], all_names, (1, 2))
    out["name_token_sort_ratio"] = _gpu_pairwise_char_cosine(s1["name_norm"], o["name_norm"], all_names, (3, 3))
    out["name_partial_ratio"] = out["name_token_sort_ratio"]  # no GPU partial-match equivalent; reuse
    len_a = s1["name_norm"].str.len().clip(lower=1)
    len_b = o["name_norm"].str.len().clip(lower=1)
    out["name_len_ratio"] = cudf.Series(cp.minimum(len_a.values, len_b.values)) / cudf.Series(
        cp.maximum(len_a.values, len_b.values)
    )
    out["name_acronym_match"] = _gpu_acronym_match(s1["name_core_tokens"], o["name_core_tokens"])
    out["name_full_exact"] = (s1["name_norm_full"] == o["name_norm_full"]).astype("int32")
    out["name_core_exact"] = (s1["name_norm"] == o["name_norm"]).astype("int32")
    out["same_script"] = (s1["name_script"] == o["name_script"]).astype("int32")

    out["addr_jaccard"] = _gpu_word_token_jaccard(s1["addr_tokens"], o["addr_tokens"])
    out["addr_lev_ratio"] = _gpu_pairwise_char_cosine(s1["addr_norm"], o["addr_norm"], all_addrs, (2, 3))
    out["addr_token_sort_ratio"] = _gpu_pairwise_char_cosine(s1["addr_norm"], o["addr_norm"], all_addrs, (3, 3))

    s1_post, o_post = s1["postal_code"], o["postal_code"]
    both_post = (s1_post != "") & (o_post != "")
    out["postal_both_present"] = both_post.astype("int32")
    out["postal_exact_match"] = (both_post & (s1_post == o_post)).astype("int32")

    s1_sn, o_sn = s1["street_number"], o["street_number"]
    both_sn = (s1_sn != "") & (o_sn != "")
    out["street_number_both_present"] = both_sn.astype("int32")
    out["street_number_match"] = (both_sn & (s1_sn == o_sn)).astype("int32")

    out["city_token_overlap"] = ((s1["city_guess"] != "") & (s1["city_guess"] == o["city_guess"])).astype("int32")

    s1_country, o_country = s1["country"], o["country"]
    both_country = (s1_country != "") & (o_country != "")
    out["country_both_present"] = both_country.astype("int32")
    out["country_match"] = (both_country & (s1_country == o_country)).astype("int32")

    out["blocking_score"] = out["score"] if "score" in out.columns else out["name_jaccard"]
    out["n_blocking_sources"] = out.get("n_blocking_sources", 1)
    out["candidate_rank"] = out.get("candidate_rank", 0)
    out["score_gap_to_next"] = out.get("score_gap_to_next", 0.0)

    return out
