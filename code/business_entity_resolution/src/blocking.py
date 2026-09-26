"""GPU candidate generation (blocking) for the Business Entity Resolution challenge.

Rewritten on RAPIDS cuDF (GPU merges) + cuML (GPU TF-IDF/cosine similarity)
instead of pandas + a Python trigram-Jaccard loop. UNVERIFIED: written from
documented cuDF/cuML APIs; I have no GPU available to execute this.

Strategy (same four blocking keys as the CPU version, still a union):
1. Token blocking: explode S1/other core-name tokens, inner-join on token
   (cuDF `.merge()` runs as a GPU hash-join kernel). Same "drop overly
   frequent tokens, restrict the other side to S1's own key vocabulary
   first" safety design as the CPU version's OOM fix -- that fix is about
   join *cardinality*, not CPU-vs-GPU, so it still matters here.
2. Postal-code / name-prefix / soundex blocking: same idea, GPU merges.
3. Rescoring: the CPU version scores candidates with a Python loop computing
   character-trigram Jaccard per pair. There is no GPU-vectorized way to do
   per-pair Python set operations, so this is replaced with a genuinely
   GPU-native equivalent: cuML's TfidfVectorizer (character n-grams) fit
   once over all names, transformed into GPU sparse vectors, with cosine
   similarity computed per (s1, candidate) pair via a GPU sparse row-wise
   dot product. This is not numerically identical to trigram Jaccard, but
   is the same kind of signal (character-level name similarity) computed
   entirely on GPU.

Soundex is kept as plain Python (it's a tiny per-token computation over a
small number of distinct first-tokens, not a bottleneck worth a GPU kernel).
"""
from __future__ import annotations

import cudf
import cupy as cp
from cuml.feature_extraction.text import TfidfVectorizer

MAX_TOKEN_ABS_FREQ = 2000
FALLBACK_MAX_PER_TOKEN = 200
TOP_K_CANDIDATES = 30
NAME_PREFIX_LEN = 4


def _soundex(token: str) -> str:
    """Unchanged from the CPU version -- operates on Python strings (one per
    distinct first-token, via cuDF's .to_pandas() bridge below), not a
    per-row GPU op; too small a workload to be worth a GPU kernel."""
    if not token:
        return ""
    token = token.upper()
    codes = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    first = token[0]
    tail = []
    prev = codes.get(first, "")
    for ch in token[1:]:
        code = codes.get(ch, "")
        if code and code != prev:
            tail.append(code)
        prev = code if ch not in "HW" else prev
    return (first + "".join(tail) + "000")[:4]


def _explode_tokens(df: cudf.DataFrame, id_col: str, token_col: str) -> cudf.DataFrame:
    ex = df[[id_col, token_col]].explode(token_col)
    ex = ex.rename(columns={token_col: "token"})
    ex = ex[ex["token"].notna() & (ex["token"].str.len() >= 2)]
    return ex


def _frequent_tokens(ex: cudf.DataFrame) -> cudf.Series:
    if len(ex) == 0:
        return cudf.Series([], dtype="object")
    doc_freq = ex.groupby("token")[ex.columns[0]].nunique()
    return doc_freq[doc_freq > MAX_TOKEN_ABS_FREQ].index.to_series()


def _token_join(s1_ex: cudf.DataFrame, other_ex: cudf.DataFrame, drop_tokens: cudf.Series
                 ) -> cudf.DataFrame:
    s1f = s1_ex[~s1_ex["token"].isin(drop_tokens)]
    otf = other_ex[~other_ex["token"].isin(drop_tokens)]
    pairs = s1f.merge(otf, on="token", how="inner")
    pairs = pairs.iloc[:, [0, 2]]
    pairs.columns = ["s1_id", "cand_id"]
    return pairs


def _gpu_char_ngram_cosine(s1_ids: cudf.Series, s1_names: cudf.Series,
                            cand_ids: cudf.Series, cand_names: cudf.Series,
                            all_names_for_fit: cudf.Series) -> cp.ndarray:
    """Fit a char n-gram TF-IDF vectorizer once (GPU, cuML), transform both
    sides of each pair, and return the per-pair cosine similarity -- the
    GPU-native replacement for the CPU version's per-pair trigram Jaccard
    Python loop. UNVERIFIED: cuML's TfidfVectorizer analyzer="char_wb" /
    ngram_range support and exact constructor kwargs should be checked
    against the installed cuml version; this mirrors scikit-learn's API,
    which cuML's text vectorizers generally track but not always exactly.
    """
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), lowercase=False)
    # cuML's char-ngram tokenizer breaks on a raw ValueError("cannot reindex
    # on an axis with duplicate labels") when the input text has many
    # duplicate values -- fitting the vocabulary/IDF weights doesn't need
    # repeated identical documents anyway, so dedup first.
    vectorizer.fit(all_names_for_fit.unique())

    s1_vecs = vectorizer.transform(s1_names)
    cand_vecs = vectorizer.transform(cand_names)

    # row-wise cosine similarity between s1_vecs[i] and cand_vecs[i] (already
    # aligned 1:1 per pair, not an all-pairs matrix): normalize each sparse
    # row to unit length, then the dot product of matched rows *is* the
    # cosine similarity. cupy.sparse elementwise-multiply + sum-per-row keeps
    # this on GPU throughout.
    def _row_norms(mat):
        sq = mat.multiply(mat)
        return cp.sqrt(cp.asarray(sq.sum(axis=1)).ravel())

    s1_norms = _row_norms(s1_vecs)
    cand_norms = _row_norms(cand_vecs)
    dots = cp.asarray(s1_vecs.multiply(cand_vecs).sum(axis=1)).ravel()
    denom = s1_norms * cand_norms
    denom[denom == 0] = 1.0
    return dots / denom


def _rescore_and_topk(pairs: cudf.DataFrame, s1_norm: cudf.Series, cand_norm: cudf.Series,
                       all_names_for_fit: cudf.Series, k: int) -> cudf.DataFrame:
    n_sources = pairs.groupby(["s1_id", "cand_id"]).size()
    pairs = pairs.drop_duplicates(["s1_id", "cand_id"])
    pairs = pairs.set_index(["s1_id", "cand_id"])
    pairs["n_blocking_sources"] = n_sources
    pairs = pairs.reset_index()

    s1_names = pairs["s1_id"].map(s1_norm)
    cand_names = pairs["cand_id"].map(cand_norm)
    pairs["score"] = _gpu_char_ngram_cosine(
        pairs["s1_id"], s1_names, pairs["cand_id"], cand_names, all_names_for_fit
    )

    pairs = pairs.sort_values(["s1_id", "score"], ascending=[True, False])
    pairs["candidate_rank"] = pairs.groupby("s1_id").cumcount()
    pairs = pairs[pairs["candidate_rank"] < k]
    top_score = pairs.groupby("s1_id")["score"].transform("max")
    pairs["score_gap_to_next"] = top_score - pairs["score"]
    return pairs


def build_candidates(s1_df: cudf.DataFrame, other_df: cudf.DataFrame, k: int = TOP_K_CANDIDATES
                      ) -> cudf.DataFrame:
    """cuDF equivalent of blocking.build_candidates. s1_df, other_df: GPU
    normalized frames (output of the GPU normalize.build_normalized_frame).
    Returns columns: s1_id, cand_id, score, n_blocking_sources,
    candidate_rank, score_gap_to_next (top-k per s1_id, deduped, scored).
    """
    s1_ex = _explode_tokens(s1_df.assign(entity_id=s1_df["entity_id"]), "entity_id", "name_core_tokens")
    s1_tokens = s1_ex["token"].unique()

    oth_ex_all = _explode_tokens(other_df.assign(entity_id=other_df["entity_id"]), "entity_id", "name_core_tokens")
    oth_ex = oth_ex_all[oth_ex_all["token"].isin(s1_tokens)]
    del oth_ex_all

    drop_tokens = cudf.concat([_frequent_tokens(oth_ex), _frequent_tokens(s1_ex)]).unique()
    token_pairs = _token_join(s1_ex, oth_ex, drop_tokens)

    covered_s1 = token_pairs["s1_id"].unique()
    uncovered = s1_ex[~s1_ex["entity_id"].isin(covered_s1)]
    if len(uncovered):
        token_doc_freq = oth_ex.groupby("token").size().reset_index()
        token_doc_freq.columns = ["token", "df"]
        rarest = uncovered.merge(token_doc_freq, on="token", how="left")
        rarest["df"] = rarest["df"].fillna(0)
        rarest = rarest.sort_values("df").drop_duplicates("entity_id")

        fallback_pool = oth_ex[oth_ex["token"].isin(rarest["token"].unique())]
        fallback_rank = fallback_pool.groupby("token").cumcount()
        fallback_candidates = fallback_pool[fallback_rank < FALLBACK_MAX_PER_TOKEN]

        fallback_pairs = rarest.rename(columns={"entity_id": "s1_id"})[["s1_id", "token"]].merge(
            fallback_candidates.rename(columns={"entity_id": "cand_id"}), on="token", how="inner"
        )[["s1_id", "cand_id"]]
        token_pairs = cudf.concat([token_pairs, fallback_pairs], ignore_index=True)

    s1_postal = s1_df.loc[s1_df["postal_code"] != "", "postal_code"].unique()
    postal_pairs = s1_df[s1_df["postal_code"] != ""][["entity_id", "postal_code"]].merge(
        other_df[other_df["postal_code"].isin(s1_postal)][["entity_id", "postal_code"]],
        on="postal_code", suffixes=("_s1", "_o"),
    )[["entity_id_s1", "entity_id_o"]]
    postal_pairs.columns = ["s1_id", "cand_id"]

    s1_df = s1_df.copy()
    other_df = other_df.copy()
    s1_df["name_prefix"] = s1_df["name_norm"].str.slice(0, NAME_PREFIX_LEN)
    other_df["name_prefix"] = other_df["name_norm"].str.slice(0, NAME_PREFIX_LEN)
    s1_prefixes = s1_df.loc[s1_df["name_prefix"] != "", "name_prefix"].unique()
    prefix_pairs = s1_df[s1_df["name_prefix"] != ""][["entity_id", "name_prefix"]].merge(
        other_df[other_df["name_prefix"].isin(s1_prefixes)][["entity_id", "name_prefix"]],
        on="name_prefix", suffixes=("_s1", "_o"),
    )[["entity_id_s1", "entity_id_o"]]
    prefix_pairs.columns = ["s1_id", "cand_id"]

    # soundex: small per-distinct-first-token computation, done via the
    # pandas/CPU bridge (.to_pandas()) rather than a GPU kernel -- see
    # _soundex's docstring for why.
    s1_df["soundex1"] = cudf.Series(
        s1_df["name_core_tokens"].to_pandas().map(lambda ts: _soundex(ts[0]) if len(ts) else "")
    )
    other_df["soundex1"] = cudf.Series(
        other_df["name_core_tokens"].to_pandas().map(lambda ts: _soundex(ts[0]) if len(ts) else "")
    )
    s1_soundex = s1_df.loc[s1_df["soundex1"] != "", "soundex1"].unique()
    sdx_pairs = s1_df[s1_df["soundex1"] != ""][["entity_id", "soundex1"]].merge(
        other_df[other_df["soundex1"].isin(s1_soundex)][["entity_id", "soundex1"]],
        on="soundex1", suffixes=("_s1", "_o"),
    )[["entity_id_s1", "entity_id_o"]]
    sdx_pairs.columns = ["s1_id", "cand_id"]

    all_pairs = cudf.concat([token_pairs, postal_pairs, prefix_pairs, sdx_pairs], ignore_index=True)
    if len(all_pairs) == 0:
        return cudf.DataFrame({"s1_id": [], "cand_id": [], "score": []})

    s1_norm = s1_df.set_index("entity_id")["name_norm"]
    cand_norm = other_df.set_index("entity_id")["name_norm"]
    all_names_for_fit = cudf.concat([s1_norm, cand_norm]).reset_index(drop=True)
    result = _rescore_and_topk(all_pairs, s1_norm, cand_norm, all_names_for_fit, k)
    return result
