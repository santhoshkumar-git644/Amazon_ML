"""GPU candidate generation (blocking) for the Business Entity Resolution challenge.

Rewritten on RAPIDS cuDF (GPU merges + native char n-gram string ops)
instead of pandas + a Python trigram-Jaccard loop. UNVERIFIED: written from
documented cuDF APIs; I have no GPU available to execute this.

Strategy (same four blocking keys as the CPU version, still a union):
1. Token blocking: explode S1/other core-name tokens, inner-join on token
   (cuDF `.merge()` runs as a GPU hash-join kernel). Same "drop overly
   frequent tokens, restrict the other side to S1's own key vocabulary
   first" safety design as the CPU version's OOM fix -- that fix is about
   join *cardinality*, not CPU-vs-GPU, so it still matters here.
2. Postal-code / name-prefix / soundex blocking: same idea, GPU merges.
3. Rescoring: the CPU version scores candidates with a Python loop computing
   character-trigram Jaccard per pair. Reimplemented here with
   gpu_char_ngram_jaccard() below, using cuDF's native `.str.character_ngrams()`
   plus the same explode/merge/groupby pattern as features.py's word-token
   Jaccard -- not cuML, after cuML's TfidfVectorizer proved to crash on real
   cluster data (see git history) with what looks like an internal bug, not
   something fixable from the calling code.

Soundex is kept as plain Python (it's a tiny per-token computation over a
small number of distinct first-tokens, not a bottleneck worth a GPU kernel).
"""
from __future__ import annotations

import cudf
import cupy as cp

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


def _keyed_join(s1_df: cudf.DataFrame, other_df: cudf.DataFrame, key_col: str,
                 max_freq: int = MAX_TOKEN_ABS_FREQ) -> cudf.DataFrame:
    """Join s1_df/other_df on key_col (postal_code, name_prefix, or
    soundex1), restricted to (a) keys that actually occur in s1_df -- the
    same lossless-for-an-inner-join restriction the token join uses -- and
    (b) keys that aren't so common in either side they'd blow the join up
    disproportionately, the same MAX_TOKEN_ABS_FREQ cap the token join uses.

    Real crash this fixes: at 50,000 S1 entities, the soundex join alone
    tried to allocate ~9.9GB and died. Soundex codes have very limited
    cardinality (~26,000 possible 4-character codes) -- once S1 is large
    enough, by pigeonhole it covers most of that space, so restriction (a)
    alone stops helping (nearly the whole other-source table still passes
    the filter). Postal code and name-prefix are lower-cardinality than
    name tokens for the same reason, just less extreme than soundex. The
    token join has always had this frequency cap (that's what fixed the
    original CPU OOM); postal/prefix/soundex never did until now.
    """
    s1_valid = s1_df[key_col] != ""
    other_valid = other_df[key_col] != ""
    s1_keys = s1_df.loc[s1_valid, key_col]
    other_keys = other_df.loc[other_valid, key_col]

    s1_freq = s1_keys.value_counts()
    other_freq = other_keys.value_counts()
    drop_keys = cudf.concat([
        s1_freq[s1_freq > max_freq].index.to_series(),
        other_freq[other_freq > max_freq].index.to_series(),
    ]).unique()

    s1_key_set = s1_keys.unique()
    s1_side = s1_df[s1_valid & (~s1_df[key_col].isin(drop_keys))][["entity_id", key_col]]
    other_side = other_df[
        other_valid & other_df[key_col].isin(s1_key_set) & (~other_df[key_col].isin(drop_keys))
    ][["entity_id", key_col]]

    pairs = s1_side.merge(other_side, on=key_col, suffixes=("_s1", "_o"))[
        ["entity_id_s1", "entity_id_o"]
    ]
    pairs.columns = ["s1_id", "cand_id"]
    return pairs


def _score_unique_batch(batch: cudf.DataFrame, n: int) -> cudf.DataFrame:
    """Score one batch of unique (a, b) pairs. batch must have columns
    a, b, u_row_id (0..len(batch)-1). Returns batch with a "score" column
    added. Split out of gpu_char_ngram_jaccard so peak memory is bounded by
    one batch's explode, not the full unique-pair set's."""
    u_row_id = batch["u_row_id"]
    padded_a = " " + batch["a"].fillna("") + " "
    padded_b = " " + batch["b"].fillna("") + " "
    tg_a = padded_a.str.character_ngrams(n, as_list=True)
    tg_b = padded_b.str.character_ngrams(n, as_list=True)

    ea = cudf.DataFrame({"u_row_id": u_row_id, "tg": tg_a}).explode("tg").dropna()
    eb = cudf.DataFrame({"u_row_id": u_row_id, "tg": tg_b}).explode("tg").dropna()
    ea = ea.drop_duplicates(["u_row_id", "tg"])
    eb = eb.drop_duplicates(["u_row_id", "tg"])

    inter = ea.merge(eb, on=["u_row_id", "tg"], how="inner").groupby("u_row_id").size()
    union = cudf.concat([ea, eb]).drop_duplicates(["u_row_id", "tg"]).groupby("u_row_id").size()
    inter = inter.reindex(u_row_id.values).fillna(0)
    union = union.reindex(u_row_id.values).fillna(0)
    batch = batch.copy()
    batch["score"] = (inter / union.where(union != 0, 1)).fillna(0.0).values
    return batch


def gpu_char_ngram_jaccard(a: cudf.Series, b: cudf.Series, n: int = 3,
                           batch_size: int = 20_000) -> cp.ndarray:
    """Row-aligned character n-gram Jaccard similarity between a[i] and b[i]
    -- GPU-native reimplementation of the CPU version's per-pair trigram
    Jaccard Python loop, using cuDF's own `.str.character_ngrams()` (core
    cuDF string functionality, not cuML's separate text-vectorizer layer).

    Switched to this from a cuML TfidfVectorizer-based cosine similarity
    after that approach crashed on real cluster data with a raw
    ValueError("cannot reindex on an axis with duplicate labels") from
    inside cuml's own tokenizer -- an apparent bug in that part of cuML I
    can't debug without GPU access. This reuses the same
    explode -> pairwise-intersect/union -> groupby-count pattern as
    features.py's _gpu_word_token_jaccard, which is plain cuDF
    (merge/groupby/explode), not cuML.

    Two real OOM crashes from the cluster shaped this function's current
    structure:
    1. Exploding character n-grams for every ROW (not just unique values)
       ran out of GPU memory once the pre-top-k candidate pool got large
       (~6.8GB single allocation failure at 5000 S1 entities). Fixed by
       deduplicating to unique (a, b) string pairs first -- pure caching,
       since the score only depends on the string values, not row identity.
    2. Deduplication alone wasn't enough at the same 5000-entity scale --
       the unique-pair set itself was still large enough that a *second*
       allocation failed (~4.4GB) inside .dropna(), with nvidia-smi showing
       usage climb past 36GB first. That's consistent with needing roughly
       2x memory during explode/dropna (the new filtered/compacted buffer
       has to be allocated before the old one is freed). Fixed by batching:
       process batch_size unique pairs at a time instead of all of them in
       one GPU operation, so peak memory is bounded by one batch regardless
       of how large the total unique-pair set grows. cupy's memory pool is
       explicitly flushed between batches so freed memory is actually
       returned, not just cached for reuse within the pool.
    """
    n_rows = len(a)
    orig_row_id = cp.arange(n_rows)
    pairs_df = cudf.DataFrame({
        "orig_row_id": orig_row_id,
        "a": a.reset_index(drop=True),
        "b": b.reset_index(drop=True),
    })
    unique_pairs = pairs_df[["a", "b"]].drop_duplicates().reset_index(drop=True)
    n_unique = len(unique_pairs)

    scored_parts = []
    for start in range(0, n_unique, batch_size):
        batch = unique_pairs.iloc[start:start + batch_size].reset_index(drop=True)
        batch["u_row_id"] = cudf.Series(cp.arange(len(batch)))
        scored_batch = _score_unique_batch(batch, n)
        scored_parts.append(scored_batch[["a", "b", "score"]])
        del batch, scored_batch
        cp.get_default_memory_pool().free_all_blocks()

    unique_scored = cudf.concat(scored_parts, ignore_index=True)

    # Map back to the original row order. A merge doesn't guarantee it
    # preserves left-frame row order, so sort by the explicit orig_row_id
    # afterward rather than assuming the merge kept it -- getting this
    # wrong would silently assign the wrong score to the wrong pair.
    scored = pairs_df.merge(unique_scored, on=["a", "b"], how="left")
    scored = scored.sort_values("orig_row_id")
    return cp.asarray(scored["score"].values)


def _rescore_and_topk(pairs: cudf.DataFrame, s1_norm: cudf.Series, cand_norm: cudf.Series,
                       k: int) -> cudf.DataFrame:
    n_sources = pairs.groupby(["s1_id", "cand_id"]).size()
    pairs = pairs.drop_duplicates(["s1_id", "cand_id"])
    pairs = pairs.set_index(["s1_id", "cand_id"])
    pairs["n_blocking_sources"] = n_sources
    pairs = pairs.reset_index()

    s1_names = pairs["s1_id"].map(s1_norm)
    cand_names = pairs["cand_id"].map(cand_norm)
    pairs["score"] = gpu_char_ngram_jaccard(s1_names, cand_names, n=3)

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

    s1_df = s1_df.copy()
    other_df = other_df.copy()
    s1_df["name_prefix"] = s1_df["name_norm"].str.slice(0, NAME_PREFIX_LEN)
    other_df["name_prefix"] = other_df["name_norm"].str.slice(0, NAME_PREFIX_LEN)

    # soundex: small per-distinct-first-token computation, done via the
    # pandas/CPU bridge (.to_pandas()) rather than a GPU kernel -- see
    # _soundex's docstring for why.
    s1_df["soundex1"] = cudf.Series(
        s1_df["name_core_tokens"].to_pandas().map(lambda ts: _soundex(ts[0]) if len(ts) else "")
    )
    other_df["soundex1"] = cudf.Series(
        other_df["name_core_tokens"].to_pandas().map(lambda ts: _soundex(ts[0]) if len(ts) else "")
    )

    postal_pairs = _keyed_join(s1_df, other_df, "postal_code")
    prefix_pairs = _keyed_join(s1_df, other_df, "name_prefix")
    sdx_pairs = _keyed_join(s1_df, other_df, "soundex1")

    all_pairs = cudf.concat([token_pairs, postal_pairs, prefix_pairs, sdx_pairs], ignore_index=True)
    if len(all_pairs) == 0:
        return cudf.DataFrame({"s1_id": [], "cand_id": [], "score": []})

    s1_norm = s1_df.set_index("entity_id")["name_norm"]
    cand_norm = other_df.set_index("entity_id")["name_norm"]
    result = _rescore_and_topk(all_pairs, s1_norm, cand_norm, k)
    return result
