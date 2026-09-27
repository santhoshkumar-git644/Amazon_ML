"""Candidate generation (blocking) for the Business Entity Resolution challenge.

Standard inverted-index blocking: records that share a sufficiently rare
"blocking key" are paired up as candidates. Four independent key types are
unioned together, because an empirical pass over the training data showed
each catches matches the others miss:

- **Name tokens.** Business names draw heavily on a shared vocabulary of
  generic descriptor words ("global", "solutions", "center", "consultancy",
  ...) — after stripping legal suffixes and the ~70 most generic business
  words, most of what's left is still only moderately rare, since names are
  short (median 2 significant tokens).
- **Address tokens.** Addresses are longer (median 5 tokens) and contain much
  more specific content — plot/door numbers, street and locality names — so
  they carry more blocking signal, especially for a business whose *name* is
  entirely generic (e.g. "Global Health Solutions").
- **Postal code** exact match, and **name prefix** (first 4 characters with
  spaces stripped) as a fallback for near-duplicate names blocking missed.

At multi-million-row scale two things keep this close to linear instead of
quadratic:

1. **Per-row rarest-token selection.** Each row contributes only its
   ``top_k`` least-frequent tokens (by document frequency across the two
   sides being blocked) as keys, rather than every token — this is what lets
   an address with one genuinely rare street name still produce a candidate
   even though it also contains "delhi" and "west".
2. **A hard per-key document-frequency cap** as a backstop, so even a
   rare-ish token that still has a heavy postings list on one side can't blow
   up the pairwise product.

Token lists are never materialized as persistent per-row columns (expensive in
memory at multi-million-row scale); tokenization happens here, transiently, on
plain string arrays.

Everything here is rule-based/statistical over the given data only; no
external services, gazetteers or geocoders are used.
"""
import numpy as np
import pandas as pd

from normalize import significant_tokens, significant_address_tokens

# (top_k, max_document_frequency) per key type. See module docstring for why
# each exists; values were tuned empirically against blocking recall on
# train_ground_truth.tsv vs. candidate-set size / memory on this dataset.
NAME_TOKEN_TOP_K, NAME_TOKEN_MAX_DF = 1, 50
ADDR_TOKEN_TOP_K, ADDR_TOKEN_MAX_DF = 1, 30
MAX_PREFIX_DF = 300
MAX_POSTAL_DF = 1500
# Hard safety cap on a single key's pairwise product, regardless of the
# per-side DF caps above (belt and suspenders against a pathological key).
MAX_BUCKET_PRODUCT = 100_000


def _explode(values: np.ndarray, tokenizer):
    """Tokenize a string array into (row_position, token) arrays."""
    row_pos_list = []
    flat_tokens = []
    for i, s in enumerate(values):
        toks = tokenizer(s) if isinstance(s, str) else []
        if toks:
            row_pos_list.append(np.full(len(toks), i, dtype=np.int64))
            flat_tokens.extend(toks)
    if not flat_tokens:
        return np.empty(0, dtype=np.int64), []
    return np.concatenate(row_pos_list), flat_tokens


def explode_name_tokens(name_norm: np.ndarray):
    return _explode(name_norm, significant_tokens)


def explode_addr_tokens(addr_norm: np.ndarray):
    return _explode(addr_norm, significant_address_tokens)


def _keep_top_k_rarest(row, freq_per_entry, k):
    """Boolean mask keeping, per distinct value of ``row``, the k entries
    with the smallest ``freq_per_entry`` (ties broken arbitrarily)."""
    order = np.lexsort((freq_per_entry, row))
    row_sorted = row[order]
    change = np.empty(len(row_sorted), dtype=bool)
    change[0] = True
    if len(row_sorted) > 1:
        change[1:] = row_sorted[1:] != row_sorted[:-1]
    group_id = np.cumsum(change) - 1
    first_idx_of_group = np.nonzero(change)[0]
    rank_within_group = np.arange(len(row_sorted)) - first_idx_of_group[group_id]
    keep_sorted = rank_within_group < k
    keep = np.zeros(len(row), dtype=bool)
    keep[order[keep_sorted]] = True
    return keep


def _bucket_pairs(s1_row_sorted, s1_start, s1_cnt, o_row_sorted, o_start, o_cnt,
                   common_codes, o_pos_of, max_df, max_product):
    pair_keys = []
    for i, code in enumerate(common_codes):
        j = o_pos_of.get(code)
        if j is None:
            continue
        n1, n2 = s1_cnt[i], o_cnt[j]
        if n1 > max_df or n2 > max_df or n1 * n2 > max_product:
            continue
        a_ids = s1_row_sorted[s1_start[i]: s1_start[i] + n1]
        b_ids = o_row_sorted[o_start[j]: o_start[j] + n2]
        a = np.repeat(a_ids, n2).astype(np.int64)
        b = np.tile(b_ids, n1).astype(np.int64)
        pair_keys.append((a << np.int64(32)) | b)
    return pair_keys


_EMPTY_TOKEN_RESULT_TEMPLATE = {"s1_pos": "int64", "other_pos": "int64"}


def _empty_result(count_col):
    return pd.DataFrame({"s1_pos": pd.Series(dtype="int64"),
                          "other_pos": pd.Series(dtype="int64"),
                          count_col: pd.Series(dtype="int64")})


def block_on_tokens(s1_explode, other_explode, top_k, max_df, count_col):
    """Many-to-many blocking from two precomputed ``_explode`` results.

    Each row contributes only its ``top_k`` least-frequent tokens (frequency
    measured jointly across both sides) as candidate keys. Returns a
    DataFrame [s1_pos, other_pos, <count_col>] (number of shared keys).
    """
    s1_row, s1_flat = s1_explode
    o_row, o_flat = other_explode
    if len(s1_flat) == 0 or len(o_flat) == 0:
        return _empty_result(count_col)

    codes, _ = pd.factorize(np.asarray(s1_flat + o_flat, dtype=object), sort=False)
    s1_codes = codes[: len(s1_flat)]
    o_codes = codes[len(s1_flat):]
    freq = np.bincount(codes)
    del codes

    s1_keep = _keep_top_k_rarest(s1_row, freq[s1_codes], top_k)
    o_keep = _keep_top_k_rarest(o_row, freq[o_codes], top_k)
    s1_row, s1_codes = s1_row[s1_keep], s1_codes[s1_keep]
    o_row, o_codes = o_row[o_keep], o_codes[o_keep]

    s1_order = np.argsort(s1_codes, kind="stable")
    o_order = np.argsort(o_codes, kind="stable")
    s1_codes_sorted = s1_codes[s1_order]
    o_codes_sorted = o_codes[o_order]
    s1_row_sorted = s1_row[s1_order]
    o_row_sorted = o_row[o_order]

    s1_tok, s1_start, s1_cnt = np.unique(s1_codes_sorted, return_index=True, return_counts=True)
    o_tok, o_start, o_cnt = np.unique(o_codes_sorted, return_index=True, return_counts=True)
    o_pos_of = {tok: i for i, tok in enumerate(o_tok)}

    pair_keys = _bucket_pairs(s1_row_sorted, s1_start, s1_cnt, o_row_sorted, o_start, o_cnt,
                               s1_tok, o_pos_of, max_df, MAX_BUCKET_PRODUCT)
    if not pair_keys:
        return _empty_result(count_col)

    all_keys = np.concatenate(pair_keys)
    uniq_keys, counts = np.unique(all_keys, return_counts=True)
    s1_pos = (uniq_keys >> np.int64(32)).astype(np.int64)
    other_pos = (uniq_keys & np.int64(0xFFFFFFFF)).astype(np.int64)
    return pd.DataFrame({"s1_pos": s1_pos, "other_pos": other_pos, count_col: counts})


def block_on_scalar_key(s1_keys, other_keys, feature_name, max_df):
    """Exact-match blocking on a scalar key column (e.g. postal code, name prefix).

    Empty-string keys never match (an absent postal code isn't a signal).
    Returns [s1_pos, other_pos, <feature_name>] with the feature set to 1.
    """
    empty = _empty_result(feature_name)
    s1_keys = np.asarray(s1_keys, dtype=object)
    other_keys = np.asarray(other_keys, dtype=object)

    s1_valid = np.array([bool(k) for k in s1_keys])
    o_valid = np.array([bool(k) for k in other_keys])
    if not s1_valid.any() or not o_valid.any():
        return empty

    s1_idx = np.nonzero(s1_valid)[0]
    o_idx = np.nonzero(o_valid)[0]
    codes, _ = pd.factorize(np.concatenate([s1_keys[s1_idx], other_keys[o_idx]]), sort=False)
    s1_codes = codes[: len(s1_idx)]
    o_codes = codes[len(s1_idx):]

    s1_order = np.argsort(s1_codes, kind="stable")
    o_order = np.argsort(o_codes, kind="stable")
    s1_codes_sorted = s1_codes[s1_order]
    o_codes_sorted = o_codes[o_order]
    s1_row_sorted = s1_idx[s1_order]
    o_row_sorted = o_idx[o_order]

    s1_tok, s1_start, s1_cnt = np.unique(s1_codes_sorted, return_index=True, return_counts=True)
    o_tok, o_start, o_cnt = np.unique(o_codes_sorted, return_index=True, return_counts=True)
    o_pos_of = {tok: i for i, tok in enumerate(o_tok)}

    pair_keys = _bucket_pairs(s1_row_sorted, s1_start, s1_cnt, o_row_sorted, o_start, o_cnt,
                               s1_tok, o_pos_of, max_df, MAX_BUCKET_PRODUCT)
    if not pair_keys:
        return empty

    all_keys = np.unique(np.concatenate(pair_keys))
    s1_pos = (all_keys >> np.int64(32)).astype(np.int64)
    other_pos = (all_keys & np.int64(0xFFFFFFFF)).astype(np.int64)
    return pd.DataFrame({"s1_pos": s1_pos, "other_pos": other_pos, feature_name: 1})


def explode_all(df):
    """Precompute all reusable per-side token explodes (name + address).

    Pass the result in as ``s1_explodes`` to ``generate_candidates`` to avoid
    re-tokenizing the Source-1 side once per "other" source (Source-2 *and*
    Source-3).
    """
    return {
        "name": explode_name_tokens(df["name_norm"].to_numpy()),
        "addr": explode_addr_tokens(df["addr_norm"].to_numpy()),
    }


def generate_candidates(s1_df, other_df, s1_explodes=None):
    """Union of name-token / address-token / postal / prefix blocking.

    Both frames must be the output of ``normalize.normalize_frame`` with a
    fresh 0..N-1 positional index (``reset_index(drop=True)``). Pass
    ``s1_explodes`` (from ``explode_all``) to avoid re-tokenizing the
    Source-1 side when blocking it against both Source-2 and Source-3.
    Returns one row per (s1_pos, other_pos) candidate pair with per-key
    features: n_shared_name_tokens, n_shared_addr_tokens, postal_match,
    prefix_match (0 when that key type didn't produce the pair).
    """
    if s1_explodes is None:
        s1_explodes = explode_all(s1_df)
    other_explodes = explode_all(other_df)

    name_tok = block_on_tokens(s1_explodes["name"], other_explodes["name"],
                                NAME_TOKEN_TOP_K, NAME_TOKEN_MAX_DF, "n_shared_name_tokens")
    addr_tok = block_on_tokens(s1_explodes["addr"], other_explodes["addr"],
                                ADDR_TOKEN_TOP_K, ADDR_TOKEN_MAX_DF, "n_shared_addr_tokens")
    postal = block_on_scalar_key(s1_df["postal"].to_numpy(), other_df["postal"].to_numpy(),
                                  "postal_match", MAX_POSTAL_DF)
    prefix = block_on_scalar_key(s1_df["name_prefix4"].to_numpy(), other_df["name_prefix4"].to_numpy(),
                                  "prefix_match", MAX_PREFIX_DF)

    merged = name_tok.merge(addr_tok, on=["s1_pos", "other_pos"], how="outer")
    merged = merged.merge(postal, on=["s1_pos", "other_pos"], how="outer")
    merged = merged.merge(prefix, on=["s1_pos", "other_pos"], how="outer")
    merged["n_shared_name_tokens"] = merged["n_shared_name_tokens"].fillna(0).astype(np.int16)
    merged["n_shared_addr_tokens"] = merged["n_shared_addr_tokens"].fillna(0).astype(np.int16)
    merged["postal_match"] = merged["postal_match"].fillna(0).astype(np.int8)
    merged["prefix_match"] = merged["prefix_match"].fillna(0).astype(np.int8)
    merged["s1_pos"] = merged["s1_pos"].astype(np.int32)
    merged["other_pos"] = merged["other_pos"].astype(np.int32)
    return merged
