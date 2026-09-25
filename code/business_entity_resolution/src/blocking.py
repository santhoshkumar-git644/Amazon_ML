"""Candidate generation (blocking) for the Business Entity Resolution challenge.

Strategy (multi-key union, all vectorized pandas joins — no external services,
no per-pair Python loops):

1. Token blocking: explode each record's normalized "core" name tokens and
   inner-join S1 tokens against S2/S3 tokens. Tokens that are too frequent
   (appear in more than MAX_TOKEN_ABS_FREQ records, e.g. generic words like
   "market") are dropped from the join key set — they would blow up the
   candidate space without adding discriminative signal — but every record
   keeps at least its rarest token so nothing is blocked out entirely.
2. Postal-code blocking: exact match on the extracted trailing digit run
   (works for US ZIP, Indian PIN, French postal code alike — no country-specific
   hardcoding).
3. Name-prefix blocking: first 4 characters of the normalized name, to catch
   near-duplicates that don't share a full token (helps short names).
4. Soundex-of-first-token blocking: phonetic key, to catch typos when the
   token-level match fails.

All four keyed joins are unioned per (s1_id -> {s2/s3 ids}), then the union is
re-scored with a cheap vectorized character-trigram Jaccard and truncated to
the top-K per S1 entity. That truncated, re-scored set is what gets written to
candidate_pairs.tsv and is exactly what the matching model scores.

Scaling note: every join below first restricts the *other* source's join keys
(tokens, postal codes, name prefixes, soundex codes) to values that actually
occur somewhere in s1_df. This is lossless for an inner join — a key that
never appears in s1_df can never produce a matching row regardless of how the
join is filtered afterward — but it bounds every join's cost by s1_df's own
size instead of always scanning/joining against the full multi-million-row
other-source corpus. Skipping this restriction is what causes an OOM kill on
a small `--max-s1` dev run: even with only a few thousand S1 entities, the old
code still built join keys from and merged against the *entire* other source.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MAX_TOKEN_ABS_FREQ = 2000    # drop join tokens present in more than this many records,
                              # regardless of corpus size (an absolute, not relative, cap)
FALLBACK_MAX_PER_TOKEN = 200  # cap candidates contributed per fallback token (see build_candidates)
TOP_K_CANDIDATES = 30        # final candidates kept per S1 entity after rescoring
NAME_PREFIX_LEN = 4


def _soundex(token: str) -> str:
    """Minimal soundex implementation (stdlib-only, no external phonetic lib)."""
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


def _explode_tokens(df: pd.DataFrame, id_col: str, token_col: str) -> pd.DataFrame:
    ex = df[[id_col, token_col]].explode(token_col)
    ex = ex[ex[token_col].notna() & (ex[token_col].str.len() >= 2)]
    ex = ex.rename(columns={token_col: "token"})
    return ex


def _frequent_tokens(ex: pd.DataFrame) -> set:
    if ex.empty:
        return set()
    doc_freq = ex.groupby("token")[ex.columns[0]].nunique()
    return set(doc_freq[doc_freq > MAX_TOKEN_ABS_FREQ].index)


def _token_join(s1_ex: pd.DataFrame, other_ex: pd.DataFrame, drop_tokens: set) -> pd.DataFrame:
    """Join S1 exploded tokens against other-source exploded tokens, dropping
    over-frequent tokens from *both* sides first (fallback keeps rarest token
    per record so no record is silently excluded — see build_candidates)."""
    s1f = s1_ex[~s1_ex["token"].isin(drop_tokens)]
    otf = other_ex[~other_ex["token"].isin(drop_tokens)]
    pairs = s1f.merge(otf, on="token", how="inner")
    return pairs.iloc[:, [0, 2]].set_axis(["s1_id", "cand_id"], axis=1)


def _trigrams(s: str) -> set:
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _rescore_and_topk(pairs: pd.DataFrame, s1_norm: pd.Series, cand_norm: pd.Series,
                       k: int) -> pd.DataFrame:
    """pairs: columns s1_id, cand_id (may contain duplicates from multiple blocking
    strategies). Dedup (keeping a count of how many strategies found each pair),
    score with char-trigram Jaccard on normalized name, keep top-k per s1_id."""
    n_sources = pairs.groupby(["s1_id", "cand_id"]).size().rename("n_blocking_sources")
    pairs = pairs.drop_duplicates(["s1_id", "cand_id"]).set_index(["s1_id", "cand_id"])
    pairs["n_blocking_sources"] = n_sources
    pairs = pairs.reset_index()

    s1_tg = s1_norm.map(_trigrams)
    cand_tg = cand_norm.map(_trigrams)

    s1_tg_arr = pairs["s1_id"].map(s1_tg)
    cand_tg_arr = pairs["cand_id"].map(cand_tg)

    def jaccard(a, b):
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    pairs["score"] = [jaccard(a, b) for a, b in zip(s1_tg_arr, cand_tg_arr)]
    pairs = pairs.sort_values(["s1_id", "score"], ascending=[True, False])
    pairs["candidate_rank"] = pairs.groupby("s1_id").cumcount()
    pairs = pairs[pairs["candidate_rank"] < k].copy()
    top_score = pairs.groupby("s1_id")["score"].transform("max")
    pairs["score_gap_to_next"] = top_score - pairs["score"]
    return pairs


def build_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame, k: int = TOP_K_CANDIDATES
                      ) -> pd.DataFrame:
    """s1_df, other_df: normalized frames (output of normalize.build_normalized_frame)
    for Source-1 and a single other source (S2 or S3).
    Returns columns: s1_id, cand_id, score  (top-k per s1_id, deduped, scored).
    """
    s1_ex = _explode_tokens(s1_df.assign(entity_id=s1_df["entity_id"]), "entity_id", "name_core_tokens")
    s1_tokens = set(s1_ex["token"].unique())

    oth_ex_all = _explode_tokens(other_df.assign(entity_id=other_df["entity_id"]), "entity_id", "name_core_tokens")
    oth_ex = oth_ex_all[oth_ex_all["token"].isin(s1_tokens)].copy()
    del oth_ex_all  # free the (much larger) unfiltered explode as soon as we no longer need it

    drop_tokens = _frequent_tokens(oth_ex) | _frequent_tokens(s1_ex)
    token_pairs = _token_join(s1_ex, oth_ex, drop_tokens)

    # Fallback: records whose every token got dropped as "too frequent" still get
    # a chance via their single rarest token, so no record is blocked out
    # entirely just because its words are common. This intentionally bypasses
    # the frequency cap (that's the point -- it's the last resort), so it caps
    # candidates per fallback token directly instead, to stay bounded even if
    # many records share the same still-fairly-common fallback token.
    covered_s1 = set(token_pairs["s1_id"].unique())
    uncovered = s1_ex[~s1_ex["entity_id"].isin(covered_s1)]
    if len(uncovered):
        token_doc_freq = oth_ex.groupby("token").size().rename("df").reset_index()
        rarest = uncovered.merge(token_doc_freq, on="token", how="left")
        rarest["df"] = rarest["df"].fillna(0)
        rarest = rarest.sort_values("df").drop_duplicates("entity_id")
        fallback_pool = oth_ex[oth_ex["token"].isin(set(rarest["token"]))]
        fallback_rank = fallback_pool.groupby("token").cumcount()
        fallback_candidates = fallback_pool[fallback_rank < FALLBACK_MAX_PER_TOKEN]
        fallback_pairs = rarest.rename(columns={"entity_id": "s1_id"})[["s1_id", "token"]].merge(
            fallback_candidates.rename(columns={"entity_id": "cand_id"}), on="token", how="inner"
        )[["s1_id", "cand_id"]]
        token_pairs = pd.concat([token_pairs, fallback_pairs], ignore_index=True)

    # Same principle as the token join above: restrict each key column's
    # "other side" to values that actually occur in s1_df before merging, so
    # a common postal code / prefix / soundex code in the other source can't
    # blow up the join with candidates s1_df could never have matched anyway.
    s1_postal = set(s1_df.loc[s1_df["postal_code"] != "", "postal_code"])
    postal_pairs = s1_df[s1_df["postal_code"] != ""][["entity_id", "postal_code"]].merge(
        other_df[other_df["postal_code"].isin(s1_postal)][["entity_id", "postal_code"]],
        on="postal_code", suffixes=("_s1", "_o"),
    )[["entity_id_s1", "entity_id_o"]].set_axis(["s1_id", "cand_id"], axis=1)

    s1_df = s1_df.copy()
    other_df = other_df.copy()
    s1_df["name_prefix"] = s1_df["name_norm"].str.slice(0, NAME_PREFIX_LEN)
    other_df["name_prefix"] = other_df["name_norm"].str.slice(0, NAME_PREFIX_LEN)
    s1_prefixes = set(s1_df.loc[s1_df["name_prefix"] != "", "name_prefix"])
    prefix_pairs = s1_df[s1_df["name_prefix"] != ""][["entity_id", "name_prefix"]].merge(
        other_df[other_df["name_prefix"].isin(s1_prefixes)][["entity_id", "name_prefix"]],
        on="name_prefix", suffixes=("_s1", "_o"),
    )[["entity_id_s1", "entity_id_o"]].set_axis(["s1_id", "cand_id"], axis=1)

    s1_df["soundex1"] = s1_df["name_core_tokens"].map(lambda ts: _soundex(ts[0]) if ts else "")
    other_df["soundex1"] = other_df["name_core_tokens"].map(lambda ts: _soundex(ts[0]) if ts else "")
    s1_soundex = set(s1_df.loc[s1_df["soundex1"] != "", "soundex1"])
    sdx_pairs = s1_df[s1_df["soundex1"] != ""][["entity_id", "soundex1"]].merge(
        other_df[other_df["soundex1"].isin(s1_soundex)][["entity_id", "soundex1"]],
        on="soundex1", suffixes=("_s1", "_o"),
    )[["entity_id_s1", "entity_id_o"]].set_axis(["s1_id", "cand_id"], axis=1)

    all_pairs = pd.concat([token_pairs, postal_pairs, prefix_pairs, sdx_pairs], ignore_index=True)
    if all_pairs.empty:
        return pd.DataFrame(columns=["s1_id", "cand_id", "score"])

    s1_norm = s1_df.set_index("entity_id")["name_norm"]
    cand_norm = other_df.set_index("entity_id")["name_norm"]
    result = _rescore_and_topk(all_pairs, s1_norm, cand_norm, k)
    return result
