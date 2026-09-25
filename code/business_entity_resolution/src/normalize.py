"""GPU text normalization shared by blocking and feature extraction.

Rewritten on RAPIDS cuDF (GPU dataframes) instead of pandas. cuDF's string
column type (`cudf.Series.str`) runs vectorized string operations as CUDA
kernels across the whole column at once, so the design here is different
from a CPU implementation: instead of a Python function called once per row
(`.map(my_func)`, which cuDF cannot run on the GPU -- arbitrary Python
callables have no CUDA equivalent), every operation is expressed as a
whole-column vectorized call (`.str.lower()`, `.str.replace(regex, ...)`,
`.str.extractall(...)`) that cuDF compiles down to a GPU kernel.

UNVERIFIED: written from documented cuDF APIs; I have no GPU available to
execute this. Test on a small sample on the cluster before trusting it.

Known limitations versus the CPU version (src/normalize.py on the
perf-parallel-oom-fix branch):
- cuDF's regex engine is RE2-based (via nvidia/cudf's libcudf), not Python's
  `re` -- it supports standard patterns (character classes, anchors,
  quantifiers) but not backreferences or lookaround assertions. The patterns
  used here were kept simple enough to be RE2-safe.
- No per-token dict lookup: the legal-suffix / address-abbreviation synonym
  maps are applied as a sequence of whole-column regex substitutions (one
  `.str.replace()` per synonym rule) instead of a Python dict.get() per
  token. Functionally equivalent, but each rule is now a separate GPU pass
  over the column rather than one Python dict lookup per token.
- Accent stripping (`unicodedata.normalize` in the CPU version) has no cuDF
  equivalent -- skipped here. Business names/addresses in this dataset are
  predominantly Latin or Devanagari script; this mainly affects rare
  accented Latin characters (e.g. "café") not being folded to their
  unaccented form before comparison.
- name_core_tokens / addr_tokens are still produced as list-columns (cuDF
  supports list dtype columns and `.explode()` natively), used by
  blocking.py's GPU token-join and features.py's GPU token-Jaccard.
"""
from __future__ import annotations

import cudf

# --- name normalization -----------------------------------------------------

# Applied as whole-column regex substitutions, in order. Each entry is
# (pattern matching the whole token bounded by word boundaries, replacement).
# RE2 supports \b word boundaries, so this stays a vectorized GPU op per rule
# instead of a per-token Python dict lookup.
_LEGAL_SYNONYM_RULES = [
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcompany\b", "co"),
    (r"\blimited\b", "ltd"),
    (r"\bprivate\b", "pvt"),
    (r"&", " and "),
]

NAME_STOPWORDS = {
    "inc", "corp", "co", "ltd", "llc", "llp", "lp", "pvt", "private",
    "limited", "corporation", "incorporated", "company", "plc", "gmbh",
    "sarl", "sas", "and", "the", "of",
}

_ADDR_SYNONYM_RULES = [
    (r"\brd\b", "road"),
    (r"\bst\b", "street"),
    (r"\bstr\b", "street"),
    (r"\bave\b", "avenue"),
    (r"\bav\b", "avenue"),
    (r"\bblvd\b", "boulevard"),
    (r"\bdr\b", "drive"),
    (r"\bln\b", "lane"),
    (r"\bct\b", "court"),
    (r"\bapt\b", "unit"),
    (r"\bapartment\b", "unit"),
    (r"\bste\b", "unit"),
    (r"\bsuite\b", "unit"),
    (r"\bhwy\b", "highway"),
    (r"\bpl\b", "place"),
    (r"\bsq\b", "square"),
    (r"\bpkwy\b", "parkway"),
    (r"\bno\b", "number"),
]

_PUNCT_PATTERN = r"[^\w\s&]"
_WS_PATTERN = r"\s+"
_WORD_PATTERN = r"[\w&]+"

# Trailing digit run: matches US 5(+4) zip, Indian 6-digit PIN, French
# 5-digit code alike -- same pattern family as the CPU version.
_POSTAL_PATTERN = r"(\d{4,6}(?:-\d{4})?)\s*$"
_LEADING_NUM_PATTERN = r"^(\d+[\w/-]*)"


def _basic_clean(col: cudf.Series) -> cudf.Series:
    """Lowercase, strip punctuation, collapse whitespace -- vectorized."""
    col = col.fillna("").str.lower()
    col = col.str.replace(_PUNCT_PATTERN, " ", regex=True)
    col = col.str.replace(_WS_PATTERN, " ", regex=True).str.strip()
    return col


def _apply_synonym_rules(col: cudf.Series, rules: list[tuple[str, str]]) -> cudf.Series:
    for pattern, replacement in rules:
        col = col.str.replace(pattern, replacement, regex=True)
    # collapse whitespace again in case a rule introduced double spaces (e.g. "&" -> " and ")
    return col.str.replace(_WS_PATTERN, " ", regex=True).str.strip()


def _tokenize(col: cudf.Series) -> cudf.Series:
    """Whole-column tokenization into a list<string> column via extractall,
    the cuDF-native way to get "all regex matches per row" without a Python
    loop. Returns a list-dtype Series, one list of tokens per row."""
    matches = col.str.findall(_WORD_PATTERN)
    return matches


def script_type(col: cudf.Series) -> cudf.Series:
    """Rough classification: latin / devanagari / other / empty, vectorized
    via a Devanagari-block character-range regex test (cuDF `.str.match`/
    `.str.contains` run as GPU kernels)."""
    has_devanagari = col.str.contains(r"[ऀ-ॿ]", regex=True)
    has_latin = col.str.contains(r"[A-Za-zÀ-ɏḀ-ỿ]", regex=True)
    is_empty = col.str.len() == 0
    result = cudf.Series(["other"] * len(col), index=col.index)
    result[has_latin] = "latin"
    result[has_devanagari] = "devanagari"
    result[is_empty] = "empty"
    return result


def extract_postal_code(col: cudf.Series) -> cudf.Series:
    extracted = col.str.extract(_POSTAL_PATTERN)[0]
    return extracted.fillna("")


def extract_street_number(col: cudf.Series) -> cudf.Series:
    extracted = col.str.extract(_LEADING_NUM_PATTERN)[0]
    return extracted.fillna("")


def extract_city_guess(col: cudf.Series) -> cudf.Series:
    """Best-effort city token: second-to-last comma-separated segment.
    cuDF's list-dtype split + list indexing keeps this vectorized."""
    parts = col.str.split(",")
    n_parts = parts.list.len()
    # second-to-last element index per row; rows with <2 parts get -1 (invalid)
    idx = (n_parts - 2).clip(lower=0)
    city = parts.list.get(idx)
    city = city.where(n_parts >= 2, "")
    return _basic_clean(city.fillna(""))


def build_normalized_frame(df: cudf.DataFrame) -> cudf.DataFrame:
    """Given a raw source cuDF dataframe (entity_id, business_name,
    business_address, country), return a new cuDF dataframe with normalized
    derived columns used by both blocking and feature extraction. GPU
    equivalent of normalize.build_normalized_frame in the CPU pipeline.
    """
    out = df.copy()
    out["business_name"] = out["business_name"].fillna("")
    out["business_address"] = out["business_address"].fillna("")
    out["country"] = out["country"].fillna("")

    name_clean = _basic_clean(out["business_name"])
    name_clean = _apply_synonym_rules(name_clean, _LEGAL_SYNONYM_RULES)
    name_tokens_all = _tokenize(name_clean)
    out["name_norm_full"] = name_tokens_all.list.astype("str").str.join(sep=" ")
    out["name_script"] = script_type(out["business_name"])

    # core tokens = name_tokens_all with NAME_STOPWORDS removed, falling back
    # to the full token list for rows where every token was a stopword.
    # UNVERIFIED, highest-risk section of this file: cuDF list columns don't
    # have a direct "per-row filter against a set" op the way a Python
    # .map(lambda tokens: [t for t in tokens if t not in STOPWORDS]) does on
    # the CPU. This uses the explode -> filter -> group-back-into-list
    # pattern instead: flatten to one row per token (keeping the original
    # row's position via a synthetic key), drop stopword rows with .isin(),
    # then re-collect into a list per original row. If `.agg(list)` isn't
    # supported by the cuDF version on the cluster, the likely fix is
    # `.collect()` or `.agg("collect")` instead -- check
    # `cudf.__version__` and that groupby aggregation's exact name first.
    row_id = cudf.Series(range(len(out)), index=out.index)
    exploded = cudf.DataFrame({"row_id": row_id, "token": name_tokens_all}).explode("token")
    is_stopword = exploded["token"].isin(cudf.Series(list(NAME_STOPWORDS)))
    core_exploded = exploded[~is_stopword]
    core_by_row = core_exploded.groupby("row_id")["token"].agg(list)
    core_tokens = core_by_row.reindex(row_id.values).reset_index(drop=True)
    core_tokens.index = out.index
    # rows where every token was a stopword (or the name was empty) got no
    # group at all -- fall back to the full token list for those, matching
    # the CPU version's "never return empty" behavior.
    had_core_tokens = row_id.isin(core_by_row.index)
    core_tokens = core_tokens.where(had_core_tokens, name_tokens_all)

    out["name_core_tokens"] = core_tokens
    out["name_norm"] = core_tokens.list.astype("str").str.join(sep=" ")

    addr_clean = _basic_clean(out["business_address"])
    addr_clean = _apply_synonym_rules(addr_clean, _ADDR_SYNONYM_RULES)
    addr_tokens = _tokenize(addr_clean)
    out["addr_tokens"] = addr_tokens
    out["addr_norm"] = addr_tokens.list.astype("str").str.join(sep=" ")

    out["postal_code"] = extract_postal_code(out["business_address"])
    out["street_number"] = extract_street_number(out["business_address"])
    out["city_guess"] = extract_city_guess(out["business_address"])

    return out
