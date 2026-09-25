"""Text normalization shared by blocking and feature extraction.

Everything here is pure string processing on the provided fields
(business_name, business_address, country) — no external lookups.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# --- name normalization -----------------------------------------------------

# word -> normalized form. Applied with word-boundary regex, both directions
# collapse to the same canonical token so "Corp" and "Corporation" compare equal.
_LEGAL_SYNONYMS = {
    "incorporated": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "company": "co",
    "co": "co",
    "limited": "ltd",
    "ltd": "ltd",
    "llc": "llc",
    "llp": "llp",
    "lp": "lp",
    "pvt": "pvt",
    "private": "pvt",
    "plc": "plc",
    "gmbh": "gmbh",
    "sarl": "sarl",
    "sas": "sas",
    "and": "and",
    "&": "and",
}

# Tokens that carry no discriminative signal for name matching once the
# legal-suffix flag has been extracted separately.
NAME_STOPWORDS = {
    "inc", "corp", "co", "ltd", "llc", "llp", "lp", "pvt", "private",
    "limited", "corporation", "incorporated", "company", "plc", "gmbh",
    "sarl", "sas", "and", "the", "of",
}

_PUNCT_RE = re.compile(r"[^\w\s&]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[\w&]+", re.UNICODE)


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def basic_clean(text: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    if text is None or (isinstance(text, float)):
        return ""
    text = str(text).lower()
    text = strip_accents(text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name_tokens(text: str) -> list[str]:
    cleaned = basic_clean(text)
    tokens = _WORD_RE.findall(cleaned)
    return [_LEGAL_SYNONYMS.get(t, t) for t in tokens]


def has_legal_suffix(tokens: list[str]) -> bool:
    return any(t in NAME_STOPWORDS for t in tokens)


def name_core_tokens(tokens: list[str]) -> list[str]:
    """Tokens with legal/stopwords removed — the discriminative "core" name."""
    core = [t for t in tokens if t not in NAME_STOPWORDS]
    return core if core else tokens  # never return empty if the name is *only* legal words


def script_type(text: str) -> str:
    """Rough Unicode-block classification: latin / devanagari / other / empty."""
    if not text:
        return "empty"
    for ch in text:
        if ch.isalpha():
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F:
                return "devanagari"
            if (0x0041 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
                return "latin"
            return "other"
    return "empty"


# --- address normalization ---------------------------------------------------

_ADDR_SYNONYMS = {
    "rd": "road", "road": "road",
    "st": "street", "str": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "avenue": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard",
    "dr": "drive", "drive": "drive",
    "ln": "lane", "lane": "lane",
    "ct": "court", "court": "court",
    "apt": "unit", "apartment": "unit", "unit": "unit", "ste": "unit", "suite": "unit",
    "hwy": "highway", "highway": "highway",
    "pl": "place", "place": "place",
    "sq": "square", "square": "square",
    "pkwy": "parkway", "parkway": "parkway",
    "no": "number", "number": "number",
}

# generic trailing digit run: matches US 5(+4) zip, Indian 6-digit PIN, French 5-digit code
_POSTAL_RE = re.compile(r"(\d{4,6}(?:-\d{4})?)\s*$")
_LEADING_NUM_RE = re.compile(r"^(\d+[\w/-]*)")


def normalize_addr_tokens(text: str) -> list[str]:
    cleaned = basic_clean(text)
    tokens = _WORD_RE.findall(cleaned)
    return [_ADDR_SYNONYMS.get(t, t) for t in tokens]


def extract_postal_code(raw_address: str) -> str:
    if not raw_address:
        return ""
    m = _POSTAL_RE.search(str(raw_address).strip())
    return m.group(1) if m else ""


def extract_street_number(raw_address: str) -> str:
    if not raw_address:
        return ""
    cleaned = str(raw_address).strip()
    m = _LEADING_NUM_RE.match(cleaned)
    return m.group(1) if m else ""


def extract_city_guess(raw_address: str) -> str:
    """Best-effort city token: second-to-last comma-separated segment."""
    if not raw_address:
        return ""
    parts = [p.strip() for p in str(raw_address).split(",") if p.strip()]
    if len(parts) >= 2:
        return basic_clean(parts[-2])
    return ""


def build_normalized_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Given a raw source dataframe (entity_id, business_name, business_address,
    country), return a new dataframe with normalized derived columns used by
    both blocking and feature extraction.
    """
    out = df.copy()
    out["business_name"] = out["business_name"].fillna("")
    out["business_address"] = out["business_address"].fillna("")
    out["country"] = out["country"].fillna("")

    # name_tokens (all tokens, pre-stopword-filter) is only needed transiently to
    # derive name_norm_full and core_tokens below -- not stored as an output
    # column since nothing downstream reads it, and it's a full list-of-strings
    # per row (expensive at multi-million-row scale). Same for has_legal_suffix:
    # computed for potential future use but never actually consumed by
    # blocking/features, so not worth the column's memory either.
    name_tokens = out["business_name"].map(normalize_name_tokens)
    core_tokens = name_tokens.map(name_core_tokens)
    out["name_core_tokens"] = core_tokens
    out["name_norm"] = core_tokens.map(lambda ts: " ".join(ts))
    out["name_norm_full"] = name_tokens.map(lambda ts: " ".join(ts))
    out["name_script"] = out["business_name"].map(script_type)

    addr_tokens = out["business_address"].map(normalize_addr_tokens)
    out["addr_tokens"] = addr_tokens
    out["addr_norm"] = addr_tokens.map(lambda ts: " ".join(ts))
    out["postal_code"] = out["business_address"].map(extract_postal_code)
    out["street_number"] = out["business_address"].map(extract_street_number)
    out["city_guess"] = out["business_address"].map(extract_city_guess)

    return out
