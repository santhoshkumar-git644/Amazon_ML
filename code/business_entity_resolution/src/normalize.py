"""Text normalization utilities for business names and addresses.

Kept dependency-free (stdlib + numpy/pandas only) and rule-based: no external
lookups, geocoders or gazetteers are used anywhere in this module, per the
challenge's fair-play rules.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

# Legal-entity suffix/connector tokens that vary across sources for the *same*
# business (Corp vs Corporation, Pvt vs Private, Ltd vs Limited, ...), plus
# generic business-descriptor words that are individually so frequent across
# *unrelated* businesses (per an empirical token-frequency pass over the
# training data — e.g. "solutions", "global", "center", "services" each occur
# in 1-3% of all Source-1 names) that they carry almost no blocking signal and
# would otherwise dominate a record's token set. Dropping them both here and
# from token-overlap features leaves the tokens that actually identify a
# specific business (proper nouns, place names, ...) as the dominant signal.
LEGAL_STOPWORDS = frozenset({
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "ltd", "limited", "llc", "llp", "lp", "plc", "pvt", "private", "pte",
    "dba", "and", "the", "of", "a", "an", "group", "holdings", "pc", "pllc",
    "trust", "foundation", "associates", "partners", "center", "centre",
    "clinic", "care", "health", "healthcare", "medicine", "medical",
    "solutions", "global", "trading", "international", "brothers",
    "technologies", "technology", "tech", "industries", "enterprises",
    "consultants", "consulting", "ventures", "systems", "products",
    "marketing", "engineering", "management", "services", "service",
    "india", "usa", "us", "america", "american", "national", "world",
    "worldwide", "general", "professional", "advisors", "advisory",
})

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")
# US ZIP (5 or 5+4) / Indian PIN (6 digits) — treated purely as a string of
# digits pulled out of the free-text address, not looked up against any postal
# database.
_POSTAL_RE = re.compile(r"\b(\d{6}|\d{5}(?:-\d{4})?)\b")


def _ascii_fold(s: str) -> str:
    """Best-effort fold of accented Latin characters to ASCII.

    Does not transliterate non-Latin scripts (e.g. Devanagari) — that would
    require an external transliteration table, which we deliberately avoid.
    """
    norm = unicodedata.normalize("NFKD", s)
    return "".join(c for c in norm if not unicodedata.combining(c))


def basic_clean(s: str) -> str:
    """Lowercase, ASCII-fold, replace '&' with 'and', strip punctuation."""
    if not isinstance(s, str) or not s:
        return ""
    s = s.lower()
    s = _ascii_fold(s)
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(s: str) -> list:
    if not s:
        return []
    return s.split(" ")


# Address boilerplate: street-type abbreviations, unit/floor/landmark
# connectors and directionals. Left in place: house/door numbers, PIN/ZIP
# digits and actual street/locality/landmark names — the tokens that make two
# addresses for the same physical business actually match.
ADDRESS_STOPWORDS = frozenset({
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr", "lane", "ln",
    "boulevard", "blvd", "highway", "hwy", "circle", "cir", "court", "ct",
    "place", "pl", "way", "terrace", "parkway", "pkwy", "square", "sq",
    "suite", "ste", "floor", "fl", "unit", "apartment", "apt", "room", "no",
    "door", "block", "building", "bldg", "tower", "complex", "plaza", "near",
    "opp", "opposite", "behind", "above", "below", "next", "to", "at", "in",
    "of", "the", "and", "a", "an", "po", "box", "care", "c/o",
    "north", "south", "east", "west", "new", "old",
})


def significant_tokens(s: str) -> list:
    """Tokens with legal-suffix / generic-business stopwords removed, for
    name blocking and name-similarity features. Single-character tokens
    (leftover from stripped possessives like "Orelee's" -> "orelee", "s", or
    US middle-initial patterns like "John A Smith PC") are dropped too: they
    are not discriminative and inflate a token's document frequency."""
    return [t for t in tokenize(s) if len(t) > 1 and t not in LEGAL_STOPWORDS]


def significant_address_tokens(s: str) -> list:
    """Address tokens with boilerplate removed, for address blocking/features."""
    return [t for t in tokenize(s) if len(t) > 1 and t not in ADDRESS_STOPWORDS]


def extract_postal_code(address: str) -> str:
    """Pull a 5-6 digit postal code out of a raw address string, if present."""
    if not isinstance(address, str) or not address:
        return ""
    m = _POSTAL_RE.search(address)
    return m.group(1).replace("-", "") if m else ""


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized-name / normalized-address / postal columns.

    Expects columns: entity_id, business_name, business_address, country.
    Deliberately keeps only *string* columns (no per-row Python list objects)
    so multi-million-row frames stay compact; tokenization happens on demand,
    transiently, inside blocking/features.
    """
    out = df[["entity_id", "country"]].copy()
    out["name_norm"] = df["business_name"].map(basic_clean)
    out["addr_norm"] = df["business_address"].map(basic_clean)
    out["postal"] = df["business_address"].map(extract_postal_code)
    out["name_prefix4"] = out["name_norm"].str.replace(" ", "", regex=False).str.slice(0, 4)
    return out
