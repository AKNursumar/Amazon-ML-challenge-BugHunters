"""
Enhanced Normalization Layer for Business Entity Resolution.
Amazon ML Challenge 2026 — Team BugHunters.

Applies conservative, discriminative normalization for names, addresses, and countries:
- Unicode NFKD accent stripping.
- Legal form stripping for core name comparison.
- Full preservation of discriminative numbers, unit codes, and postal/PIN codes.
- Robust support for arbitrary country labels (US, India, France, etc.).
"""

import re
import unicodedata
from typing import Optional, Set, Tuple

# Comprehensive legal business suffixes across English, Indian, and French corporate forms
LEGAL_FORM_SUFFIXES = {
    "ltd", "limited", "corp", "corporation", "inc", "incorporated",
    "pvt", "private", "llc", "llp", "gmbh", "sa", "sarl", "sas", "co",
    "company", "enterprises", "enterprise", "holdings", "holding",
    "solutions", "services", "technologies", "tech", "group", "ventures",
    "industries", "associates", "consulting", "international", "intl",
    "trading", "traders", "agency", "agencies", "foundation", "society",
    "trust", "club", "association", "institute", "center", "centre"
}

GENERIC_BUSINESS_TERMS = {
    "and", "the", "for", "all", "of", "in", "at", "by", "on", "with",
    "services", "service", "solutions", "solution", "technologies", "technology",
    "enterprises", "enterprise", "group", "holdings", "holding", "industries",
    "industry", "associates", "consulting", "consultants", "consultant",
    "international", "intl", "national", "global", "ventures", "venture",
    "trading", "traders", "trader", "commercial", "products", "product",
    "systems", "system", "management", "consultancy", "logistics"
}

ADDRESS_ABBREVIATIONS = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "ct": "court", "cir": "circle", "bldg": "building", "fl": "floor",
    "ground": "ground", "apt": "apartment", "ste": "suite", "unit": "unit",
    "dept": "department", "no": "number", "nr": "near", "opp": "opposite",
    "behind": "behind", "dist": "district", "sec": "sector", "ph": "phase",
    "bl": "block", "blk": "block", "nagar": "nagar", "marg": "marg",
    "colony": "colony", "puram": "puram", "road": "road", "street": "street"
}

COUNTRY_CANONICAL_MAP = {
    "us": "US", "usa": "US", "united states": "US", "united states of america": "US",
    "in": "India", "ind": "India", "india": "India", "bharat": "India",
    "fr": "France", "fra": "France", "france": "France", "republique francaise": "France",
}


def strip_accents(text: str) -> str:
    """Strip diacritics and accents using Unicode NFKD decomposition."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_country(country: Optional[str]) -> str:
    """Normalize country string to canonical representation (supports arbitrary country names)."""
    if not country or pd_isna(country):
        return "UNKNOWN"
    c_clean = str(country).strip().lower()
    return COUNTRY_CANONICAL_MAP.get(c_clean, str(country).strip().upper())


def pd_isna(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and val != val:
        return True
    s = str(val).strip().lower()
    return s in {"", "nan", "none", "null", "<na>"}


def normalize_name(name: Optional[str]) -> str:
    """
    Standardize business name:
    - Lowercase & Unicode NFKD accent stripping.
    - Replace '&' with 'and'.
    - Remove punctuation while keeping alphanumeric tokens.
    """
    if not name or pd_isna(name):
        return ""
    text = strip_accents(str(name)).lower()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)


def clean_core_name(norm_name: str) -> str:
    """
    Strip trailing/leading legal form suffixes to extract the core discriminative business name.
    """
    if not norm_name:
        return ""
    tokens = norm_name.split()
    if not tokens:
        return ""

    # Strip trailing legal forms
    while len(tokens) > 1 and tokens[-1] in LEGAL_FORM_SUFFIXES:
        tokens.pop()

    # Strip leading common prefixes (e.g. 'the')
    if len(tokens) > 1 and tokens[0] in {"the", "a", "an"}:
        tokens.pop(0)

    return " ".join(tokens)


def extract_postal_code(addr: str, country: str = "") -> str:
    """
    Extract postal / PIN code from normalized address string:
    - India: 6-digit code (e.g., '560001', '110001')
    - US: 5-digit ZIP code (e.g., '98101', '90210')
    - France: 5-digit postal code (e.g., '75008', '69002')
    """
    if not addr:
        return ""
    tokens = addr.split()
    country_upper = country.upper() if country else ""

    if country_upper in {"INDIA", "IN"}:
        for tok in reversed(tokens):
            clean = tok.strip("#,.-/ ")
            if clean.isdigit() and len(clean) == 6 and clean[0] in "123456789":
                return clean
    elif country_upper in {"US", "USA", "FRANCE", "FR"}:
        for tok in reversed(tokens):
            clean = tok.strip("#,.-/ ")
            if clean.isdigit() and len(clean) == 5:
                return clean

    # Generic fallback: look for 5 or 6 digit numbers
    for tok in reversed(tokens):
        clean = tok.strip("#,.-/ ")
        if clean.isdigit() and len(clean) in {5, 6}:
            return clean
    return ""


def extract_numeric_tokens(addr: str) -> Set[str]:
    """Extract all standalone or hyphenated/slashed numeric tokens from address."""
    if not addr:
        return set()
    tokens = set()
    for tok in addr.split():
        clean = tok.strip("#,.-/ ")
        if clean.isdigit():
            tokens.add(clean)
        elif ("-" in tok or "/" in tok) and any(c.isdigit() for c in tok):
            tokens.add(clean.replace("-", "").replace("/", ""))
    return tokens


def extract_building_no(addr: str) -> str:
    """Extract leading building, plot, shop or house digit sequence."""
    if not addr:
        return ""
    for tok in addr.split():
        clean = tok.strip("#,.-/ ")
        if clean.isdigit() and 1 <= len(clean) <= 6:
            return clean
    return ""


def normalize_address(address: Optional[str]) -> str:
    """
    Standardize address string:
    - Lowercase & Unicode NFKD accent stripping.
    - Standardize street/road/unit abbreviations.
    - Preserve all building numbers, plot numbers, postal codes.
    """
    if not address or pd_isna(address):
        return ""
    text = strip_accents(str(address)).lower()
    text = re.sub(r"[#,\.]", " ", text)
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    norm_tokens = []
    for tok in tokens:
        clean_tok = tok.strip(" ,.-/")
        if not clean_tok:
            continue
        norm_tok = ADDRESS_ABBREVIATIONS.get(clean_tok, clean_tok)
        norm_tokens.append(norm_tok)

    return " ".join(norm_tokens)
