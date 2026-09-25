"""
Normalization Module for Business Entity Resolution.
Amazon ML Challenge 2026.

Provides safe, robust, and reproducible normalization functions for:
- Business names (diacritics, noise prefixes, legal entity types, abbreviations, domains)
- Addresses (street suffixes, unit types, state mappings, leading-zero numbers)
- Countries
"""

import re
import unicodedata
from typing import Optional, Set, Tuple


def strip_accents(text: str) -> str:
    """
    Remove European diacritics and accents (e.g. 'Énterprises' -> 'Enterprises').
    Preserves Indic, Cyrillic, and other scripts intact.
    """
    if not text:
        return ""
    # Normalize unicode to decomposed form and discard non-spacing marks for Latin
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_country(country: Optional[str]) -> str:
    """
    Normalize country code or name.
    """
    if not country or str(country).lower() in ("nan", "none", "null", ""):
        return "UNKNOWN"
    c = str(country).strip()
    c_lower = c.lower()
    if c_lower in ("us", "usa", "united states", "united states of america"):
        return "US"
    elif c_lower in ("india", "ind", "in"):
        return "India"
    elif c_lower in ("france", "fr", "fra"):
        return "France"
    return c.title()


# Common legal business form replacements (ordered from longer to shorter)
LEGAL_ENTITY_PATTERNS = [
    (r"\bprivate\s+limited\b", "pvt ltd"),
    (r"\bpvt\.?\s*ltd\.?\b", "pvt ltd"),
    (r"\bpvt\s+limited\b", "pvt ltd"),
    (r"\bprivate\s+ltd\.?\b", "pvt ltd"),
    (r"\blimited\s+liability\s+company\b", "llc"),
    (r"\bl\.l\.c\.?\b", "llc"),
    (r"\bl\.p\.?\b", "lp"),
    (r"\bpublic\s+limited\s+(?:company)?\b", "plc"),
    (r"\bp\.l\.c\.?\b", "plc"),
    (r"\blimited\b", "ltd"),
    (r"\bltd\.?\b", "ltd"),
    (r"\bincorporated\b", "inc"),
    (r"\binc\.?\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcorp\.?\b", "corp"),
    (r"\bcompany\b", "co"),
    (r"\bco\.?\b", "co"),
    (r"\bs\.a\.r\.l\.?\b", "sarl"),
    (r"\bs\.a\.s\.?\b", "sas"),
    (r"\bs\.a\.?\b", "sa"),
    (r"\bg\.m\.b\.h\.?\b", "gmbh"),
    (r"\bl\.l\.p\.?\b", "llp"),
]

LEGAL_SUFFIXES_SET: Set[str] = {
    "pvt ltd", "ltd", "inc", "llc", "corp", "co", "plc", "llp", "sarl", "sas", "sa", "gmbh", "lp", "pvt", "private"
}

# Generic business stopwords that carry low discrimination power
GENERIC_BUSINESS_TERMS: Set[str] = {
    "pvt", "ltd", "inc", "llc", "corp", "co", "plc", "llp", "sarl", "sas", "sa", "gmbh", "lp",
    "limited", "private", "corporation", "company", "incorporated",
    "enterprises", "services", "solutions", "technologies", "international", "group", "holdings",
    "consulting", "management", "industries", "associates", "ventures", "systems", "trading",
    "financial", "logistics", "retail", "products", "center", "global", "national"
}

# Domain extensions to strip from business names
DOMAIN_PATTERN = re.compile(r"\b(?:https?:\/\/|www\.)?[a-zA-Z0-9\-\.]+\.(?:com|org|net|in|co\.in|co|io|biz|info)\b", re.IGNORECASE)
DOMAIN_SUFFIX_PATTERN = re.compile(r"\.(?:com|org|net|in|co\.in|co|io|biz|info)$", re.IGNORECASE)

# Leading noise patterns like >>, --, <<, ##, //
LEADING_NOISE_PATTERN = re.compile(r"^[\s\-><#\*\/\+]+")


def normalize_name(name: Optional[str]) -> str:
    """
    Robust normalization for business names:
    - Diacritics stripping (É -> E)
    - Lowercase
    - Strip noise prefixes (>>, --, <<)
    - Domain suffix removal (.com, www.)
    - Handle 'm/s' prefix (Messrs.)
    - Legal form normalization (private limited -> pvt ltd)
    - Ampersand normalization (& -> and)
    - Punctuation removal
    - Whitespace normalization
    """
    if not name or str(name).lower() in ("nan", "none", "null", ""):
        return ""

    text = str(name).strip()
    text = strip_accents(text)
    text = LEADING_NOISE_PATTERN.sub("", text).strip()
    text = text.lower()

    # Remove leading 'the '
    text = re.sub(r"^the\s+", "", text)

    # Normalize OCR/digit substitutions inside words (e.g. k0ch -> koch, neighb0rhood -> neighborhood)
    text = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", text)
    text = re.sub(r"(?<=[a-z])1(?=[a-z])", "l", text)

    # Remove leading 'm/s' or 'm / s'
    text = re.sub(r"^\s*m\s*\/\s*s\s+", "", text)
    text = re.sub(r"^\s*m\s*\.\s*s\s*\.?\s+", "", text)

    # Normalize domain names (e.g., 'maurewilliamscolombier.com' -> 'maurewilliamscolombier')
    text = DOMAIN_SUFFIX_PATTERN.sub("", text)
    text = re.sub(r"^www\.", "", text)

    # Normalize ampersand
    text = re.sub(r"\s*&\s*", " and ", text)

    # Normalize legal forms
    for pat, rep in LEGAL_ENTITY_PATTERNS:
        text = re.sub(pat, f" {rep} ", text)

    # Remove punctuation, keep letters, digits, and unicode words
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_core_name(normalized_name: str) -> str:
    """
    Extract the core business name by removing trailing/leading legal form suffixes.
    Also handles 'aka', 'dba', 'formerly', 'nee' alias prefixes.
    Example: 'clemons silver eastern inc' -> 'clemons silver eastern'
             'xylosol sys formerly beryle kent steel inc' -> 'beryle kent steel'
    """
    if not normalized_name:
        return ""

    text = normalized_name

    # Check for alias delimiters: formerly, nee, aka, dba, fka
    alias_delims = [r"\bformerly\b", r"\bnee\b", r"\bf\/k\/a\b", r"\bfka\b", r"\baka\b", r"\ba\/k\/a\b", r"\bd\/b\/a\b", r"\bdba\b"]
    for delim in alias_delims:
        parts = re.split(delim, text)
        if len(parts) > 1:
            # Prefer the longer component or second component (often original name)
            text = max(parts, key=len).strip()
            break

    tokens = text.split()

    # Strip legal entity words from ends
    while tokens and tokens[-1] in LEGAL_SUFFIXES_SET:
        tokens.pop()
    while tokens and tokens[0] in LEGAL_SUFFIXES_SET:
        tokens.pop(0)

    core = " ".join(tokens).strip()
    return core if core else normalized_name


# Street type normalization mappings
STREET_MAPPINGS = [
    (r"\broad\b", "rd"),
    (r"\brd\.?\b", "rd"),
    (r"\bstreet\b", "st"),
    (r"\bst\.?\b", "st"),
    (r"\bsaint\b", "st"),
    (r"\bavenue\b", "ave"),
    (r"\bave\.?\b", "ave"),
    (r"\bboulevard\b", "blvd"),
    (r"\bblvd\.?\b", "blvd"),
    (r"\bdrive\b", "dr"),
    (r"\bdr\.?\b", "dr"),
    (r"\blane\b", "ln"),
    (r"\bln\.?\b", "ln"),
    (r"\bhighway\b", "hwy"),
    (r"\bhwy\.?\b", "hwy"),
    (r"\bcircle\b", "cir"),
    (r"\bcir\.?\b", "cir"),
    (r"\bcourt\b", "ct"),
    (r"\bct\.?\b", "ct"),
    (r"\bparkway\b", "pkwy"),
    (r"\bpkwy\.?\b", "pkwy"),
    (r"\bterrace\b", "ter"),
    (r"\bter\.?\b", "ter"),
    (r"\bexpressway\b", "expy"),
    (r"\bexpy\.?\b", "expy"),
    (r"\btrail\b", "trl"),
    (r"\bplace\b", "pl"),
    (r"\bsquare\b", "sq"),
]

# Unit and building indicator mappings
UNIT_MAPPINGS = [
    (r"\bapartment\b", "apt"),
    (r"\bapt\.?\b", "apt"),
    (r"\bsuite\b", "ste"),
    (r"\bste\.?\b", "ste"),
    (r"\bbuilding\b", "bldg"),
    (r"\bbldg\.?\b", "bldg"),
    (r"\bfloor\b", "fl"),
    (r"\bfl\.?\b", "fl"),
    (r"\bdoor\s+no\.?\b", "door"),
    (r"\bshop\s+no\.?\b", "shop"),
    (r"\bplot\s+no\.?\b", "plot"),
    (r"\bflat\s+no\.?\b", "flat"),
    (r"\b(?:house\s+no|h\.?\s*no)\.?\b", "hno"),
    (r"\bkh\s*(?:no\.?|-)\s*", "kh "),
    (r"\bp\.?\s*o\.?\s*box\b", "pobox"),
    (r"\bpo\s+box\b", "pobox"),
    (r"\bnull\b", " "),
]

# US and Indian State name standardizations
STATE_MAPPINGS = [
    (r"\bnorth\s+carolina\b", "nc"),
    (r"\bsouth\s+carolina\b", "sc"),
    (r"\bnew\s+york\b", "ny"),
    (r"\bcalifornia\b", "ca"),
    (r"\btexas\b", "tx"),
    (r"\bflorida\b", "fl"),
    (r"\billinois\b", "il"),
    (r"\bmissouri\b", "mo"),
    (r"\bmaryland\b", "md"),
    (r"\bohio\b", "oh"),
    (r"\bconnecticut\b", "ct"),
    (r"\boklahoma\b", "ok"),
    (r"\barizona\b", "az"),
    (r"\bpennsylvania\b", "pa"),
    (r"\bmichigan\b", "mi"),
    (r"\bgeorgia\b", "ga"),
    (r"\bvirginia\b", "va"),
    (r"\btennessee\b", "tn"),
    (r"\bcolorado\b", "co"),
    (r"\bnew\s+jersey\b", "nj"),
    (r"\buttarakhand\b", "uk"),
    (r"\buttarkhand\b", "uk"),
    (r"\buttar\s+pradesh\b", "up"),
    (r"\btamil\s+nadu\b", "tn"),
    (r"\bmadhya\s+pradesh\b", "mp"),
    (r"\bwest\s+bengal\b", "wb"),
    (r"\bkarnataka\b", "ka"),
    (r"\bmaharashtra\b", "mh"),
    (r"\bdelhi\b", "dl"),
    (r"\brajasthan\b", "rj"),
    (r"\bgujarat\b", "gj"),
    (r"\bharyana\b", "hr"),
    (r"\bandhra\s+pradesh\b", "ap"),
    (r"\btelangana\b", "ts"),
    (r"\bkerala\b", "kl"),
    (r"\bpunjab\b", "pb"),
    (r"\bbihar\b", "br"),
]


def normalize_address(address: Optional[str]) -> str:
    """
    Robust address normalization:
    - Diacritics stripping
    - Lowercase
    - Street type standardizations (road -> rd, street -> st, saint -> st)
    - Unit type standardizations (apartment -> apt, suite -> ste, shop no -> shop)
    - State standardizations (north carolina -> nc, uttar pradesh -> up)
    - Number cleanups (stripping leading zeros, e.g. 0189 -> 189, #44 -> 44)
    - Safe punctuation handling
    """
    if not address or str(address).lower() in ("nan", "none", "null", ""):
        return ""

    text = str(address).strip()
    text = strip_accents(text)
    text = text.lower()

    # Remove noise like '#' or 'no.' before digits
    text = re.sub(r"#\s*", "", text)
    text = re.sub(r"\bno\.?\s*(?=\d)", "", text)

    # Street types
    for pat, rep in STREET_MAPPINGS:
        text = re.sub(pat, f" {rep} ", text)

    # Units
    for pat, rep in UNIT_MAPPINGS:
        text = re.sub(pat, f" {rep} ", text)

    # States
    for pat, rep in STATE_MAPPINGS:
        text = re.sub(pat, f" {rep} ", text)

    # Normalize isolated fractions like 1/2
    text = re.sub(r"\b1\/2\b", "", text)

    # Normalize leading zeros in isolated numbers (e.g. 0189 -> 189)
    text = re.sub(r"\b0+(\d+)\b", r"\1", text)

    # Punctuation to space, except keeping alphanumeric and hyphens/slashes in building numbers like AF-684 or 59/101
    text = re.sub(r"[^\w\s\-\/]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_address_key(normalized_addr: str) -> str:
    """
    Extract a discriminative address blocking key:
    First alphanumeric building/street number + first significant street/locality token.
    Example:
    '1619 julia park dr spring tx' -> '1619_julia'
    '3315 fremont st peoria il' -> '3315_fremont'
    'af-684 nandgram near mother india...' -> 'af-684_nandgram'
    'j-215 block j saket new dl' -> 'j-215_saket'
    """
    if not normalized_addr:
        return ""

    ignored_tokens = {
        "apt", "ste", "unit", "shop", "door", "hno", "block", "floor", "fl", "ground",
        "near", "opp", "opposite", "behind", "road", "rd", "street", "st", "lane", "ln",
        "avenue", "ave", "drive", "dr", "highway", "hwy", "court", "ct", "circle", "cir",
        "terrace", "ter", "expressway", "expy", "parkway", "pkwy", "trail", "trl", "pl", "sq"
    }

    tokens = [t for t in normalized_addr.split() if t not in ignored_tokens]
    if not tokens:
        return ""

    # Find the first token with digits (house/shop/street number)
    num_idx = -1
    for idx, t in enumerate(tokens):
        if any(c.isdigit() for c in t):
            num_idx = idx
            break

    if num_idx != -1:
        num_token = tokens[num_idx]
        # Find next significant alpha token of length >= 3
        street_token = ""
        for t in tokens[num_idx + 1:]:
            clean = t.replace("-", "").replace("/", "")
            if len(clean) >= 3 and clean not in ignored_tokens:
                street_token = t
                break
        if street_token:
            return f"{num_token}_{street_token}"
        return num_token

    return ""
