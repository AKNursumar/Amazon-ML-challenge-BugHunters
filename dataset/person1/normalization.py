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
    (r"\bs\.a\.s\.u\.?\b", "sasu"),
    (r"\bs\.a\.s\.?\b", "sas"),
    (r"\bs\.a\.?\b", "sa"),
    (r"\be\.u\.r\.l\.?\b", "eurl"),
    (r"\bs\.c\.i\.?\b", "sci"),
    (r"\bs\.n\.c\.?\b", "snc"),
    (r"\bs\.c\.a\.?\b", "sca"),
    (r"\bg\.m\.b\.h\.?\b", "gmbh"),
    (r"\bl\.l\.p\.?\b", "llp"),
]

LEGAL_SUFFIXES_SET: Set[str] = {
    "pvt ltd", "ltd", "inc", "llc", "corp", "co", "plc", "llp", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "sca", "gmbh", "lp", "pvt", "private"
}

# Generic business stopwords that carry low discrimination power
GENERIC_BUSINESS_TERMS: Set[str] = {
    "pvt", "ltd", "inc", "llc", "corp", "co", "plc", "llp", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "sca", "gmbh", "lp",
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


def _extract_root_keyword(pattern_str: str) -> str:
    s = pattern_str.replace(r"\b", "").replace(r"\\b", "").replace(r"\.?", "").replace(r"\.", "").replace(r"\s+", " ").replace(r"\s*", " ")
    s = re.sub(r"[\(\)\?\:\-\#]", "", s).strip()
    return s.split()[0].lower() if s else ""


COMPILED_LEGAL_ENTITY_PATTERNS = [
    (_extract_root_keyword(pat), re.compile(pat, re.IGNORECASE), f" {rep} ") for pat, rep in LEGAL_ENTITY_PATTERNS
]

RE_THE = re.compile(r"^the\s+", re.IGNORECASE)
RE_OCR_0 = re.compile(r"(?<=[a-z])0(?=[a-z])", re.IGNORECASE)
RE_OCR_1 = re.compile(r"(?<=[a-z])1(?=[a-z])", re.IGNORECASE)
RE_MS1 = re.compile(r"^\s*m\s*\/\s*s\s+", re.IGNORECASE)
RE_MS2 = re.compile(r"^\s*m\s*\.\s*s\s*\.?\s+", re.IGNORECASE)
RE_WWW = re.compile(r"^www\.", re.IGNORECASE)
RE_AMP = re.compile(r"\s*&\s*")
RE_PUNCT_NAME = re.compile(r"[^\w\s]")
RE_SPACE = re.compile(r"\s+")

RE_ADDR_HASH = re.compile(r"#\s*")
RE_ADDR_NO_DIGIT = re.compile(r"\bno\.?\s*(?=\d)", re.IGNORECASE)
RE_ADDR_FRAC = re.compile(r"\b1\/2\b")
RE_ADDR_LEAD_ZERO = re.compile(r"\b0+(\d+)\b")
RE_ADDR_PUNCT = re.compile(r"[^\w\s\-\/]")


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
    text = LEADING_NOISE_PATTERN.sub("", text).strip().lower()

    # Remove leading 'the '
    text = RE_THE.sub("", text)

    # Normalize OCR/digit substitutions inside words (e.g. k0ch -> koch, neighb0rhood -> neighborhood)
    text = RE_OCR_0.sub("o", text)
    text = RE_OCR_1.sub("l", text)

    # Remove leading 'm/s' or 'm / s'
    text = RE_MS1.sub("", text)
    text = RE_MS2.sub("", text)

    # Normalize domain names (e.g., 'maurewilliamscolombier.com' -> 'maurewilliamscolombier')
    text = DOMAIN_SUFFIX_PATTERN.sub("", text)
    text = RE_WWW.sub("", text)

    # Normalize ampersand
    text = RE_AMP.sub(" and ", text)

    # Normalize legal forms using keyword-guarded precompiled regexes
    for kw, pat, rep in COMPILED_LEGAL_ENTITY_PATTERNS:
        if not kw or kw in text:
            text = pat.sub(rep, text)

    # Remove punctuation, keep letters, digits, and unicode words
    text = RE_PUNCT_NAME.sub(" ", text)
    # Collapse multiple whitespaces
    text = RE_SPACE.sub(" ", text).strip()

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
    # French street types
    (r"\brue\b", "st"),
    (r"\bchemin\b", "ch"),
    (r"\bimpasse\b", "imp"),
    (r"\ball[eé]e\b", "all"),
    (r"\broute\b", "rte"),
    (r"\bquai\b", "quai"),
    (r"\bcours\b", "crs"),
    (r"\bpassage\b", "pass"),
    (r"\bfaubourg\b", "fbg"),
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
    # French unit and building types
    (r"\bb[aâ]timent\b", "bldg"),
    (r"\bb[aâ]t\.?\b", "bldg"),
    (r"\b[eé]tage\b", "fl"),
    (r"\br[eé]sidence\b", "res"),
    (r"\bporte\b", "door"),
    (r"\bb\.?\s*p\.?\b", "pobox"),
    (r"\bcedex\b", "cedex"),
    (r"\bz\.?\s*i\.?\b", "zi"),
    (r"\bz\.?\s*a\.?\b", "za"),
    (r"\bz\.?\s*a\.?\s*c\.?\b", "zac"),
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


COMPILED_ADDR_PATTERNS = [
    (_extract_root_keyword(pat), re.compile(pat, re.IGNORECASE), f" {rep} ")
    for pat, rep in STREET_MAPPINGS + UNIT_MAPPINGS + STATE_MAPPINGS
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
    text = strip_accents(text).lower()

    # Remove noise like '#' or 'no.' before digits
    text = RE_ADDR_HASH.sub("", text)
    text = RE_ADDR_NO_DIGIT.sub("", text)

    # Street types, Units, and States using keyword-guarded precompiled regexes
    for kw, pat, rep in COMPILED_ADDR_PATTERNS:
        if not kw or kw in text:
            text = pat.sub(rep, text)

    # Normalize isolated fractions like 1/2
    text = RE_ADDR_FRAC.sub("", text)

    # Normalize leading zeros in isolated numbers (e.g. 0189 -> 189)
    text = RE_ADDR_LEAD_ZERO.sub(r"\1", text)

    # Punctuation to space, except keeping alphanumeric and hyphens/slashes in building numbers like AF-684 or 59/101
    text = RE_ADDR_PUNCT.sub(" ", text)
    # Collapse multiple whitespaces
    text = RE_SPACE.sub(" ", text).strip()

    return text


RE_INDIAN_LANDMARKS = re.compile(
    r"\b(?:opp(?:osite)?(?:\s+to)?|near|nr\.?|behind|beside|in\s+front\s+of|adjacent\s+to|next\s+to)\b|"
    r"\b(?:railway\s+station|bus\s+stand|bus\s+stop|metro\s+station|post\s+office|police\s+station|petrol\s+pump)\b|"
    r"\b(?:gidc(?:\s+phase\s+\d+)?|midc|industrial\s+area|industrial\s+estate|phase\s+\d+|sector\s+\d+)\b|"
    r"\b(?:main\s+market|subji\s+mandi|anaj\s+mandi|bazaar|chowk|chauraha|circle|cross\s+road)\b|"
    r"\b(?:commercial\s+complex|shopping\s+complex|plaza|tower|towers|arcade|chambers|enclave)\b",
    re.IGNORECASE,
)


def strip_landmarks(address: Optional[str]) -> str:
    """Strip common Indian and generic transit/commercial landmark phrases."""
    if not address or str(address).lower() in ("nan", "none", "null", ""):
        return ""
    text = RE_INDIAN_LANDMARKS.sub(" ", str(address))
    return RE_SPACE.sub(" ", text).strip()


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
        "terrace", "ter", "expressway", "expy", "parkway", "pkwy", "trail", "trl", "pl", "sq",
        # Indian landmark & commercial stop words
        "railway", "station", "stand", "stop", "metro", "chowk", "circle", "bazaar", "market",
        "complex", "plaza", "tower", "towers", "arcade", "chamber", "chambers", "enclave",
        "gidc", "midc", "industrial", "estate", "phase", "sector", "colony", "nagar",
        # French street and address stop words
        "rue", "ch", "chemin", "impasse", "imp", "all", "allee", "route", "rte", "quai",
        "cours", "crs", "pass", "passage", "fbg", "faubourg", "cedex", "res", "bldg",
        "de", "la", "du", "des", "le", "les", "en", "sur"
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

