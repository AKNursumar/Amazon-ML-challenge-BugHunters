"""
Feature Engineering Module for Entity Resolution (Person 2).
Amazon ML Challenge 2026.

Extracts interpretable similarity features across business names, addresses, and structured fields.
Reuses Person 1's normalization logic without duplicating or altering it.
"""

import os
import sys
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

# Ensure both dataset/ and repo root are in sys.path
cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.dirname(cur_dir)
repo_root = os.path.dirname(dataset_dir)
for p in [cur_dir, dataset_dir, repo_root]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from person1.normalization import (
        normalize_name,
        clean_core_name,
        normalize_address,
        normalize_country,
        extract_address_key,
        GENERIC_BUSINESS_TERMS,
    )
except ImportError:
    from dataset.person1.normalization import (
        normalize_name,
        clean_core_name,
        normalize_address,
        normalize_country,
        extract_address_key,
        GENERIC_BUSINESS_TERMS,
    )

# Common street type tokens to ignore for locality matching
IGNORED_ADDR_TOKENS = {
    "rd", "st", "ave", "dr", "ln", "hwy", "blvd", "ct", "cir", "fl", "apt", "ste",
    "unit", "shop", "near", "opp", "opposite", "behind", "hotel", "road", "street",
    "highway", "lane", "floor", "building", "bldg", "cross", "main", "nagar", "block"
}

# Feature groups definitions
NAME_FEATURES = [
    "name_exact",
    "name_normalized_exact",
    "name_levenshtein",
    "name_jaccard",
    "name_token_overlap",
    "name_token_set_ratio",
    "name_char_3gram_jaccard",
    "name_length_difference",
    "name_token_count_difference",
    "common_name_tokens",
    "name_contains_other",
]

ADDRESS_FEATURES = [
    "address_exact",
    "address_normalized_exact",
    "address_levenshtein",
    "address_jaccard",
    "address_token_overlap",
    "address_token_set_ratio",
    "address_char_3gram_jaccard",
    "address_length_difference",
    "address_token_count_difference",
    "common_address_tokens",
    "address_contains_other",
    "building_number_match",
    "locality_match",
]

STRUCTURED_FEATURES = [
    "country_match",
    "candidate_source_is_s2",
]

ALL_FEATURES = NAME_FEATURES + ADDRESS_FEATURES + STRUCTURED_FEATURES


def extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from cleaned string."""
    clean = re.sub(r"\s+", "", text.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def extract_building_no(addr: str) -> str:
    """Extract leading building, plot, or house digit sequence."""
    if not addr:
        return ""
    for tok in addr.split():
        clean = tok.strip("#,.-/").replace("-", "").replace("/", "")
        if clean.isdigit() and len(clean) <= 6:
            return clean
    return ""


def extract_locality_tokens(norm_addr: str) -> Set[str]:
    """Extract significant alphabetic locality words from normalized address."""
    if not norm_addr:
        return set()
    tokens = norm_addr.split()
    return {w for w in tokens if len(w) >= 4 and w.isalpha() and w not in IGNORED_ADDR_TOKENS}


def preprocess_entity_record(
    raw_name: str, raw_addr: str, raw_country: str
) -> Dict[str, object]:
    """Preprocess and pre-tokenize a single record for fast feature lookups."""
    raw_name = str(raw_name) if pd.notna(raw_name) else ""
    raw_addr = str(raw_addr) if pd.notna(raw_addr) else ""
    raw_country = str(raw_country) if pd.notna(raw_country) else ""

    norm_c = normalize_country(raw_country)
    norm_n = normalize_name(raw_name)
    core_n = clean_core_name(norm_n)
    norm_a = normalize_address(raw_addr)

    name_tokens = norm_n.split()
    name_token_set = set(name_tokens)
    core_token_set = set(core_n.split())

    addr_tokens = norm_a.split()
    addr_token_set = set(addr_tokens)

    bldg_no = extract_building_no(norm_a)
    locality_set = extract_locality_tokens(norm_a)

    return {
        "raw_name": raw_name,
        "norm_name": norm_n,
        "core_name": core_n,
        "name_tokens": name_tokens,
        "name_token_set": name_token_set,
        "core_token_set": core_token_set,
        "name_char_3grams": extract_char_ngrams(norm_n, 3),
        "raw_addr": raw_addr,
        "norm_addr": norm_a,
        "addr_tokens": addr_tokens,
        "addr_token_set": addr_token_set,
        "addr_char_3grams": extract_char_ngrams(norm_a, 3),
        "bldg_no": bldg_no,
        "locality_set": locality_set,
        "country": norm_c,
    }


def load_entity_lookups(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    needed_s1_ids: Optional[Set[str]] = None,
    needed_cand_ids: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Build preprocessed lookup mapping for all unique entity IDs."""
    lookup: Dict[str, Dict[str, object]] = {}

    # 1. Source 1
    s1_iter = s1_df
    if needed_s1_ids is not None:
        s1_iter = s1_df[s1_df["entity_id"].isin(needed_s1_ids)]
    for eid, bname, baddr, ctry in zip(
        s1_iter["entity_id"], s1_iter["business_name"], s1_iter["business_address"], s1_iter["country"]
    ):
        lookup[eid] = preprocess_entity_record(bname, baddr, ctry)

    # 2. Source 2
    s2_iter = s2_df
    if needed_cand_ids is not None:
        s2_iter = s2_df[s2_df["entity_id"].isin(needed_cand_ids)]
    for eid, bname, baddr, ctry in zip(
        s2_iter["entity_id"], s2_iter["business_name"], s2_iter["business_address"], s2_iter["country"]
    ):
        lookup[eid] = preprocess_entity_record(bname, baddr, ctry)

    # 3. Source 3
    s3_iter = s3_df
    if needed_cand_ids is not None:
        s3_iter = s3_df[s3_df["entity_id"].isin(needed_cand_ids)]
    for eid, bname, baddr, ctry in zip(
        s3_iter["entity_id"], s3_iter["business_name"], s3_iter["business_address"], s3_iter["country"]
    ):
        lookup[eid] = preprocess_entity_record(bname, baddr, ctry)

    return lookup


def compute_pair_features(
    s1: Dict[str, object], cand: Dict[str, object], cand_source: str
) -> Dict[str, float]:
    """Compute numerical features for a single (Source 1, Candidate) pair."""
    # Fallbacks for missing/empty
    if not s1 or not cand:
        return {feat: 0.0 for feat in ALL_FEATURES}

    # --- NAME FEATURES ---
    raw_n1, raw_n2 = s1["raw_name"], cand["raw_name"]
    norm_n1, norm_n2 = s1["norm_name"], cand["norm_name"]
    core_n1, core_n2 = s1["core_name"], cand["core_name"]

    name_exact = 1.0 if raw_n1 == raw_n2 and raw_n1 else 0.0
    name_norm_exact = 1.0 if norm_n1 == norm_n2 and norm_n1 else 0.0

    # Levenshtein similarity on normalized name
    name_lev = Levenshtein.normalized_similarity(norm_n1, norm_n2) if (norm_n1 and norm_n2) else 0.0

    # Token overlap & Jaccard
    t_set1, t_set2 = s1["core_token_set"], cand["core_token_set"]
    name_common = len(t_set1.intersection(t_set2))
    name_union = len(t_set1.union(t_set2))
    name_jaccard = (name_common / name_union) if name_union > 0 else 0.0
    min_tokens = min(len(t_set1), len(t_set2))
    name_overlap = (name_common / min_tokens) if min_tokens > 0 else 0.0

    # Token set ratio (handles word reorderings and substrings)
    name_set_ratio = fuzz.token_set_ratio(norm_n1, norm_n2) / 100.0 if (norm_n1 and norm_n2) else 0.0

    # Character 3-gram Jaccard
    g1, g2 = s1["name_char_3grams"], cand["name_char_3grams"]
    name_gram_jaccard = (len(g1.intersection(g2)) / len(g1.union(g2))) if (g1 and g2) else 0.0

    name_len_diff = abs(len(norm_n1) - len(norm_n2))
    name_tok_diff = abs(len(s1["name_tokens"]) - len(cand["name_tokens"]))
    name_contains = 1.0 if (norm_n1 in norm_n2 or norm_n2 in norm_n1) and (norm_n1 and norm_n2) else 0.0

    # --- ADDRESS FEATURES ---
    raw_a1, raw_a2 = s1["raw_addr"], cand["raw_addr"]
    norm_a1, norm_a2 = s1["norm_addr"], cand["norm_addr"]

    addr_exact = 1.0 if raw_a1 == raw_a2 and raw_a1 else 0.0
    addr_norm_exact = 1.0 if norm_a1 == norm_a2 and norm_a1 else 0.0

    # Levenshtein similarity on normalized address
    addr_lev = Levenshtein.normalized_similarity(norm_a1, norm_a2) if (norm_a1 and norm_a2) else 0.0

    # Address token overlap & Jaccard
    at_set1, at_set2 = s1["addr_token_set"], cand["addr_token_set"]
    addr_common = len(at_set1.intersection(at_set2))
    addr_union = len(at_set1.union(at_set2))
    addr_jaccard = (addr_common / addr_union) if addr_union > 0 else 0.0
    min_atoks = min(len(at_set1), len(at_set2))
    addr_overlap = (addr_common / min_atoks) if min_atoks > 0 else 0.0

    # Address token set ratio
    addr_set_ratio = fuzz.token_set_ratio(norm_a1, norm_a2) / 100.0 if (norm_a1 and norm_a2) else 0.0

    # Address 3-gram Jaccard
    ag1, ag2 = s1["addr_char_3grams"], cand["addr_char_3grams"]
    addr_gram_jaccard = (len(ag1.intersection(ag2)) / len(ag1.union(ag2))) if (ag1 and ag2) else 0.0

    addr_len_diff = abs(len(norm_a1) - len(norm_a2))
    addr_tok_diff = abs(len(s1["addr_tokens"]) - len(cand["addr_tokens"]))
    addr_contains = 1.0 if (norm_a1 in norm_a2 or norm_a2 in norm_a1) and (norm_a1 and norm_a2) else 0.0

    # Building number comparison
    b1, b2 = s1["bldg_no"], cand["bldg_no"]
    if not b1 or not b2:
        bldg_match = 0.5  # Unknown / neutral
    else:
        bldg_match = 1.0 if b1 == b2 else 0.0

    # Locality comparison
    loc1, loc2 = s1["locality_set"], cand["locality_set"]
    if not loc1 or not loc2:
        loc_match = 0.5
    else:
        loc_inter = len(loc1.intersection(loc2))
        loc_match = 1.0 if loc_inter >= 2 else (0.75 if loc_inter == 1 else 0.0)

    # --- STRUCTURED FEATURES ---
    c1, c2 = s1["country"], cand["country"]
    country_match = 1.0 if c1 == c2 and c1 else 0.0
    cand_src_is_s2 = 1.0 if cand_source == "source2" else 0.0

    return {
        "name_exact": name_exact,
        "name_normalized_exact": name_norm_exact,
        "name_levenshtein": name_lev,
        "name_jaccard": name_jaccard,
        "name_token_overlap": name_overlap,
        "name_token_set_ratio": name_set_ratio,
        "name_char_3gram_jaccard": name_gram_jaccard,
        "name_length_difference": float(name_len_diff),
        "name_token_count_difference": float(name_tok_diff),
        "common_name_tokens": float(name_common),
        "name_contains_other": name_contains,
        "address_exact": addr_exact,
        "address_normalized_exact": addr_norm_exact,
        "address_levenshtein": addr_lev,
        "address_jaccard": addr_jaccard,
        "address_token_overlap": addr_overlap,
        "address_token_set_ratio": addr_set_ratio,
        "address_char_3gram_jaccard": addr_gram_jaccard,
        "address_length_difference": float(addr_len_diff),
        "address_token_count_difference": float(addr_tok_diff),
        "common_address_tokens": float(addr_common),
        "address_contains_other": addr_contains,
        "building_number_match": bldg_match,
        "locality_match": loc_match,
        "country_match": country_match,
        "candidate_source_is_s2": cand_src_is_s2,
    }


def generate_feature_matrix(
    candidate_pairs_df: pd.DataFrame,
    entity_lookups: Dict[str, Dict[str, object]],
    ground_truth_pairs: Optional[Set[Tuple[str, str]]] = None,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    """
    Generate feature matrix for all rows in candidate_pairs_df.
    Computes all numerical similarity features and assigns true label if ground_truth is provided.
    """
    feature_rows = []
    has_gt = ground_truth_pairs is not None

    empty_record = {feat: 0.0 for feat in ALL_FEATURES}

    s1_ids = candidate_pairs_df["source1_id"].tolist()
    c_ids = candidate_pairs_df["candidate_id"].tolist()
    sources = candidate_pairs_df["candidate_source"].tolist()

    labels = [] if has_gt else None

    for s1_id, c_id, src in zip(s1_ids, c_ids, sources):
        s1_rec = entity_lookups.get(s1_id)
        c_rec = entity_lookups.get(c_id)

        feat_dict = compute_pair_features(s1_rec, c_rec, src)
        feature_rows.append(feat_dict)

        if has_gt:
            labels.append(1 if (s1_id, c_id) in ground_truth_pairs else 0)

    feat_df = pd.DataFrame(feature_rows)

    # Attach identifiers
    feat_df["source1_id"] = s1_ids
    feat_df["candidate_id"] = c_ids
    feat_df["candidate_source"] = sources

    if has_gt:
        feat_df["label"] = labels

    return feat_df
