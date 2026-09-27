"""
Enhanced Feature Engineering Module for Entity Resolution.
Amazon ML Challenge 2026 — Team BugHunters.

Extracts 45 discriminative features across:
1. Business Names (exact, Levenshtein, token Jaccard, token set/sort ratio, 3/4-gram Jaccard, IDF-weighted overlap, prefix match, rare tokens)
2. Addresses (exact, Levenshtein, token Jaccard, 3-gram Jaccard, building number match, locality match, postal/PIN code match, numeric overlap, numeric conflict, missing address handling)
3. Structured & Candidate Meta-Features (country match, candidate source, blocking agreement score, blocking rank, blocking score ratio)
"""

import os
import sys
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.dirname(cur_dir)
repo_root = os.path.dirname(dataset_dir)
for p in [cur_dir, dataset_dir, repo_root]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from dataset.person1.enhanced_normalization import (
    GENERIC_BUSINESS_TERMS,
    clean_core_name,
    extract_building_no,
    extract_numeric_tokens,
    extract_postal_code,
    normalize_address,
    normalize_country,
    normalize_name,
)

IGNORED_ADDR_TOKENS = {
    "road", "street", "avenue", "drive", "lane", "highway", "court", "circle",
    "floor", "building", "apartment", "suite", "unit", "near", "opposite", "behind",
    "block", "sector", "nagar", "colony", "cross", "main"
}

ENHANCED_NAME_FEATURES = [
    "name_exact",
    "name_normalized_exact",
    "name_core_exact",
    "name_levenshtein",
    "name_jaccard",
    "name_token_overlap",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_char_3gram_jaccard",
    "name_char_4gram_jaccard",
    "name_prefix_similarity",
    "name_length_difference",
    "name_token_count_difference",
    "common_name_tokens",
    "name_contains_other",
    "name_first_token_match",
    "name_rare_token_overlap",
    "name_idf_weighted_jaccard",
]

ENHANCED_ADDRESS_FEATURES = [
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
    "postal_code_match",
    "numeric_token_overlap",
    "numeric_token_conflict",
    "has_both_addresses",
    "either_address_empty",
]

ENHANCED_STRUCTURED_FEATURES = [
    "country_match",
    "candidate_source_is_s2",
    "blocking_score",
    "blocking_rank",
    "blocking_score_ratio",
]

ALL_ENHANCED_FEATURES = (
    ENHANCED_NAME_FEATURES + ENHANCED_ADDRESS_FEATURES + ENHANCED_STRUCTURED_FEATURES
)


def extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from cleaned string."""
    clean = re.sub(r"\s+", "", text.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def extract_locality_tokens(norm_addr: str) -> Set[str]:
    """Extract significant alphabetic locality words from normalized address."""
    if not norm_addr:
        return set()
    tokens = norm_addr.split()
    return {w for w in tokens if len(w) >= 4 and w.isalpha() and w not in IGNORED_ADDR_TOKENS}


def compute_corpus_idf(corpus_tokens_list: List[List[str]]) -> Dict[str, float]:
    """Compute token inverse document frequency (IDF) weights from corpus token lists."""
    doc_count = len(corpus_tokens_list)
    df_counts = Counter()
    for doc in corpus_tokens_list:
        for tok in set(doc):
            df_counts[tok] += 1

    idf_weights = {}
    for tok, count in df_counts.items():
        idf_weights[tok] = float(np.log((doc_count + 1) / (count + 1)) + 1.0)
    return idf_weights


def preprocess_entity_record(
    raw_name: str,
    raw_addr: str,
    raw_country: str,
    norm_name_cached: Optional[str] = None,
    core_name_cached: Optional[str] = None,
    norm_addr_cached: Optional[str] = None,
    norm_country_cached: Optional[str] = None,
) -> Dict[str, object]:
    """Preprocess and pre-tokenize a single entity record."""
    raw_name = str(raw_name) if pd.notna(raw_name) else ""
    raw_addr = str(raw_addr) if pd.notna(raw_addr) else ""
    raw_country = str(raw_country) if pd.notna(raw_country) else ""

    norm_c = norm_country_cached if norm_country_cached is not None else normalize_country(raw_country)
    norm_n = norm_name_cached if norm_name_cached is not None else normalize_name(raw_name)
    core_n = core_name_cached if core_name_cached is not None else clean_core_name(norm_n)
    norm_a = norm_addr_cached if norm_addr_cached is not None else normalize_address(raw_addr)

    name_tokens = norm_n.split()
    name_token_set = set(name_tokens)
    core_tokens = core_n.split()
    core_token_set = set(core_tokens)

    addr_tokens = norm_a.split()
    addr_token_set = set(addr_tokens)

    bldg_no = extract_building_no(norm_a)
    locality_set = extract_locality_tokens(norm_a)
    postal_code = extract_postal_code(norm_a, norm_c)
    numeric_tokens = extract_numeric_tokens(norm_a)

    return {
        "raw_name": raw_name,
        "norm_name": norm_n,
        "core_name": core_n,
        "name_tokens": name_tokens,
        "name_token_set": name_token_set,
        "core_tokens": core_tokens,
        "core_token_set": core_token_set,
        "name_char_3grams": extract_char_ngrams(norm_n, 3),
        "name_char_4grams": extract_char_ngrams(norm_n, 4),
        "raw_addr": raw_addr,
        "norm_addr": norm_a,
        "addr_tokens": addr_tokens,
        "addr_token_set": addr_token_set,
        "addr_char_3grams": extract_char_ngrams(norm_a, 3),
        "bldg_no": bldg_no,
        "locality_set": locality_set,
        "postal_code": postal_code,
        "numeric_tokens": numeric_tokens,
        "country": norm_c,
    }


def load_entity_lookups_enhanced(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    needed_s1_ids: Optional[Set[str]] = None,
    needed_cand_ids: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Build preprocessed lookup mapping using cached normalized columns if available."""
    lookup: Dict[str, Dict[str, object]] = {}

    def process_df(df, needed_ids):
        if df is None or len(df) == 0:
            return
        sub = df if needed_ids is None else df[df["entity_id"].isin(needed_ids)]
        has_norm = "norm_name" in sub.columns
        for row in sub.itertuples(index=False):
            eid = row.entity_id
            bname = row.business_name
            baddr = row.business_address
            ctry = row.country
            nn = getattr(row, "norm_name", None) if has_norm else None
            cn = getattr(row, "core_name", None) if has_norm else None
            na = getattr(row, "norm_addr", None) if has_norm else None
            nc = getattr(row, "norm_country", None) if has_norm else None
            lookup[eid] = preprocess_entity_record(bname, baddr, ctry, nn, cn, na, nc)

    process_df(s1_df, needed_s1_ids)
    process_df(s2_df, needed_cand_ids)
    process_df(s3_df, needed_cand_ids)
    return lookup


def compute_enhanced_pair_features(
    s1: Dict[str, object],
    cand: Dict[str, object],
    cand_source: str,
    blocking_score: float = 1.0,
    blocking_rank: float = 1.0,
    blocking_score_ratio: float = 1.0,
    idf_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute 45 numerical features for a single (Source 1, Candidate) pair."""
    if not s1 or not cand:
        return {feat: 0.0 for feat in ALL_ENHANCED_FEATURES}

    # --- NAME FEATURES ---
    raw_n1, raw_n2 = s1["raw_name"], cand["raw_name"]
    norm_n1, norm_n2 = s1["norm_name"], cand["norm_name"]
    core_n1, core_n2 = s1["core_name"], cand["core_name"]

    name_exact = 1.0 if raw_n1 == raw_n2 and raw_n1 else 0.0
    name_norm_exact = 1.0 if norm_n1 == norm_n2 and norm_n1 else 0.0
    name_core_exact = 1.0 if core_n1 == core_n2 and core_n1 else 0.0

    name_lev = Levenshtein.normalized_similarity(norm_n1, norm_n2) if (norm_n1 and norm_n2) else 0.0

    t_set1, t_set2 = s1["core_token_set"], cand["core_token_set"]
    inter_tokens = t_set1.intersection(t_set2)
    union_tokens = t_set1.union(t_set2)

    name_common = len(inter_tokens)
    name_union = len(union_tokens)
    name_jaccard = (name_common / name_union) if name_union > 0 else 0.0
    min_tokens = min(len(t_set1), len(t_set2))
    name_token_overlap = (name_common / min_tokens) if min_tokens > 0 else 0.0

    name_set_ratio = fuzz.token_set_ratio(norm_n1, norm_n2) / 100.0 if (norm_n1 and norm_n2) else 0.0
    name_sort_ratio = fuzz.token_sort_ratio(norm_n1, norm_n2) / 100.0 if (norm_n1 and norm_n2) else 0.0

    g1_3, g2_3 = s1["name_char_3grams"], cand["name_char_3grams"]
    name_3gram_jaccard = (len(g1_3.intersection(g2_3)) / len(g1_3.union(g2_3))) if (g1_3 and g2_3) else 0.0

    g1_4, g2_4 = s1["name_char_4grams"], cand["name_char_4grams"]
    name_4gram_jaccard = (len(g1_4.intersection(g2_4)) / len(g1_4.union(g2_4))) if (g1_4 and g2_4) else 0.0

    # Prefix similarity
    p1 = norm_n1[:4] if len(norm_n1) >= 4 else norm_n1
    p2 = norm_n2[:4] if len(norm_n2) >= 4 else norm_n2
    name_prefix_sim = 1.0 if p1 == p2 and p1 else (0.5 if (p1 and p2 and p1[:3] == p2[:3]) else 0.0)

    name_len_diff = float(abs(len(norm_n1) - len(norm_n2)))
    name_tok_diff = float(abs(len(s1["name_tokens"]) - len(cand["name_tokens"])))
    name_contains = 1.0 if (norm_n1 in norm_n2 or norm_n2 in norm_n1) and (norm_n1 and norm_n2) else 0.0

    # First token match
    t1_first = s1["core_tokens"][0] if s1["core_tokens"] else ""
    t2_first = cand["core_tokens"][0] if cand["core_tokens"] else ""
    name_first_match = 1.0 if t1_first == t2_first and t1_first else 0.0

    # Rare tokens shared
    rare_shared = sum(1 for t in inter_tokens if t not in GENERIC_BUSINESS_TERMS and len(t) >= 4)

    # IDF-weighted Jaccard
    if idf_weights and union_tokens:
        inter_weight = sum(idf_weights.get(t, 1.0) for t in inter_tokens)
        union_weight = sum(idf_weights.get(t, 1.0) for t in union_tokens)
        name_idf_jaccard = inter_weight / union_weight if union_weight > 0 else 0.0
    else:
        name_idf_jaccard = name_jaccard

    # --- ADDRESS FEATURES ---
    raw_a1, raw_a2 = s1["raw_addr"], cand["raw_addr"]
    norm_a1, norm_a2 = s1["norm_addr"], cand["norm_addr"]

    has_both_addr = 1.0 if (norm_a1 and norm_a2) else 0.0
    either_empty = 1.0 if (not norm_a1 or not norm_a2) else 0.0

    addr_exact = 1.0 if raw_a1 == raw_a2 and raw_a1 else 0.0
    addr_norm_exact = 1.0 if norm_a1 == norm_a2 and norm_a1 else 0.0
    addr_lev = Levenshtein.normalized_similarity(norm_a1, norm_a2) if (norm_a1 and norm_a2) else 0.0

    at_set1, at_set2 = s1["addr_token_set"], cand["addr_token_set"]
    addr_inter = at_set1.intersection(at_set2)
    addr_union = at_set1.union(at_set2)

    addr_common = len(addr_inter)
    addr_jaccard = (addr_common / len(addr_union)) if addr_union else 0.0
    min_atoks = min(len(at_set1), len(at_set2))
    addr_overlap = (addr_common / min_atoks) if min_atoks > 0 else 0.0

    addr_set_ratio = fuzz.token_set_ratio(norm_a1, norm_a2) / 100.0 if (norm_a1 and norm_a2) else 0.0

    ag1, ag2 = s1["addr_char_3grams"], cand["addr_char_3grams"]
    addr_3gram_jaccard = (len(ag1.intersection(ag2)) / len(ag1.union(ag2))) if (ag1 and ag2) else 0.0

    addr_len_diff = float(abs(len(norm_a1) - len(norm_a2)))
    addr_tok_diff = float(abs(len(s1["addr_tokens"]) - len(cand["addr_tokens"])))
    addr_contains = 1.0 if (norm_a1 in norm_a2 or norm_a2 in norm_a1) and (norm_a1 and norm_a2) else 0.0

    # Building number
    b1, b2 = s1["bldg_no"], cand["bldg_no"]
    bldg_match = 0.5 if (not b1 or not b2) else (1.0 if b1 == b2 else 0.0)

    # Locality match
    loc1, loc2 = s1["locality_set"], cand["locality_set"]
    if not loc1 or not loc2:
        loc_match = 0.5
    else:
        inter_loc = len(loc1.intersection(loc2))
        loc_match = 1.0 if inter_loc >= 2 else (0.75 if inter_loc == 1 else 0.0)

    # Postal code match
    p_code1, p_code2 = s1["postal_code"], cand["postal_code"]
    if not p_code1 or not p_code2:
        postal_match = 0.5
    else:
        postal_match = 1.0 if p_code1 == p_code2 else 0.0

    # Numeric tokens overlap & conflict
    nums1, nums2 = s1["numeric_tokens"], cand["numeric_tokens"]
    if nums1 and nums2:
        num_inter = len(nums1.intersection(nums2))
        num_overlap = float(num_inter)
        num_conflict = 1.0 if num_inter == 0 else 0.0
    else:
        num_overlap = 0.0
        num_conflict = 0.0

    # --- STRUCTURED & META FEATURES ---
    c1, c2 = s1["country"], cand["country"]
    country_match = 1.0 if c1 == c2 and c1 else 0.0
    cand_src_is_s2 = 1.0 if cand_source == "source2" else 0.0

    return {
        "name_exact": name_exact,
        "name_normalized_exact": name_norm_exact,
        "name_core_exact": name_core_exact,
        "name_levenshtein": name_lev,
        "name_jaccard": name_jaccard,
        "name_token_overlap": name_token_overlap,
        "name_token_set_ratio": name_set_ratio,
        "name_token_sort_ratio": name_sort_ratio,
        "name_char_3gram_jaccard": name_3gram_jaccard,
        "name_char_4gram_jaccard": name_4gram_jaccard,
        "name_prefix_similarity": name_prefix_sim,
        "name_length_difference": name_len_diff,
        "name_token_count_difference": name_tok_diff,
        "common_name_tokens": float(name_common),
        "name_contains_other": name_contains,
        "name_first_token_match": name_first_match,
        "name_rare_token_overlap": float(rare_shared),
        "name_idf_weighted_jaccard": name_idf_jaccard,
        "address_exact": addr_exact,
        "address_normalized_exact": addr_norm_exact,
        "address_levenshtein": addr_lev,
        "address_jaccard": addr_jaccard,
        "address_token_overlap": addr_overlap,
        "address_token_set_ratio": addr_set_ratio,
        "address_char_3gram_jaccard": addr_3gram_jaccard,
        "address_length_difference": addr_len_diff,
        "address_token_count_difference": addr_tok_diff,
        "common_address_tokens": float(addr_common),
        "address_contains_other": addr_contains,
        "building_number_match": bldg_match,
        "locality_match": loc_match,
        "postal_code_match": postal_match,
        "numeric_token_overlap": num_overlap,
        "numeric_token_conflict": num_conflict,
        "has_both_addresses": has_both_addr,
        "either_address_empty": either_empty,
        "country_match": country_match,
        "candidate_source_is_s2": cand_src_is_s2,
        "blocking_score": float(blocking_score),
        "blocking_rank": float(blocking_rank),
        "blocking_score_ratio": float(blocking_score_ratio),
    }


def generate_enhanced_feature_matrix(
    candidate_pairs_df: pd.DataFrame,
    entity_lookups: Dict[str, Dict[str, object]],
    ground_truth_pairs: Optional[Set[Tuple[str, str]]] = None,
    idf_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Generate 45-feature matrix with meta-features and ground truth labels."""
    has_gt = ground_truth_pairs is not None
    feature_rows = []

    s1_ids = candidate_pairs_df["source1_id"].tolist()
    c_ids = candidate_pairs_df["candidate_id"].tolist()
    sources = candidate_pairs_df["candidate_source"].tolist()

    has_bscore = "blocking_score" in candidate_pairs_df.columns
    has_brank = "blocking_rank" in candidate_pairs_df.columns
    has_bratio = "blocking_score_ratio" in candidate_pairs_df.columns

    b_scores = candidate_pairs_df["blocking_score"].tolist() if has_bscore else [1.0] * len(s1_ids)
    b_ranks = candidate_pairs_df["blocking_rank"].tolist() if has_brank else [1.0] * len(s1_ids)
    b_ratios = candidate_pairs_df["blocking_score_ratio"].tolist() if has_bratio else [1.0] * len(s1_ids)

    labels = [] if has_gt else None

    for i in range(len(s1_ids)):
        s1_id = s1_ids[i]
        c_id = c_ids[i]
        src = sources[i]

        s1_rec = entity_lookups.get(s1_id)
        c_rec = entity_lookups.get(c_id)

        feat_dict = compute_enhanced_pair_features(
            s1=s1_rec,
            cand=c_rec,
            cand_source=src,
            blocking_score=b_scores[i],
            blocking_rank=b_ranks[i],
            blocking_score_ratio=b_ratios[i],
            idf_weights=idf_weights,
        )
        feature_rows.append(feat_dict)

        if has_gt:
            labels.append(1 if (s1_id, c_id) in ground_truth_pairs else 0)

    feat_df = pd.DataFrame(feature_rows)
    feat_df["source1_id"] = s1_ids
    feat_df["candidate_id"] = c_ids
    feat_df["candidate_source"] = sources

    if has_gt:
        feat_df["label"] = labels

    return feat_df
