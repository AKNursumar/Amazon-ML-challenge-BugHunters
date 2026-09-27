"""
Enhanced Multi-Pass Blocking Module for High-Recall Candidate Retrieval.
Amazon ML Challenge 2026 — Team BugHunters.

Implements multi-strategy inverted indexing:
1. Country Partition (Hard partition)
2. Name Prefixes (NP4, NP6)
3. Name Distinctive Tokens (TOK, TOK3)
4. Name Compressed / Domain Key (NCOMP)
5. Name Sorted Tokens (NSORT)
6. Distinctive Name Pairs (NPAIR)
7. Postal / PIN Code (ZIP)
8. Name Token + Postal Code (NTOK_ZIP)
9. Alphanumeric Unit & Plot numbers (UNIT, PLOT)
10. Address Locality Pairs (ADDR_LOC)

Tracks blocking meta-features (rule agreement count, score, rank).
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from .enhanced_normalization import (
    GENERIC_BUSINESS_TERMS,
    clean_core_name,
    extract_building_no,
    extract_numeric_tokens,
    extract_postal_code,
    normalize_address,
    normalize_country,
    normalize_name,
)


def get_record_blocking_keys(
    country: str,
    norm_name: str,
    core_name: str,
    norm_addr: str,
    strategies: Optional[Set[str]] = None,
) -> Set[str]:
    """Generate multi-pass blocking keys for a single normalized entity record."""
    keys = set()
    if not country:
        country = "UNKNOWN"

    active = strategies or {
        "name_prefix_4",
        "name_prefix_6",
        "name_tokens",
        "name_tokens_3",
        "name_sorted",
        "name_compressed",
        "name_pairs",
        "postal_code",
        "name_postal",
        "address_unit",
        "address_locality",
    }

    compressed_core = core_name.replace(" ", "")

    # 1. Name Prefix 4 & 6
    if "name_prefix_4" in active and len(compressed_core) >= 4:
        keys.add(f"{country}|NP4|{compressed_core[:4]}")
    if "name_prefix_6" in active and len(compressed_core) >= 6:
        keys.add(f"{country}|NP6|{compressed_core[:6]}")

    # 2. Name Compressed
    if "name_compressed" in active and len(compressed_core) >= 5:
        keys.add(f"{country}|NCOMP|{compressed_core}")

    # 3. Name Tokens & Acronyms
    tokens = core_name.split() if core_name else []
    sig_tokens = [t for t in tokens if len(t) >= 3 and t not in GENERIC_BUSINESS_TERMS]

    if "name_tokens" in active:
        for tok in sig_tokens:
            if len(tok) >= 4:
                keys.add(f"{country}|TOK|{tok}")
            elif len(tok) == 3 and tok.isalpha():
                keys.add(f"{country}|TOK3|{tok}")

    # 4. Name Sorted Tokens
    if "name_sorted" in active and len(sig_tokens) >= 2:
        sorted_k = "_".join(sorted(sig_tokens[:4]))
        keys.add(f"{country}|NSORT|{sorted_k}")

    # 5. Distinctive Name Pairs
    if "name_pairs" in active and len(sig_tokens) >= 2:
        for i in range(min(3, len(sig_tokens))):
            for j in range(i + 1, min(4, len(sig_tokens))):
                pair_k = "_".join(sorted([sig_tokens[i], sig_tokens[j]]))
                keys.add(f"{country}|NPAIR|{pair_k}")

    # 6. Postal Code & Postal + Name Token
    postal = extract_postal_code(norm_addr, country)
    if postal:
        if "postal_code" in active and len(sig_tokens) >= 1:
            # Pair postal code with first distinctive token to avoid Cartesian product on postal
            keys.add(f"{country}|ZIP_TOK|{postal}_{sig_tokens[0]}")
        if "name_postal" in active and len(compressed_core) >= 4:
            keys.add(f"{country}|NP4_ZIP|{compressed_core[:4]}_{postal}")

    # 7. Address Unit, Plot, and Building Numbers
    if norm_addr:
        addr_tokens = norm_addr.split()
        if "address_unit" in active:
            for tok in addr_tokens:
                clean_tok = tok.replace("-", "").replace("/", "")
                # Skip ordinals like 1st, 2nd, 3rd, 4th
                if len(clean_tok) >= 3 and clean_tok[-2:] in {"st", "nd", "rd", "th"} and clean_tok[:-2].isdigit():
                    continue
                # Alphanumeric unit codes like d062, af684, a40
                if any(c.isdigit() for c in clean_tok) and any(c.isalpha() for c in clean_tok) and 3 <= len(clean_tok) <= 8:
                    keys.add(f"{country}|UNIT|{clean_tok}")
                elif "/" in tok and any(c.isdigit() for c in tok) and len(tok) <= 8:
                    keys.add(f"{country}|PLOT|{tok}")

        # 8. Address Locality Pairs
        if "address_locality" in active:
            ignored = {
                "road", "street", "avenue", "drive", "lane", "highway", "court", "circle",
                "building", "floor", "apartment", "suite", "unit", "near", "opposite", "behind"
            }
            sig_addr = [w for w in addr_tokens if len(w) >= 4 and w not in ignored and w.isalpha()]
            if len(sig_addr) >= 2:
                keys.add(f"{country}|ADDR_LOC|{sig_addr[0]}_{sig_addr[1]}")

    return keys


def generate_blocking_keys(
    df: pd.DataFrame,
    strategies: Optional[Set[str]] = None,
) -> List[Set[str]]:
    """Generate blocking key sets for all records in DataFrame."""
    if "norm_name" not in df.columns:
        df["norm_name"] = df["business_name"].apply(normalize_name)
    if "core_name" not in df.columns:
        df["core_name"] = df["norm_name"].apply(clean_core_name)
    if "norm_addr" not in df.columns:
        df["norm_addr"] = df["business_address"].apply(normalize_address)
    if "norm_country" not in df.columns:
        df["norm_country"] = df["country"].apply(normalize_country)

    all_keys = []
    for c, nn, cn, na in zip(df["norm_country"], df["norm_name"], df["core_name"], df["norm_addr"]):
        keys = get_record_blocking_keys(c, nn, cn, na, strategies=strategies)
        all_keys.append(keys)
    return all_keys


class EnhancedBlockingIndex:
    """High-performance inverted blocking index with block pruning and candidate scoring."""

    def __init__(self, max_block_size: int = 1000):
        self.max_block_size = max_block_size
        self.index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add_candidates(self, df: pd.DataFrame, source_name: str, strategies: Optional[Set[str]] = None):
        keys_list = generate_blocking_keys(df, strategies=strategies)
        entity_ids = df["entity_id"].tolist()
        for eid, keys in zip(entity_ids, keys_list):
            item = (eid, source_name)
            for k in keys:
                self.index[k].append(item)

    def prune_large_blocks(self) -> int:
        pruned_keys = [k for k, v in self.index.items() if len(v) > self.max_block_size]
        for k in pruned_keys:
            del self.index[k]
        return len(pruned_keys)
