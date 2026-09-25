"""
Blocking Module for Business Entity Resolution.
Amazon ML Challenge 2026.

Implements multiple independent blocking strategies:
1. Country Partition (Hard partition)
2. Name Prefix (4-char and 6-char prefixes)
3. Name Distinctive Tokens (excluding high-frequency stopwords)
4. Name Sorted Tokens (permutation-invariant token key)
5. Name Compressed / Domain Key (exact alphanumeric match for domains/spacing differences)
6. Address Key (building/house/shop number + street/locality token)

Supports strategy comparison and multi-strategy union blocking.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from .normalization import (
    GENERIC_BUSINESS_TERMS,
    clean_core_name,
    extract_address_key,
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
    """
    Generate blocking keys for a single normalized record.

    Args:
        country: Normalized country code ('US', 'India', 'France', etc.).
        norm_name: Normalized full business name.
        core_name: Core business name without legal form suffixes.
        norm_addr: Normalized address string.
        strategies: Set of strategy names to include. If None, all are included.

    Returns:
        Set of blocking key strings prefixed by country.
    """
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
        "address_key",
        "address_locality",
    }

    # Clean core name without spaces
    compressed_core = core_name.replace(" ", "")

    # 1. Name Prefix 4
    if "name_prefix_4" in active and len(compressed_core) >= 4:
        keys.add(f"{country}|NP4|{compressed_core[:4]}")

    # 2. Name Prefix 6
    if "name_prefix_6" in active and len(compressed_core) >= 6:
        keys.add(f"{country}|NP6|{compressed_core[:6]}")

    # 3. Name Compressed (Matches domain names and spaceless variations)
    if "name_compressed" in active and len(compressed_core) >= 5:
        keys.add(f"{country}|NCOMP|{compressed_core}")

    # 4. Name Distinctive Tokens & 3-Letter Acronyms
    if core_name:
        tokens = core_name.split()
        if "name_tokens" in active:
            for tok in tokens:
                if len(tok) >= 4 and tok not in GENERIC_BUSINESS_TERMS:
                    keys.add(f"{country}|TOK|{tok}")
        if "name_tokens_3" in active:
            for tok in tokens:
                if len(tok) == 3 and tok.isalpha() and tok not in GENERIC_BUSINESS_TERMS and tok not in {"and", "the", "for", "all"}:
                    keys.add(f"{country}|TOK3|{tok}")

    # 5. Name Sorted Tokens (Order-invariant)
    if "name_sorted" in active and core_name:
        sig_tokens = [t for t in core_name.split() if t not in GENERIC_BUSINESS_TERMS and len(t) >= 3]
        if len(sig_tokens) >= 2:
            sorted_key = "_".join(sorted(sig_tokens[:4]))
            keys.add(f"{country}|NSORT|{sorted_key}")

    # 6. Address Key, Unit/Plot Codes, and Non-numeric Locality Pairs
    if norm_addr:
        addr_tokens = norm_addr.split()
        if "address_key" in active:
            addr_k = extract_address_key(norm_addr)
            if addr_k and len(addr_k) >= 3:
                keys.add(f"{country}|ADDR|{addr_k}")

            # Alphanumeric unit/plot/flat numbers like 'd-062', 'a-40', 'af-684', 'wz-187c', '126/4'
            for tok in addr_tokens:
                clean_tok = tok.replace("-", "").replace("/", "")
                # Skip ordinals like 1st, 2nd, 3rd, 4th, 22nd
                if len(clean_tok) >= 3 and clean_tok[-2:] in {"st", "nd", "rd", "th"} and clean_tok[:-2].isdigit():
                    continue
                # Alphanumeric codes like d062, af684, a40
                if any(c.isdigit() for c in clean_tok) and any(c.isalpha() for c in clean_tok) and 3 <= len(clean_tok) <= 8:
                    keys.add(f"{country}|UNIT|{clean_tok}")
                # Slash plot numbers like 126/4 or 59/101
                elif "/" in tok and any(c.isdigit() for c in tok) and len(tok) <= 8:
                    keys.add(f"{country}|PLOT|{tok}")

        # 7. Non-numeric Locality Key (for commercial / highway addresses without house numbers)
        if "address_locality" in active:
            ignored_addr = {
                "rd", "st", "ave", "dr", "ln", "hwy", "blvd", "ct", "cir", "fl", "apt", "ste",
                "unit", "shop", "near", "opp", "opposite", "behind", "hotel", "road", "street",
                "highway", "lane", "avenue", "drive", "floor", "ground", "block", "sector", "building"
            }
            sig_addr_words = [w for w in addr_tokens if len(w) >= 4 and w not in ignored_addr and w.isalpha()]
            if len(sig_addr_words) >= 2:
                keys.add(f"{country}|ADDR_LOC|{sig_addr_words[0]}_{sig_addr_words[1]}")

    return keys


def generate_blocking_keys(
    df: pd.DataFrame,
    strategies: Optional[Set[str]] = None,
) -> List[Set[str]]:
    """
    Generate blocking keys for all records in a DataFrame.
    Assumes or creates normalized columns ('norm_name', 'core_name', 'norm_addr', 'norm_country').
    """
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


class BlockingIndex:
    """
    Inverted Index for high-performance blocking and candidate retrieval.
    Stores mapping from blocking keys to candidate (entity_id, source) pairs.
    Includes maximum block size filtering to prevent Cartesian explosion on generic terms.
    """

    def __init__(self, max_block_size: int = 500):
        self.max_block_size = max_block_size
        self.index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add_candidates(self, df: pd.DataFrame, source_name: str, strategies: Optional[Set[str]] = None):
        """
        Index candidate records from a source (e.g. source2 or source3).
        """
        keys_list = generate_blocking_keys(df, strategies=strategies)
        entity_ids = df["entity_id"].tolist()

        for eid, keys in zip(entity_ids, keys_list):
            item = (eid, source_name)
            for k in keys:
                self.index[k].append(item)

    def prune_large_blocks(self):
        """
        Remove blocking keys that exceed max_block_size to avoid Cartesian product on stopwords.
        """
        pruned_keys = [k for k, v in self.index.items() if len(v) > self.max_block_size]
        for k in pruned_keys:
            del self.index[k]
        return len(pruned_keys)

    def get_candidates(self, keys: Set[str]) -> Set[Tuple[str, str]]:
        """
        Retrieve candidate (entity_id, source) tuples matching any of the given keys.
        """
        candidates = set()
        for k in keys:
            if k in self.index:
                cand_list = self.index[k]
                if len(cand_list) <= self.max_block_size:
                    candidates.update(cand_list)
        return candidates
