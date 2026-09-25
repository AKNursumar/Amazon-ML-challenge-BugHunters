"""
Candidate Generation Module for Amazon ML Challenge 2026.
Produces high-recall candidate sets by indexing Source 2 and Source 3 and querying Source 1.
"""

from collections import defaultdict
from typing import List, Optional, Set, Tuple
import pandas as pd

from .blocking import BlockingIndex, generate_blocking_keys
from .normalization import (
    clean_core_name,
    normalize_address,
    normalize_country,
    normalize_name,
)


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    strategies: Optional[Set[str]] = None,
    max_block_size: int = 1000,
    max_candidates_per_query: int = 300,
) -> pd.DataFrame:
    """
    Generate candidate pairs between Source 1 and (Source 2 + Source 3).

    Pipeline:
    1. Normalization of names, addresses, countries (preserving original columns).
    2. Inverted index construction for Source 2 and Source 3.
    3. Block pruning to prevent Cartesian product on generic terms.
    4. Querying index for each Source 1 record.
    5. Deduplication and deterministic formatting.

    Args:
        source1: DataFrame containing Source 1 records.
        source2: DataFrame containing Source 2 records.
        source3: DataFrame containing Source 3 records.
        strategies: Set of blocking strategies to activate (None = all default strategies).
        max_block_size: Maximum records allowed in a single block before pruning.
        max_candidates_per_query: Maximum candidates to retain per Source 1 entity.

    Returns:
        pd.DataFrame with columns: ['source1_id', 'candidate_id', 'candidate_source']
    """
    # Step 1: Normalize required fields if not already present
    for df in [source1, source2, source3]:
        if "norm_name" not in df.columns:
            df["norm_name"] = df["business_name"].apply(normalize_name)
        if "core_name" not in df.columns:
            df["core_name"] = df["norm_name"].apply(clean_core_name)
        if "norm_addr" not in df.columns:
            df["norm_addr"] = df["business_address"].apply(normalize_address)
        if "norm_country" not in df.columns:
            df["norm_country"] = df["country"].apply(normalize_country)

    # Step 2: Build Blocking Index from Source 2 and Source 3
    index = BlockingIndex(max_block_size=max_block_size)
    if source2 is not None and len(source2) > 0:
        index.add_candidates(source2, source_name="source2", strategies=strategies)
    if source3 is not None and len(source3) > 0:
        index.add_candidates(source3, source_name="source3", strategies=strategies)

    # Step 3: Prune oversized blocks
    index.prune_large_blocks()

    # Step 4: Query index for Source 1 records
    s1_keys_list = generate_blocking_keys(source1, strategies=strategies)
    s1_ids = source1["entity_id"].tolist()

    records: List[Tuple[str, str, str]] = []

    for s1_id, keys in zip(s1_ids, s1_keys_list):
        cand_scores = defaultdict(int)
        cand_source_map = {}

        for k in keys:
            if k in index.index:
                cand_list = index.index[k]
                if len(cand_list) <= index.max_block_size:
                    for cand_id, cand_src in cand_list:
                        cand_scores[cand_id] += 1
                        cand_source_map[cand_id] = cand_src

        # Sort candidates by:
        # 1. Negative score (highest overlap count first)
        # 2. Candidate ID (for deterministic tie-breaking)
        sorted_cands = sorted(cand_scores.items(), key=lambda x: (-x[1], x[0]))

        if max_candidates_per_query and len(sorted_cands) > max_candidates_per_query:
            sorted_cands = sorted_cands[:max_candidates_per_query]

        for cand_id, _ in sorted_cands:
            records.append((s1_id, cand_id, cand_source_map[cand_id]))

    # Step 5: Construct candidate pairs DataFrame
    candidate_pairs_df = pd.DataFrame(
        records,
        columns=["source1_id", "candidate_id", "candidate_source"],
    )

    # Deduplicate deterministically
    candidate_pairs_df = candidate_pairs_df.drop_duplicates(
        subset=["source1_id", "candidate_id"]
    ).reset_index(drop=True)

    return candidate_pairs_df
