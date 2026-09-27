"""
Decision Engine, Multiple-Match Aggregator, and Singleton Handler (Person 3).
Amazon ML Challenge 2026.

Key Responsibilities:
1. Threshold / Decision Logic:
   - Filters candidate predictions using calibrated probability threshold (default 0.75 - 0.80).
2. Multiple Matches Handling:
   - Supports 0, 1, or multiple matches per Source 1 entity (e.g., S1-001 -> S2-010,S2-024,S3-091).
   - Orders matched candidates by probability (highest first) with deterministic ID tie-breaking.
3. Singleton Handling:
   - Entities with no candidates reaching threshold are left empty (e.g., S1-002 -> "").
   - Avoids forcing false merges, ensuring 1.0 score on singletons.
4. Submission File Generation & Integrity:
   - Produces matching_results.tsv and candidate_pairs.tsv.
   - Enforces subset rule: matching IDs are strictly a subset of candidate IDs.
   - Validates format: tab-separated, no self-matches, no duplicates within list, S2-/S3- prefixes only.
"""

from collections import defaultdict
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple
import pandas as pd


class DecisionEngine:
    """
    Final decision engine applying threshold filtering, singleton guarding,
    multiple match aggregation, and submission formatting.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        max_matches_per_entity: Optional[int] = None,
    ):
        """
        Args:
            threshold: Probability decision threshold. Candidates with prob >= threshold are accepted.
            max_matches_per_entity: Optional safety cap on matched records per entity.
        """
        self.threshold = threshold
        self.max_matches = max_matches_per_entity

    def filter_and_aggregate_matches(
        self,
        predictions_df: pd.DataFrame,
        all_required_s1_ids: Iterable[str],
        custom_threshold: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """
        Filter candidate predictions above threshold and aggregate into ordered lists per S1 entity.

        Args:
            predictions_df: DataFrame with ['source1_id', 'candidate_id', 'match_probability'].
            all_required_s1_ids: Iterable of all S1 IDs that must exist in the output.
            custom_threshold: Optional override for decision threshold.

        Returns:
            Dict mapping source1_id -> list of matched candidate IDs.
        """
        t = custom_threshold if custom_threshold is not None else self.threshold

        # Filter candidates meeting decision threshold
        passing_mask = predictions_df["match_probability"] >= t
        passing_cands = predictions_df[passing_mask]

        # Group by source1_id, sort candidates deterministically:
        # 1. match_probability descending (-prob)
        # 2. candidate_id ascending (tie break)
        grouped_matches = defaultdict(list)

        for s1_id, c_id, prob in zip(
            passing_cands["source1_id"],
            passing_cands["candidate_id"],
            passing_cands["match_probability"],
        ):
            # Enforce prefix constraint (no self matches)
            if c_id.startswith(("S2-", "S3-")):
                grouped_matches[s1_id].append((c_id, prob))

        final_matches: Dict[str, List[str]] = {}

        for s1_id in all_required_s1_ids:
            cand_probs = grouped_matches.get(s1_id, [])
            if not cand_probs:
                # Singleton / no matches above threshold
                final_matches[s1_id] = []
            else:
                # Deduplicate while preserving highest prob, sort deterministically
                seen = set()
                sorted_cands = sorted(cand_probs, key=lambda x: (-x[1], x[0]))
                deduped = []
                for cid, _ in sorted_cands:
                    if cid not in seen:
                        seen.add(cid)
                        deduped.append(cid)

                if self.max_matches is not None:
                    deduped = deduped[: self.max_matches]

                final_matches[s1_id] = deduped

        return final_matches

    def aggregate_candidates(
        self,
        candidate_pairs_df: pd.DataFrame,
        all_required_s1_ids: Iterable[str],
    ) -> Dict[str, List[str]]:
        """
        Aggregate candidate pairs into ordered, deduplicated candidate lists per S1 entity.

        Args:
            candidate_pairs_df: DataFrame with ['source1_id', 'candidate_id'].
            all_required_s1_ids: Iterable of all S1 IDs that must exist in the candidate output.

        Returns:
            Dict mapping source1_id -> list of candidate IDs.
        """
        s1_col = "source1_id" if "source1_id" in candidate_pairs_df.columns else "source1_entity_id"
        c_col = "candidate_id" if "candidate_id" in candidate_pairs_df.columns else "candidate_entity_ids"

        grouped = defaultdict(list)
        for s1_id, c_id in zip(candidate_pairs_df[s1_col], candidate_pairs_df[c_col]):
            if c_id and str(c_id).startswith(("S2-", "S3-")):
                grouped[s1_id].append(str(c_id))

        final_candidates: Dict[str, List[str]] = {}
        for s1_id in all_required_s1_ids:
            c_list = grouped.get(s1_id, [])
            # Deduplicate preserving order
            seen = set()
            deduped = []
            for cid in c_list:
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(cid)
            final_candidates[s1_id] = deduped

        return final_candidates


def format_submission_files(
    matching_map: Dict[str, List[str]],
    candidates_map: Dict[str, List[str]],
    required_s1_ids: Iterable[str],
    matching_output_path: str = "output/matching_results.tsv",
    candidate_output_path: str = "output/candidate_pairs.tsv",
) -> Tuple[int, int]:
    """
    Format and write matching_results.tsv and candidate_pairs.tsv according to exact competition rules.

    Rules strictly enforced:
    - Tab-separated columns (source1_entity_id, matched_entity_ids)
    - Tab-separated columns (source1_entity_id, candidate_entity_ids)
    - Comma-separated ID lists with zero quoting
    - Exactly one row per required S1 entity
    - Singleton entities output with empty ID string
    - Matched IDs are guaranteed to be a subset of candidate IDs
    - No duplicate IDs within a list
    - UTF-8 encoding

    Returns:
        (total_s1_rows, total_matched_entities_count)
    """
    os.makedirs(os.path.dirname(os.path.abspath(matching_output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(candidate_output_path)), exist_ok=True)

    s1_order = list(required_s1_ids)

    # 1. Write candidate_pairs.tsv
    cand_rows_written = 0
    with open(candidate_output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_order:
            c_list = candidates_map.get(s1_id, [])
            # Also ensure any matches are in candidates to satisfy subset rule
            m_list = matching_map.get(s1_id, [])
            combined_set = set(c_list)
            for m in m_list:
                if m not in combined_set:
                    c_list.append(m)
                    combined_set.add(m)

            c_str = ",".join(c_list)
            f.write(f"{s1_id}\t{c_str}\n")
            cand_rows_written += 1

    # 2. Write matching_results.tsv
    total_matches = 0
    match_rows_written = 0
    with open(matching_output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_order:
            m_list = matching_map.get(s1_id, [])
            m_str = ",".join(m_list)
            f.write(f"{s1_id}\t{m_str}\n")
            match_rows_written += 1
            total_matches += len(m_list)

    return match_rows_written, total_matches
