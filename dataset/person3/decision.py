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


BRANCH_DIRECTIONS = {
    "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "central",
}
BRANCH_DIVISIONS = {
    "retail", "wholesale", "express", "logistics", "distribution", "services", "capital", "holdings",
}


def has_branch_conflict(name1: str, name2: str) -> bool:
    """Return True if two business names have contradictory directional or division tags."""
    if not name1 or not name2:
        return False
    toks1 = set(str(name1).lower().split())
    toks2 = set(str(name2).lower().split())

    dirs1 = toks1.intersection(BRANCH_DIRECTIONS)
    dirs2 = toks2.intersection(BRANCH_DIRECTIONS)
    if dirs1 and dirs2 and dirs1 != dirs2:
        return True

    divs1 = toks1.intersection(BRANCH_DIVISIONS)
    divs2 = toks2.intersection(BRANCH_DIVISIONS)
    if divs1 and divs2 and divs1 != divs2:
        return True

    return False


def has_building_clash(bldg1: str, bldg2: str, core1: str, core2: str) -> bool:
    """
    Return True if both entities have extracted non-empty building/plot numbers
    and they differ, unless core business names are identical.
    """
    if bldg1 and bldg2 and bldg1 != bldg2:
        if core1 != core2:
            return True
    return False


def is_near_duplicate_record(rec1: dict, rec2: dict) -> bool:
    """
    Verify if two candidate records in the same source are internal duplicate filings:
    - Same core business name, or
    - High token overlap on normalized name (>= 0.88), and
    - Compatible building numbers (not clashing).
    """
    if not rec1 or not rec2:
        return False

    core1 = rec1.get("core_name", "")
    core2 = rec2.get("core_name", "")
    if core1 and core2 and core1 == core2:
        b1 = rec1.get("bldg_no", "")
        b2 = rec2.get("bldg_no", "")
        if b1 and b2 and b1 != b2:
            return False
        return True

    name1 = rec1.get("norm_name", "")
    name2 = rec2.get("norm_name", "")
    toks1 = set(name1.split())
    toks2 = set(name2.split())
    if toks1 and toks2:
        jaccard = len(toks1 & toks2) / len(toks1 | toks2)
        if jaccard >= 0.88:
            b1 = rec1.get("bldg_no", "")
            b2 = rec2.get("bldg_no", "")
            if b1 and b2 and b1 != b2:
                return False
            return True

    return False


class GlobalBipartiteMatcher:
    """
    Global Greedy Bipartite Matching Engine:
    - Enforces strict mutual disjointness: no S2 or S3 entity is assigned to more than 1 S1 entity.
    - Applies per-source cardinality cap: at most 1 primary match per source,
      plus at most verified near-identical textual duplicates.
    - Applies margin gap cutoff: if an S1 has competing ambiguous candidates, requires elite confidence.
    - Applies hard incompatibility vetoes (branch directionals, building number clash).
    """

    def __init__(
        self,
        base_threshold: float = 0.88,
        margin_cutoff: float = 0.15,
        elite_threshold: float = 0.92,
        max_matches_per_source: int = 2,
    ):
        self.base_threshold = base_threshold
        self.margin_cutoff = margin_cutoff
        self.elite_threshold = elite_threshold
        self.max_matches_per_source = max_matches_per_source

    def assign(
        self,
        candidate_edges: List[dict],
        all_required_s1_ids: Iterable[str],
        lookups: Dict[str, dict],
    ) -> Dict[str, List[str]]:
        """
        candidate_edges: list of dicts with:
            's1_id': str
            'cand_id': str
            'prob': float
            'cand_src': str ('S2' or 'S3')
            'rank': int
        """
        # Step 1: Precompute top 2 candidate probabilities per (s1_id, cand_src)
        top_probs_by_src = defaultdict(list)
        for e in candidate_edges:
            s1_id = e["s1_id"]
            c_src = e["cand_src"]
            prob = e["prob"]
            top_probs_by_src[(s1_id, c_src)].append(prob)

        margin_abstain_set = set()
        for key, p_list in top_probs_by_src.items():
            if len(p_list) >= 2:
                p_list.sort(reverse=True)
                p1, p2 = p_list[0], p_list[1]
                if (p1 - p2 < self.margin_cutoff) and p1 < self.elite_threshold:
                    margin_abstain_set.add(key)

        # Step 2: Sort all candidate edges across the entire catalog by prob descending
        candidate_edges.sort(key=lambda x: (-x["prob"], x["cand_id"]))

        assigned_cands: Dict[str, str] = {}  # cand_id -> s1_id (Strict 1-to-1 from candidate side)
        s1_matches = defaultdict(list)        # s1_id -> list of cand_ids

        for e in candidate_edges:
            prob = e["prob"]
            s1_id = e["s1_id"]
            cand_id = e["cand_id"]
            cand_src = e["cand_src"]

            # Guard 1: Candidate already claimed by a higher-probability S1
            if cand_id in assigned_cands:
                continue

            # Guard 2: Dynamic threshold by rank
            rk = e.get("rank", 1)
            req_t = self.base_threshold
            if rk > 3:
                req_t += 0.03
            if rk > 6:
                req_t += 0.03

            if prob < req_t:
                continue

            # Guard 3: Margin Gap (if ambiguous between distractors, require elite confidence)
            if (s1_id, cand_src) in margin_abstain_set and prob < self.elite_threshold:
                continue

            # Guard 4: Incompatibility Vetoes
            s1_rec = lookups.get(s1_id)
            c_rec = lookups.get(cand_id)
            if s1_rec and c_rec:
                if has_branch_conflict(s1_rec.get("norm_name", ""), c_rec.get("norm_name", "")):
                    continue
                if has_building_clash(
                    s1_rec.get("bldg_no", ""),
                    c_rec.get("bldg_no", ""),
                    s1_rec.get("core_name", ""),
                    c_rec.get("core_name", ""),
                ):
                    continue

            # Guard 5: Per-Source Cardinality Cap & Duplicate Verification
            existing_cands = s1_matches[s1_id]
            same_source_existing = [c for c in existing_cands if c.startswith(cand_src)]

            if not same_source_existing:
                # First match from this source catalog
                assigned_cands[cand_id] = s1_id
                s1_matches[s1_id].append(cand_id)
            else:
                # Secondary match in the same source:
                # MUST be a near-duplicate filing of the primary match!
                if len(same_source_existing) < self.max_matches_per_source:
                    primary_id = same_source_existing[0]
                    rec_prim = lookups.get(primary_id)
                    if rec_prim and c_rec and is_near_duplicate_record(rec_prim, c_rec):
                        assigned_cands[cand_id] = s1_id
                        s1_matches[s1_id].append(cand_id)

        # Cross-Source Triangular Verification:
        final_map: Dict[str, List[str]] = {}
        for s1_id in all_required_s1_ids:
            cands = s1_matches.get(s1_id, [])
            if len(cands) >= 2:
                s2_c = [c for c in cands if c.startswith("S2-")]
                s3_c = [c for c in cands if c.startswith("S3-")]
                if s2_c and s3_c:
                    rec2 = lookups.get(s2_c[0])
                    rec3 = lookups.get(s3_c[0])
                    if rec2 and rec3:
                        b2 = rec2.get("bldg_no", "")
                        b3 = rec3.get("bldg_no", "")
                        if b2 and b3 and b2 != b3:
                            # Drop conflicting S3 candidate
                            cands = [c for c in cands if not c.startswith("S3-")]
            final_map[s1_id] = cands

        return final_map


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
