"""
Complete End-to-End Prediction Pipeline for Business Entity Resolution (Person 3).
Amazon ML Challenge 2026.

Integrates all three roles seamlessly:
Raw Data
   ↓
Person 1 preprocessing & normalization (normalization.py)
   ↓
Person 1 blocking & candidate generation (blocking.py, candidate_generation.py)
   ↓
Person 2 feature engineering (features.py - 26 features)
   ↓
Person 2 ML model inference (LightGBM champion classifier)
   ↓
Person 3 threshold / decision logic & singleton handling (decision.py)
   ↓
Final output validation & generation (matching_results.tsv, candidate_pairs.tsv)
"""

import os
import sys
import io
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple
import joblib
import pandas as pd
import numpy as np

# Ensure path resolution
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
    )
    from person1.candidate_generation import generate_candidates
    from person1.data_loader import load_source
    from person2.features import (
        ALL_FEATURES,
        generate_feature_matrix,
        load_entity_lookups,
    )
    from person3.decision import DecisionEngine, format_submission_files
    from person3.metrics import evaluate_entity_resolution
except ImportError:
    from dataset.person1.normalization import (
        normalize_name,
        clean_core_name,
        normalize_address,
        normalize_country,
    )
    from dataset.person1.candidate_generation import generate_candidates
    from dataset.person1.data_loader import load_source
    from dataset.person2.features import (
        ALL_FEATURES,
        generate_feature_matrix,
        load_entity_lookups,
    )
    from dataset.person3.decision import DecisionEngine, format_submission_files
    from dataset.person3.metrics import evaluate_entity_resolution


class EndToEndPipeline:
    """
    Unified end-to-end Entity Resolution Pipeline orchestrating
    Person 1, Person 2, and Person 3 components.
    """

    def __init__(
        self,
        model=None,
        model_path: Optional[str] = None,
        threshold: float = 0.75,
        max_matches_per_entity: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.threshold = threshold
        self.feature_names = feature_names or ALL_FEATURES
        self.decision_engine = DecisionEngine(
            threshold=threshold, max_matches_per_entity=max_matches_per_entity
        )

        self.model = model
        if self.model is None and model_path and os.path.exists(model_path):
            payload = joblib.load(model_path)
            if isinstance(payload, dict):
                self.model = payload.get("model")
                self.feature_names = payload.get("feature_names", self.feature_names)
                if "optimal_threshold" in payload and threshold == 0.75:
                    self.threshold = payload["optimal_threshold"]
                    self.decision_engine.threshold = self.threshold
            else:
                self.model = payload

    def run(
        self,
        source1_df: pd.DataFrame,
        source2_df: pd.DataFrame,
        source3_df: pd.DataFrame,
        candidate_pairs_df: Optional[pd.DataFrame] = None,
        precomputed_predictions_df: Optional[pd.DataFrame] = None,
        output_matching_path: str = "output/matching_results.tsv",
        output_candidate_path: str = "output/candidate_pairs.tsv",
        max_block_size: int = 1000,
        max_candidates_per_query: int = 300,
    ) -> Dict[str, object]:
        """
        Execute the complete multi-stage prediction pipeline.

        Returns:
            Dictionary containing prediction summaries, output paths, and execution times.
        """
        t0 = time.time()
        print("=" * 85)
        print("PERSON 3: END-TO-END ENTITY RESOLUTION PIPELINE")
        print("=" * 85)

        all_required_s1_ids = list(source1_df["entity_id"].astype(str))
        print(f"Total required Source 1 entities: {len(all_required_s1_ids):,}")

        # -------------------------------------------------------------
        # STAGE 1 & 2: Person 1 Normalization & Candidate Generation
        # -------------------------------------------------------------
        if precomputed_predictions_df is not None:
            print("\n[Stage 1-4] Using precomputed candidate predictions...")
            pred_df = precomputed_predictions_df
            cand_pairs_df = candidate_pairs_df or pred_df[["source1_id", "candidate_id", "candidate_source"]]
        elif candidate_pairs_df is not None:
            print(f"\n[Stage 1-2] Using provided candidate set ({len(candidate_pairs_df):,} pairs)...")
            cand_pairs_df = candidate_pairs_df
            pred_df = self._extract_features_and_predict(
                source1_df, source2_df, source3_df, cand_pairs_df
            )
        else:
            print(f"\n[Stage 1-2] Person 1: Normalizing data and generating blocking index...")
            t_block = time.time()
            cand_pairs_df = generate_candidates(
                source1=source1_df,
                source2=source2_df,
                source3=source3_df,
                max_block_size=max_block_size,
                max_candidates_per_query=max_candidates_per_query,
            )
            print(f"  Generated {len(cand_pairs_df):,} candidate pairs in {time.time()-t_block:.2f}s")
            pred_df = self._extract_features_and_predict(
                source1_df, source2_df, source3_df, cand_pairs_df
            )

        # -------------------------------------------------------------
        # STAGE 5: Person 3 Decision Logic, Singleton Guard & Formatting
        # -------------------------------------------------------------
        print(f"\n[Stage 5] Person 3: Applying Decision Logic (Threshold = {self.threshold:.2f})...")
        t_dec = time.time()

        # 1. Aggregate candidate pairs
        candidates_map = self.decision_engine.aggregate_candidates(
            cand_pairs_df, all_required_s1_ids
        )

        # 2. Filter matches and handle singletons / multiple matches
        matching_map = self.decision_engine.filter_and_aggregate_matches(
            pred_df, all_required_s1_ids, custom_threshold=self.threshold
        )

        # Statistics
        matched_s1_count = sum(1 for m in matching_map.values() if len(m) > 0)
        singleton_s1_count = len(all_required_s1_ids) - matched_s1_count
        total_matched_links = sum(len(m) for m in matching_map.values())

        print(f"  Entities with >= 1 match: {matched_s1_count:,} ({matched_s1_count/len(all_required_s1_ids)*100:.2f}%)")
        print(f"  Singletons (zero match):  {singleton_s1_count:,} ({singleton_s1_count/len(all_required_s1_ids)*100:.2f}%)")
        print(f"  Total matched links:      {total_matched_links:,} (avg {total_matched_links/max(1, matched_s1_count):.2f} per matched entity)")
        print(f"  Decision logic completed in {time.time()-t_dec:.2f}s")

        # -------------------------------------------------------------
        # STAGE 6: Write Submission Files & Verify Subset Rule
        # -------------------------------------------------------------
        print(f"\n[Stage 6] Person 3: Writing final submission files...")
        n_rows, n_links = format_submission_files(
            matching_map=matching_map,
            candidates_map=candidates_map,
            required_s1_ids=all_required_s1_ids,
            matching_output_path=output_matching_path,
            candidate_output_path=output_candidate_path,
        )

        match_mb = os.path.getsize(output_matching_path) / (1024 * 1024)
        cand_mb = os.path.getsize(output_candidate_path) / (1024 * 1024)
        print(f"  Saved matching_results: {output_matching_path} ({match_mb:.2f} MB, {n_rows:,} rows)")
        print(f"  Saved candidate_pairs:  {output_candidate_path} ({cand_mb:.2f} MB, {n_rows:,} rows)")

        total_elapsed = time.time() - t0
        print("=" * 85)
        print(f"END-TO-END PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.2f}s")
        print("=" * 85)

        return {
            "matching_output_path": output_matching_path,
            "candidate_output_path": output_candidate_path,
            "total_s1_entities": len(all_required_s1_ids),
            "matched_s1_count": matched_s1_count,
            "singleton_count": singleton_s1_count,
            "total_matches": total_matched_links,
            "runtime_sec": total_elapsed,
            "matching_map": matching_map,
            "candidates_map": candidates_map,
        }

    def _extract_features_and_predict(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        cand_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract Person 2 features and predict candidate match probabilities."""
        print(f"\n[Stage 3] Person 2: Extracting 26 similarity features for {len(cand_df):,} candidate pairs...")
        t_feat = time.time()

        needed_s1 = set(cand_df["source1_id"].unique())
        needed_cands = set(cand_df["candidate_id"].unique())

        lookups = load_entity_lookups(s1_df, s2_df, s3_df, needed_s1, needed_cands)
        feat_df = generate_feature_matrix(cand_df, lookups)
        print(f"  Feature matrix constructed in {time.time()-t_feat:.2f}s")

        print(f"\n[Stage 4] Person 2: Running ML classifier inference...")
        t_inf = time.time()
        X = feat_df[self.feature_names].values

        if self.model is not None:
            probs = self.model.predict_proba(X)[:, 1]
        else:
            raise ValueError("No ML model provided or loaded for EndToEndPipeline.")

        pred_df = pd.DataFrame({
            "source1_id": feat_df["source1_id"],
            "candidate_id": feat_df["candidate_id"],
            "candidate_source": feat_df["candidate_source"],
            "match_probability": probs.round(6),
            "predicted_match": (probs >= self.threshold).astype(int),
        })
        print(f"  Probabilities predicted in {time.time()-t_inf:.2f}s")
        return pred_df
