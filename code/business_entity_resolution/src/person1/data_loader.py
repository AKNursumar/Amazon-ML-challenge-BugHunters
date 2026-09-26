"""
Data Loader Module for Amazon ML Challenge 2026.
Handles robust loading and parsing of TSV data files for Sources 1, 2, 3, and Ground Truth.
"""

import os
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd


def load_source(
    file_path: str,
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Safely load a Source TSV file (source1, source2, or source3).

    Args:
        file_path: Path to the TSV file.
        nrows: Optional maximum number of rows to read.
        usecols: Optional list of column names to load.

    Returns:
        pd.DataFrame with string types and missing values filled or preserved safely.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    dtype_spec = {
        "entity_id": "string",
        "business_name": "string",
        "business_address": "string",
        "country": "string",
    }
    if usecols:
        dtype_spec = {k: v for k, v in dtype_spec.items() if k in usecols}

    df = pd.read_csv(
        file_path,
        sep="\t",
        nrows=nrows,
        usecols=usecols,
        dtype=dtype_spec,
        keep_default_na=True,
        na_values=["", "null", "NULL", "None", "NaN", "nan"],
    )

    # Fill NA for string operations safely
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df


def load_ground_truth(
    file_path: str,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Safely load the ground truth TSV file.

    Args:
        file_path: Path to train_ground_truth.tsv.
        nrows: Optional maximum number of rows to read.

    Returns:
        pd.DataFrame with source1_entity_id and matched_entity_ids.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        nrows=nrows,
        dtype={"source1_entity_id": "string", "matched_entity_ids": "string"},
    )
    return df


def get_true_pairs(ground_truth_df: pd.DataFrame) -> Set[Tuple[str, str]]:
    """
    Convert ground truth DataFrame into a set of (source1_id, candidate_id) tuples.

    Args:
        ground_truth_df: DataFrame with source1_entity_id and matched_entity_ids.

    Returns:
        Set of (s1_id, cand_id) tuples representing all true positive pairs.
    """
    true_pairs = set()
    valid_df = ground_truth_df.dropna(subset=["matched_entity_ids"])

    for s1_id, matched_str in zip(valid_df["source1_entity_id"], valid_df["matched_entity_ids"]):
        if not matched_str or pd.isna(matched_str):
            continue
        targets = str(matched_str).split(",")
        for t in targets:
            clean_t = t.strip()
            if clean_t:
                true_pairs.add((s1_id, clean_t))

    return true_pairs
