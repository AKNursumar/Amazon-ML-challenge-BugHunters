"""
Person 1: Data Loading, Normalization, Blocking, and Candidate Generation.
Amazon ML Challenge 2026.
"""

from .data_loader import load_source, load_ground_truth, get_true_pairs
from .normalization import normalize_name, normalize_address, normalize_country, clean_core_name
from .blocking import BlockingIndex, generate_blocking_keys
from .candidate_generation import generate_candidates

__all__ = [
    "load_source",
    "load_ground_truth",
    "get_true_pairs",
    "normalize_name",
    "normalize_address",
    "normalize_country",
    "clean_core_name",
    "BlockingIndex",
    "generate_blocking_keys",
    "generate_candidates",
]
