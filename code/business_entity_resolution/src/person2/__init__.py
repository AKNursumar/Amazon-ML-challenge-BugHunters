"""
Person 2: Candidate-Pair Feature Engineering, ML Training, Evaluation, and Prediction.
Amazon ML Challenge 2026.
"""

from .features import (
    NAME_FEATURES,
    ADDRESS_FEATURES,
    STRUCTURED_FEATURES,
    ALL_FEATURES,
    generate_feature_matrix,
    load_entity_lookups,
)
from .evaluate import (
    calculate_metrics,
    calculate_f_beta,
    tune_threshold,
    evaluate_blocking_vs_classification,
)

__all__ = [
    "NAME_FEATURES",
    "ADDRESS_FEATURES",
    "STRUCTURED_FEATURES",
    "ALL_FEATURES",
    "generate_feature_matrix",
    "load_entity_lookups",
    "calculate_metrics",
    "calculate_f_beta",
    "tune_threshold",
    "evaluate_blocking_vs_classification",
]
