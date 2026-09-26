"""
Person 3 Package: Evaluation, Decision Logic & Final Prediction Pipeline.
Amazon ML Challenge 2026.

Modules:
- metrics: Official Macro-averaged F0.5, Precision, Recall, and Singleton metrics.
- validation: Group-aware train/val splitting and threshold sweeping.
- decision: Multi-match handling, singleton logic, thresholding, and TSV formatting.
- pipeline: Complete end-to-end inference pipeline connecting Person 1, Person 2, and Person 3.
- validate_submission_runner: Wrapper and auditor for the official submission validator.
"""

from .metrics import calculate_entity_metrics, evaluate_entity_resolution
from .decision import DecisionEngine, format_submission_files
from .validation import ValidationSystem, run_validation_experiments
from .pipeline import EndToEndPipeline

__all__ = [
    "calculate_entity_metrics",
    "evaluate_entity_resolution",
    "DecisionEngine",
    "format_submission_files",
    "ValidationSystem",
    "run_validation_experiments",
    "EndToEndPipeline",
]
