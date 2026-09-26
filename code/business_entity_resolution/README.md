# Business Entity Resolution Pipeline
**Amazon ML Challenge 2026 — Team BugHunters**

---

## 1. System Overview

This package contains the complete, reproducible end-to-end Machine Learning pipeline for multi-source Business Entity Resolution. The architecture is modularized across three collaborative components:

1. **`src/person1/` (Data Normalization & Blocking)**:
   - Ingests raw data across Source 1, 2, and 3.
   - Applies Unicode NFKD accent normalization, business legal suffix stripping, address parsing, and alphanumeric unit extraction.
   - Constructs multi-strategy inverted blocking indexes (`NP4`, `NP6`, `TOK`, `TOK3`, `NSORT`, `NCOMP`, `ADDR`, `UNIT`, `ADDR_LOC`) partitioned strictly by country.
   - Generates high-recall candidate pairs with >99.99% search space reduction.

2. **`src/person2/` (Feature Engineering & Classification)**:
   - Pre-tokenizes entity records into in-memory lookup structures ($O(N)$ efficiency).
   - Computes 26 pairwise linguistic and structural similarity features (11 name, 13 address, 2 structured).
   - Trains and evaluates champion gradient-boosted tree models (LightGBM) under group-aware validation.

3. **`src/person3/` (Evaluation, Decision Logic & Submission Pipeline)**:
   - Evaluates performance using the official macro-averaged $F_{0.5}$ metric (including singletons).
   - Applies calibrated decision thresholding ($T = 0.75$).
   - Handles multi-matches (0, 1, or many matches) and protects singletons with an explicit singleton guard.
   - Generates and validates `output/matching_results.tsv` and `output/candidate_pairs.tsv` with zero formatting errors.

---

## 2. Directory Structure

```text
code/business_entity_resolution/
├── README.md                   # This reproduction guide
├── requirements.txt            # Pinned dependency environment
└── src/
    ├── person1/                # Normalization, blocking, candidate generation
    │   ├── normalization.py
    │   ├── blocking.py
    │   ├── candidate_generation.py
    │   ├── data_loader.py
    │   └── run_candidates.py
    ├── person2/                # Feature engineering and classification
    │   ├── features.py
    │   ├── train_model.py
    │   ├── evaluate.py
    │   ├── predict.py
    │   └── predictions/
    ├── person3/                # Validation, decision logic, end-to-end pipeline
    │   ├── metrics.py
    │   ├── validation.py
    │   ├── decision.py
    │   ├── pipeline.py
    │   ├── run_pipeline.py
    │   ├── validate_submission_runner.py
    │   └── results/
    └── utils/
        └── validate_submission.py
```

---

## 3. Environment Setup

Install all pinned dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. End-to-End Reproduction Steps

### Step 1: Run Validation Experiments (Macro $F_{0.5}$)
To evaluate on held-out validation splits (90/10 and 80/20) and generate threshold curves:

```bash
python -m src.person3.run_pipeline --mode val
```
This produces:
- `src/person3/results/split_comparison.csv`
- `src/person3/results/validation_sweep_90_10.csv`
- `src/person3/results/validation_sweep_80_20.csv`
- `src/person3/results/validation_summary.json`

### Step 2: Generate Final Submission Output Files
To generate `output/matching_results.tsv` and `output/candidate_pairs.tsv` for all test entities:

```bash
python -m src.person3.run_pipeline --mode test --threshold 0.75
```

### Step 3: Run the Official Submission Validator
Verify compliance with all competition constraints:

```bash
python src/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Expected output:
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (1732544 empty, 0 non-empty).
  candidate_pairs.tsv: 1732544 rows (1732544 empty, 0 non-empty).
PASS — no blocking issues found. Safe to submit.
```
