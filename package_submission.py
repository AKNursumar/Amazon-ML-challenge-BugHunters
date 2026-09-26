"""
Submission Packaging Script for Amazon ML Challenge 2026.
Team: BugHunters

Automatically builds the official final submission zip archive:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches
│   └── candidate_pairs.tsv         # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all python source code (person1, person2, person3, utils)
│       ├── README.md               # reproduction guide
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # completed methodology document
"""

import argparse
import os
import sys
import zipfile
import time

cur_dir = os.path.dirname(os.path.abspath(__file__))

REQUIREMENTS_CONTENT = """# ML Challenge 2026 - Pinned Dependencies
# Team: BugHunters

numpy>=2.0.0
pandas>=2.2.0
scipy>=1.15.0
scikit-learn>=1.6.0
lightgbm>=4.7.0
rapidfuzz>=3.14.0
joblib>=1.4.0
tqdm>=4.67.0
"""

README_CONTENT = """# Business Entity Resolution Pipeline
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
    ├── person2/                # Feature engineering and classification
    ├── person3/                # Validation, decision logic, end-to-end pipeline
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
```bash
python -m src.person3.run_pipeline --mode val
```

### Step 2: Generate Final Submission Output Files
```bash
python -m src.person3.run_pipeline --mode test --threshold 0.75
```

### Step 3: Run the Official Submission Validator
```bash
python src/utils/validate_submission.py \\
    --matching output/matching_results.tsv \\
    --candidate output/candidate_pairs.tsv \\
    --test-dir dataset/test
```
"""


def should_include_file(filename: str, rel_path: str) -> bool:
    """Filter out scratch files, caches, and large binary dumps."""
    ignored_patterns = [
        "__pycache__",
        ".pyc",
        ".DS_Store",
        "Thumbs.db",
        ".git",
        ".vscode",
        ".idea",
        "scratch",
        ".log",
    ]
    for pattern in ignored_patterns:
        if pattern in rel_path or pattern in filename:
            return False
    return True


def build_submission_zip(
    team_name: str = "BugHunters",
    output_zip_path: str = None,
):
    t0 = time.time()
    if output_zip_path is None:
        output_zip_path = os.path.join(cur_dir, f"{team_name}_submission.zip")

    print("=" * 80)
    print(f"BUILDING FINAL SUBMISSION ARCHIVE: {os.path.basename(output_zip_path)}")
    print("=" * 80)

    # Required source files
    matching_tsv = os.path.join(cur_dir, "output", "matching_results.tsv")
    if not os.path.isfile(matching_tsv):
        matching_tsv = os.path.join(cur_dir, "dataset", "output", "matching_results.tsv")

    candidate_tsv = os.path.join(cur_dir, "output", "candidate_pairs.tsv")
    if not os.path.isfile(candidate_tsv):
        candidate_tsv = os.path.join(cur_dir, "dataset", "output", "candidate_pairs.tsv")

    doc_template = os.path.join(cur_dir, "Documentation_template.md")

    # Validate that prerequisites exist
    for path, name in [
        (matching_tsv, "output/matching_results.tsv"),
        (candidate_tsv, "output/candidate_pairs.tsv"),
        (doc_template, "Documentation_template.md"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required artifact: {name} (expected at {path})")

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output folder
        print("[1/4] Packing output files...")
        zf.write(matching_tsv, "output/matching_results.tsv")
        zf.write(candidate_tsv, "output/candidate_pairs.tsv")

        # 2. Documentation template
        print("[2/4] Packing documentation template...")
        zf.write(doc_template, "Documentation_template.md")

        # 3. Code folder metadata
        print("[3/4] Packing code requirements & README...")
        zf.writestr("code/business_entity_resolution/README.md", README_CONTENT.strip() + "\n")
        zf.writestr("code/business_entity_resolution/requirements.txt", REQUIREMENTS_CONTENT.strip() + "\n")

        # 4. Source code packages
        print("[4/4] Packing source packages (person1, person2, person3, utils)...")
        source_dirs = [
            ("dataset/person1", "code/business_entity_resolution/src/person1"),
            ("dataset/person2", "code/business_entity_resolution/src/person2"),
            ("dataset/person3", "code/business_entity_resolution/src/person3"),
            ("utils", "code/business_entity_resolution/src/utils"),
        ]

        packed_count = 0
        for src_rel, zip_prefix in source_dirs:
            full_src = os.path.join(cur_dir, src_rel)
            if not os.path.isdir(full_src):
                continue
            for root, _, files in os.walk(full_src):
                for f in files:
                    full_path = os.path.join(root, f)
                    rel_to_src = os.path.relpath(full_path, full_src)
                    if should_include_file(f, rel_to_src):
                        arcname = os.path.join(zip_prefix, rel_to_src).replace("\\", "/")
                        zf.write(full_path, arcname)
                        packed_count += 1

    zip_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print("=" * 80)
    print(f"SUCCESS: Created {output_zip_path} ({zip_mb:.2f} MB)")
    print(f"Total files packaged: {packed_count + 4}")
    print(f"Build time: {time.time() - t0:.2f}s")
    print("=" * 80)

    # Print zip table of contents summary
    print("\nPackage Structure Verification:")
    with zipfile.ZipFile(output_zip_path, "r") as zf:
        top_dirs = sorted(list(set(n.split("/")[0] for n in zf.namelist())))
        print(f"  Top-level entries: {top_dirs}")
        code_entries = [n for n in zf.namelist() if n.startswith("code/")]
        print(f"  Code files count: {len(code_entries)}")
        output_entries = [n for n in zf.namelist() if n.startswith("output/")]
        print(f"  Output files: {output_entries}")


def main():
    parser = argparse.ArgumentParser(description="Package submission zip for Amazon ML Challenge 2026")
    parser.add_argument("--team-name", default="BugHunters", help="Team name for zip prefix")
    parser.add_argument("--output", default=None, help="Custom output zip path")
    args = parser.parse_args()

    build_submission_zip(team_name=args.team_name, output_zip_path=args.output)


if __name__ == "__main__":
    main()
