# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** BugHunters  
**Team Members:** Person 1 (Data & Blocking), Person 2 (Features & ML), Person 3 (Evaluation & Pipeline)  
**Submission Date:** September 26, 2026  

---

## 1. Executive Summary

We present an end-to-end Machine Learning pipeline for multi-source Business Entity Resolution that links noisy, fragmented business records across three heterogeneous data sources into canonical entities. Our solution couples a multi-strategy inverted blocking index (retaining **93.78% candidate recall** with **>99.991% search space reduction**) with a high-throughput 26-dimensional pairwise similarity engine (processing **49,200 pairs/sec**) and a champion **LightGBM gradient-boosted classifier**. Evaluated under strict group-aware holdout validation, our calibrated decision thresholding ($T = 0.75$) achieves a peak **Macro $F_{0.5}$ of 0.9491** (Precision: **97.63%**, Recall: **90.26%**, Singleton Accuracy: **94.44%**), fully compliant with the competition submission validator.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across 12.5M training records and 11.7M test records, we identified critical domain patterns:
1. **Zero Cross-Country Matching Invariance**: Across all 7,638,365 ground-truth pairs, exactly **0 cross-country matches exist**. Enforcing a strict hard partition by `country` eliminates over 60% of pairwise comparisons with zero recall penalty, even when generalizing to unseen test countries (e.g., France).
2. **Noise and Typographical Variations**:
   - *Business Names*: Pervasive legal suffixes (`Pvt Ltd`, `LLC`, `Corp`, `Inc`, `LLP`), trade aliases (`aka`, `dba`, `formerly known as`), domain extensions (`.com`, `.in`), and character substitutions/OCR noise.
   - *Addresses*: Inconsistent formatting (Street vs St, Road vs Rd), missing postal codes or states, landmark-based Indian addresses (`Near SBI ATM`), and alphanumeric building/plot descriptors (`D-062`, `WZ-187C`, `Plot 126/4`).
3. **Cardinality Spectrum & Singletons**:
   - Non-singleton Source 1 entities match an average of 3.66 records across Source 2 and Source 3 (ranging up to 11 records).
   - Approximately **5.7% of Source 1 entities are singletons** (no match in Source 2 or 3). Because the official macro-averaged metric awards **1.0** for correctly predicting empty and penalizes false merges on singletons to **0.0**, guarding against low-confidence spurious merges is paramount.
4. **Multilingual & Indic Scripts**: Source 2 and Source 3 contain regional Indian script names (Devanagari, Tamil, Telugu) paired with Latin script addresses.

### 2.2 Solution Strategy
Our architecture follows a disciplined three-stage funnel:

```text
Raw Data (S1, S2, S3) 
   → Person 1: Text Normalization & Multi-Key Blocking (93.78% recall, >99.99% reduction)
   → Person 2: Pairwise Feature Engineering (26 features) & LightGBM Inference
   → Person 3: Calibrated Decision Engine (T = 0.75) & Singleton Guard (Macro F0.5 = 0.9491)
   → Validated Output TSVs (matching_results.tsv, candidate_pairs.tsv)
```

**Approach Type:** Hybrid Multi-Index Inverted Blocking + Pre-tokenized Pairwise Feature Engineering + Gradient Boosted Decision Trees + Calibrated Singleton/Multi-Match Decision Engine.  
**Core Innovation:** Decoupling high-recall candidate generation (acronym indexing `TOK3`, compound locality matching `ADDR_LOC`) from high-precision probability calibration, paired with an explicit singleton guard that maximizes macro-averaged $F_{0.5}$ by penalizing false merges 2× over missed links.

---

## 3. Candidate Generation (Blocking)

Person 1 implemented an inverted indexing architecture designed to maximize recall while maintaining sub-linear comparison scaling:

- **Blocking Keys Used:**
  1. `NP4` / `NP6`: 4-character and 6-character alphanumeric name prefixes on core normalized names.
  2. `TOK`: Distinctive name tokens ($\ge 4$ characters) excluding a curated list of high-frequency corporate stopwords.
  3. `TOK3`: 3-letter acronym tokens (`ONP`, `PUN`, `TXO`) recovering compact corporate entities.
  4. `NSORT`: Permutation-invariant token keys (sorting top 4 significant tokens) resolving word-order transpositions.
  5. `NCOMP`: Fully compressed, space-stripped name strings resolving domain names and concatenation variations.
  6. `ADDR`: Building/plot number combined with first alphabetic street token.
  7. `UNIT` / `PLOT`: Alphanumeric sub-unit identifiers (`D-062`, `AF-684`) and slash-plot numbers (`126/4`).
  8. `ADDR_LOC`: Non-numeric locality pair keys linking records sharing neighborhood and landmark tokens.
- **Candidate Pairs Generated:**
  - Evaluated on the 5,000 S1 benchmark: generated **1,059,281 candidate pairs** (compact average of **188.5 candidates per S1 entity**).
  - Search space reduction exceeds **99.991%**, eliminating over 10.65 billion negative Cartesian pairs.
- **Recall Preservation:**
  - Standard baseline recall of 90.01% was boosted to **93.78% candidate recall** (+3.77% absolute gain, **+752 true matches recovered**), verified via diagnostic error analysis (`blocking_error_analysis.csv`).

---

## 4. Matching Model

Person 2 engineered an ultra-fast in-memory feature extraction pipeline and evaluated four machine learning classifier families:

### Features Used (26 Features across 3 Groups)
1. **Name Similarity Features (11 features):**
   - Exact string equality (`name_exact`) and normalized core equality (`name_normalized_exact`).
   - C++ RapidFuzz normalized Levenshtein distance (`name_levenshtein`).
   - Token-level Jaccard similarity (`name_jaccard`) and token containment ratio (`name_token_overlap`).
   - Word-order invariant token set ratio (`name_token_set_ratio`).
   - Character 3-gram Jaccard coefficient (`name_char_3gram_jaccard`) for OCR/leetspeak robustness.
   - Character length difference, token count difference, common token count, and substring containment.
2. **Address Similarity Features (13 features):**
   - Raw and normalized address exact matches (`address_exact`, `address_normalized_exact`).
   - Normalized Levenshtein distance (`address_levenshtein`).
   - Address token Jaccard, token overlap, token set ratio, and character 3-gram Jaccard.
   - Address length difference, token count difference, common address tokens, and substring containment.
   - Structured building number match (`building_number_match`: $1.0$ match, $0.0$ mismatch, $0.5$ missing).
   - Locality token overlap coefficient (`locality_match`).
3. **Structured Features (2 features):**
   - Country match indicator (`country_match`).
   - Candidate source origin indicator (`candidate_source_is_s2`).

### Model Type & Benchmark Comparison
Models were evaluated under group-aware splitting on `source1_id` (Seed = 42):

| Model Architecture | Preprocessing | Optimal Threshold | Precision | Recall | Pairwise $F_{0.5}$ | Fit Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Champion)** | None (Tree-based) | **0.80** | **0.9871** | **0.9490** | **0.9792** | **4.20s** |
| HistGradientBoosting | None (Histogram) | 0.80 | 0.9870 | 0.9423 | 0.9778 | 6.22s |
| Random Forest | Balanced subsample | 0.85 | 0.9731 | 0.9520 | 0.9688 | 35.97s |
| Logistic Regression | StandardScaler | 0.95 | 0.9443 | 0.9777 | 0.9508 | 3.74s |

**Champion Selection:** LightGBM achieved the top pairwise $F_{0.5}$ (0.9792) with rapid 4.2-second training, outperforming linear models and random forests.

### Threshold Selection & Decision Engine
Person 3 tuned decision thresholds to maximize the competition **Macro-averaged $F_{0.5}$** across all entities, incorporating singletons. Sweeping $T \in [0.40, 0.95]$ revealed **$T = 0.75$** as the global optimum on both 90/10 and 80/20 splits.

---

## 5. Results & Error Analysis

### 5.1 Validation Results (Macro $F_{0.5}$)

Evaluated across strict group-aware splits on `source1_id`:

| Evaluation Split | Evaluated S1 Queries | Optimal Threshold | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **90% Train / 10% Val (Challenge Recommended)** | **499** | **0.75** | **0.9491** | **0.9763** | **0.9026** | **94.44%** |
| **80% Train / 20% Val (Standard Holdout)** | **999** | **0.75** | **0.9481** | **0.9754** | **0.8999** | **93.44%** |

- **Macro $F_{0.5}$:** **0.9491** (primary evaluation metric).
- **Macro Precision:** **97.63%** (low false merge rate).
- **Singleton Accuracy:** **94.44%** (34 of 36 singletons correctly predicted empty).

### 5.2 Error Diagnostics
1. **Common False Positives (Wrong Merges):**
   - *Corporate Branch Collisions*: Identical franchise names sharing street names but having different building numbers (e.g., *Chapa Mountain Fuel LLC* at `2798 K-V Road` vs *Chapa Mountain Fuel North LLC* at `809 K-V Road`). Mitigated by penalizing non-matching `building_number_match`.
   - *Generic Name Overlap*: Businesses sharing common generic words (*All Investment Private Limited* vs *-- All Investments*). Suppressed by high threshold $T = 0.75$.
2. **Common False Negatives (Missed Matches):**
   - *Candidate Address Missing*: ~62% of false negatives have `NaN` address in Source 2/3.
   - *Indic / Regional Script Names*: Trade names written in Devanagari or Malayalam paired with Latin query names without address numbers.

---

## 6. Conclusion

By unifying Person 1's high-recall inverted blocking, Person 2's rapid 26-feature similarity matrix and champion LightGBM classifier, and Person 3's calibrated macro-$F_{0.5}$ decision engine and singleton guard, Team BugHunters achieved **0.9491 Macro $F_{0.5}$** with **97.63% precision**. The end-to-end pipeline processes over 1.73 million test entities, handles 0, 1, or multiple matches deterministically, and passes the official `utils/validate_submission.py` submission check with zero errors.

---

## Appendix

### A. Code Artefacts
All reproducible code is organized inside `code/business_entity_resolution/`:
- `src/person1/`: Preprocessing, normalization, and candidate generation.
- `src/person2/`: Feature engineering, model training, and probability prediction.
- `src/person3/`: Validation system, macro metrics, decision engine, and pipeline orchestrator.
- `src/utils/validate_submission.py`: Submission validator tool.
- Entry points:
  - Validation: `python -m src.person3.run_pipeline --mode val`
  - Final Output Generation: `python -m src.person3.run_pipeline --mode test --threshold 0.75`

### B. Additional Results

#### Feature Ablation Experiments (`feature_experiments.csv`)
| Feature Set | Features | Optimal Threshold | Precision | Recall | $F_{0.5}$ | $F_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Name Only | 11 | 0.55 | 0.6922 | 0.5094 | 0.6459 | 0.5869 |
| Name + Address | 24 | 0.85 | 0.9778 | 0.9318 | 0.9682 | 0.9542 |
| **All (Name + Address + Structured)** | **26** | **0.80** | **0.9871** | **0.9490** | **0.9792** | **0.9677** |

Incorporating physical address features yields a **+0.3223 boost in $F_{0.5}$** over name-only matching, proving that physical location features are essential for disambiguating commercial businesses.
