# PERSON 2: Feature Engineering, ML Classification & Probability Prediction
**Amazon ML Challenge 2026**

---

## 1. Overview & Architecture

Person 2 is responsible for ingesting Person 1's candidate pairs, performing pairwise entity record joins, extracting 26 interpretable linguistic and structural similarity features, executing a strict group-aware train/validation split, training and tuning classical ML classifiers optimizing for $F_{0.5}$, and predicting calibrated match probabilities for every candidate pair.

```text
       candidate_pairs.tsv (from Person 1: 1,059,281 pairs)
                      ↓
  [features.py] (Lookup Cache: 288,468 pre-tokenized entities)
                      ↓
  Pairwise Feature Matrix (26 features: Name, Address, Structured)
                      ↓
  [train_model.py] (Group-Aware Split on source1_id: 80% Train / 20% Val)
                      ↓
  Baseline Models Training (Logistic Regression, Random Forest, LightGBM)
                      ↓
  [evaluate.py] (Threshold Tuning: Optimal T=0.80, Peak F0.5 = 0.9792)
                      ↓
    best_model.pkl (Serialized Champion LightGBM Model)
                      ↓
   [predict.py] (Batch Inference across 1.05M candidate pairs)
                      ↓
candidate_predictions.tsv  (For Person 3: source1_id, candidate_id, probability)
```

---

## 2. Candidate Universe & Class Imbalance Profiling

Person 2 operates strictly on Person 1's candidate-generation output without computing full Cartesian joins ($O(N \times M)$):

| Dataset Split | S1 Entities | Total Candidate Pairs | True Matches (Positives) | Hard Negative Pairs | Class Balance (% Pos) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Candidate Universe** | 4,998 | 1,059,281 | 16,129 | 1,043,152 | 1.52% |
| **Train Set (80% Group Split)** | 3,999 | 845,512 | 12,817 | 832,695 | 1.52% |
| **Validation Set (20% Group Split)**| 999 | 213,769 | 3,312 | 210,457 | 1.55% |

### Critical Validation Guarantees:
1. **Zero Entity Leakage**: Data splitting is grouped strictly by `source1_id` (Seed = 42). All candidate pairs belonging to any validation query entity are completely excluded from training.
2. **Extreme Class Imbalance (1:64)**: For every 1 true match, the candidate set contains ~64 hard negative candidates sharing name prefixes or localities.
3. **No 1-to-1 Cardinality Assumption**: Ground truth labels reflect actual 1-to-many business relationships (averaging 3.67 true targets per query entity).

---

## 3. Feature Engineering Architecture (`features.py`)

Person 2 reuses Person 1's [`normalization.py`](file:///c:/Users/ASUS/Desktop/Amazon-ML-challenge-BugHunters/dataset/person1/normalization.py) (accent stripping, legal suffixes, whitespace normalization) without alteration. Entity records are pre-tokenized into an in-memory lookup cache ($O(N)$), enabling pairwise feature generation at **49,200 pairs/second**:

### A. Name Features (11 features)
- `name_exact`: Exact match on raw business name ($1/0$).
- `name_normalized_exact`: Exact match on normalized core name ($1/0$).
- `name_levenshtein`: Normalized Levenshtein similarity via C++ RapidFuzz ($[0.0, 1.0]$).
- `name_jaccard`: Token-level Jaccard coefficient ($|T_1 \cap T_2| / |T_1 \cup T_2|$).
- `name_token_overlap`: Token containment ($|T_1 \cap T_2| / \min(|T_1|, |T_2|)$).
- `name_token_set_ratio`: Word re-ordering invariant token set ratio ($[0.0, 1.0]$).
- `name_char_3gram_jaccard`: Character 3-gram Jaccard similarity (resilient to OCR typos and leetspeak).
- `name_length_difference`: Absolute character length discrepancy ($|L_1 - L_2|$).
- `name_token_count_difference`: Absolute difference in token counts.
- `common_name_tokens`: Cardinality of shared distinctive tokens ($|T_1 \cap T_2|$).
- `name_contains_other`: Substring containment ($1/0$).

### B. Address Features (13 features)
- `address_exact`, `address_normalized_exact`: Exact match on raw and normalized addresses.
- `address_levenshtein`, `address_jaccard`, `address_token_overlap`, `address_token_set_ratio`: Address similarity metrics.
- `address_char_3gram_jaccard`: Character 3-gram similarity for addresses.
- `address_length_difference`, `address_token_count_difference`, `common_address_tokens`, `address_contains_other`: Address length and token difference metrics.
- `building_number_match`: $1.0$ if extracted house/building numbers match, $0.0$ if both exist and differ, $0.5$ if missing in either.
- `locality_match`: Match coefficient on non-numeric locality tokens.

### C. Structured Features (2 features)
- `country_match`: $1.0$ if countries match, else $0.0$.
- `candidate_source_is_s2`: $1.0$ for Source 2, $0.0$ for Source 3.

---

## 4. Measured Experimental Results

### A. Baseline Model Comparison (`model_comparison.csv`)
Evaluated on the 20% validation split (213,769 candidate pairs, 3,312 true matches) using the primary competition metric $F_{0.5}$ (where $\beta = 0.5$ weights precision twice as heavily as recall):

$$F_{0.5} = (1 + 0.25) \cdot \frac{\text{Precision} \times \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

| Model | Preprocessing | Optimal Threshold | Precision | Recall | $F_{0.5}$ (Primary) | $F_1$ | ROC-AUC | Fit Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Champion)** | None (Tree-based) | **0.80** | **0.9871** | **0.9490** | **0.9792** | **0.9677** | **0.9999** | **4.20s** |
| **HistGradientBoosting** | None (Histogram) | 0.80 | 0.9870 | 0.9423 | 0.9778 | 0.9642 | 0.9999 | 6.22s |
| **Random Forest** | Balanced subsample | 0.85 | 0.9731 | 0.9520 | 0.9688 | 0.9625 | 0.9999 | 35.97s |
| **Logistic Regression** | StandardScaler | 0.95 | 0.9443 | 0.9777 | 0.9508 | 0.9607 | 0.9998 | 3.74s |

---

### B. Probability Threshold Tuning (`threshold_results.csv`)
Sweeping probability decision thresholds on the validation set for the champion LightGBM model:

| Threshold | Precision | Recall | $F_{0.5}$ | $F_1$ | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.50 | 0.9751 | 0.9704 | 0.9742 | 0.9728 | 3,214 | 82 | 98 |
| 0.55 | 0.9768 | 0.9665 | 0.9747 | 0.9716 | 3,201 | 76 | 111 |
| 0.60 | 0.9791 | 0.9635 | 0.9760 | 0.9712 | 3,191 | 68 | 121 |
| 0.65 | 0.9806 | 0.9623 | 0.9769 | 0.9714 | 3,187 | 63 | 125 |
| 0.70 | 0.9833 | 0.9595 | 0.9784 | 0.9713 | 3,178 | 54 | 134 |
| 0.75 | 0.9851 | 0.9565 | 0.9792 | 0.9706 | 3,168 | 48 | 144 |
| **0.80\*** | **0.9871** | **0.9490** | **0.9792** | **0.9677** | **3,143** | **41** | **169** |
| 0.85 | 0.9892 | 0.9408 | 0.9791 | 0.9644 | 3,116 | 34 | 196 |
| 0.90 | 0.9922 | 0.9275 | 0.9786 | 0.9588 | 3,072 | 24 | 240 |
| 0.95 | 0.9940 | 0.9058 | 0.9750 | 0.9479 | 3,000 | 18 | 312 |

* **Optimal Threshold Choice**: **$T = 0.80$** achieves the peak $F_{0.5} = 0.9792$, suppressing false positive pairs down to only **41 out of 210,457 non-matching pairs** (Precision: 98.71%).

---

### C. Feature Ablation Experiments (`feature_experiments.csv`)
Controlled ablation isolating the contribution of each feature group:

| Experiment | Feature Group | Features | Optimal Threshold | Precision | Recall | $F_{0.5}$ | $F_1$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | Name Only | 11 | 0.55 | 0.6922 | 0.5094 | 0.6459 | 0.5869 |
| **Exp B** | Name + Address | 24 | 0.85 | 0.9778 | 0.9318 | 0.9682 | 0.9542 |
| **Exp C** | All Features (Name + Addr + Structured) | **26** | **0.80** | **0.9871** | **0.9490** | **0.9792** | **0.9677** |

* **Ablation Finding**: Name-only features fail on commercial entities ($F_{0.5} = 0.6459$) due to generic business stopword collisions (e.g. *Consultancy*, *Enterprises*). Incorporating physical address and building features yields a massive **+0.3223 boost in $F_{0.5}$**, and structured features push performance to **0.9792**.

---

## 5. Recall Decomposition & Error Analysis

### A. Blocking vs. Classification Recall Decomposition
Separating Person 1's blocking coverage from Person 2's classifier performance:

```text
Validation True Ground Truth Matches:       3,538
True Matches Present in Candidate Pairs:    3,312  (Blocking Recall:       93.61%)
True Matches Correctly Predicted by ML:     3,143  (Classification Recall: 94.90%)

Overall End-to-End System Recall:           88.84%

Failure Breakdown:
  - Blocking Failures (never reached Person 2):     226 matches (57.2% of total misses)
  - Classification Failures (predicted non-match):  169 matches (42.8% of total misses)
```

$$\text{System Recall} = \text{Blocking Recall} \times \text{Classification Recall} = 93.61\% \times 94.90\% = 88.84\%$$

---

### B. Diagnostic Error Analysis (`false_positives.csv` & `false_negatives.csv`)

#### False Positive Patterns (Predicted Match, Ground Truth = 0):
1. **Similar Business Name, Different Street / Building Number (58%)**:
   - Example: *Chapa Mountain Fuel LLC* (`2798 K-V Road`) vs *Chapa Mountain Fuel North LLC* (`809 K-V Road`) (Prob: 0.9973). Same street and corporate entity prefix, but distinct branch or franchise locations.
2. **Common Corporate Entity Collision (27%)**:
   - Example: *All Investment Private Limited* vs *-- All Investments* (Prob: 0.9986). High token overlap on generic commercial words across different cities.
3. **Identical Name, Different City / State (15%)**:
   - National service providers with duplicate names in different administrative districts.

#### False Negative Patterns (Predicted Non-Match, Ground Truth = 1):
1. **Candidate Address Missing (62%)**:
   - Example: *Southern Technologies Private Limited* vs *Southern ടെക്നോളജീസ് പ്രൈവറ്റ് ലിമിറ്റഡ്* (Candidate address is `NaN`, Prob: 0.0000). When candidate address is missing, address similarity features drop to zero.
2. **Indic Script / Multilingual Script Mismatch (24%)**:
   - Candidate trade names rendered in Malayalam, Tamil, or Devanagari paired with Latin query names without address overlaps.
3. **Severe Spelling Distortions & Unlisted Aliases (14%)**:
   - Example: *Varun Services LLP* vs *Iriaria* at the same physical address (Prob: 0.0016).

---

## 6. Reusable Integration API (For Person 3)

Person 3 can load the trained model and run predictions programmatically:

```python
import joblib
import pandas as pd
from person2.features import generate_feature_matrix, load_entity_lookups

# 1. Load serialized champion model and optimal threshold
payload = joblib.load("person2/models/best_model.pkl")
model = payload["model"]
threshold = payload["optimal_threshold"]  # 0.80
feature_names = payload["feature_names"]

# 2. Load entity tables and build lookups
s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
s2_df = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3_df = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
cand_df = pd.read_csv("dataset/candidate_pairs.tsv", sep="\t")

lookups = load_entity_lookups(s1_df, s2_df, s3_df)

# 3. Generate feature matrix and predict match probabilities
features_df = generate_feature_matrix(cand_df, lookups)
match_probs = model.predict_proba(features_df[feature_names].values)[:, 1]

# 4. Attach probabilities
cand_df["match_probability"] = match_probs.round(6)
cand_df["predicted_match"] = (match_probs >= threshold).astype(int)
```

Or via CLI:

```bash
python -m person2.predict \
  --candidates dataset/candidate_pairs.tsv \
  --source1 dataset/train/train_source1.tsv \
  --source2 dataset/train/train_source2.tsv \
  --source3 dataset/train/train_source3.tsv \
  --model person2/models/best_model.pkl \
  --output person2/predictions/candidate_predictions.tsv \
  --threshold 0.80
```

---

## 7. Generated Deliverables

1. **`models/best_model.pkl`**: Serialized champion LightGBM model, feature list, and calibrated optimal threshold ($T = 0.80$).
2. **`predictions/candidate_predictions.tsv`**: Full predictions table for all 1,059,281 candidate pairs with columns: `['source1_id', 'candidate_id', 'candidate_source', 'match_probability', 'predicted_match']`.
3. **`results/model_comparison.csv`**: Comprehensive 4-model validation benchmark (Logistic Regression, Random Forest, HistGradientBoosting, LightGBM).
4. **`results/threshold_results.csv`**: Complete probability threshold curve ($0.50$ to $0.95$) with Precision, Recall, $F_1$, and $F_{0.5}$.
5. **`results/feature_experiments.csv`**: Feature ablation results comparing Name-only, Name+Address, and Full feature sets.
6. **`results/false_positives.csv`**: Categorized diagnosis of top false positive candidate pairs.
7. **`results/false_negatives.csv`**: Categorized diagnosis of top false negative true pairs.
8. **`results/validation_results.json`**: Machine-readable metadata and recall decomposition statistics.
9. **`person2/` Package**:
   - `features.py`: Ultra-fast (49k pairs/s) pre-tokenized similarity engine (26 features).
   - `train_model.py`: Group-aware split, training, and evaluation pipeline.
   - `evaluate.py`: $F_{0.5}$ metric, threshold tuner, and diagnostic error analyzer.
   - `predict.py`: Batch probability prediction runner.
