"""
Training, Validation, and Model Comparison Pipeline for Entity Resolution (Person 2).
Amazon ML Challenge 2026.

Orchestrates:
1. Data loading and feature generation
2. Group-aware train/validation split (grouped by source1_id)
3. Multi-model baseline training (Logistic Regression, Random Forest, LightGBM, HistGradientBoosting)
4. Probability threshold tuning for F0.5
5. Feature ablation experiments (Name only, Name+Address, All)
6. False positive & false negative error analysis
7. Model selection and artifact serialization (best_model.pkl)
"""

import os
import sys
import io
import time
import json
import joblib
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# Ensure paths
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.dirname(cur_dir)
repo_root = os.path.dirname(dataset_dir)
for p in [cur_dir, dataset_dir, repo_root]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from person1.data_loader import get_true_pairs, load_ground_truth
    from person2.features import (
        ALL_FEATURES,
        NAME_FEATURES,
        ADDRESS_FEATURES,
        STRUCTURED_FEATURES,
        generate_feature_matrix,
        load_entity_lookups,
    )
    from person2.evaluate import (
        calculate_metrics,
        tune_threshold,
        evaluate_blocking_vs_classification,
        analyze_errors,
    )
except ImportError:
    from dataset.person1.data_loader import get_true_pairs, load_ground_truth
    from dataset.person2.features import (
        ALL_FEATURES,
        NAME_FEATURES,
        ADDRESS_FEATURES,
        STRUCTURED_FEATURES,
        generate_feature_matrix,
        load_entity_lookups,
    )
    from dataset.person2.evaluate import (
        calculate_metrics,
        tune_threshold,
        evaluate_blocking_vs_classification,
        analyze_errors,
    )

def resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    if os.path.exists(os.path.join(dataset_dir, p)):
        return os.path.join(dataset_dir, p)
    if os.path.exists(os.path.join(repo_root, p)):
        return os.path.join(repo_root, p)
    if p.startswith("dataset/") and os.path.exists(p[8:]):
        return p[8:]
    return p


def run_pipeline(
    candidate_pairs_path: str = "dataset/candidate_pairs.tsv",
    ground_truth_path: str = "dataset/train/train_ground_truth.tsv",
    source1_path: str = "dataset/train/train_source1.tsv",
    source2_path: str = "dataset/train/train_source2.tsv",
    source3_path: str = "dataset/train/train_source3.tsv",
    output_dir: str = "person2",
    val_ratio: float = 0.20,
    random_seed: int = 42,
):
    print("=" * 85)
    print("PERSON 2: CANDIDATE-PAIR CLASSIFICATION & ML TRAINING PIPELINE")
    print("=" * 85)
    t_start = time.time()

    candidate_pairs_path = resolve_path(candidate_pairs_path)
    ground_truth_path = resolve_path(ground_truth_path)
    source1_path = resolve_path(source1_path)
    source2_path = resolve_path(source2_path)
    source3_path = resolve_path(source3_path)
    output_dir = resolve_path(output_dir)

    models_dir = os.path.join(output_dir, "models")
    results_dir = os.path.join(output_dir, "results")
    predictions_dir = os.path.join(output_dir, "predictions")
    for d in [models_dir, results_dir, predictions_dir]:
        os.makedirs(d, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1 & 2: Load Candidate Pairs and Ground Truth
    # ---------------------------------------------------------
    print("\n[Step 1/8] Ingesting candidate pairs and ground truth labels...")
    cand_df = pd.read_csv(candidate_pairs_path, sep="\t")
    print(f"  Loaded candidate pairs: {len(cand_df):,} rows")

    # Load ground truth
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str).fillna("")
    all_true_pairs = get_true_pairs(gt_df)
    print(f"  Loaded full ground truth true pairs: {len(all_true_pairs):,}")

    # Restrict ground truth to S1 entities in candidate_pairs
    unique_s1_in_cand = set(cand_df["source1_id"].unique())
    s1_gt_pairs = {p for p in all_true_pairs if p[0] in unique_s1_in_cand}
    print(f"  Ground truth true pairs for candidate universe: {len(s1_gt_pairs):,}")

    cand_pairs_set = set(zip(cand_df["source1_id"], cand_df["candidate_id"]))
    hits_in_cand = len(cand_pairs_set.intersection(s1_gt_pairs))
    blocking_rec = hits_in_cand / max(1, len(s1_gt_pairs))
    print(f"  True matches present in candidate_pairs.tsv: {hits_in_cand:,} ({blocking_rec*100:.2f}% blocking recall)")
    print(f"  Blocking failures (missed by blocking): {len(s1_gt_pairs) - hits_in_cand:,}")

    # ---------------------------------------------------------
    # STEP 3 & 4: Load Entity Lookups and Extract Features
    # ---------------------------------------------------------
    print("\n[Step 2/8] Loading entity lookups for candidate pairs...")
    t_feat_start = time.time()
    unique_cands = set(cand_df["candidate_id"].unique())
    print(f"  Unique S1 queries: {len(unique_s1_in_cand):,} | Unique candidates: {len(unique_cands):,}")

    # Load only necessary records from sources
    s1_df = pd.read_csv(source1_path, sep="\t", dtype=str).fillna("")
    s1_df = s1_df[s1_df["entity_id"].isin(unique_s1_in_cand)]

    # Stream S2 and S3 to extract required candidates
    s2_chunks = []
    for chunk in pd.read_csv(source2_path, sep="\t", chunksize=300_000, dtype=str, keep_default_na=False):
        m = chunk["entity_id"].isin(unique_cands)
        if m.any():
            s2_chunks.append(chunk[m])
    s2_df = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    s3_chunks = []
    for chunk in pd.read_csv(source3_path, sep="\t", chunksize=300_000, dtype=str, keep_default_na=False):
        m = chunk["entity_id"].isin(unique_cands)
        if m.any():
            s3_chunks.append(chunk[m])
    s3_df = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    print(f"  Retrieved {len(s1_df):,} S1, {len(s2_df):,} S2, {len(s3_df):,} S3 records.")
    entity_lookups = load_entity_lookups(s1_df, s2_df, s3_df, unique_s1_in_cand, unique_cands)
    print(f"  Preprocessed lookup table built ({len(entity_lookups):,} entities) in {time.time()-t_feat_start:.2f}s")

    print("\n[Step 3/8] Generating pairwise similarity features and labels...")
    t_mat = time.time()
    feature_matrix_df = generate_feature_matrix(cand_df, entity_lookups, ground_truth_pairs=s1_gt_pairs)
    print(f"  Feature matrix built: {feature_matrix_df.shape} in {time.time()-t_mat:.2f}s")
    pos_count = feature_matrix_df["label"].sum()
    neg_count = len(feature_matrix_df) - pos_count
    print(f"  Class balance: Positive matches = {pos_count:,} ({pos_count/len(feature_matrix_df)*100:.2f}%), Negative pairs = {neg_count:,} ({neg_count/len(feature_matrix_df)*100:.2f}%)")

    # ---------------------------------------------------------
    # STEP 5: Group-Aware Train / Validation Split
    # ---------------------------------------------------------
    print(f"\n[Step 4/8] Executing group-aware train/validation split (Group = source1_id)...")
    np.random.seed(random_seed)
    all_s1_ids = np.array(sorted(list(unique_s1_in_cand)))
    np.random.shuffle(all_s1_ids)

    n_val_s1 = int(len(all_s1_ids) * val_ratio)
    val_s1_set = set(all_s1_ids[:n_val_s1])
    train_s1_set = set(all_s1_ids[n_val_s1:])

    train_mask = feature_matrix_df["source1_id"].isin(train_s1_set)
    val_mask = feature_matrix_df["source1_id"].isin(val_s1_set)

    train_df = feature_matrix_df[train_mask].reset_index(drop=True)
    val_df = feature_matrix_df[val_mask].reset_index(drop=True)

    print(f"  Split configuration: {int((1-val_ratio)*100)}% Train / {int(val_ratio*100)}% Validation (Seed={random_seed})")
    print(f"  Train: {len(train_s1_set):,} S1 entities | {len(train_df):,} candidate pairs (Pos: {train_df['label'].sum():,}, Neg: {len(train_df)-train_df['label'].sum():,})")
    print(f"  Val:   {len(val_s1_set):,} S1 entities | {len(val_df):,} candidate pairs (Pos: {val_df['label'].sum():,}, Neg: {len(val_df)-val_df['label'].sum():,})")

    X_train_full = train_df[ALL_FEATURES].values
    y_train = train_df["label"].values
    X_val_full = val_df[ALL_FEATURES].values
    y_val = val_df["label"].values

    # ---------------------------------------------------------
    # STEP 6: Multi-Model Baseline Training & Evaluation
    # ---------------------------------------------------------
    print("\n[Step 5/8] Training multiple baseline ML models...")
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_seed, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            n_jobs=-1,
            random_state=random_seed,
            class_weight="balanced_subsample",
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=10,
            random_state=random_seed,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=8,
            random_state=random_seed,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    model_comparison_results = []
    trained_models = {}
    model_val_probs = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        t_m = time.time()
        model.fit(X_train_full, y_train)
        fit_time = time.time() - t_m

        # Probabilities
        val_probs = model.predict_proba(X_val_full)[:, 1]
        trained_models[name] = model
        model_val_probs[name] = val_probs

        # Default threshold (0.50) metrics
        val_pred_50 = (val_probs >= 0.50).astype(int)
        m_50 = calculate_metrics(y_val, val_pred_50, val_probs)

        # Tuned threshold metrics
        best_t, _ = tune_threshold(y_val, val_probs)
        val_pred_tuned = (val_probs >= best_t).astype(int)
        m_tuned = calculate_metrics(y_val, val_pred_tuned, val_probs)

        print(f"    {name} fitted in {fit_time:.2f}s | Default (0.50) F0.5: {m_50['f05']:.4f} | Optimal ({best_t:.2f}) F0.5: {m_tuned['f05']:.4f} (Prec: {m_tuned['precision']:.4f}, Rec: {m_tuned['recall']:.4f})")

        model_comparison_results.append({
            "model": name,
            "features": "All (26)",
            "default_threshold": 0.50,
            "default_f05": round(m_50["f05"], 4),
            "optimal_threshold": best_t,
            "precision": round(m_tuned["precision"], 4),
            "recall": round(m_tuned["recall"], 4),
            "f05": round(m_tuned["f05"], 4),
            "f1": round(m_tuned["f1"], 4),
            "accuracy": round(m_tuned["accuracy"], 4),
            "roc_auc": round(m_tuned["roc_auc"], 4),
            "fit_time_sec": round(fit_time, 2),
        })

    model_comp_df = pd.DataFrame(model_comparison_results).sort_values(by="f05", ascending=False)
    comp_path = os.path.join(results_dir, "model_comparison.csv")
    model_comp_df.to_csv(comp_path, index=False)
    print(f"\n  Model comparison saved to: {comp_path}")

    # Select best model
    best_model_name = model_comp_df.iloc[0]["model"]
    best_threshold = float(model_comp_df.iloc[0]["optimal_threshold"])
    best_model = trained_models[best_model_name]
    best_val_probs = model_val_probs[best_model_name]
    print(f"\n  Selected Best Model: {best_model_name} (Threshold: {best_threshold:.2f}, F0.5: {model_comp_df.iloc[0]['f05']})")

    # ---------------------------------------------------------
    # STEP 7: Detailed Threshold Sweeping for Best Model
    # ---------------------------------------------------------
    print("\n[Step 6/8] Sweeping probability thresholds for selected best model...")
    _, thresh_df = tune_threshold(y_val, best_val_probs)
    thresh_path = os.path.join(results_dir, "threshold_results.csv")
    thresh_df.to_csv(thresh_path, index=False)
    print(f"  Threshold sweep saved to: {thresh_path}")
    print(f"  Threshold Curve:")
    for _, r in thresh_df.iterrows():
        star = " *" if r["threshold"] == best_threshold else ""
        print(f"    Thresh {r['threshold']:.2f}: Prec: {r['precision']:.4f} | Rec: {r['recall']:.4f} | F0.5: {r['f05']:.4f} | F1: {r['f1']:.4f}{star}")

    # ---------------------------------------------------------
    # STEP 8: Feature Ablation Experiments
    # ---------------------------------------------------------
    print("\n[Step 7/8] Running Feature Ablation Experiments...")
    ablation_experiments = [
        ("Exp A: Name Only", NAME_FEATURES),
        ("Exp B: Name + Address", NAME_FEATURES + ADDRESS_FEATURES),
        ("Exp C: Name + Address + Structured", ALL_FEATURES),
    ]

    ablation_results = []
    for exp_title, feat_cols in ablation_experiments:
        X_tr = train_df[feat_cols].values
        X_v = val_df[feat_cols].values

        abl_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=8,
            random_state=random_seed,
            n_jobs=-1,
            verbose=-1,
        )
        abl_model.fit(X_tr, y_train)
        abl_probs = abl_model.predict_proba(X_v)[:, 1]
        abl_best_t, _ = tune_threshold(y_val, abl_probs)
        abl_pred = (abl_probs >= abl_best_t).astype(int)
        m = calculate_metrics(y_val, abl_pred, abl_probs)

        print(f"  {exp_title:<35}: {len(feat_cols):>2} features | Threshold: {abl_best_t:.2f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F0.5: {m['f05']:.4f}")
        ablation_results.append({
            "experiment": exp_title,
            "feature_count": len(feat_cols),
            "features_used": ";".join(feat_cols),
            "optimal_threshold": abl_best_t,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f05": round(m["f05"], 4),
            "f1": round(m["f1"], 4),
            "roc_auc": round(m["roc_auc"], 4),
        })

    ablation_df = pd.DataFrame(ablation_results)
    abl_path = os.path.join(results_dir, "feature_experiments.csv")
    ablation_df.to_csv(abl_path, index=False)
    print(f"  Feature ablation results saved to: {abl_path}")

    # ---------------------------------------------------------
    # STEP 9: False Positive & False Negative Error Analysis
    # ---------------------------------------------------------
    print("\n[Step 8/8] Conducting False Positive & False Negative Error Analysis...")
    val_preds_best = (best_val_probs >= best_threshold).astype(int)
    fp_df, fn_df = analyze_errors(val_df, y_val, val_preds_best, best_val_probs, entity_lookups, top_n=15)

    fp_path = os.path.join(results_dir, "false_positives.csv")
    fn_path = os.path.join(results_dir, "false_negatives.csv")
    fp_df.to_csv(fp_path, index=False)
    fn_df.to_csv(fn_path, index=False)
    print(f"  Saved top false positives to: {fp_path}")
    print(f"  Saved top false negatives to: {fn_path}")

    # Blocking vs Classification Recall decomposition
    total_val_gt_pairs = sum(len(str(gt_df[gt_df['source1_entity_id'] == s1_id]['matched_entity_ids'].iloc[0]).split(',')) for s1_id in val_s1_set if not gt_df[gt_df['source1_entity_id'] == s1_id].empty and gt_df[gt_df['source1_entity_id'] == s1_id]['matched_entity_ids'].iloc[0])
    val_true_in_cands = int(y_val.sum())
    val_correctly_classified = int(np.sum((y_val == 1) & (val_preds_best == 1)))

    decomp = evaluate_blocking_vs_classification(
        total_ground_truth=total_val_gt_pairs,
        true_matches_in_candidates=val_true_in_cands,
        correctly_classified_matches=val_correctly_classified,
    )

    print("\n" + "=" * 85)
    print("RECALL DECOMPOSITION (BLOCKING VS CLASSIFICATION):")
    print(f"  Validation Total Ground Truth Pairs:    {decomp['total_ground_truth']:,}")
    print(f"  True Matches in Candidate Pairs:        {decomp['true_matches_in_candidates']:,} (Blocking Recall: {decomp['blocking_recall']*100:.2f}%)")
    print(f"  True Matches Correctly Predicted Match: {decomp['correctly_classified_matches']:,} (Classification Recall: {decomp['classification_recall']*100:.2f}%)")
    print(f"  Overall End-to-End System Recall:       {decomp['system_recall']*100:.2f}%")
    print(f"  Blocking Failures (Never Entered Cands):{decomp['blocking_failures']:,}")
    print(f"  Classification Failures (Predicted 0):  {decomp['classification_failures']:,}")
    print("=" * 85)

    # ---------------------------------------------------------
    # SAVE BEST MODEL & VALIDATION SUMMARY
    # ---------------------------------------------------------
    model_save_path = os.path.join(models_dir, "best_model.pkl")
    model_payload = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": ALL_FEATURES,
        "optimal_threshold": best_threshold,
        "val_metrics": {
            "precision": float(model_comp_df.iloc[0]["precision"]),
            "recall": float(model_comp_df.iloc[0]["recall"]),
            "f05": float(model_comp_df.iloc[0]["f05"]),
            "f1": float(model_comp_df.iloc[0]["f1"]),
            "roc_auc": float(model_comp_df.iloc[0]["roc_auc"]),
        },
        "recall_decomposition": decomp,
    }
    joblib.dump(model_payload, model_save_path)
    print(f"\nSerialized best model payload to: {model_save_path}")

    # Save summary JSON
    val_json_path = os.path.join(results_dir, "validation_results.json")
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_model_name,
            "optimal_threshold": best_threshold,
            "metrics": model_payload["val_metrics"],
            "recall_decomposition": decomp,
            "total_candidates": len(cand_df),
            "train_pairs": len(train_df),
            "val_pairs": len(val_df),
            "train_s1_count": len(train_s1_set),
            "val_s1_count": len(val_s1_set),
        }, f, indent=2)
    print(f"Saved validation summary to: {val_json_path}")

    total_time = time.time() - t_start
    print(f"\nPERSON 2 ML PIPELINE COMPLETED IN {total_time:.2f}s")
    print("=" * 85)
    return model_payload, feature_matrix_df, entity_lookups


if __name__ == "__main__":
    run_pipeline()
