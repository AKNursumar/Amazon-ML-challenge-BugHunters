"""
Inference and Prediction Pipeline for Entity Resolution (Person 2).
Amazon ML Challenge 2026.

Predicts match probabilities and classifications for every candidate pair in candidate_pairs.tsv.
Output schema:
['source1_id', 'candidate_id', 'candidate_source', 'match_probability', 'predicted_match']
"""

import os
import sys
import io
import time
import argparse
import joblib
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.dirname(cur_dir)
repo_root = os.path.dirname(dataset_dir)
for p in [cur_dir, dataset_dir, repo_root]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from person2.features import ALL_FEATURES, generate_feature_matrix, load_entity_lookups
except ImportError:
    from dataset.person2.features import ALL_FEATURES, generate_feature_matrix, load_entity_lookups

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

def predict_candidate_pairs(
    candidate_pairs_path: str = "dataset/candidate_pairs.tsv",
    source1_path: str = "dataset/train/train_source1.tsv",
    source2_path: str = "dataset/train/train_source2.tsv",
    source3_path: str = "dataset/train/train_source3.tsv",
    model_path: str = None,
    output_path: str = None,
    custom_threshold: float = None,
    precomputed_matrix: pd.DataFrame = None,
):
    if model_path is None:
        model_path = os.path.join(cur_dir, "models", "best_model.pkl")
    if output_path is None:
        output_path = os.path.join(cur_dir, "predictions", "candidate_predictions.tsv")

    candidate_pairs_path = resolve_path(candidate_pairs_path)
    source1_path = resolve_path(source1_path)
    source2_path = resolve_path(source2_path)
    source3_path = resolve_path(source3_path)
    model_path = resolve_path(model_path)
    print("=" * 80)
    print("PERSON 2: INFERENCE & MATCH PROBABILITY PREDICTION PIPELINE")
    print("=" * 80)
    t0 = time.time()

    # 1. Load trained model payload
    print(f"[1/4] Loading trained model from: {model_path}")
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_names = payload.get("feature_names", ALL_FEATURES)
    threshold = custom_threshold if custom_threshold is not None else payload.get("optimal_threshold", 0.50)
    print(f"  Loaded model: {payload.get('model_name', type(model).__name__)}")
    print(f"  Classification Threshold: {threshold:.4f}")
    print(f"  Features required: {len(feature_names)}")

    # 2. Prepare Feature Matrix
    if precomputed_matrix is not None:
        print("[2/4] Using precomputed feature matrix...")
        feat_df = precomputed_matrix
        cand_df = feat_df[["source1_id", "candidate_id", "candidate_source"]]
    else:
        print(f"[2/4] Loading candidate pairs from: {candidate_pairs_path}")
        cand_df = pd.read_csv(candidate_pairs_path, sep="\t")
        print(f"  Loaded {len(cand_df):,} candidate pairs to predict.")

        print("  Loading source records and building lookup table...")
        unique_s1 = set(cand_df["source1_id"].unique())
        unique_cands = set(cand_df["candidate_id"].unique())

        s1_df = pd.read_csv(source1_path, sep="\t", dtype=str).fillna("")
        s1_df = s1_df[s1_df["entity_id"].isin(unique_s1)]

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

        entity_lookups = load_entity_lookups(s1_df, s2_df, s3_df, unique_s1, unique_cands)
        print("  Generating pairwise features...")
        feat_df = generate_feature_matrix(cand_df, entity_lookups)

    # 3. Model Inference
    print(f"\n[3/4] Predicting match probabilities for {len(feat_df):,} pairs...")
    t_pred = time.time()
    X = feat_df[feature_names].values
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    print(f"  Predictions generated in {time.time()-t_pred:.2f}s")
    print(f"  Predicted Matches:     {preds.sum():,} ({preds.sum()/len(preds)*100:.2f}%)")
    print(f"  Predicted Non-Matches: {len(preds)-preds.sum():,} ({(len(preds)-preds.sum())/len(preds)*100:.2f}%)")

    # 4. Save Output Table
    print(f"\n[4/4] Writing candidate predictions table...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_df = pd.DataFrame({
        "source1_id": feat_df["source1_id"],
        "candidate_id": feat_df["candidate_id"],
        "candidate_source": feat_df["candidate_source"],
        "match_probability": probs.round(6),
        "predicted_match": preds,
    })

    out_df.to_csv(output_path, sep="\t", index=False)
    file_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Successfully saved {len(out_df):,} predictions to: {output_path} ({file_mb:.2f} MB)")
    print(f"  Inference pipeline finished in {time.time()-t0:.2f}s")
    print("=" * 80)
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Person 2 Match Probability Prediction")
    parser.add_argument("--candidates", default="dataset/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--source1", default="dataset/train/train_source1.tsv", help="Path to Source 1 TSV")
    parser.add_argument("--source2", default="dataset/train/train_source2.tsv", help="Path to Source 2 TSV")
    parser.add_argument("--source3", default="dataset/train/train_source3.tsv", help="Path to Source 3 TSV")
    parser.add_argument("--model", default="person2/models/best_model.pkl", help="Path to best_model.pkl")
    parser.add_argument("--output", default="person2/predictions/candidate_predictions.tsv", help="Path to output TSV")
    parser.add_argument("--threshold", type=float, default=None, help="Optional probability threshold override")
    args = parser.parse_args()

    predict_candidate_pairs(
        candidate_pairs_path=args.candidates,
        source1_path=args.source1,
        source2_path=args.source2,
        source3_path=args.source3,
        model_path=args.model,
        output_path=args.output,
        custom_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
