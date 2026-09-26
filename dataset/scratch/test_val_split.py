import pandas as pd
import numpy as np
from collections import defaultdict

pred_df = pd.read_csv('Amazon-ML-challenge-BugHunters/dataset/person2/predictions/candidate_predictions.tsv', sep='\t')
gt_path = 'Amazon-ML-challenge-BugHunters/dataset/train/train_ground_truth.tsv'
gt_df = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')

s1_entities = np.array(sorted(pred_df['source1_id'].unique()))
gt_subset = gt_df[gt_df['source1_entity_id'].isin(s1_entities)]

gt_map = {}
for s1_id, m_str in zip(gt_subset['source1_entity_id'], gt_subset['matched_entity_ids']):
    m_str = m_str.strip()
    gt_map[s1_id] = set(m_str.split(',')) if m_str else set()

preds_by_s1 = defaultdict(list)
for s1, c_id, prob in zip(pred_df['source1_id'], pred_df['candidate_id'], pred_df['match_probability']):
    preds_by_s1[s1].append((c_id, prob))

def eval_split(split_name, s1_eval_ids, threshold):
    f05_scores, prec_scores, rec_scores, sing_acc = [], [], [], []
    for s1_id in s1_eval_ids:
        true_set = gt_map.get(s1_id, set())
        pred_set = {c_id for c_id, prob in preds_by_s1.get(s1_id, []) if prob >= threshold}
        
        if len(true_set) == 0:
            if len(pred_set) == 0:
                f05_scores.append(1.0)
                prec_scores.append(1.0)
                rec_scores.append(1.0)
                sing_acc.append(1.0)
            else:
                f05_scores.append(0.0)
                prec_scores.append(0.0)
                rec_scores.append(1.0)
                sing_acc.append(0.0)
        else:
            if len(pred_set) == 0:
                f05_scores.append(0.0)
                prec_scores.append(0.0)
                rec_scores.append(0.0)
            else:
                tp = len(pred_set.intersection(true_set))
                prec = tp / len(pred_set)
                rec = tp / len(true_set)
                prec_scores.append(prec)
                rec_scores.append(rec)
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
                f05_scores.append(f05)
    return {
        'split': split_name,
        'n': len(s1_eval_ids),
        'threshold': threshold,
        'macro_f05': np.mean(f05_scores),
        'macro_prec': np.mean(prec_scores),
        'macro_rec': np.mean(rec_scores),
        'singleton_acc': np.mean(sing_acc) if sing_acc else 0.0
    }

# 90/10 split (seed=42)
np.random.seed(42)
shuffled_s1 = np.random.permutation(s1_entities)
n_val_10 = int(len(s1_entities) * 0.10)
val_s1_10 = shuffled_s1[:n_val_10]
train_s1_90 = shuffled_s1[n_val_10:]

# 80/20 split (seed=42)
n_val_20 = int(len(s1_entities) * 0.20)
val_s1_20 = shuffled_s1[:n_val_20]
train_s1_80 = shuffled_s1[n_val_20:]

print('--- 90/10 Split (10% Validation = 499 S1 entities) ---')
for t in [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    res = eval_split('Val-10%', val_s1_10, t)
    print(f"Thresh {t:.2f} -> Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_prec']:.4f} | Rec: {res['macro_rec']:.4f} | SingAcc: {res['singleton_acc']:.4f}")

print('\n--- 80/20 Split (20% Validation = 999 S1 entities) ---')
for t in [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    res = eval_split('Val-20%', val_s1_20, t)
    print(f"Thresh {t:.2f} -> Macro F0.5: {res['macro_f05']:.4f} | Prec: {res['macro_prec']:.4f} | Rec: {res['macro_rec']:.4f} | SingAcc: {res['singleton_acc']:.4f}")
