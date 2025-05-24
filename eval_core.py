import pandas as pd
import numpy as np
from collections import Counter

def expand_with_inverts(df, head_col='head_entity', tail_col='tail_entity', rel_col='relation', inv_col='invert_relation'):
    """
    For each row, generate both direct and inverse relation rows.
    If invert_relation == 1, swap head/tail for the inverse, and (if relevant) flip the relation string with a marker.
    """
    records = []
    for idx, row in df.iterrows():
        # direct
        records.append({head_col: row[head_col],
                        tail_col: row[tail_col],
                        rel_col: row[rel_col],
                        inv_col: 0})
        # inverse
        records.append({head_col: row[tail_col],
                        tail_col: row[head_col],
                        rel_col: row[rel_col] + '_invert',  # mark as inverted relation
                        inv_col: 1})
    return pd.DataFrame(records)

def eval_dual_relation_f1(agent_output_path, ground_truth_path):
    # Load and normalize column names
    agent_df = pd.read_csv(agent_output_path)
    gt_df = pd.read_csv(ground_truth_path)

    agent_df.columns = [c.strip().lower().replace(' ', '_') for c in agent_df.columns]
    gt_df.columns = [c.strip().lower().replace(' ', '_') for c in gt_df.columns]

    # Normalize column mappings if necessary
    rename_map = {
        'entity': 'head_entity', 'target_entity': 'tail_entity',
        'e1_name': 'head_entity', 'e2_name': 'tail_entity'
    }
    agent_df = agent_df.rename(columns={k: v for k, v in rename_map.items() if k in agent_df.columns})
    gt_df = gt_df.rename(columns={k: v for k, v in rename_map.items() if k in gt_df.columns})
    # fill missing columns if needed, but expected: head_entity, tail_entity, relation, invert_relation

    # For both dataframes, expand to double rows: (head, tail, relation, 0) and (tail, head, relation_invert, 1)
    agent_all = expand_with_inverts(agent_df)
    gt_all = expand_with_inverts(gt_df)

    # Compose a label for each instance: (head, tail, relation, invert_relation)
    key_cols = ['head_entity', 'tail_entity', 'relation', 'invert_relation']
    agent_labels = set(tuple(row) for row in agent_all[key_cols].values)
    gt_labels = set(tuple(row) for row in gt_all[key_cols].values)

    TP = len(agent_labels & gt_labels)
    FP = len(agent_labels - gt_labels)
    FN = len(gt_labels - agent_labels)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"F1: {f1:.4f}  (P={precision:.4f}, R={recall:.4f})")
    return dict(f1=f1, precision=precision, recall=recall, tp=TP, fp=FP, fn=FN)

# Example usage:
# eval_dual_relation_f1('relationships_core.csv', 'test_core.csv')

def eval_relationship_extraction(agent_output_path, ground_truth_path):
    """
    Computes micro and macro F1 scores for relationship extraction.
    The input files are expected to have columns:
        - Entity (or head_entity_text)
        - Target Entity (or tail_entity_text)
        - Relationship (or relation)
    'no_relation' relations are included in the calculation.
    """
    agent_df = pd.read_csv(agent_output_path)
    gt_df = pd.read_csv(ground_truth_path)

    # Accept alternate header names
    agent_df.columns = [c.strip().lower().replace(' ', '_') for c in agent_df.columns]
    gt_df.columns = [c.strip().lower().replace(' ', '_') for c in gt_df.columns]

    rename_map = {
        'e1_name': 'head_entity',
        'entity': 'head_entity',
        'e2_name': 'tail_entity',
        'target_entity': 'tail_entity',
    }
    agent_df = agent_df.rename(columns={k: v for k, v in rename_map.items() if k in agent_df.columns})
    gt_df = gt_df.rename(columns={k: v for k, v in rename_map.items() if k in gt_df.columns})

    # Ensure expected columns exist
    required_cols = ['head_entity', 'tail_entity', 'relation']
    for col in required_cols:
        if col not in agent_df.columns or col not in gt_df.columns:
            raise ValueError(f"Missing column '{col}' in agent output or ground truth CSV.")

    # For fair comparison, make sure all Entity, Target Entity pairs in both
    # Also, join on (head_entity, tail_entity)
    key_cols = ['head_entity', 'tail_entity']
    merged = pd.merge(
        gt_df[key_cols + ['relation']].rename(columns={'relation': 'true_relation'}),
        agent_df[key_cols + ['relation']].rename(columns={'relation': 'pred_relation'}),
        on=key_cols,
        how='left'
    )
    # If prediction not found, treat as 'no_relation'
    merged['pred_relation'] = merged['pred_relation'].fillna('undefined')

    # All unique labels
    all_labels = sorted(set(merged['true_relation'].unique()) | set(merged['pred_relation'].unique()))

    # Compute micro metrics
    TP = Counter()
    FP = Counter()
    FN = Counter()

    for label in all_labels:
        TP[label] = int(np.sum((merged['true_relation'] == label) & (merged['pred_relation'] == label)))
        FP[label] = int(np.sum((merged['true_relation'] != label) & (merged['pred_relation'] == label)))
        FN[label] = int(np.sum((merged['true_relation'] == label) & (merged['pred_relation'] != label)))

    # Per-label precision/recall/f1
    precisions, recalls, f1s = {}, {}, {}
    for label in all_labels:
        p = TP[label] / (TP[label] + FP[label]) if (TP[label] + FP[label]) > 0 else 0
        r = TP[label] / (TP[label] + FN[label]) if (TP[label] + FN[label]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precisions[label] = p
        recalls[label] = r
        f1s[label] = f1

    # Macro: mean over all classes
    macro_f1 = np.mean(list(f1s.values()))
    # Micro: aggregate TP, FP, FN
    total_TP = sum(TP.values())
    total_FP = sum(FP.values())
    total_FN = sum(FN.values())
    micro_p = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    micro_r = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Micro F1:  {micro_f1:.4f}")
    print("Category F1s (including 'undefined'):")
    for label in all_labels:
        print(f"  {label:20} F1: {f1s[label]:.4f}  (P={precisions[label]:.4f} R={recalls[label]:.4f})")

    # Optionally return results as dict
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_label_f1": f1s
    }

# Example usage:
# eval_relationship_extraction('relationships.csv', 'test_refined_filtered_lite.csv')

if __name__ == "__main__":
    print(eval_relationship_extraction('relationships_core.csv', 'test_core.csv'))