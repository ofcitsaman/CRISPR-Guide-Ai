"""
CRISPR Guide AI: Cross-Cell-Line Transferability and External Generalization
Evaluates models trained on one cell line when transferred to evaluate on all other cell lines.
Quantifies domain shift, cell-type-specific vs sequence-generalizable efficacy,
and documents the status of external benchmark datasets (including DeepCRISPR HEL).
Outputs:
- publication_audit/external_validation_results.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score, accuracy_score, f1_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app.features.extractor import CRISPRFeatureExtractor

AUDIT_DIR = os.path.join(BASE_DIR, "publication_audit")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "backend", "data", "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "backend", "saved_models")

CELL_LINES = ["hela", "hct116", "hek293", "hl60"]
DISPLAY_NAMES = {"hela": "HeLa", "hct116": "HCT116", "hek293": "HEK293T", "hl60": "HL60"}

extractor = CRISPRFeatureExtractor()

print("=" * 80)
print("EVALUATING CROSS-CELL-LINE GENERALIZATION AND EXTERNAL TRANSFERABILITY")
print("=" * 80)

# Pre-extract features for all datasets
data_dict = {}
for cell in CELL_LINES:
    csv_path = os.path.join(PROCESSED_DATA_DIR, f"{cell}.csv")
    df = pd.read_csv(csv_path)
    
    split_path = os.path.join(SAVED_MODELS_DIR, cell, "split_assignments.json")
    with open(split_path) as f:
        splits = json.load(f)
        
    dev_idx = np.array(splits.get("dev_indices", splits.get("train_indices", [])))
    hold_idx = np.array(splits.get("holdout_indices", []))
    
    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    hold_df = df.iloc[hold_idx].reset_index(drop=True)
    
    print(f"Pre-extracting {cell}...")
    X_dev = extractor.extract_features_df(dev_df["sgRNA"]).values
    X_hold = extractor.extract_features_df(hold_df["sgRNA"]).values
    
    data_dict[cell] = {
        "dev_df": dev_df,
        "hold_df": hold_df,
        "X_dev": X_dev,
        "X_hold": X_hold,
        "y_dev_reg": dev_df["efficiency_score"].values,
        "y_hold_reg": hold_df["efficiency_score"].values,
        "y_dev_clf": dev_df["binary_label"].values,
        "y_hold_clf": hold_df["binary_label"].values
    }

transfer_rows = []

for train_cell in CELL_LINES:
    train_disp = DISPLAY_NAMES[train_cell]
    tr_data = data_dict[train_cell]
    
    # Train Ridge and LogisticRegression on train_cell dev set
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(tr_data["X_dev"])
    
    reg = Ridge(alpha=100.0)
    reg.fit(X_tr_sc, tr_data["y_dev_reg"])
    
    clf = LogisticRegression(C=0.1, max_iter=500, solver="lbfgs")
    clf.fit(X_tr_sc, tr_data["y_dev_clf"])
    
    for test_cell in CELL_LINES:
        test_disp = DISPLAY_NAMES[test_cell]
        te_data = data_dict[test_cell]
        
        # Evaluate on test_cell holdout set
        X_te_sc = sc.transform(te_data["X_hold"])
        
        # Regression
        p_preds = reg.predict(X_te_sc)
        p_val, p_p = pearsonr(p_preds, te_data["y_hold_reg"])
        s_val, s_p = spearmanr(p_preds, te_data["y_hold_reg"])
        r2_val = r2_score(te_data["y_hold_reg"], p_preds)
        mae_val = mean_absolute_error(te_data["y_hold_reg"], p_preds)
        rmse_val = np.sqrt(mean_squared_error(te_data["y_hold_reg"], p_preds))
        
        # Classification
        c_probs = clf.predict_proba(X_te_sc)[:, 1]
        roc_val = roc_auc_score(te_data["y_hold_clf"], c_probs)
        pr_val = average_precision_score(te_data["y_hold_clf"], c_probs)
        
        is_intra_dataset = (train_cell == test_cell)
        print(f"  Train: {train_disp:>7} -> Test: {test_disp:>7} | Pearson={p_val:.4f}, Spearman={s_val:.4f}, ROC-AUC={roc_val:.4f} {'(Internal)' if is_intra_dataset else '(Cross-Cell)'}")
        
        transfer_rows.append({
            "training_cell_line": train_disp,
            "evaluation_cell_line": test_disp,
            "transfer_type": "Internal Holdout" if is_intra_dataset else "Cross-Cell-Line Transfer",
            "test_sample_size_N": len(te_data["hold_df"]),
            "transfer_pearson": round(float(p_val), 4),
            "transfer_pearson_p": float(p_p),
            "transfer_spearman": round(float(s_val), 4),
            "transfer_spearman_p": float(s_p),
            "transfer_r2": round(float(r2_val), 4),
            "transfer_mae": round(float(mae_val), 4),
            "transfer_rmse": round(float(rmse_val), 4),
            "transfer_roc_auc": round(float(roc_val), 4),
            "transfer_pr_auc": round(float(pr_val), 4)
        })

# Document DeepCRISPR HEL cell line status
transfer_rows.append({
    "training_cell_line": "All Available Cell Lines",
    "evaluation_cell_line": "HEL (Human Erythroleukemia)",
    "transfer_type": "External Dataset Audit",
    "test_sample_size_N": 0,
    "transfer_pearson": None,
    "transfer_pearson_p": None,
    "transfer_spearman": None,
    "transfer_spearman_p": None,
    "transfer_r2": None,
    "transfer_mae": None,
    "transfer_rmse": None,
    "transfer_roc_auc": None,
    "transfer_pr_auc": None
})

out_csv = os.path.join(AUDIT_DIR, "external_validation_results.csv")
pd.DataFrame(transfer_rows).to_csv(out_csv, index=False)
print(f"\nSaved cross-cell-line transferability and external validation results to: {out_csv}")
