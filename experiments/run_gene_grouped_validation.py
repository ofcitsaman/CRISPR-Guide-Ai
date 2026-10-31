"""
CRISPR Guide AI: Gene/Locus Grouped Cross-Validation Script
Evaluates model generalization to completely unseen genomic loci (GroupKFold).
Compares standard stratified CV vs locus-grouped CV to quantify locus-leakage effects.
Outputs:
- publication_audit/grouped_validation_results.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

def assign_locus_clusters(df: pd.DataFrame, max_gap_bp: int = 50000) -> np.ndarray:
    """Clusters guides on the same chromosome within max_gap_bp into genomic loci."""
    df_sorted = df.copy()
    df_sorted["orig_idx"] = np.arange(len(df))
    df_sorted = df_sorted.sort_values(["chr", "start"])
    
    clusters = np.zeros(len(df_sorted), dtype=int)
    curr_chr = None
    curr_end = -1
    curr_cluster = 0
    
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if row["chr"] != curr_chr or row["start"] > curr_end + max_gap_bp:
            curr_cluster += 1
            curr_chr = row["chr"]
            curr_end = row["end"]
        else:
            curr_end = max(curr_end, row["end"])
        clusters[i] = curr_cluster
        
    df_sorted["cluster_id"] = clusters
    # Map back to original order
    df_sorted = df_sorted.sort_values("orig_idx")
    return df_sorted["cluster_id"].values

print("=" * 80)
print("RUNNING GENE/LOCUS GROUPED CROSS-VALIDATION")
print("=" * 80)

grouped_results = []

for cell in CELL_LINES:
    disp = DISPLAY_NAMES[cell]
    csv_path = os.path.join(PROCESSED_DATA_DIR, f"{cell}.csv")
    df = pd.read_csv(csv_path)
    
    split_path = os.path.join(SAVED_MODELS_DIR, cell, "split_assignments.json")
    with open(split_path) as f:
        splits = json.load(f)
        
    dev_idx = np.array(splits.get("dev_indices", splits.get("train_indices", [])))
    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    
    print(f"\nProcessing {disp}: {len(dev_df)} development guides...")
    
    # Assign clusters
    clusters = assign_locus_clusters(dev_df, max_gap_bp=50000)
    num_clusters = len(np.unique(clusters))
    print(f"  Genomic locus clusters identified: {num_clusters}")
    
    # Extract features
    X_raw = extractor.extract_features_df(dev_df["sgRNA"]).values
    y_reg = dev_df["efficiency_score"].values
    y_clf = dev_df["binary_label"].values
    
    n_splits = 5 if cell == "hek293" else 10
    gkf = GroupKFold(n_splits=n_splits)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for split_type, cv_gen in [("Locus-Grouped CV (Unseen Loci)", gkf.split(X_raw, y_reg, groups=clusters)),
                              ("Standard K-Fold CV (Random Loci)", kf.split(X_raw, y_reg))]:
        
        reg_pearsons, reg_spearmans, reg_r2s = [], [], []
        clf_rocs, clf_prs = [], []
        
        for train_idx, val_idx in cv_gen:
            X_tr, X_val = X_raw[train_idx], X_raw[val_idx]
            y_tr_reg, y_val_reg = y_reg[train_idx], y_reg[val_idx]
            y_tr_clf, y_val_clf = y_clf[train_idx], y_clf[val_idx]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)
            
            # Regression: Ridge
            reg = Ridge(alpha=100.0)
            reg.fit(X_tr_sc, y_tr_reg)
            pred_reg = reg.predict(X_val_sc)
            
            p_val, _ = pearsonr(pred_reg, y_val_reg)
            s_val, _ = spearmanr(pred_reg, y_val_reg)
            r2_val = r2_score(y_val_reg, pred_reg)
            
            reg_pearsons.append(p_val if not np.isnan(p_val) else 0.0)
            reg_spearmans.append(s_val if not np.isnan(s_val) else 0.0)
            reg_r2s.append(r2_val)
            
            # Classification: Logistic Regression
            clf = LogisticRegression(C=0.1, max_iter=500, solver="lbfgs")
            clf.fit(X_tr_sc, y_tr_clf)
            prob_clf = clf.predict_proba(X_val_sc)[:, 1]
            
            if len(np.unique(y_val_clf)) > 1:
                roc_val = roc_auc_score(y_val_clf, prob_clf)
                pr_val = average_precision_score(y_val_clf, prob_clf)
            else:
                roc_val, pr_val = 0.5, float(y_tr_clf.mean())
                
            clf_rocs.append(roc_val)
            clf_prs.append(pr_val)
            
        mean_p, std_p = float(np.mean(reg_pearsons)), float(np.std(reg_pearsons))
        mean_s, std_s = float(np.mean(reg_spearmans)), float(np.std(reg_spearmans))
        mean_r2, std_r2 = float(np.mean(reg_r2s)), float(np.std(reg_r2s))
        mean_roc, std_roc = float(np.mean(clf_rocs)), float(np.std(clf_rocs))
        mean_pr, std_pr = float(np.mean(clf_prs)), float(np.std(clf_prs))
        
        print(f"  [{split_type}] Reg Pearson={mean_p:.4f}±{std_p:.4f}, Clf ROC-AUC={mean_roc:.4f}±{std_roc:.4f}")
        
        grouped_results.append({
            "dataset_key": cell,
            "cell_line": disp,
            "evaluation_scheme": split_type,
            "n_splits": n_splits,
            "number_of_genomic_clusters": num_clusters,
            "reg_model": "Ridge (alpha=100.0)",
            "reg_pearson_mean": round(mean_p, 4),
            "reg_pearson_std": round(std_p, 4),
            "reg_spearman_mean": round(mean_s, 4),
            "reg_spearman_std": round(std_s, 4),
            "reg_r2_mean": round(mean_r2, 4),
            "reg_r2_std": round(std_r2, 4),
            "clf_model": "Logistic Regression (C=0.1)",
            "clf_roc_auc_mean": round(mean_roc, 4),
            "clf_roc_auc_std": round(std_roc, 4),
            "clf_pr_auc_mean": round(mean_pr, 4),
            "clf_pr_auc_std": round(std_pr, 4)
        })

out_csv = os.path.join(AUDIT_DIR, "grouped_validation_results.csv")
pd.DataFrame(grouped_results).to_csv(out_csv, index=False)
print(f"\nSaved gene/locus grouped validation results to: {out_csv}")
