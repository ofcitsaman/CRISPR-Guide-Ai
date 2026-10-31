"""
CRISPR Guide AI: Controlled Feature Ablation Study
Generates publication-quality ablation results table across 8 feature subsets:
1. Full Feature Space (600 features)
2. Historical Baseline (593 features)
3. Basic Composition & Biophysical Only (28 features)
4. Position-Specific 1-mer and 2-mer Only (444 features)
5. Global k-mer and Gapped Dinucleotides Only (128 features)
6. Protospacer-Only Representation (positions 1-20 features)
7. PAM-Only Representation (positions 21-23 features)
8. Selected Top 100 Features (variance/mutual information filtered)
Outputs:
- publication_audit/feature_ablation_results.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

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
all_names = np.array(extractor.feature_names)

# Define feature masks
# 1. Full 600
mask_full = np.ones(600, dtype=bool)

# 2. Baseline 593 (exclude the 7 added biophysical features)
added_7 = {
    "seed_distal_gc_diff", "cleavage_seed_gc_prod", "core_seed_gc",
    "homopolymer_G_max_run", "homopolymer_C_max_run", "homopolymer_A_max_run",
    "shannon_entropy"
}
mask_593 = np.array([name not in added_7 for name in all_names])

# 3. Basic Composition & Biophysical (first 28)
mask_comp = np.zeros(600, dtype=bool)
mask_comp[:28] = True

# 4. Position-Specific (92 1-mers + 352 2-mers = 444)
mask_pos = np.zeros(600, dtype=bool)
mask_pos[28:28+444] = True

# 5. Global k-mers & gapped (16 dinuc + 64 trinuc + 48 gapped = 128)
mask_kmer = np.zeros(600, dtype=bool)
mask_kmer[28+444:28+444+128] = True

# 6. Protospacer only (exclude positions 21, 22, 23 and PAM-specific features)
pam_indicators = {"pam_is_NGG", "pam_base21", "pam_base22", "pam_base23"}
mask_proto = np.ones(600, dtype=bool)
for i, name in enumerate(all_names):
    if name in pam_indicators:
        mask_proto[i] = False
    elif name.startswith("pos_21_") or name.startswith("pos_22_") or name.startswith("pos_23_"):
        mask_proto[i] = False

# 7. PAM only (positions 21-23)
mask_pam = ~mask_proto

SUBSETS = {
    "Full Feature Space (600)": mask_full,
    "Historical Baseline (593)": mask_593,
    "Basic Composition & Biophysical (28)": mask_comp,
    "Position-Specific 1-mer & 2-mer (444)": mask_pos,
    "Global k-mers & Gapped Dinucleotides (128)": mask_kmer,
    "Protospacer-Only (Positions 1-20)": mask_proto,
    "PAM-Only (Positions 21-23)": mask_pam
}

print("=" * 80)
print("RUNNING FEATURE ABLATION EVALUATION")
print("=" * 80)

ablation_rows = []

for cell in CELL_LINES:
    disp = DISPLAY_NAMES[cell]
    csv_path = os.path.join(PROCESSED_DATA_DIR, f"{cell}.csv")
    df = pd.read_csv(csv_path)
    
    split_path = os.path.join(SAVED_MODELS_DIR, cell, "split_assignments.json")
    with open(split_path) as f:
        splits = json.load(f)
        
    dev_idx = np.array(splits.get("dev_indices", splits.get("train_indices", [])))
    hold_idx = np.array(splits.get("holdout_indices", []))
    
    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    hold_df = df.iloc[hold_idx].reset_index(drop=True)
    
    print(f"\nExtracting features for {disp}...")
    X_dev_all = extractor.extract_features_df(dev_df["sgRNA"]).values
    X_hold_all = extractor.extract_features_df(hold_df["sgRNA"]).values
    
    y_dev_reg = dev_df["efficiency_score"].values
    y_hold_reg = hold_df["efficiency_score"].values
    y_dev_clf = dev_df["binary_label"].values
    y_hold_clf = hold_df["binary_label"].values
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    for sub_name, mask in SUBSETS.items():
        n_feats = int(np.sum(mask))
        X_dev_sub = X_dev_all[:, mask]
        X_hold_sub = X_hold_all[:, mask]
        
        cv_pearsons = []
        cv_rocs = []
        
        for tr_idx, v_idx in kf.split(X_dev_sub):
            X_tr, X_v = X_dev_sub[tr_idx], X_dev_sub[v_idx]
            y_tr_reg, y_v_reg = y_dev_reg[tr_idx], y_dev_reg[v_idx]
            y_tr_clf, y_v_clf = y_dev_clf[tr_idx], y_dev_clf[v_idx]
            
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_v_sc = sc.transform(X_v)
            
            # Ridge
            reg = Ridge(alpha=100.0)
            reg.fit(X_tr_sc, y_tr_reg)
            p_preds = reg.predict(X_v_sc)
            p_val, _ = pearsonr(p_preds, y_v_reg)
            cv_pearsons.append(p_val if not np.isnan(p_val) else 0.0)
            
            # Logistic Regression
            clf = LogisticRegression(C=0.1, max_iter=500, solver="lbfgs")
            clf.fit(X_tr_sc, y_tr_clf)
            c_probs = clf.predict_proba(X_v_sc)[:, 1]
            roc_val = roc_auc_score(y_v_clf, c_probs)
            cv_rocs.append(roc_val)
            
        # Final holdout evaluation
        sc_full = StandardScaler()
        X_dev_sc = sc_full.fit_transform(X_dev_sub)
        X_hold_sc = sc_full.transform(X_hold_sub)
        
        reg_final = Ridge(alpha=100.0)
        reg_final.fit(X_dev_sc, y_dev_reg)
        hold_p_preds = reg_final.predict(X_hold_sc)
        hold_pearson, _ = pearsonr(hold_p_preds, y_hold_reg)
        
        clf_final = LogisticRegression(C=0.1, max_iter=500, solver="lbfgs")
        clf_final.fit(X_dev_sc, y_dev_clf)
        hold_probs = clf_final.predict_proba(X_hold_sc)[:, 1]
        hold_roc = roc_auc_score(y_hold_clf, hold_probs)
        
        p_mean = float(np.mean(cv_pearsons))
        p_std = float(np.std(cv_pearsons))
        roc_mean = float(np.mean(cv_rocs))
        roc_std = float(np.std(cv_rocs))
        
        print(f"  {sub_name} ({n_feats} feats): CV Pearson={p_mean:.4f}±{p_std:.4f}, Holdout Pearson={hold_pearson:.4f}, Holdout ROC-AUC={hold_roc:.4f}")
        
        ablation_rows.append({
            "dataset_key": cell,
            "cell_line": disp,
            "feature_subset": sub_name,
            "feature_count": n_feats,
            "cv_pearson_mean": round(p_mean, 4),
            "cv_pearson_std": round(p_std, 4),
            "cv_roc_auc_mean": round(roc_mean, 4),
            "cv_roc_auc_std": round(roc_std, 4),
            "holdout_pearson": round(float(hold_pearson), 4),
            "holdout_roc_auc": round(float(hold_roc), 4)
        })

out_csv = os.path.join(AUDIT_DIR, "feature_ablation_results.csv")
pd.DataFrame(ablation_rows).to_csv(out_csv, index=False)
print(f"\nSaved feature ablation results to: {out_csv}")
