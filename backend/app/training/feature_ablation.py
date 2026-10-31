"""
Controlled Feature Ablation Experiment Suite for CRISPR Guide AI.
Evaluates the predictive utility of feature subsets strictly within the 80% development pool:
1. Full 600 features (593 sequence-derived + 7 biophysical)
2. Baseline 593 features (original sequence-derived features)
3. Basic composition only (28 features)
4. Position-specific features only (1-mer + 2-mer = 444 features)
5. k-mer & gapped features only (count 2-mer, count 3-mer, gapped di = 128 features)
6. Selected top-k features (SelectKBest f_regression k=100)
Saves results to experiments/feature_ablation_report.json
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_regression

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app.features.extractor import CRISPRFeatureExtractor
PROCESSED_DIR = os.path.join(BASE_DIR, "backend", "data", "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "backend", "saved_models")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
OUTPUT_PATH = os.path.join(EXPERIMENTS_DIR, "feature_ablation_report.json")

CELL_LINES = ["hela", "hct116", "a375", "hl60"]
RANDOM_SEED = 42


def get_feature_subsets(extractor: CRISPRFeatureExtractor):
    all_names = extractor.feature_names  # 600
    
    # 7 newly added biophysical features
    new_biophys = {
        "seed_distal_gc_diff", "cleavage_seed_gc_prod", "core_seed_gc",
        "homopolymer_G_max_run", "homopolymer_C_max_run", "homopolymer_A_max_run",
        "shannon_entropy"
    }
    baseline_593 = [f for f in all_names if f not in new_biophys]
    
    basic_features = [
        f for f in all_names if (
            f.startswith("count_") and len(f) <= 7 or
            f.startswith("pct_") or
            f in ["seq_length", "gc_content", "at_content", "distal_gc", "seed_gc",
                  "seed_purine_pct", "poly_T_max_run", "has_terminator_TTTT",
                  "pam_is_NGG", "pam_base21", "pam_base22", "pam_base23",
                  "cleavage_flank_gc"] or
            f in new_biophys
        )
    ]
    
    pos_features = [f for f in all_names if f.startswith("pos_")]
    kmer_gapped_features = [f for f in all_names if f.startswith("gap") or (f.startswith("count_") and len(f) > 7)]
    
    return {
        "full_600": all_names,
        "baseline_593": baseline_593,
        "basic_composition_only": basic_features,
        "position_specific_only": pos_features,
        "kmer_and_gapped_only": kmer_gapped_features
    }


def run_feature_ablation():
    print("=" * 70)
    print("CRISPR Guide AI: Controlled Feature Ablation Experiments")
    print("Strictly on 80% Development Pool (Zero Touch on Holdout)")
    print("=" * 70)

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    extractor = CRISPRFeatureExtractor()
    feature_subsets = get_feature_subsets(extractor)

    for name, feats in feature_subsets.items():
        print(f"  Subset '{name}': {len(feats)} features")

    report = {
        "audit_name": "Controlled Feature Ablation Study",
        "timestamp": pd.Timestamp.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "subsets": {k: len(v) for k, v in feature_subsets.items()},
        "results": {}
    }

    for cell in CELL_LINES:
        print(f"\n[EVALUATING CELL LINE] {cell.upper()}...")
        proc_path = os.path.join(PROCESSED_DIR, f"{cell}.csv")
        df = pd.read_csv(proc_path)

        # Load split assignments to get exact 80% dev pool
        split_path = os.path.join(SAVED_MODELS_DIR, cell, "split_assignments.json")
        with open(split_path, "r") as f:
            split_data = json.load(f)
        dev_indices = split_data.get("dev_indices", split_data.get("development_indices"))

        dev_df = df.iloc[dev_indices].reset_index(drop=True)
        sequences = dev_df["sgRNA"].tolist()
        y_eff = dev_df["efficiency_score"].values
        y_bin = dev_df["binary_label"].values

        # Extract all features once
        print(f"  Extracting features for {len(sequences)} development sequences...")
        X_df = extractor.extract_features_df(sequences)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        cell_results = {}

        for subset_name, feat_cols in feature_subsets.items():
            X_sub = X_df[feat_cols].values

            pearson_scores = []
            auc_scores = []

            for train_idx, val_idx in cv.split(X_sub, y_bin):
                X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
                y_tr_eff, y_val_eff = y_eff[train_idx], y_eff[val_idx]
                y_tr_bin, y_val_bin = y_bin[train_idx], y_bin[val_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)

                # Regression: Ridge
                reg = Ridge(alpha=100.0)
                reg.fit(X_tr_s, y_tr_eff)
                pred_eff = reg.predict(X_val_s)
                try:
                    r, _ = stats.pearsonr(y_val_eff, pred_eff)
                    if np.isnan(r):
                        r = 0.0
                except Exception:
                    r = 0.0
                pearson_scores.append(float(r))

                # Classification: LogisticRegression
                clf = LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_SEED)
                clf.fit(X_tr_s, y_tr_bin)
                pred_prob = clf.predict_proba(X_val_s)[:, 1]
                try:
                    auc = float(roc_auc_score(y_val_bin, pred_prob))
                except Exception:
                    auc = 0.5
                auc_scores.append(auc)

            mean_r = float(np.mean(pearson_scores))
            std_r = float(np.std(pearson_scores))
            mean_auc = float(np.mean(auc_scores))
            std_auc = float(np.std(auc_scores))

            cell_results[subset_name] = {
                "num_features": len(feat_cols),
                "cv_pearson_mean": round(mean_r, 4),
                "cv_pearson_std": round(std_r, 4),
                "cv_roc_auc_mean": round(mean_auc, 4),
                "cv_roc_auc_std": round(std_auc, 4)
            }
            print(f"    {subset_name:25s} ({len(feat_cols):3d} feats) -> Pearson r: {mean_r:.4f} +/- {std_r:.4f} | ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

        # Also evaluate SelectKBest (k=100) inside CV
        k_pearson = []
        k_auc = []
        X_all = X_df[extractor.feature_names].values
        for train_idx, val_idx in cv.split(X_all, y_bin):
            X_tr, X_val = X_all[train_idx], X_all[val_idx]
            y_tr_eff, y_val_eff = y_eff[train_idx], y_eff[val_idx]
            y_tr_bin, y_val_bin = y_bin[train_idx], y_bin[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            sel = SelectKBest(f_regression, k=min(100, X_tr_s.shape[1]))
            X_tr_k = sel.fit_transform(X_tr_s, y_tr_eff)
            X_val_k = sel.transform(X_val_s)

            reg = Ridge(alpha=100.0)
            reg.fit(X_tr_k, y_tr_eff)
            pred_eff = reg.predict(X_val_k)
            try:
                r, _ = stats.pearsonr(y_val_eff, pred_eff)
                if np.isnan(r):
                    r = 0.0
            except Exception:
                r = 0.0
            k_pearson.append(float(r))

            clf = LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_SEED)
            clf.fit(X_tr_k, y_tr_bin)
            pred_prob = clf.predict_proba(X_val_k)[:, 1]
            try:
                auc = float(roc_auc_score(y_val_bin, pred_prob))
            except Exception:
                auc = 0.5
            k_auc.append(auc)

        cell_results["selected_top_100_features"] = {
            "num_features": 100,
            "cv_pearson_mean": round(float(np.mean(k_pearson)), 4),
            "cv_pearson_std": round(float(np.std(k_pearson)), 4),
            "cv_roc_auc_mean": round(float(np.mean(k_auc)), 4),
            "cv_roc_auc_std": round(float(np.std(k_auc)), 4)
        }
        print(f"    {'selected_top_100_features':25s} (100 feats) -> Pearson r: {np.mean(k_pearson):.4f} +/- {np.std(k_pearson):.4f} | ROC-AUC: {np.mean(k_auc):.4f} +/- {np.std(k_auc):.4f}")

        report["results"][cell] = cell_results

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nAblation study results saved to: {OUTPUT_PATH}")
    return report


if __name__ == "__main__":
    run_feature_ablation()
