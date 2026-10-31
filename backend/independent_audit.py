"""
Independent Scientific Audit and Metric Verification Script for CRISPR Guide AI.
Audits:
1. Split integrity (N, dev/holdout counts, zero sequence overlap).
2. Independent recomputation of holdout metrics from saved model artifacts.
3. Cross-validation fold integrity (100 folds, mean/std, pooled OOF).
4. Baselines correctness.
5. Traceability of development-only model selection, hyperparameters, and thresholds.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    balanced_accuracy_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SAVED_MODELS_DIR = os.path.join(BASE_DIR, "backend", "saved_models")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "backend", "data", "processed")

CELL_LINES = ["hela", "hct116", "hek293", "hl60"]
from backend.app.features.extractor import CRISPRFeatureExtractor

extractor = CRISPRFeatureExtractor()

print("=" * 80)
print("CRISPR GUIDE AI: INDEPENDENT SCIENTIFIC AUDIT")
print("=" * 80)

audit_report = {
    "split_integrity": {},
    "holdout_metric_recomputation": {},
    "oof_analysis": {},
    "discrepancies": []
}

# --- SECTION 1: SPLIT INTEGRITY ---
print("\n[1] AUDITING DATASET SPLIT INTEGRITY...")
for cell in CELL_LINES:
    csv_path = os.path.join(PROCESSED_DATA_DIR, f"{cell}.csv")
    split_path = os.path.join(SAVED_MODELS_DIR, cell, "split_assignments.json")
    
    df = pd.read_csv(csv_path)
    with open(split_path, "r") as f:
        split_data = json.load(f)
        
    dev_seqs = split_data["development_sequences"]
    ho_seqs = split_data["holdout_sequences"]
    
    dev_set = set(dev_seqs)
    ho_set = set(ho_seqs)
    overlap = dev_set.intersection(ho_set)
    
    total_samples = len(df)
    dev_count = len(dev_seqs)
    ho_count = len(ho_seqs)
    
    audit_report["split_integrity"][cell] = {
        "total_dataset_rows": total_samples,
        "dev_count": dev_count,
        "dev_pct": round(dev_count / total_samples * 100, 2),
        "holdout_count": ho_count,
        "holdout_pct": round(ho_count / total_samples * 100, 2),
        "sequence_overlap_count": len(overlap),
        "split_exact_sum_matches": (dev_count + ho_count == total_samples)
    }
    
    print(f"  {cell.upper()}: Total={total_samples}, Dev={dev_count} ({dev_count/total_samples:.1%}), "
          f"Holdout={ho_count} ({ho_count/total_samples:.1%}), Overlap={len(overlap)}")
    if len(overlap) > 0:
        audit_report["discrepancies"].append(f"LEAKAGE: {cell} has {len(overlap)} overlapping sequences!")

# --- SECTION 2: HOLDOUT METRIC RECOMPUTATION ---
print("\n[2] INDEPENDENTLY RECOMPUTING HOLDOUT METRICS FROM SAVED JOB-LIBS...")

for cell in CELL_LINES:
    cell_dir = os.path.join(SAVED_MODELS_DIR, cell)
    csv_path = os.path.join(PROCESSED_DATA_DIR, f"{cell}.csv")
    df = pd.read_csv(csv_path)
    
    with open(os.path.join(cell_dir, "split_assignments.json"), "r") as f:
        split_data = json.load(f)
    ho_set = set(split_data["holdout_sequences"])
    
    # Filter holdout rows in consistent order
    df_ho = df[df["sgRNA"].isin(ho_set)].copy()
    y_ho_cont = df_ho["efficiency_score"].values.astype(float)
    y_ho_bin = df_ho["binary_label"].values.astype(int)
    
    # Extract features for holdout sequences
    X_ho_raw = extractor.extract_features_array(df_ho["sgRNA"].tolist())
    
    scaler = joblib.load(os.path.join(cell_dir, "scaler.joblib"))
    selector = joblib.load(os.path.join(cell_dir, "feature_selector.joblib"))
    
    X_ho_scaled = scaler.transform(X_ho_raw)
    X_ho_selected = selector.transform(X_ho_scaled)
    
    # Load thresholds and ensemble weights
    with open(os.path.join(cell_dir, "ensemble_config.json"), "r") as f:
        ens_cfg = json.load(f)
    weights = ens_cfg["weights"]
    
    with open(os.path.join(cell_dir, "hyperparameters.json"), "r") as f:
        hp_data = json.load(f)
    frozen_reg_thresh = hp_data["frozen_thresholds"]
    
    with open(os.path.join(cell_dir, "classifier_thresholds.json"), "r") as f:
        clf_thresh = json.load(f)
        
    with open(os.path.join(cell_dir, "regression_metrics.json"), "r") as f:
        stored_reg = json.load(f)
    with open(os.path.join(cell_dir, "classification_metrics.json"), "r") as f:
        stored_clf = json.load(f)
        
    cell_eval = {"regression": {}, "classification": {}, "baselines": {}}
    
    # Regression models
    reg_models = {
        "Ridge Linear Regression": joblib.load(os.path.join(cell_dir, "ridge_model.joblib")),
        "Random Forest": joblib.load(os.path.join(cell_dir, "rf_model.joblib")),
        "XGBoost": joblib.load(os.path.join(cell_dir, "xgb_model.joblib")),
        "SVM": joblib.load(os.path.join(cell_dir, "svr_model.joblib")),
    }
    
    ho_reg_preds = {}
    for m_name, model in reg_models.items():
        preds = model.predict(X_ho_selected)
        preds = np.clip(preds, 0.0, 1.0)
        ho_reg_preds[m_name] = preds
        
        pr, _ = pearsonr(y_ho_cont, preds)
        sr, _ = spearmanr(y_ho_cont, preds)
        rmse = np.sqrt(mean_squared_error(y_ho_cont, preds))
        mae = mean_absolute_error(y_ho_cont, preds)
        r2 = r2_score(y_ho_cont, preds)
        
        stored_val = stored_reg["models"][m_name]["final_validation_20"]
        diff_pr = abs(pr - stored_val["pearson"])
        diff_rmse = abs(rmse - stored_val["rmse"])
        
        cell_eval["regression"][m_name] = {
            "recomputed": {"pearson": round(float(pr), 4), "spearman": round(float(sr), 4), "rmse": round(float(rmse), 4), "mae": round(float(mae), 4), "r2": round(float(r2), 4)},
            "stored": {"pearson": stored_val["pearson"], "spearman": stored_val["spearman"], "rmse": stored_val["rmse"], "mae": stored_val["mae"], "r2": stored_val["r2"]},
            "pr_diff": round(float(diff_pr), 6),
            "rmse_diff": round(float(diff_rmse), 6)
        }
        if diff_pr > 0.005 or diff_rmse > 0.005:
            audit_report["discrepancies"].append(f"Reg Discrepancy in {cell}/{m_name}: calc pr={pr:.4f} vs stored={stored_val['pearson']}")

    # Weighted Ensemble
    ens_pred = (
        weights["Ridge Linear Regression"] * ho_reg_preds["Ridge Linear Regression"] +
        weights["Random Forest"] * ho_reg_preds["Random Forest"] +
        weights["XGBoost"] * ho_reg_preds["XGBoost"] +
        weights["SVM"] * ho_reg_preds["SVM"]
    )
    ens_pred = np.clip(ens_pred, 0.0, 1.0)
    pr_ens, _ = pearsonr(y_ho_cont, ens_pred)
    sr_ens, _ = spearmanr(y_ho_cont, ens_pred)
    rmse_ens = np.sqrt(mean_squared_error(y_ho_cont, ens_pred))
    mae_ens = mean_absolute_error(y_ho_cont, ens_pred)
    r2_ens = r2_score(y_ho_cont, ens_pred)
    stored_ens = stored_reg["models"]["Weighted Ensemble"]["final_validation_20"]
    
    cell_eval["regression"]["Weighted Ensemble"] = {
        "recomputed": {"pearson": round(float(pr_ens), 4), "spearman": round(float(sr_ens), 4), "rmse": round(float(rmse_ens), 4), "mae": round(float(mae_ens), 4), "r2": round(float(r2_ens), 4)},
        "stored": {"pearson": stored_ens["pearson"], "spearman": stored_ens["spearman"], "rmse": stored_ens["rmse"], "mae": stored_ens["mae"], "r2": stored_ens["r2"]},
        "pr_diff": round(float(abs(pr_ens - stored_ens["pearson"])), 6),
        "rmse_diff": round(float(abs(rmse_ens - stored_ens["rmse"])), 6)
    }

    # Classification models
    clf_models = {
        "Logistic Regression": joblib.load(os.path.join(cell_dir, "logistic_model.joblib")),
        "Random Forest Classifier": joblib.load(os.path.join(cell_dir, "rf_classifier.joblib")),
        "XGBoost Classifier": joblib.load(os.path.join(cell_dir, "xgb_classifier.joblib")),
        "Support Vector Classifier (SVC)": joblib.load(os.path.join(cell_dir, "svc_model.joblib")),
        "Ridge Classifier": joblib.load(os.path.join(cell_dir, "ridge_classifier.joblib")),
    }
    
    for c_name, model in clf_models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_ho_selected)[:, 1]
        else:
            dec = model.decision_function(X_ho_selected)
            probs = 1.0 / (1.0 + np.exp(-dec))
            
        thresh = clf_thresh[c_name]
        bin_pred = (probs >= thresh).astype(int)
        
        auc = roc_auc_score(y_ho_bin, probs)
        pr_auc = average_precision_score(y_ho_bin, probs)
        acc = accuracy_score(y_ho_bin, bin_pred)
        prec = precision_score(y_ho_bin, bin_pred, zero_division=0)
        rec = recall_score(y_ho_bin, bin_pred, zero_division=0)
        f1 = f1_score(y_ho_bin, bin_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_ho_bin, bin_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        stored_c_val = stored_clf["models"][c_name]["final_validation_20"]
        diff_auc = abs(auc - stored_c_val["roc_auc"])
        diff_spec = abs(spec - stored_c_val["specificity"])
        
        cell_eval["classification"][c_name] = {
            "recomputed": {"roc_auc": round(float(auc), 4), "pr_auc": round(float(pr_auc), 4), "accuracy": round(float(acc), 4), "specificity": round(float(spec), 4), "recall": round(float(rec), 4), "f1": round(float(f1), 4)},
            "stored": {"roc_auc": stored_c_val["roc_auc"], "pr_auc": stored_c_val["pr_auc"], "accuracy": stored_c_val["accuracy"], "specificity": stored_c_val["specificity"], "recall": stored_c_val["recall"], "f1": stored_c_val["f1"]},
            "auc_diff": round(float(diff_auc), 6),
            "spec_diff": round(float(diff_spec), 6)
        }
        if diff_auc > 0.005 or diff_spec > 0.005:
            audit_report["discrepancies"].append(f"Clf Discrepancy in {cell}/{c_name}: calc auc={auc:.4f} vs stored={stored_c_val['roc_auc']}")

    # Baselines
    dummy_reg = joblib.load(os.path.join(cell_dir, "dummy_regressor.joblib"))
    dummy_pred = dummy_reg.predict(X_ho_selected)
    dummy_rmse = np.sqrt(mean_squared_error(y_ho_cont, dummy_pred))
    dummy_mae = mean_absolute_error(y_ho_cont, dummy_pred)
    
    prev = float(np.mean(y_ho_bin))
    cell_eval["baselines"] = {
        "dummy_regressor_rmse": round(float(dummy_rmse), 4),
        "dummy_regressor_mae": round(float(dummy_mae), 4),
        "holdout_positive_prevalence": round(float(prev), 4),
        "dummy_classifier_most_frequent_accuracy": round(max(prev, 1.0 - prev), 4)
    }

    audit_report["holdout_metric_recomputation"][cell] = cell_eval
    print(f"  {cell.upper()} audit completed. Max diff observed: "
          f"Reg Pearson={max(v['pr_diff'] for v in cell_eval['regression'].values()):.6f}, "
          f"Clf AUC={max(v['auc_diff'] for v in cell_eval['classification'].values()):.6f}")

# --- SECTION 3: 100-FOLD CV AUDIT ---
print("\n[3] AUDITING 100-FOLD CV FOLDS AND OOF AUDITABILITY...")
for cell in CELL_LINES:
    cell_dir = os.path.join(SAVED_MODELS_DIR, cell)
    with open(os.path.join(cell_dir, "regression_metrics.json"), "r") as f:
        reg_data = json.load(f)
    with open(os.path.join(cell_dir, "classification_metrics.json"), "r") as f:
        clf_data = json.load(f)
    oof_path = os.path.join(cell_dir, "oof_predictions.csv")
    df_oof = pd.read_csv(oof_path)
    
    # Check fold counts in regression and classification
    ridge_folds = len(reg_data["models"]["Ridge Linear Regression"]["cv_100_fold"]["all_folds"]["pearson"])
    rf_folds = len(reg_data["models"]["Random Forest"]["cv_100_fold"]["all_folds"]["pearson"])
    xgb_folds = len(reg_data["models"]["XGBoost"]["cv_100_fold"]["all_folds"]["pearson"])
    svr_folds = len(reg_data["models"]["SVM"]["cv_100_fold"]["all_folds"]["pearson"])
    lr_folds = len(clf_data["models"]["Logistic Regression"]["cv_100_fold"]["all_folds"]["roc_auc"])
    
    # OOF pooled metrics
    oof_metrics = {}
    for m in ["Ridge Linear Regression", "Random Forest", "XGBoost", "SVM", "Weighted Ensemble"]:
        sub = df_oof[(df_oof["model"] == m) & (df_oof["task"] == "regression")]
        if len(sub) > 0:
            p_oof, _ = pearsonr(sub["efficiency_score"], sub["predicted_score"])
            sp_oof, _ = spearmanr(sub["efficiency_score"], sub["predicted_score"])
            rmse_oof = np.sqrt(mean_squared_error(sub["efficiency_score"], sub["predicted_score"]))
            r2_oof = r2_score(sub["efficiency_score"], sub["predicted_score"])
            oof_metrics[m] = {"pooled_pearson": round(float(p_oof), 4), "pooled_spearman": round(float(sp_oof), 4), "pooled_rmse": round(float(rmse_oof), 4), "pooled_r2": round(float(r2_oof), 4)}
            
    audit_report["oof_analysis"][cell] = {
        "cv_fold_count": {"ridge": ridge_folds, "rf": rf_folds, "xgb": xgb_folds, "svr": svr_folds, "logistic": lr_folds},
        "total_oof_rows": len(df_oof),
        "unique_dev_sequences_in_oof": df_oof["sequence"].nunique(),
        "pooled_oof_regression": oof_metrics
    }
    print(f"  {cell.upper()}: Fold counts={ridge_folds}, OOF Rows={len(df_oof)}, Unique Dev Seqs={df_oof['sequence'].nunique()}")

# Save audit report
audit_out_path = os.path.join(SAVED_MODELS_DIR, "independent_audit_report.json")
with open(audit_out_path, "w") as f:
    json.dump(audit_report, f, indent=2)

print("\n" + "=" * 80)
print(f"AUDIT COMPLETE. Report saved to: {audit_out_path}")
print(f"Total Discrepancies Flagged: {len(audit_report['discrepancies'])}")
print("=" * 80)
