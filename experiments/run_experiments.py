"""
CRISPR Guide AI: Systematic Development-Only Optimization Engine.

Phases 1-12 of the scientific improvement protocol:
- Uses ONLY the 80% Development Pool for all feature, hyperparameter, target, and ensemble choices.
- Completely isolates the 20% Holdout (never loaded, touched, or peeked at).
- Logs all experiments to experiments/<dataset>/<exp_id>/ and master_comparison.csv.
"""

import os
import sys
import json
import time
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app.features.extractor import CRISPRFeatureExtractor

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
CELL_LINES = ["hela", "hct116", "hek293", "hl60"]
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
MASTER_CSV_PATH = os.path.join(EXPERIMENTS_DIR, "master_comparison.csv")


def compute_metrics(y_true_cont: np.ndarray, y_pred_cont: np.ndarray, y_true_bin: np.ndarray, threshold: float = None) -> Dict[str, float]:
    """Computes continuous regression and thresholded classification metrics."""
    mae = float(mean_absolute_error(y_true_cont, y_pred_cont))
    mse = float(mean_squared_error(y_true_cont, y_pred_cont))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_cont, y_pred_cont))
    
    try:
        r, _ = stats.pearsonr(y_true_cont, y_pred_cont)
        if np.isnan(r): r = 0.0
    except Exception:
        r = 0.0
        
    try:
        rho, _ = stats.spearmanr(y_true_cont, y_pred_cont)
        if np.isnan(rho): rho = 0.0
    except Exception:
        rho = 0.0

    # Continuous AUCs
    roc_auc = 0.5
    pr_auc = float(np.mean(y_true_bin))
    if len(np.unique(y_true_bin)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_bin, y_pred_cont))
        except Exception:
            pass
        try:
            pr_auc = float(average_precision_score(y_true_bin, y_pred_cont))
        except Exception:
            pass

    # Thresholded classification
    if threshold is None:
        try:
            fpr, tpr, threshs = roc_curve(y_true_bin, y_pred_cont)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            threshold = float(threshs[best_idx])
            if np.isinf(threshold) or threshold > 1.0 or threshold < 0.0:
                threshold = float(np.median(y_pred_cont))
        except Exception:
            threshold = float(np.median(y_pred_cont))

    y_pred_bin = (y_pred_cont >= threshold).astype(int)
    acc = float(accuracy_score(y_true_bin, y_pred_bin))
    prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    except Exception:
        spec = 0.0

    return {
        "pearson_r": round(float(r), 4),
        "spearman_rho": round(float(rho), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1), 4),
        "threshold": round(float(threshold), 4)
    }


def log_experiment_to_master(row: Dict[str, Any]):
    """Appends experiment results to master_comparison.csv."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    file_exists = os.path.exists(MASTER_CSV_PATH)
    
    fieldnames = [
        "dataset", "experiment_id", "model", "feature_count", "target_transform",
        "pearson_r", "spearman_rho", "rmse", "mae", "r2", "roc_auc", "pr_auc",
        "accuracy", "precision", "recall", "specificity", "f1", "threshold", "notes"
    ]
    
    with open(MASTER_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_experiment_artifacts(dataset: str, exp_id: str, config: Dict[str, Any], cv_metrics: Dict[str, Any], oof_df: pd.DataFrame, selected_features: List[str], model_params: Dict[str, Any], notes: str):
    """Saves dedicated experiment directory."""
    exp_dir = os.path.join(EXPERIMENTS_DIR, dataset, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(exp_dir, "cv_metrics.json"), "w") as f:
        json.dump(cv_metrics, f, indent=2)
    with open(os.path.join(exp_dir, "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=2)
    with open(os.path.join(exp_dir, "model_params.json"), "w") as f:
        json.dump(model_params, f, indent=2)
    with open(os.path.join(exp_dir, "notes.txt"), "w") as f:
        f.write(notes)
    if oof_df is not None:
        oof_df.to_csv(os.path.join(exp_dir, "oof_predictions.csv"), index=False)


def extract_base_and_extended_features(df: pd.DataFrame, extractor: CRISPRFeatureExtractor) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Extracts base 593 features as well as candidate extended biological sequence features:
    - seed_distal_gc_diff: (seed_gc - distal_gc)
    - cleavage_seed_gc_prod: cleavage_flank_gc * seed_gc
    - core_seed_gc: GC content of positions 15-20 (critical 6 nt seed next to PAM)
    - homopolymer_G_max_run: Max consecutive Gs (G-quadruplex risk)
    - homopolymer_C_max_run: Max consecutive Cs
    - homopolymer_A_max_run: Max consecutive As
    - shannon_entropy: Sequence entropy across base distribution
    """
    base_feats = []
    ext_feats = []
    
    base_names = extractor.feature_names
    ext_names = list(base_names) + [
        "seed_distal_gc_diff",
        "cleavage_seed_gc_prod",
        "core_seed_gc",
        "homopolymer_G_max_run",
        "homopolymer_C_max_run",
        "homopolymer_A_max_run",
        "shannon_entropy"
    ]
    
    import re
    
    for seq in df["sgRNA"]:
        b_dict = extractor.extract_features_single(seq)
        base_feats.append([b_dict[k] for k in base_names])
        
        # Extended calculations
        s_gc = b_dict.get("seed_gc", 0.5)
        d_gc = b_dict.get("distal_gc", 0.5)
        c_gc = b_dict.get("cleavage_flank_gc", 0.5)
        
        # Core seed (15-20, 6 nt)
        core_sub = seq[14:20]
        core_gc = (core_sub.count("G") + core_sub.count("C")) / 6.0
        
        # Homopolymer runs
        g_runs = re.findall(r"G+", seq)
        c_runs = re.findall(r"C+", seq)
        a_runs = re.findall(r"A+", seq)
        max_g = float(max([len(r) for r in g_runs])) if g_runs else 0.0
        max_c = float(max([len(r) for r in c_runs])) if c_runs else 0.0
        max_a = float(max([len(r) for r in a_runs])) if a_runs else 0.0
        
        # Shannon entropy
        counts = [seq.count(b) for b in ["A", "C", "G", "T"]]
        probs = [c / len(seq) for c in counts if c > 0]
        entropy = float(-sum(p * np.log2(p) for p in probs))
        
        extra_vals = [
            float(s_gc - d_gc),
            float(c_gc * s_gc),
            float(core_gc),
            max_g,
            max_c,
            max_a,
            entropy
        ]
        ext_feats.append([b_dict[k] for k in base_names] + extra_vals)
        
    return np.array(base_feats, dtype=np.float32), base_names, np.array(ext_feats, dtype=np.float32), ext_names


def load_dev_data_only(cell_line: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
    """
    Loads raw dataset, performs the exact 80/20 train_test_split (seed 42),
    and returns ONLY the 80% development pool.
    The 20% holdout is strictly discarded here so exploratory code cannot touch it.
    """
    csv_path = os.path.join(BASE_DIR, "backend", "data", "processed", f"{cell_line}.csv")
    df = pd.read_csv(csv_path)
    
    n_samples = len(df)
    indices = np.arange(n_samples)
    y_bin = df["binary_label"].values.astype(int)
    
    dev_idx, holdout_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_bin
    )
    
    dev_df = df.iloc[dev_idx].copy().reset_index(drop=True)
    extractor = CRISPRFeatureExtractor()
    
    X_base, base_names, X_ext, ext_names = extract_base_and_extended_features(dev_df, extractor)
    y_dev_cont = dev_df["efficiency_score"].values.astype(np.float32)
    y_dev_bin = dev_df["binary_label"].values.astype(int)
    seqs_dev = dev_df["sgRNA"].values
    
    return X_base, base_names, X_ext, ext_names, y_dev_cont, y_dev_bin, seqs_dev


def evaluate_pipeline_on_dev(
    X_dev: np.ndarray,
    feature_names: List[str],
    y_dev_cont: np.ndarray,
    y_dev_bin: np.ndarray,
    model_factory,
    k_val: Any = "all",
    target_transform: str = "none",
    clip_output: bool = True,
    n_splits: int = 5
) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
    """
    Evaluates a model configuration using stratified CV strictly inside the 80% development pool.
    Every preprocessing operation (StandardScaler, SelectKBest) is fitted strictly inside each training fold!
    Returns mean metrics, out-of-fold predictions, and selected features.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(y_dev_cont), dtype=np.float32)
    
    selected_feature_names = feature_names
    
    for tr_idx, val_idx in skf.split(X_dev, y_dev_bin):
        X_tr, X_val = X_dev[tr_idx], X_dev[val_idx]
        y_tr, y_val = y_dev_cont[tr_idx], y_dev_cont[val_idx]
        
        # Fit scaler ONLY on fold training data
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        # Fit feature selector ONLY on fold training data
        if k_val != "all":
            k_int = min(int(k_val), X_tr_s.shape[1])
            fs = SelectKBest(f_regression, k=k_int)
            X_tr_s = fs.fit_transform(X_tr_s, y_tr)
            X_val_s = fs.transform(X_val_s)
            mask = fs.get_support()
            selected_feature_names = [feature_names[i] for i, b in enumerate(mask) if b]
            
        # Target transform if specified
        y_tr_fit = y_tr
        if target_transform == "logit":
            eps = 1e-4
            y_clip = np.clip(y_tr, eps, 1.0 - eps)
            y_tr_fit = np.log(y_clip / (1.0 - y_clip))
            
        model = model_factory()
        model.fit(X_tr_s, y_tr_fit)
        
        raw_pred = model.predict(X_val_s)
        if target_transform == "logit":
            pred = 1.0 / (1.0 + np.exp(-raw_pred))
        else:
            pred = raw_pred
            
        if clip_output:
            pred = np.clip(pred, 0.0, 1.0)
            
        oof_preds[val_idx] = pred
        
    metrics = compute_metrics(y_dev_cont, oof_preds, y_dev_bin)
    return metrics, oof_preds, selected_feature_names


def optimize_ensemble_weights(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Finds non-negative ensemble weights summing to 1 that maximize Pearson correlation with y_true on dev OOF.
    """
    n_models = oof_matrix.shape[1]
    
    def loss(weights):
        w = np.array(weights)
        w = w / (np.sum(w) + 1e-9)
        ens_pred = oof_matrix @ w
        r, _ = stats.pearsonr(y_true, ens_pred)
        return -r if not np.isnan(r) else 0.0
        
    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    
    res = minimize(loss, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        w_opt = res.x / np.sum(res.x)
    else:
        w_opt = init_weights
    return w_opt


def run_all_development_experiments():
    """
    Main driver executing Phases 1 to 12 across HeLa, HCT116, HEK293, and HL60.
    """
    print("=" * 80)
    print("CRISPR GUIDE AI: PHASES 1-12 SYSTEMATIC DEVELOPMENT OPTIMIZATION")
    print("=" * 80)
    
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    
    best_dataset_configs = {}
    
    for cell_line in CELL_LINES:
        print(f"\n{'='*70}\n>>> OPTIMIZING DEVELOPMENT POOL FOR: {cell_line.upper()}\n{'='*70}")
        
        # Load ONLY dev pool
        X_base, base_names, X_ext, ext_names, y_dev_cont, y_dev_bin, seqs_dev = load_dev_data_only(cell_line)
        print(f"Dev Pool N = {len(y_dev_cont)} | Base features = {X_base.shape[1]} | Extended features = {X_ext.shape[1]}")
        
        # -------------------------------------------------------------
        # PHASE 2: FEATURE OPTIMIZATION (Inner 5-fold CV on Dev Pool)
        # -------------------------------------------------------------
        print("\n--- PHASE 2: Feature Selection & Representation Search ---")
        k_candidates = [50, 75, 100, 150, 200, 300, "all"]
        feat_results = {}
        
        for k in k_candidates:
            exp_id = f"feat_k_{k}"
            m_res, oof_p, sel_names = evaluate_pipeline_on_dev(
                X_base, base_names, y_dev_cont, y_dev_bin,
                model_factory=lambda: Ridge(alpha=100.0, random_state=RANDOM_SEED),
                k_val=k,
                target_transform="none",
                clip_output=True
            )
            feat_results[str(k)] = m_res
            print(f"  k={str(k):<4} -> Pearson r={m_res['pearson_r']:.4f}, Spearman rho={m_res['spearman_rho']:.4f}, RMSE={m_res['rmse']:.4f}, ROC-AUC={m_res['roc_auc']:.4f}")
            log_experiment_to_master({
                "dataset": cell_line,
                "experiment_id": exp_id,
                "model": "Ridge (screening)",
                "feature_count": len(sel_names),
                "target_transform": "none",
                **m_res,
                "notes": f"Feature selection evaluation k={k}"
            })
            
        # Test Extended features
        for k in [100, 150, 200, "all"]:
            exp_id = f"feat_ext_k_{k}"
            m_res, oof_p, sel_names = evaluate_pipeline_on_dev(
                X_ext, ext_names, y_dev_cont, y_dev_bin,
                model_factory=lambda: Ridge(alpha=100.0, random_state=RANDOM_SEED),
                k_val=k,
                target_transform="none",
                clip_output=True
            )
            print(f"  Extended k={str(k):<4} -> Pearson r={m_res['pearson_r']:.4f}, ROC-AUC={m_res['roc_auc']:.4f}")
            log_experiment_to_master({
                "dataset": cell_line,
                "experiment_id": exp_id,
                "model": "Ridge (screening)",
                "feature_count": len(sel_names),
                "target_transform": "none",
                **m_res,
                "notes": f"Extended biological features k={k}"
            })
            feat_results[f"ext_{k}"] = m_res

        # Select best feature setup
        best_feat_key = max(feat_results, key=lambda k: feat_results[k]["pearson_r"])
        print(f"  >> Optimal Feature Setup for {cell_line}: {best_feat_key} (r = {feat_results[best_feat_key]['pearson_r']:.4f})")
        
        use_ext = best_feat_key.startswith("ext_")
        chosen_k = best_feat_key.replace("ext_", "")
        chosen_k = "all" if chosen_k == "all" else int(chosen_k)
        active_X = X_ext if use_ext else X_base
        active_names = ext_names if use_ext else base_names
        
        # -------------------------------------------------------------
        # PHASE 6 & 7: TARGET HANDLING & DOMAIN CONSTRAINTS
        # -------------------------------------------------------------
        print("\n--- PHASE 6 & 7: Target Transformation & Domain Constraints ---")
        for mode in ["direct", "logit"]:
            for clip in [True, False]:
                exp_id = f"target_{mode}_clip_{clip}"
                m_res, _, _ = evaluate_pipeline_on_dev(
                    active_X, active_names, y_dev_cont, y_dev_bin,
                    model_factory=lambda: Ridge(alpha=100.0, random_state=RANDOM_SEED),
                    k_val=chosen_k,
                    target_transform="logit" if mode == "logit" else "none",
                    clip_output=clip
                )
                print(f"  Mode={mode:<6} Clip={str(clip):<5} -> Pearson r={m_res['pearson_r']:.4f}, RMSE={m_res['rmse']:.4f}, ROC={m_res['roc_auc']:.4f}")
                log_experiment_to_master({
                    "dataset": cell_line,
                    "experiment_id": exp_id,
                    "model": "Ridge (target test)",
                    "feature_count": chosen_k if chosen_k != "all" else len(active_names),
                    "target_transform": mode,
                    **m_res,
                    "notes": f"Target test mode={mode}, clip={clip}"
                })

        # -------------------------------------------------------------
        # PHASE 5: RIDGE OPTIMIZATION
        # -------------------------------------------------------------
        print("\n--- PHASE 5: Ridge Regularization Search ---")
        ridge_alphas = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0, 300.0, 500.0, 1000.0]
        best_ridge_alpha = 100.0
        best_ridge_r = -1.0
        best_ridge_oof = None
        best_ridge_metrics = None
        
        for a in ridge_alphas:
            exp_id = f"ridge_alpha_{a}"
            m_res, oof_p, _ = evaluate_pipeline_on_dev(
                active_X, active_names, y_dev_cont, y_dev_bin,
                model_factory=lambda alpha=a: Ridge(alpha=alpha, random_state=RANDOM_SEED),
                k_val=chosen_k,
                clip_output=True
            )
            if m_res["pearson_r"] > best_ridge_r:
                best_ridge_r = m_res["pearson_r"]
                best_ridge_alpha = a
                best_ridge_oof = oof_p
                best_ridge_metrics = m_res
            log_experiment_to_master({
                "dataset": cell_line,
                "experiment_id": exp_id,
                "model": "Ridge Linear Regression",
                "feature_count": chosen_k if chosen_k != "all" else len(active_names),
                "target_transform": "direct",
                **m_res,
                "notes": f"Ridge alpha={a}"
            })
        print(f"  >> Best Ridge: alpha={best_ridge_alpha} (r = {best_ridge_r:.4f})")

        # -------------------------------------------------------------
        # PHASE 3: XGBOOST SYSTEMATIC OPTIMIZATION
        # -------------------------------------------------------------
        print("\n--- PHASE 3: XGBoost Fine-Tuning ---")
        xgb_candidates = [
            {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 2, "reg_alpha": 0.1, "reg_lambda": 1.0},
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.2, "reg_lambda": 2.0},
            {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.7, "min_child_weight": 2, "reg_alpha": 0.5, "reg_lambda": 3.0},
            {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 4, "reg_alpha": 1.0, "reg_lambda": 4.0},
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.8, "min_child_weight": 2, "reg_alpha": 0.1, "reg_lambda": 2.0},
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.025, "subsample": 0.75, "colsample_bytree": 0.6, "min_child_weight": 5, "reg_alpha": 0.5, "reg_lambda": 3.0}
        ]
        
        best_xgb_cfg = None
        best_xgb_r = -1.0
        best_xgb_oof = None
        best_xgb_metrics = None
        
        for idx, cfg in enumerate(xgb_candidates):
            exp_id = f"xgb_tune_{idx+1}"
            m_res, oof_p, _ = evaluate_pipeline_on_dev(
                active_X, active_names, y_dev_cont, y_dev_bin,
                model_factory=lambda c=cfg: XGBRegressor(
                    n_estimators=c["n_estimators"],
                    max_depth=c["max_depth"],
                    learning_rate=c["learning_rate"],
                    subsample=c["subsample"],
                    colsample_bytree=c["colsample_bytree"],
                    min_child_weight=c["min_child_weight"],
                    reg_alpha=c["reg_alpha"],
                    reg_lambda=c["reg_lambda"],
                    tree_method="hist",
                    random_state=RANDOM_SEED,
                    n_jobs=4
                ),
                k_val=chosen_k,
                clip_output=True
            )
            print(f"  XGB {idx+1}: depth={cfg['max_depth']}, lr={cfg['learning_rate']}, sub={cfg['subsample']}, col={cfg['colsample_bytree']} -> r={m_res['pearson_r']:.4f}, ROC={m_res['roc_auc']:.4f}")
            if m_res["pearson_r"] > best_xgb_r:
                best_xgb_r = m_res["pearson_r"]
                best_xgb_cfg = cfg
                best_xgb_oof = oof_p
                best_xgb_metrics = m_res
            log_experiment_to_master({
                "dataset": cell_line,
                "experiment_id": exp_id,
                "model": "XGBoost",
                "feature_count": chosen_k if chosen_k != "all" else len(active_names),
                "target_transform": "direct",
                **m_res,
                "notes": f"XGB tune: depth={cfg['max_depth']}, lr={cfg['learning_rate']}, n_est={cfg['n_estimators']}"
            })
        print(f"  >> Best XGBoost: {best_xgb_cfg} (r = {best_xgb_r:.4f})")

        # -------------------------------------------------------------
        # PHASE 4: RANDOM FOREST SYSTEMATIC OPTIMIZATION
        # -------------------------------------------------------------
        print("\n--- PHASE 4: Random Forest Fine-Tuning ---")
        rf_candidates = [
            {"n_estimators": 200, "max_depth": 10, "max_features": 0.3, "min_samples_leaf": 2, "min_samples_split": 4},
            {"n_estimators": 250, "max_depth": 12, "max_features": 0.3, "min_samples_leaf": 2, "min_samples_split": 4},
            {"n_estimators": 250, "max_depth": 14, "max_features": "sqrt", "min_samples_leaf": 2, "min_samples_split": 4},
            {"n_estimators": 300, "max_depth": 12, "max_features": 0.25, "min_samples_leaf": 3, "min_samples_split": 6},
            {"n_estimators": 250, "max_depth": 16, "max_features": 0.35, "min_samples_leaf": 1, "min_samples_split": 2},
        ]
        
        best_rf_cfg = None
        best_rf_r = -1.0
        best_rf_oof = None
        best_rf_metrics = None
        
        for idx, cfg in enumerate(rf_candidates):
            exp_id = f"rf_tune_{idx+1}"
            m_res, oof_p, _ = evaluate_pipeline_on_dev(
                active_X, active_names, y_dev_cont, y_dev_bin,
                model_factory=lambda c=cfg: RandomForestRegressor(
                    n_estimators=c["n_estimators"],
                    max_depth=c["max_depth"],
                    max_features=c["max_features"],
                    min_samples_leaf=c["min_samples_leaf"],
                    min_samples_split=c["min_samples_split"],
                    random_state=RANDOM_SEED,
                    n_jobs=4
                ),
                k_val=chosen_k,
                clip_output=True
            )
            print(f"  RF {idx+1}: n_est={cfg['n_estimators']}, depth={cfg['max_depth']}, feat={cfg['max_features']}, leaf={cfg['min_samples_leaf']} -> r={m_res['pearson_r']:.4f}, ROC={m_res['roc_auc']:.4f}")
            if m_res["pearson_r"] > best_rf_r:
                best_rf_r = m_res["pearson_r"]
                best_rf_cfg = cfg
                best_rf_oof = oof_p
                best_rf_metrics = m_res
            log_experiment_to_master({
                "dataset": cell_line,
                "experiment_id": exp_id,
                "model": "Random Forest",
                "feature_count": chosen_k if chosen_k != "all" else len(active_names),
                "target_transform": "direct",
                **m_res,
                "notes": f"RF tune: n_est={cfg['n_estimators']}, depth={cfg['max_depth']}, feat={cfg['max_features']}"
            })
        print(f"  >> Best Random Forest: {best_rf_cfg} (r = {best_rf_r:.4f})")

        # -------------------------------------------------------------
        # PHASE 5: SVR SYSTEMATIC OPTIMIZATION
        # -------------------------------------------------------------
        print("\n--- PHASE 5: SVR Fine-Tuning ---")
        svr_candidates = [
            {"kernel": "rbf", "C": 1.0, "epsilon": 0.05, "gamma": "scale"},
            {"kernel": "rbf", "C": 1.5, "epsilon": 0.05, "gamma": "scale"},
            {"kernel": "rbf", "C": 2.0, "epsilon": 0.08, "gamma": 0.005},
            {"kernel": "rbf", "C": 2.5, "epsilon": 0.06, "gamma": 0.004},
            {"kernel": "rbf", "C": 3.0, "epsilon": 0.05, "gamma": "scale"},
            {"kernel": "linear", "C": 0.5, "epsilon": 0.05, "gamma": "scale"},
        ]
        
        best_svr_cfg = None
        best_svr_r = -1.0
        best_svr_oof = None
        best_svr_metrics = None
        
        for idx, cfg in enumerate(svr_candidates):
            exp_id = f"svr_tune_{idx+1}"
            m_res, oof_p, _ = evaluate_pipeline_on_dev(
                active_X, active_names, y_dev_cont, y_dev_bin,
                model_factory=lambda c=cfg: SVR(
                    kernel=c["kernel"],
                    C=c["C"],
                    epsilon=c["epsilon"],
                    gamma=c["gamma"],
                    tol=1e-2,
                    max_iter=2000,
                    cache_size=1000
                ),
                k_val=chosen_k,
                clip_output=True
            )
            print(f"  SVR {idx+1}: kernel={cfg['kernel']}, C={cfg['C']}, eps={cfg['epsilon']}, gamma={cfg['gamma']} -> r={m_res['pearson_r']:.4f}, ROC={m_res['roc_auc']:.4f}")
            if m_res["pearson_r"] > best_svr_r:
                best_svr_r = m_res["pearson_r"]
                best_svr_cfg = cfg
                best_svr_oof = oof_p
                best_svr_metrics = m_res
            log_experiment_to_master({
                "dataset": cell_line,
                "experiment_id": exp_id,
                "model": "SVM",
                "feature_count": chosen_k if chosen_k != "all" else len(active_names),
                "target_transform": "direct",
                **m_res,
                "notes": f"SVR tune: kernel={cfg['kernel']}, C={cfg['C']}, eps={cfg['epsilon']}"
            })
        print(f"  >> Best SVR: {best_svr_cfg} (r = {best_svr_r:.4f})")

        # -------------------------------------------------------------
        # PHASE 8: OUT-OF-FOLD ENSEMBLE EXPLORATION
        # -------------------------------------------------------------
        print("\n--- PHASE 8: Out-of-Fold Ensemble Experiments ---")
        oof_matrix = np.column_stack([
            best_ridge_oof,
            best_rf_oof,
            best_xgb_oof,
            best_svr_oof
        ])
        
        # A. Simple Equal Mean
        simple_mean_pred = np.mean(oof_matrix, axis=1)
        m_simple_ens = compute_metrics(y_dev_cont, simple_mean_pred, y_dev_bin)
        print(f"  Simple Mean Ensemble -> Pearson r={m_simple_ens['pearson_r']:.4f}, RMSE={m_simple_ens['rmse']:.4f}, ROC={m_simple_ens['roc_auc']:.4f}")
        log_experiment_to_master({
            "dataset": cell_line,
            "experiment_id": "ensemble_simple_mean",
            "model": "Ensemble (Simple Mean)",
            "feature_count": chosen_k if chosen_k != "all" else len(active_names),
            "target_transform": "direct",
            **m_simple_ens,
            "notes": "Equal weight (0.25 each) across Ridge, RF, XGB, SVR"
        })
        
        # B. Non-negative Constrained Weighted Average (learned only on Dev OOF)
        w_opt = optimize_ensemble_weights(oof_matrix, y_dev_cont)
        w_opt_norm = w_opt / np.sum(w_opt)
        opt_weighted_pred = oof_matrix @ w_opt_norm
        m_weighted_ens = compute_metrics(y_dev_cont, opt_weighted_pred, y_dev_bin)
        print(f"  Optimal Weighted Ensemble -> Weights [Ridge={w_opt_norm[0]:.3f}, RF={w_opt_norm[1]:.3f}, XGB={w_opt_norm[2]:.3f}, SVR={w_opt_norm[3]:.3f}]")
        print(f"    Pearson r={m_weighted_ens['pearson_r']:.4f}, RMSE={m_weighted_ens['rmse']:.4f}, ROC={m_weighted_ens['roc_auc']:.4f}")
        log_experiment_to_master({
            "dataset": cell_line,
            "experiment_id": "ensemble_optimal_weighted",
            "model": "Ensemble (Optimal Weighted)",
            "feature_count": chosen_k if chosen_k != "all" else len(active_names),
            "target_transform": "direct",
            **m_weighted_ens,
            "notes": f"Weights: Ridge={w_opt_norm[0]:.3f}, RF={w_opt_norm[1]:.3f}, XGB={w_opt_norm[2]:.3f}, SVR={w_opt_norm[3]:.3f}"
        })
        
        # Compare individual models vs ensembles
        print("\n  Summary of Candidates on 80% Development Pool:")
        candidates_summary = {
            "Ridge Linear Regression": (best_ridge_r, best_ridge_metrics),
            "Random Forest": (best_rf_r, best_rf_metrics),
            "XGBoost": (best_xgb_r, best_xgb_metrics),
            "SVM": (best_svr_r, best_svr_metrics),
            "Ensemble (Simple Mean)": (m_simple_ens["pearson_r"], m_simple_ens),
            "Ensemble (Optimal Weighted)": (m_weighted_ens["pearson_r"], m_weighted_ens)
        }
        for name, (r_score, m_data) in candidates_summary.items():
            print(f"    {name:<28} | Pearson r = {r_score:.4f} | RMSE = {m_data['rmse']:.4f} | ROC-AUC = {m_data['roc_auc']:.4f} | PR-AUC = {m_data['pr_auc']:.4f}")
            
        best_dataset_configs[cell_line] = {
            "feature_setup": {
                "use_extended": use_ext,
                "selected_k": chosen_k,
                "feature_names_count": len(active_names)
            },
            "ridge": {"alpha": best_ridge_alpha, "metrics": best_ridge_metrics},
            "rf": {"cfg": best_rf_cfg, "metrics": best_rf_metrics},
            "xgb": {"cfg": best_xgb_cfg, "metrics": best_xgb_metrics},
            "svr": {"cfg": best_svr_cfg, "metrics": best_svr_metrics},
            "ensemble_weights": {
                "Ridge Linear Regression": round(float(w_opt_norm[0]), 4),
                "Random Forest": round(float(w_opt_norm[1]), 4),
                "XGBoost": round(float(w_opt_norm[2]), 4),
                "SVM": round(float(w_opt_norm[3]), 4)
            },
            "ensemble_metrics": m_weighted_ens
        }
        
    # Save master optimized configuration JSON
    opt_summary_path = os.path.join(EXPERIMENTS_DIR, "optimized_development_configurations.json")
    with open(opt_summary_path, "w") as f:
        json.dump(best_dataset_configs, f, indent=2)
    print(f"\nSaved all development optimization configurations to: {opt_summary_path}")
    print("=" * 80)
    print("DEVELOPMENT OPTIMIZATION COMPLETE (ZERO HOLDOUT CONTAMINATION)")
    print("=" * 80)


if __name__ == "__main__":
    run_all_development_experiments()
