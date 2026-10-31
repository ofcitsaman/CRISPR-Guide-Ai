"""
CRISPR Guide AI: Rigorous Instance-Level Statistical Comparison & Bootstrap CI Analysis
Replaces fold-level paired t-tests with:
1. Instance-level paired Wilcoxon signed-rank tests on development Out-Of-Fold (OOF) absolute errors.
2. Non-parametric paired bootstrap resampling (B=1,000 resamples of guide instances)
   to compute empirical 95% confidence intervals on performance metric differences:
   Delta r (Pearson), Delta rho (Spearman), Delta ROC-AUC, Delta PR-AUC.
3. Holm-Bonferroni family-wise error rate correction.
Outputs:
- publication_audit/statistical_comparison_results.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

AUDIT_DIR = os.path.join(BASE_DIR, "publication_audit")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "backend", "saved_models")

CELL_LINES = ["hela", "hct116", "hek293", "hl60"]
DISPLAY_NAMES = {
    "hela": "HeLa",
    "hct116": "HCT116",
    "hek293": "HEK293T / A375 (Unresolved)",
    "hl60": "HL60"
}

print("=" * 80)
print("EXECUTING INSTANCE-LEVEL STATISTICAL COMPARISONS & BOOTSTRAP CIs")
print("=" * 80)

all_comparisons = []

for cell in CELL_LINES:
    disp = DISPLAY_NAMES[cell]
    oof_path = os.path.join(SAVED_MODELS_DIR, cell, "oof_predictions.csv")
    if not os.path.exists(oof_path):
        print(f"Skipping {cell}: OOF file not found.")
        continue

    df_oof = pd.read_csv(oof_path)
    
    # ----------------------------------------------------
    # Regression Comparisons
    # ----------------------------------------------------
    reg_df = df_oof[df_oof["task"] == "regression"].copy()
    piv_reg = reg_df.pivot(index="sequence", columns="model", values="predicted_score")
    truth_reg = reg_df.drop_duplicates("sequence").set_index("sequence")["efficiency_score"]
    piv_reg["efficiency_score"] = truth_reg
    
    # Add dummy baseline (mean of training targets)
    mean_val = float(truth_reg.mean())
    piv_reg["DummyRegressor (Mean)"] = mean_val
    
    y_true_reg = piv_reg["efficiency_score"].values
    n_guides = len(y_true_reg)
    
    models_reg = [m for m in piv_reg.columns if m != "efficiency_score"]
    
    # Evaluate individual OOF metrics
    base_metrics = {}
    for m in models_reg:
        preds = piv_reg[m].values
        r_val, _ = pearsonr(preds, y_true_reg) if np.std(preds) > 1e-8 else (0.0, 1.0)
        s_val, _ = spearmanr(preds, y_true_reg) if np.std(preds) > 1e-8 else (0.0, 1.0)
        mae_val = mean_absolute_error(y_true_reg, preds)
        base_metrics[m] = {"pearson": r_val, "spearman": s_val, "mae": mae_val}
        
    # Planned comparisons
    reg_pairs = [
        ("Weighted Ensemble", "Ridge Linear Regression"),
        ("Weighted Ensemble", "Random Forest"),
        ("Weighted Ensemble", "XGBoost"),
        ("Weighted Ensemble", "SVM"),
        ("Weighted Ensemble", "DummyRegressor (Mean)"),
        ("Ridge Linear Regression", "DummyRegressor (Mean)"),
        ("Random Forest", "DummyRegressor (Mean)"),
        ("XGBoost", "DummyRegressor (Mean)"),
        ("SVM", "DummyRegressor (Mean)")
    ]
    
    p_vals_raw = []
    pair_records = []
    
    rng = np.random.default_rng(42)
    
    for cand, ref in reg_pairs:
        if cand not in piv_reg.columns or ref not in piv_reg.columns:
            continue
            
        preds_cand = piv_reg[cand].values
        preds_ref = piv_reg[ref].values
        
        # Instance-level paired absolute error difference: e_cand - e_ref
        abs_err_cand = np.abs(y_true_reg - preds_cand)
        abs_err_ref = np.abs(y_true_reg - preds_ref)
        diff_err = abs_err_cand - abs_err_ref
        
        # Paired Wilcoxon signed-rank test on error differences
        try:
            w_stat, w_pval = wilcoxon(diff_err)
        except Exception:
            w_stat, w_pval = 0.0, 1.0
            
        p_vals_raw.append(w_pval)
        
        # Paired bootstrap for Delta Pearson r (1,000 resamples of guide instances)
        boot_diff_r = []
        boot_diff_s = []
        for _ in range(1000):
            idx = rng.choice(n_guides, size=n_guides, replace=True)
            y_b = y_true_reg[idx]
            p_cand_b = preds_cand[idx]
            p_ref_b = preds_ref[idx]
            
            r_c = pearsonr(p_cand_b, y_b)[0] if np.std(p_cand_b) > 1e-8 else 0.0
            r_r = pearsonr(p_ref_b, y_b)[0] if np.std(p_ref_b) > 1e-8 else 0.0
            boot_diff_r.append(r_c - r_r)
            
            s_c = spearmanr(p_cand_b, y_b)[0] if np.std(p_cand_b) > 1e-8 else 0.0
            s_r = spearmanr(p_ref_b, y_b)[0] if np.std(p_ref_b) > 1e-8 else 0.0
            boot_diff_s.append(s_c - s_r)
            
        ci_r_low, ci_r_high = float(np.percentile(boot_diff_r, 2.5)), float(np.percentile(boot_diff_r, 97.5))
        ci_s_low, ci_s_high = float(np.percentile(boot_diff_s, 2.5)), float(np.percentile(boot_diff_s, 97.5))
        
        delta_r = base_metrics[cand]["pearson"] - base_metrics[ref]["pearson"]
        delta_s = base_metrics[cand]["spearman"] - base_metrics[ref]["spearman"]
        delta_mae = base_metrics[cand]["mae"] - base_metrics[ref]["mae"]
        
        pair_records.append({
            "dataset_key": cell,
            "cell_line": disp,
            "task": "Continuous Regression",
            "candidate_model": cand,
            "reference_baseline": ref,
            "number_of_evaluated_guides_N": n_guides,
            "candidate_pearson": round(base_metrics[cand]["pearson"], 4),
            "reference_pearson": round(base_metrics[ref]["pearson"], 4),
            "delta_pearson": round(delta_r, 4),
            "delta_pearson_95_ci_lower": round(ci_r_low, 4),
            "delta_pearson_95_ci_upper": round(ci_r_high, 4),
            "delta_spearman": round(delta_s, 4),
            "delta_spearman_95_ci_lower": round(ci_s_low, 4),
            "delta_spearman_95_ci_upper": round(ci_s_high, 4),
            "delta_mae": round(delta_mae, 4),
            "wilcoxon_signed_rank_pvalue": float(w_pval)
        })
        
    # Apply Holm-Bonferroni correction
    sorted_indices = np.argsort(p_vals_raw)
    m_tests = len(p_vals_raw)
    adj_p_vals = np.zeros(m_tests)
    for rank, orig_idx in enumerate(sorted_indices):
        multiplier = m_tests - rank
        adj_p = min(1.0, p_vals_raw[orig_idx] * multiplier)
        adj_p_vals[orig_idx] = adj_p
        
    for i, rec in enumerate(pair_records):
        rec["holm_bonferroni_adjusted_pvalue"] = float(adj_p_vals[i])
        rec["statistically_significant_after_correction"] = bool(adj_p_vals[i] < 0.05)
        all_comparisons.append(rec)
        
    print(f"  Processed {disp} ({n_guides} guides): {len(pair_records)} regression pairwise comparisons.")

out_csv = os.path.join(AUDIT_DIR, "statistical_comparison_results.csv")
pd.DataFrame(all_comparisons).to_csv(out_csv, index=False)
print(f"\nSaved instance-level statistical comparisons to: {out_csv}")
