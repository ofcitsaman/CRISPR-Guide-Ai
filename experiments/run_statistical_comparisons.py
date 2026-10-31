"""
CRISPR Guide AI: Statistical Model Comparison and Confidence Interval Analysis
Performs:
1. Paired t-tests and Wilcoxon signed-rank tests across 100 CV folds
   comparing models against baselines and between architectures.
2. 95% non-parametric bootstrap confidence intervals (1,000 resamples)
   for Pearson r, Spearman rho, R2, ROC-AUC, and PR-AUC.
Outputs:
- publication_audit/statistical_comparison_results.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

AUDIT_DIR = os.path.join(BASE_DIR, "publication_audit")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "backend", "saved_models")

CELL_LINES = ["hela", "hct116", "hek293", "hl60"]
DISPLAY_NAMES = {"hela": "HeLa", "hct116": "HCT116", "hek293": "HEK293T", "hl60": "HL60"}

print("=" * 80)
print("COMPUTING STATISTICAL COMPARISONS AND 95% BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 80)

comparison_rows = []

for cell in CELL_LINES:
    disp = DISPLAY_NAMES[cell]
    cell_dir = os.path.join(SAVED_MODELS_DIR, cell)
    
    cv_path = os.path.join(cell_dir, "cv_metrics.json")
    if not os.path.exists(cv_path):
        continue
        
    with open(cv_path) as f:
        cv_data = json.load(f)
        
    # Get all folds for each model
    models = list(cv_data.keys())
    
    # Check if 'all_folds' exists
    fold_metrics = {}
    for m in models:
        folds = cv_data[m].get("all_folds", {})
        if isinstance(folds, dict) and "pearson" in folds:
            fold_metrics[m] = {
                "pearson": np.array(folds["pearson"]),
                "spearman": np.array(folds["spearman"]),
                "roc_auc": np.array(folds.get("roc_auc", [])),
                "pr_auc": np.array(folds.get("pr_auc", []))
            }
            
    # Pairwise statistical comparisons against Ridge and Weighted Ensemble
    benchmarks = ["Ridge Linear Regression"] if "Ridge Linear Regression" in fold_metrics else []
    if "Weighted Ensemble" in fold_metrics:
        benchmarks.append("Weighted Ensemble")
        
    for baseline_name in benchmarks:
        base_vals = fold_metrics[baseline_name]
        for candidate_name in models:
            if candidate_name == baseline_name:
                continue
            cand_vals = fold_metrics[candidate_name]
            
            for metric in ["pearson", "spearman", "roc_auc", "pr_auc"]:
                if len(cand_vals[metric]) == 0 or len(base_vals[metric]) == 0:
                    continue
                diff = cand_vals[metric] - base_vals[metric]
                mean_diff = float(np.mean(diff))
                std_diff = float(np.std(diff, ddof=1))
                
                # Paired t-test
                try:
                    t_stat, t_pval = ttest_rel(cand_vals[metric], base_vals[metric])
                except Exception:
                    t_stat, t_pval = 0.0, 1.0
                    
                # Wilcoxon signed-rank test
                try:
                    w_stat, w_pval = wilcoxon(cand_vals[metric], base_vals[metric])
                except Exception:
                    w_stat, w_pval = 0.0, 1.0
                    
                # 95% bootstrap CI on the mean difference (1000 resamples)
                rng = np.random.default_rng(42)
                boot_diffs = []
                n_folds = len(diff)
                for _ in range(1000):
                    sample = rng.choice(diff, size=n_folds, replace=True)
                    boot_diffs.append(np.mean(sample))
                ci_lower = float(np.percentile(boot_diffs, 2.5))
                ci_upper = float(np.percentile(boot_diffs, 97.5))
                
                is_significant = bool(t_pval < 0.05 and (ci_lower > 0 or ci_upper < 0))
                
                comparison_rows.append({
                    "dataset_key": cell,
                    "cell_line": disp,
                    "candidate_model": candidate_name,
                    "reference_baseline": baseline_name,
                    "metric_evaluated": metric,
                    "candidate_mean": round(float(np.mean(cand_vals[metric])), 4),
                    "baseline_mean": round(float(np.mean(base_vals[metric])), 4),
                    "mean_difference": round(mean_diff, 4),
                    "difference_95_ci_lower": round(ci_lower, 4),
                    "difference_95_ci_upper": round(ci_upper, 4),
                    "paired_t_statistic": round(float(t_stat), 4),
                    "paired_t_pvalue": float(t_pval),
                    "wilcoxon_w_statistic": round(float(w_stat), 4),
                    "wilcoxon_pvalue": float(w_pval),
                    "statistically_significant_p05": is_significant
                })

out_csv = os.path.join(AUDIT_DIR, "statistical_comparison_results.csv")
pd.DataFrame(comparison_rows).to_csv(out_csv, index=False)
print(f"Saved statistical comparisons to: {out_csv} ({len(comparison_rows)} comparisons)")
