"""
Creates the 4 reproducible research Jupyter notebooks for CRISPR Guide AI:
1. 01_data_exploration.ipynb
2. 02_feature_engineering.ipynb
3. 03_model_training.ipynb
4. 04_model_evaluation.ipynb
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(BASE_DIR, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


# -------------------------------------------------------------
# Notebook 1: Data Exploration
# -------------------------------------------------------------
nb1_cells = [
    md_cell("""# CRISPR Guide AI: Notebook 01 - Data Exploration & Quality Control

### Objective:
Inspect the legitimate human sgRNA on-target knockout efficacy benchmark dataset from the **DeepCRISPR** and **CRISPRpred(SEQ)** studies (Hart et al., Doench et al., Xu et al.).

Key aspects:
- 16,749 experimentally validated sgRNA screening records across 4 human cell lines: HCT116, HEK293, HeLa, HL60
- Distribution of continuous normalized knockout scores and binary active/inactive labels
- Nucleotide composition, GC-content variation, and PAM frequency analysis"""),
    code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("..", "backend", "data", "processed", "master_dataset.csv")
df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)
df.head()"""),
    code_cell("""print("Cell Line Sample Counts:")
print(df['cell_line'].value_counts())
print("\\nBinary Activity Distribution:")
print(df['binary_label'].value_counts(normalize=True))"""),
    code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Efficiency Score Distribution
axes[0].hist(df['efficiency_score'], bins=30, color='#0284c7', edgecolor='black', alpha=0.8)
axes[0].set_title("Distribution of Continuous Knockout Efficiency Scores")
axes[0].set_xlabel("Normalized Knockout Efficiency")
axes[0].set_ylabel("Guide Count")

# GC Content Distribution
gc_vals = df['sgRNA'].apply(lambda s: (s.count('G') + s.count('C')) / len(s) * 100)
axes[1].hist(gc_vals, bins=25, color='#0d9488', edgecolor='black', alpha=0.8)
axes[1].set_title("sgRNA GC Content (%) Distribution")
axes[1].set_xlabel("GC Content (%)")
axes[1].set_ylabel("Guide Count")

plt.tight_layout()
plt.show()"""),
    code_cell("""# Verify PAM requirements (positions 21-23 should end in GG for SpCas9)
pam_motifs = df['sgRNA'].apply(lambda s: s[20:23])
print("Top 5 PAM Motifs:")
print(pam_motifs.value_counts().head(5))""")
]

with open(os.path.join(NB_DIR, "01_data_exploration.ipynb"), "w") as f:
    json.dump(make_notebook(nb1_cells), f, indent=1)
print("Created 01_data_exploration.ipynb")


# -------------------------------------------------------------
# Notebook 2: Feature Engineering
# -------------------------------------------------------------
nb2_cells = [
    md_cell("""# CRISPR Guide AI: Notebook 02 - Feature Engineering Engine

### Objective:
Extract 593 biologically motivated sequence features from 23-mer guide RNAs:
1. Basic sequence composition (A, C, G, T counts and percentages, overall GC and AT)
2. Sub-region GC content: PAM-distal (positions 1-10) vs. PAM-proximal Seed region (positions 11-20)
3. Transcription termination hazards (poly-T motifs, e.g. TTTT in U6/H1 promoters)
4. Position-Specific Features (PSF: 1-mer at 23 positions, adjacent 2-mers across 22 positions)
5. Position-Independent Features (PIF: 2-mer counts, 3-mer counts)
6. n-Gapped Di-nucleotides (nGD: pairs at distance gaps 1, 2, 3)"""),
    code_cell("""import sys
sys.path.append("..")
from backend.app.features.extractor import CRISPRFeatureExtractor, validate_sgrna_sequence
import pandas as pd

extractor = CRISPRFeatureExtractor()
print(f"Total features per guide: {len(extractor.feature_names)}")"""),
    code_cell("""# Test on sample benchmark guide
sample_seq = "CTTGCTCGCGCAGGACGAGGCGG"
validated, extended = validate_sgrna_sequence(sample_seq)
feat_dict = extractor.extract_features_single(validated)

print("Sample Feature Values for:", validated)
print(f"  Overall GC%:    {feat_dict['gc_content']*100:.1f}%")
print(f"  Distal GC%:     {feat_dict['distal_gc']*100:.1f}%")
print(f"  Seed GC%:       {feat_dict['seed_gc']*100:.1f}%")
print(f"  Cleavage GC%:   {feat_dict['cleavage_flank_gc']*100:.1f}%")
print(f"  Poly-T run:     {feat_dict['poly_T_max_run']}")
print(f"  Canonical PAM:  {feat_dict['pam_is_NGG']}")"""),
    code_cell("""# Batch feature extraction demo
df_sample = pd.read_csv("../backend/data/processed/sample_test_data.csv")
X_sample = extractor.extract_features_df(df_sample['sgRNA'].tolist())
print("Sample feature matrix shape:", X_sample.shape)
X_sample.iloc[:5, :12]""")
]

with open(os.path.join(NB_DIR, "02_feature_engineering.ipynb"), "w") as f:
    json.dump(make_notebook(nb2_cells), f, indent=1)
print("Created 02_feature_engineering.ipynb")


# -------------------------------------------------------------
# Notebook 3: Model Training
# -------------------------------------------------------------
nb3_cells = [
    md_cell("""# CRISPR Guide AI: Notebook 03 - Model Training Pipeline

### Objective:
Train 4 distinct machine learning architectures under dataset-specific 80/20 holdout + 100-fold CV:
1. **Baseline Linear Model**: Ridge Regression with hyperparameter-tuned alpha
2. **Random Forest**: Decorrelated decision trees with empirical ensemble dispersion (±1.96σ)
3. **XGBoost Regressor**: Gradient-boosted trees with column/subsample regularization
4. **Support Vector Regression (SVR)**: RBF-kernel support vector machine with optimized margin and gamma"""),
    code_cell("""import sys
sys.path.append("..")
import os
import joblib
import json
from backend.app.training.train import run_training_pipeline

# Check if pre-trained models already exist
SAVED_DIR = os.path.join("..", "backend", "saved_models")
if os.path.exists(os.path.join(SAVED_DIR, "metrics.json")):
    with open(os.path.join(SAVED_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    print("Models are already trained!")
    print("Best Model Identified:", metrics.get("best_model"))
    print("\\nRandom Split Performance Summary:")
    for model_name, m_dict in metrics.get("random_split", {}).items():
        print(f"  {model_name:18} -> Pearson r: {m_dict['pearson']:.4f}, Spearman rho: {m_dict['spearman']:.4f}, RMSE: {m_dict['rmse']:.4f}")
else:
    print("Running training pipeline...")
    run_training_pipeline()""")
]

with open(os.path.join(NB_DIR, "03_model_training.ipynb"), "w") as f:
    json.dump(make_notebook(nb3_cells), f, indent=1)
print("Created 03_model_training.ipynb")


# -------------------------------------------------------------
# Notebook 4: Model Evaluation
# -------------------------------------------------------------
nb4_cells = [
    md_cell("""# CRISPR Guide AI: Notebook 04 - Rigorous Model Evaluation & Explainability

### Objective:
Analyze model performance across multiple evaluation dimensions:
1. 100-Fold Stratified Cross-Validation vs. Final 20% Holdout validation
2. Out-of-fold predictions and Youden's J operating threshold optimization
3. Empirical Ensemble Dispersion estimation (Random Forest)
4. Model-derived feature importance and empirical positional attribution"""),
    code_cell("""import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SAVED_DIR = os.path.join("..", "backend", "saved_models")
with open(os.path.join(SAVED_DIR, "metrics.json")) as f:
    metrics = json.load(f)

# Random Split comparison table
random_df = pd.DataFrame(metrics["random_split"]).T
random_df = random_df[["pearson", "spearman", "rmse", "mae", "r2", "roc_auc", "pr_auc"]]
random_df.columns = ["Pearson r", "Spearman rho", "RMSE", "MAE", "R2", "ROC-AUC", "PR-AUC"]
print("Random 75/25 Train-Test Split Results:")
random_df"""),
    code_cell("""# Leave-One-Cell-Out Cross-Validation Table
loco_data = {}
for cell, val in metrics.get("leave_cell_out", {}).items():
    m = val["metrics"]
    loco_data[f"Held-Out: {cell}"] = {
        "Samples": val["held_out_samples"],
        "Pearson r": m["pearson"],
        "Spearman rho": m["spearman"],
        "RMSE": m["rmse"],
        "ROC-AUC": m["roc_auc"]
    }
loco_df = pd.DataFrame(loco_data).T
print("Leave-One-Cell-Out Generalization Performance:")
loco_df"""),
    code_cell("""# Top 15 Predictive Sequence Features Across the Genome
with open(os.path.join(SAVED_DIR, "feature_importance.json")) as f:
    feat_imp = json.load(f)

imp_df = pd.DataFrame(feat_imp).head(15)
plt.figure(figsize=(10, 6))
plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#2563eb")
plt.title("Top 15 Most Predictive Sequence Features (XGBoost Importance)")
plt.xlabel("Relative Feature Importance")
plt.tight_layout()
plt.show()""")
]

with open(os.path.join(NB_DIR, "04_model_evaluation.ipynb"), "w") as f:
    json.dump(make_notebook(nb4_cells), f, indent=1)
print("Created 04_model_evaluation.ipynb")
