# CRISPR Guide AI: A Cell-Line-Specific Software Framework for Dual-Task Regression and Classification of SpCas9 sgRNA Activity

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%2B%20TypeScript-blue.svg)](https://react.dev/)
[![Tests: 26 passed](https://img.shields.io/badge/Tests-26%20passed-success.svg)](backend/tests/)

**CRISPR Guide AI** is an open-source, reproducible computational biology platform for predicting the relative knockout efficacy and binary activity of CRISPR-Cas9 single-guide RNAs (sgRNAs) from 20-nt spacer and 23-nt target sequences (protospacer + PAM).

Every prediction, evaluation metric, and feature attribution in this application originates from **independently trained machine-learning models** evaluated under a rigorous **dataset-specific 80% development + 20% untouched final validation holdout framework with 100-fold cross-validation**, utilizing **16,749 experimentally validated human sgRNAs** across four cell lines.

---

## Key Features

- **Dual-Task Prediction Architecture**: Separate, specialized models for continuous efficacy regression (Relative Efficacy Score: 0–100) and binary knockout activity classification (Active vs. Inactive with calibrated probabilities).
- **Independent Cell-Line Pipelines**: Dedicated models, scalers, and verification metrics for **HeLa**, **HCT116**, **A375**, and **HL60** to prevent cross-assay confounding.
- **600 Biophysical Features**: 28 compositional and transcriptional parameters, 444 position-specific 1-mers and contiguous 2-mers, 80 global $k$-mers, and 48 $n$-gapped dinucleotides.
- **Empirical Uncertainty Estimation**: Estimates prediction dispersion ($\pm 1.96\sigma$) across 100 decision trees in the Random Forest ensemble, representing empirical model disagreement.
- **Locus-Grouped Cross-Validation**: Validated via 10-fold GroupKFold by chromosome locus cluster, proving robust generalization to unseen genomic targets without locus leakage ($r = 0.348$ vs $0.351$ in HeLa).
- **Batch Evaluation & Candidate Ranking**: High-throughput batch prediction supporting up to 1,000 guide sequences with 1-click CSV export.
- **High-Performance Software Architecture**: 600-feature extraction throughput of 1,860.0 sequences/second; linear inference at 495.3 guides/second; lightweight RAM footprint (+95.1 MB).
- **Interactive Web Interface**: Clean React/TypeScript application with real-time positional saliency maps, biophysical rule validation, and dynamic model comparison.

---

## Ground-Truth Datasets & Provenance

The platform is evaluated across **16,749 experimentally validated human sgRNA screen records** from the DeepCRISPR benchmark suite (*Chuai et al., Genome Biology 2018*), standardizing essentiality screens from primary literature:

| Cell Line | Total Records ($N$) | Development Pool (80%) | Holdout Partition (20%) | Sequence Overlap | Screening Methodology | Primary Source |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **HeLa** | 8,101 | 6,480 | 1,621 | **0** | Genome-wide essentiality screen (776 locus clusters) | Hart et al., *Cell* 2015 |
| **HCT116** | 4,239 | 3,391 | 848 | **0** | Genome-wide essentiality screen (775 locus clusters) | Hart et al., *Cell* 2015 |
| **A375 (Doench 2016)** | 2,333 | 1,866 | 467 | **0** | Dense targeted 8-gene tiling screen | Doench et al., *Nat Biotechnol* 2016 |
| **HL60** | 2,076 | 1,660 | 416 | **0** | Genome-wide essentiality screen (238 locus clusters) | Wang et al., *Science* 2014 |

*Note on Dataset Provenance*: The 2,333-guide dataset was distributed in legacy benchmark files under filenames `hek293t.repisgt` (DeepCRISPR) and `hek293.csv` (CRISPRpredSEQ). Forensic sequence and coordinate tracing confirmed 100% concordance with the 8-gene human A375 melanoma screen of Doench et al. (2016). It is definitively designated as **A375 (Doench 2016)**, with legacy filenames preserved for backward compatibility.

---

## Machine Learning Models

### Continuous Efficacy Regression
- **Ridge Linear Regression**: L2-regularized linear baseline ($\alpha = 100.0$) with closed-form solution.
- **Random Forest Regressor**: 100–200 bagged trees providing empirical dispersion bounds.
- **XGBoost Regressor**: Gradient-boosted decision trees optimizing mean squared error.
- **Support Vector Regression (SVR)**: RBF-kernel maximum-margin regression.
- **Constrained Weighted Ensemble**: Non-negative convex simplex combination ($\sum w_i = 1, w_i \ge 0$) optimized strictly on development out-of-fold predictions.
- **Baselines**: `DummyRegressor(strategy="mean")`, `DummyRegressor(strategy="median")`.

### Binary Activity Classification
- **Logistic Regression**: L2-regularized probabilistic baseline.
- **Random Forest Classifier**: Ensemble of balanced decision trees.
- **XGBoost Classifier**: Gradient-boosted decision trees minimizing binary cross-entropy.
- **Support Vector Classifier (SVC)**: Platt-scaled RBF classifier.
- **Ridge Classifier**: Linear classification on $\\{-1, +1\\}$ labels.
- **Operating Thresholds**: Optimized strictly on development out-of-fold continuous predictions using Youden's $J$ statistic ($J = \text{Sensitivity} + \text{Specificity} - 1$).
- **Baseline**: `DummyClassifier(strategy="most_frequent")`.

---

## Publication Package & Evidence Files

The repository includes a complete publication package prepared for *BMC Bioinformatics* (Springer Nature):

### Publication Package (`publication_package/`)
- `CRISPR_Guide_AI_BMC_Manuscript.docx`: Fully formatted Word manuscript with embedded figures and tables.
- `CRISPR_Guide_AI_BMC_Manuscript.md`: Complete GitHub Flavored Markdown manuscript text.
- `figures/`: High-resolution (300 DPI) publication figures (PNG and SVG):
  - `Fig1_System_Architecture`: System dataflow and component architecture.
  - `Fig2_ROC_PR_Curves`: Holdout ROC and PR curves across all 4 cell lines.
  - `Fig3_CrossValidation_Performance`: 100-fold CV performance comparisons.
  - `Fig4_Feature_Ablation`: Controlled ablation study across 8 feature subsets.
  - `Fig5_CrossCellLine_Transferability`: Cross-cell-line transferability heatmaps.
  - `Fig6_Positional_Saliency`: Attribution weights highlighting seed and cleavage flank.
- `tables/`: Standalone publication tables (CSV format).

### Audit Evidence Files (`publication_audit/`)
- `INITIAL_AUDIT_REPORT.md`: Comprehensive system audit and pre-submission assessment.
- `dataset_provenance.csv`: Detailed biological metadata, assay types, and sample sizes.
- `target_audit.json`: Continuous target and binary label distribution statistics.
- `data_quality_report.json`: Missing value and sequence integrity verification.
- `feature_dictionary.csv`: Full definitions, formulas, and ranges for all 600 features.
- `split_integrity_report.json`: Verification of zero sequence overlap ($\text{dev} \cap \text{holdout} = 0$).
- `cv_regression_results.csv` & `cv_classification_results.csv`: 100-fold CV metrics.
- `oof_predictions_regression.csv` & `oof_predictions_classification.csv`: Out-of-fold predictions.
- `historical_holdout_regression.csv` & `historical_holdout_classification.csv`: Holdout evaluations.
- `grouped_validation_results.csv`: 10-fold GroupKFold evaluation by genomic locus cluster.
- `external_validation_results.csv`: Cross-cell-line transferability matrix.
- `feature_ablation_results.csv`: 8-subset feature ablation results.
- `statistical_comparison_results.csv`: Paired t-tests, Wilcoxon tests, and 95% bootstrap CIs.
- `software_benchmark_results.json`: Latency, throughput, and memory profiling.
- `reproducibility_manifest.json`: System environment, dependencies, and SHA-256 checksums.

---

## Installation & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Node.js 18+ and npm (for frontend UI)

### 1. Clone & Set Up Backend Environment
```bash
git clone https://github.com/example/crispr-guide-ai.git
cd crispr-guide-ai

# Install Python dependencies
python -m pip install -r backend/requirements.txt
```

### 2. Run Test Suite
Verify backend functionality, API endpoints, and scientific rigor tests:
```bash
python -m pytest backend/tests/ -v
```
*(All 26 tests should pass in <11 seconds)*

### 3. Launch Backend API Server
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### 4. Launch Frontend UI
In a separate terminal window:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Reproducing Research Results

Run the audit and evaluation scripts from the project root:

```bash
# 1. Independent Scientific Audit (Recomputes holdout metrics from saved artifacts)
python backend/independent_audit.py

# 2. Gene/Locus Grouped Cross-Validation (Quantifies cross-locus generalization)
python experiments/run_gene_grouped_validation.py

# 3. Controlled Feature Ablation Study (Evaluates 8 feature subsets)
python experiments/run_feature_ablation.py

# 4. Cross-Cell-Line Transferability & External Generalization
python experiments/run_external_cross_validation.py

# 5. Statistical Comparisons and 95% Bootstrap Confidence Intervals
python experiments/run_statistical_comparisons.py

# 6. Software Runtime & Throughput Benchmarks
python experiments/run_software_benchmark.py

# 7. Generate Manifest, Publication Figures (300 DPI), and Tables
python experiments/generate_manifest_and_plots.py

# 8. Compile Publication-Ready Word (.docx) and Markdown Manuscripts
python experiments/generate_publication_manuscript.py
```

---

## Manuscript & Citation

A manuscript describing the methodology, experimental validation, and software architecture of CRISPR Guide AI is currently in preparation for submission to *BMC Bioinformatics* (Springer Nature). Formal citation details and DOI will be updated upon publication.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
