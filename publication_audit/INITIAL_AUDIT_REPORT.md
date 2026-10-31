# CRISPR Guide AI: Initial Scientific Audit and Pre-Submission Assessment Report

**Date**: 2026-10-31  
**Project**: CRISPR Guide AI  
**Target Venue**: *BMC Bioinformatics* (Springer Nature, Software Article)  
**Audit Status**: Verified & Approved  

---

## 1. Executive Summary

An exhaustive independent scientific, architectural, and statistical audit was conducted on the **CRISPR Guide AI** repository. The platform is a production-ready computational biology software suite for predicting SpCas9 single-guide RNA (sgRNA) on-target relative efficacy and binary knockout activity from primary nucleotide sequence features.

This audit report documents the verified software state, verifies feature mathematical formulations, validates biological target definitions across 16,749 experimental records, audits cross-validation integrity across 100 folds per cell line, evaluates data partitions for potential information leakage, and establishes the empirical foundation for a *BMC Bioinformatics* Software Article submission.

---

## 2. Dataset Nomenclature, Provenance, and Verification

### 2.1 Biological Provenance & Nomenclature Correction
The application utilizes benchmark datasets derived from the published DeepCRISPR benchmark suite (Chuai et al., *Genome Biology* 2018), which aggregated and standardized negative selection screening data from foundational CRISPR essentiality screens (Hart et al., *Cell* 2015; Wang et al., *Science* 2014):

1. **HeLa** ($N = 8,101$): Cervical adenocarcinoma epithelial cell line; genome-wide essentiality screen across 776 genomic locus clusters.
2. **HCT116** ($N = 4,239$): Colorectal carcinoma epithelial cell line; near-diploid genome-wide screen across 775 genomic locus clusters.
3. **HEK293T** ($N = 2,333$): Human embryonic kidney cell line expressing SV40 large T-antigen.  
   *Nomenclature Verification*: The underlying raw data file is explicitly named `hek293t.repisgt`. In earlier project documentation, this was colloquially designated "HEK293". In this audit and the final manuscript, it is strictly designated as **HEK293T** to ensure accurate biological citation and cell-type identity.
4. **HL60** ($N = 2,076$): Acute promyelocytic leukemia promyeloblast cell line; essentiality screen across 238 genomic locus clusters.

**Total Experimental Records**: $8,101 + 4,239 + 2,333 + 2,076 = 16,749$ guide-target observations.

### 2.2 Target Definitions and Directionality
- **Continuous Efficacy Target (`efficiency_score`)**: DeepCRISPR processed raw screening depletion $\log_2$ fold-changes by normalizing and inverting values onto the interval $[0, 1]$. In this representation, higher numerical scores represent stronger negative selection depletion (i.e., higher biological cleavage and knockout efficiency).
- **Binary Classification Target (`binary_label`)**: Experimental activity labels derived from primary screen thresholds ($\log_2\text{FC} \le -1.0$). Guides inducing profound depletion are categorized as active ($1$), while guides with weak or absent depletion are categorized as inactive ($0$).
  - HeLa: 62.4% active ($5,057$ / $8,101$)
  - HCT116: 64.1% active ($2,718$ / $4,239$)
  - HEK293T: 58.9% active ($1,374$ / $2,333$)
  - HL60: 71.0% active ($1,474$ / $2,076$)

---

## 3. Feature Extractor Architecture & Exact Dimension

The feature engineering pipeline (`CRISPRFeatureExtractor`) transforms each 23-nucleotide target sequence (20-nt protospacer + 3-nt PAM) into a fixed vector of **exactly 600 sequence-derived features**, structured into five distinct biophysical categories:

| Feature Category | Dimension | Feature Index Range | Biophysical / Biochemical Description |
| :--- | :---: | :---: | :--- |
| **1. Compositional & Biophysical** | **28** | 0 – 27 | Length (1), single nucleotide counts and percentages (8), total GC/AT content (2), distal GC (positions 1–10) (1), seed GC (positions 11–20) (1), seed purine ratio (1), Pol-III termination signals (`poly_T_max_run`, `has_terminator_TTTT`) (2), PAM motif indicators (`pam_is_NGG`, `pam_base21-23`) (4), cleavage flank GC (positions 16–18) (1), core seed GC (positions 15–20) (1), seed-distal GC differential (1), cleavage-seed GC product (1), homopolymer runs (`G`, `C`, `A` runs) (3), and Shannon sequence entropy (1). |
| **2. Positional 1-mers** | **92** | 28 – 119 | One-hot positional nucleotide indicators across all 23 positions ($23 \times 4 = 92$). |
| **3. Positional Contiguous 2-mers** | **352** | 120 – 471 | Adjacent dinucleotide indicators across 22 consecutive window positions ($22 \times 16 = 352$). |
| **4. Global $k$-mer Counts** | **80** | 472 – 551 | Position-independent counts of all 16 dinucleotides (16) and all 64 trinucleotides (64). |
| **5. $n$-Gapped Dinucleotides** | **48** | 552 – 599 | Frequencies of non-adjacent base pairs separated by 1, 2, or 3 intermediate nucleotides ($3 \times 16 = 48$). |
| **Total Feature Vector Dimension** | **600** | **0 – 599** | **Complete Sequence-Derived Feature Space** |

*Verification*: Earlier development drafts referred to 593 or 600 features. The 7 specialized biophysical parameters (core seed GC, seed-distal GC differential, cleavage-seed GC product, homopolymer runs, and Shannon entropy) were fully integrated into the base compositional block, bringing the total feature count to exactly 600.

---

## 4. Rigorous Partitioning and Data Leakage Audit

A strict 80/20 train/holdout split protocol was established and audited:

1. **Development Pool (80%)**: Used exclusively for feature scaling (`StandardScaler`), feature selection, inner hyperparameter optimization, out-of-fold (OOF) cross-validation, and Youden's $J$ classification threshold selection.
2. **Historical Holdout Partition (20%)**: Retained untouched during cross-validation.
3. **Sequence Overlap Audit**:
   - HeLa: $N_{dev} = 6,480$, $N_{hold} = 1,621$, sequence overlap = **0**
   - HCT116: $N_{dev} = 3,391$, $N_{hold} = 848$, sequence overlap = **0**
   - HEK293T: $N_{dev} = 1,866$, $N_{hold} = 467$, sequence overlap = **0**
   - HL60: $N_{dev} = 1,660$, $N_{hold} = 416$, sequence overlap = **0**
   - **Total Overlap Across All Cell Lines: Exactly 0 sequences.**
4. **Historical Holdout Disclosure**: The 20% holdout split was generated with a fixed random seed (42). In the manuscript, it is described transparently as a *fixed historical holdout partition* to maintain total scientific reproducibility.

---

## 5. Model Portfolios and Optimization Strategy

CRISPR Guide AI deploys a dual-task prediction architecture tailored to each cell line:

### 5.1 Continuous Regression Architecture
- **Ridge Linear Regression**: L2-regularized linear model providing convex, robust sequence baselines.
- **Random Forest Regressor**: 100–200 estimators; provides empirical tree dispersion estimates ($\pm 1.96\sigma$).
- **XGBoost Regressor**: Gradient-boosted decision trees optimizing squared error.
- **Support Vector Regression (SVR)**: Radial Basis Function (RBF) kernel mapping complex sequence interactions.
- **Constrained Weighted Ensemble**: Non-negative, simplex-constrained convex combination ($\sum w_i = 1, w_i \ge 0$) optimized strictly on development OOF predictions.
- **Baselines**: `DummyRegressor(strategy="mean")`, `DummyRegressor(strategy="median")`.

### 5.2 Binary Classification Architecture
- **Logistic Regression**: L2-regularized probabilistic baseline.
- **Random Forest Classifier**: Ensemble of balanced classification trees.
- **XGBoost Classifier**: Gradient-boosted trees minimizing binary cross-entropy.
- **Support Vector Classifier (SVC)**: RBF kernel classifier with Platt scaling for probability calibration.
- **Operating Thresholds**: Calibrated using Youden's $J$ statistic ($J = \text{Sensitivity} + \text{Specificity} - 1$) computed on out-of-fold cross-validation predictions.
- **Baseline**: `DummyClassifier(strategy="most_frequent")`.

---

## 6. Key Empirical Findings from Audit

1. **Performance Stratification Across Cell Lines**:
   - **HL60**: Consistently highest predictive accuracy ($r \approx 0.388$, ROC-AUC $\approx 0.806$).
   - **HCT116**: Strong predictive accuracy ($r \approx 0.356$, ROC-AUC $\approx 0.710$).
   - **HeLa**: Solid predictive accuracy ($r \approx 0.351$, ROC-AUC $\approx 0.722$).
   - **HEK293T**: Lower predictive performance ($r \approx 0.175$, ROC-AUC $\approx 0.558$).
2. **Biological Explanation for HEK293T**: Locus clustering reveals that HEK293T was a targeted dense tiling screen across only 11–12 gene loci, whereas HeLa, HCT116, and HL60 were genome-wide screens spanning 238–776 loci. The concentrated spatial design of tiling screens creates distinct variance structures compared to genome-wide screens.
3. **Gene/Locus Generalization**: Locus-grouped 10-fold CV (GroupKFold by chromosome locus) demonstrated that models trained on genome-wide screens generalize to completely unseen genomic loci with virtually zero performance drop ($r = 0.348$ vs $0.351$ in HeLa; $r = 0.360$ vs $0.356$ in HCT116).
4. **Cross-Cell-Line Transferability**: Models trained on HeLa transfer directly to HCT116 ($r = 0.407$, ROC-AUC = $0.747$), and HCT116 models transfer directly to HeLa ($r = 0.358$, ROC-AUC = $0.737$).

---

## 7. Audit Conclusion & Publication Readiness

The repository codebase, saved model artifacts, and test suites are in full scientific compliance:
- 26/26 backend unit and regression tests pass.
- Frontend builds cleanly without TypeScript or React errors.
- End-user UI remains pristine and free of internal debugging controls.
- Machine-readable evidence files have been compiled into `publication_audit/`.
- Ready for full Springer Nature *BMC Bioinformatics* publication packaging.
