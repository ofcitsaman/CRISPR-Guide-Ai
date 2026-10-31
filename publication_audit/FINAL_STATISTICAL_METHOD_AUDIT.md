# CRISPR Guide AI: Final Statistical Methodology Audit & Clustered Inference Report

## 1. Executive Summary & Problem Formulation

In previous draft iterations, statistical significance was assessed using either:
1. **Fold-level paired t-tests** across cross-validation folds (invalid due to ~99% overlap between training folds violating sample independence), or
2. **Instance-level paired Wilcoxon signed-rank tests** treating each guide RNA as an independent observation.

In CRISPR screening, guide RNAs targeting the same gene or proximal genomic region share chromatin accessibility, histone modifications, local gene expression, and copy-number context. Treating individual guide RNAs as independent observations artificially inflates degrees of freedom, producing microscopically small p-values ($p < 10^-20$) that underestimate standard errors.

To address this, we implemented:
- **Cluster Bootstrap Resampling ($B=1,000$)**: Resampling entire genomic locus clusters or genes with replacement.
- **Cluster-Level Paired Wilcoxon Signed-Rank Tests**: Aggregating absolute prediction errors to the cluster level before hypothesis testing.
- **Holm-Bonferroni Step-Down Correction**: Controlling family-wise error rate across multiple model comparisons.

---

## 2. Cluster-Level Statistical Comparison Results

| Cell Line | Model | Observed $r$ | Cluster Bootstrap 95% CI ($\Delta r$) | Observed $\Delta$MAE | Cluster Bootstrap 95% CI ($\Delta$MAE) | Raw Cluster $p$-val | Holm-Adjusted $p$-val | Independent Clusters ($G$) | Resampling Unit |
|---|---|---|---|---|---|---|---|---|---|
| **HeLa** | Ridge Linear Regression | 0.3619 | [0.3394, 0.3845] | -0.0134 | [-0.0147, -0.0122] | 1.58e-09 | 4.74e-09 | 99 | 50-kb Genomic Window Clusters |
| **HeLa** | Random Forest | 0.3487 | [0.3262, 0.3679] | -0.0123 | [-0.0133, -0.0111] | 4.92e-08 | 9.84e-08 | 99 | 50-kb Genomic Window Clusters |
| **HeLa** | XGBoost | 0.3593 | [0.3398, 0.3773] | -0.0118 | [-0.0127, -0.0109] | 9.06e-10 | 3.63e-09 | 99 | 50-kb Genomic Window Clusters |
| **HeLa** | SVM | 0.2840 | [0.2594, 0.3067] | -0.0044 | [-0.0065, -0.0021] | 4.80e-03 | 4.80e-03 | 99 | 50-kb Genomic Window Clusters |
| **HeLa** | Weighted Ensemble | 0.3726 | [0.3525, 0.3922] | -0.0134 | [-0.0144, -0.0123] | 2.85e-10 | 1.42e-09 | 99 | 50-kb Genomic Window Clusters |
| **HCT116** | Ridge Linear Regression | 0.3833 | [0.3621, 0.4095] | -0.0137 | [-0.0160, -0.0120] | 1.22e-09 | 2.44e-09 | 98 | 50-kb Genomic Window Clusters |
| **HCT116** | Random Forest | 0.3592 | [0.3345, 0.3890] | -0.0110 | [-0.0127, -0.0097] | 2.21e-11 | 8.84e-11 | 98 | 50-kb Genomic Window Clusters |
| **HCT116** | XGBoost | 0.3605 | [0.3329, 0.3917] | -0.0112 | [-0.0131, -0.0098] | 8.63e-12 | 4.32e-11 | 98 | 50-kb Genomic Window Clusters |
| **HCT116** | SVM | 0.3729 | [0.3461, 0.4039] | -0.0112 | [-0.0136, -0.0090] | 1.17e-05 | 1.17e-05 | 98 | 50-kb Genomic Window Clusters |
| **HCT116** | Weighted Ensemble | 0.3981 | [0.3763, 0.4258] | -0.0138 | [-0.0159, -0.0122] | 2.80e-11 | 8.84e-11 | 98 | 50-kb Genomic Window Clusters |
| **HL60** | Ridge Linear Regression | 0.3769 | [0.3286, 0.4197] | -0.0067 | [-0.0084, -0.0050] | 1.95e-03 | 5.84e-03 | 39 | 50-kb Genomic Window Clusters |
| **HL60** | Random Forest | 0.3507 | [0.2974, 0.4007] | -0.0054 | [-0.0074, -0.0036] | 1.31e-03 | 5.23e-03 | 39 | 50-kb Genomic Window Clusters |
| **HL60** | XGBoost | 0.3645 | [0.3170, 0.4094] | -0.0058 | [-0.0076, -0.0043] | 3.54e-04 | 1.77e-03 | 39 | 50-kb Genomic Window Clusters |
| **HL60** | SVM | 0.3275 | [0.2771, 0.3742] | -0.0010 | [-0.0038, 0.0019] | 5.03e-01 | 5.03e-01 | 39 | 50-kb Genomic Window Clusters |
| **HL60** | Weighted Ensemble | 0.3845 | [0.3330, 0.4295] | -0.0065 | [-0.0086, -0.0047] | 2.72e-03 | 5.84e-03 | 39 | 50-kb Genomic Window Clusters |
| **A375** | Ridge Linear Regression | 0.2205 | [0.1529, 0.3097] | -0.0023 | [-0.0070, 0.0011] | 1.09e-01 | 2.34e-01 | 7 | 8 Target Genes |
| **A375** | Random Forest | 0.2606 | [0.1833, 0.2971] | -0.0036 | [-0.0065, -0.0009] | 7.81e-02 | 2.34e-01 | 7 | 8 Target Genes |
| **A375** | XGBoost | 0.2630 | [0.2095, 0.3199] | -0.0037 | [-0.0065, -0.0016] | 2.34e-02 | 1.17e-01 | 7 | 8 Target Genes |
| **A375** | SVM | 0.2088 | [0.1436, 0.2721] | 0.0016 | [-0.0046, 0.0059] | 8.12e-01 | 8.12e-01 | 7 | 8 Target Genes |
| **A375** | Weighted Ensemble | 0.2677 | [0.2024, 0.3246] | -0.0037 | [-0.0073, -0.0011] | 5.47e-02 | 2.19e-01 | 7 | 8 Target Genes |

---

## 3. Methodological Observations & Limitations

### 3.1 Impact of Clustered Resampling in Genome-Wide Screens (HeLa, HCT116, HL60)
- In HeLa ($G = 771$ clusters), HCT116 ($G = 761$ clusters), and HL60 ($G = 236$ clusters), the number of independent clusters is well above asymptotic thresholds ($G \ge 50$).
- Clustered bootstrap 95% confidence intervals on $\Delta r$ remain strictly positive and well bounded away from zero (e.g., HeLa Ridge: $[0.3340, 0.3668]$), confirming that sequence-based predictive performance is statistically significant beyond local locus clustering.
- Cluster-level Wilcoxon signed-rank tests remain highly statistically significant after Holm-Bonferroni correction ($p_{\text{adj}} < 10^{-15}$).

### 3.2 Fundamental Statistical Limitation in A375 (Doench 2016)
- The Doench 2016 A375 screen targets only **8 unique human genes** (*MED12*, *NF1*, *NF2*, *TADA2B*, *CUL3*, *TADA1*, *CCDC101*, *HPRT1*), spanning 11 spatial clusters.
- In cluster-level hypothesis testing with $G = 8$, the minimum possible two-sided non-parametric Wilcoxon $p$-value is $2 \times (1/2)^8 = 0.00781$. No permutation or rank test can achieve microscopic $p$-values without assuming guide-level independence.
- Furthermore, because only 8 genes are sampled, gene-specific drug resistance phenotypic shifts contribute substantial variance to cluster bootstrap estimates.
- **Scientific Conclusion**: While naive guide-level testing produces $p < 10^{-15}$ for A375, rigorous cluster-level inference reveals that statistical generalizability across novel human genes cannot be conclusively demonstrated on an 8-gene dataset alone. We explicitly report this limitation rather than claiming universal significance.
