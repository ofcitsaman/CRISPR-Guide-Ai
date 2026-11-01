# CRISPR Guide AI: Research Methodology

## 1. Biological Mechanism of CRISPR-Cas9
The Type II Clustered Regularly Interspaced Short Palindromic Repeats (CRISPR)-associated Cas9 endonuclease system (specifically *Streptococcus pyogenes* Cas9, or SpCas9) performs targeted DNA double-strand breaks (DSBs) directed by a single-guide RNA (sgRNA).

Recognition and cleavage proceed through well-defined biophysical phases:
1. **PAM Interrogation**: Cas9 scans genomic DNA and binds to a `5'-NGG-3'` Protospacer Adjacent Motif (PAM), located immediately 3' of the target protospacer (positions 21–23 in standard 23-mer numbering).
2. **Seed Sequence Unwinding & R-Loop Formation**: Following PAM binding, local melting of DNA begins at the PAM-proximal **seed region** (positions 11–20, especially the 8–10 nucleotides directly adjacent to the PAM). Base pairing between the guide RNA and the target DNA strand forms a stable R-loop structure.
3. **PAM-Distal Extension**: Conformational activation occurs as base pairing extends through the PAM-distal segment (positions 1–10).
4. **Endonucleolytic Cleavage**: Cas9's two catalytic nuclease domains (HNH domain cutting the target strand and RuvC domain cutting the non-target strand) cleave the phosphodiester backbone **3 base pairs upstream of the PAM** (precisely between positions 17 and 18).
5. **Host Repair & Functional Consequence**: Non-Homologous End Joining (NHEJ) repairs the DSB, frequently introducing insertion/deletion (indel) mutations that cause frameshifts and functional gene knockout.

---

## 2. Feature Engineering Architecture
Our framework extracts **600 biologically grounded sequence features** from each 23-mer sequence ($S = s_1 s_2 \dots s_{23}$):

### A. Basic Sequence & Biophysical Composition (28 features)
- Base counts and frequencies: $f_b = \frac{1}{23}\sum_{i=1}^{23}\mathbb{I}(s_i = b)$ for $b \in \{A, C, G, T\}$.
- Overall GC content ($f_G + f_C$) and AT content ($f_A + f_T$).
- Sub-region GC properties: PAM-distal GC (positions 1–10), PAM-proximal seed GC (positions 11–20), cleavage-flank GC (positions 16–18), core seed GC (positions 15–20), distal-seed GC difference, and cleavage-seed GC product.
- Homopolymer repeat runs: maximum consecutive A, C, G, and T runs.
- Poly-T transcription termination motif flag ($\mathbb{I}(\text{"TTTT"} \in S)$), which triggers premature RNA polymerase III termination in U6/H1 guide expression cassettes.
- Shannon sequence entropy ($H = -\sum_b p_b \log_2 p_b$).
- PAM motif indicators (positions 21–23, canonical SpCas9 `NGG`).

### B. Position-Specific Features (PSF, 444 features)
- **1-mer Positional Identity**: 92 binary indicators $\mathbb{I}(s_i = b)$ for $i \in \{1, \dots, 23\}$ and $b \in \{A, C, G, T\}$.
- **Adjacent 2-mer Positional Identity**: 352 binary indicators $\mathbb{I}(s_i s_{i+1} = d)$ for $i \in \{1, \dots, 22\}$ and $d \in \{AA, AC, \dots, TT\}$.

### C. Position-Independent k-mer Frequencies (PIF, 80 features)
- Count of all 16 dinucleotides across the entire sequence.
- Count of all 64 trinucleotides capturing local secondary structure tendencies and base-stacking thermodynamics.

### D. n-Gapped Di-nucleotides (nGD, 48 features)
- Pairs of nucleotides separated by gap distances $g \in \{1, 2, 3\}$, capturing longer-range non-adjacent steric and melting interactions along the spacer:
  $$\text{count}(b_1, b_2, g) = \sum_{i=1}^{23 - (g + 1)} \mathbb{I}(s_i = b_1 \land s_{i+g+1} = b_2)$$

---

## 3. Computational Models
 
### Model 1: Baseline Linear Model (Ridge Regression)
$$y = \mathbf{w}^T \mathbf{x} + b, \quad \min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$
Fitted with standard scaling and feature selection strictly on training folds. Hyperparameter $\alpha \in [100.0, 300.0]$ is optimized within cross-validation.

### Model 2: Random Forest Regressor
Ensemble of decorrelated CART regression trees with feature subsampling.
**Empirical Ensemble Dispersion Estimation**:
Unlike uncalibrated single models, Random Forest produces an empirical ensemble distribution:
$$\bar{y} = \frac{1}{B} \sum_{b=1}^B \hat{y}_b, \quad \sigma_{\text{ensemble}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B (\hat{y}_b - \bar{y})^2}$$
An empirical dispersion interval is formulated as $[\bar{y} - 1.96\sigma_{\text{ensemble}}, \bar{y} + 1.96\sigma_{\text{ensemble}}]$ representing inter-tree disagreement.

### Model 3: XGBoost (Gradient Boosted Decision Trees)
Successively fits trees to second-order gradient approximations of the regularized objective:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|$$
Provides leading predictive ranking accuracy on tabular sequence features.

### Model 4: Support Vector Regression (SVR with RBF Kernel)
Fits a flat function within an $\epsilon$-insensitive loss margin in high-dimensional Hilbert space:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)$$
Using Radial Basis Function (RBF) kernel $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$ with tuned $C, \epsilon, \gamma$ to capture epistatic sequence interactions.

---

## 4. Evaluation Strategy & Avoiding Data Leakage
1. **Dataset-Specific Evaluation**: Evaluates each cell-line dataset independently (HeLa, HCT116, A375, HL60).
2. **80/20 Sequence-Disjoint Outer Split**: Zero sequence overlap is strictly enforced between the 80% development pool and the 20% final validation holdout set ($\text{Overlap} = 0$).
3. **100-Fold Stratified Cross-Validation on Development Pool**: Preprocessing transformers (`StandardScaler`, `SelectKBest`) are fitted strictly on each fold's training slice.
4. **Out-of-Fold (OOF) Prediction & Threshold Optimization**: Operating threshold $\theta^*$ is selected via Youden's $J = \text{TPR} - \text{FPR}$ on out-of-fold continuous predictions and frozen before evaluating the holdout set.
5. **Final Validation on Fixed Historical 20% Holdout**: Models are trained on the complete 80% development pool and evaluated once on the fixed historical 20% holdout partition.
