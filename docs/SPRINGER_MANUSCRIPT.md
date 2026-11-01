# CRISPR Guide AI: A Cell-Line-Specific Software Framework for Dual-Task Regression and Classification of SpCas9 sgRNA Activity

**Journal:** *BMC Bioinformatics* (Springer Nature)  
**Article Type:** Software Article  
**Status:** Revised Publication Package (Post-Audit Reconciled Submission)  

---

## Authors & Affiliations

**[Author Names Hidden for Blind Peer Review]**  
1. Department of Computer Science and Engineering, Advanced Computational Biology Laboratory  
2. Department of Bioengineering and Biomedical Data Science  

**Corresponding Author:** [Author Email & Institutional Address]  

---

## Abstract

### Background
Predicting CRISPR-Cas9 single-guide RNA (sgRNA) on-target activity is essential for functional genomics, therapeutic genome editing, and high-throughput pooled screening. However, sgRNA efficacy varies markedly across cell lines and screening platforms due to distinct assay readouts, target gene essentiality, and library designs. Existing computational tools frequently combine heterogeneous assays, risk optimistic performance estimates through cross-validation data leakage, lack empirical uncertainty estimation, or evaluate only a single prediction modality (continuous regression or binary classification). We present **CRISPR Guide AI**, an open-source, reproducible computational software framework for cell-line-specific sgRNA activity prediction using separate continuous regression and binary classification pipelines.

### Implementation
CRISPR Guide AI combines an exhaustive 600-dimensional sequence-derived feature extraction pipeline with a cell-line-specific dual-task predictive engine. The feature space captures 28 compositional and transcriptional parameters, 444 position-specific 1-mers and contiguous 2-mers, 80 global $k$-mers, and 48 $n$-gapped dinucleotides. For continuous efficacy, the platform integrates Ridge Regression, Random Forest, XGBoost, Support Vector Regression (SVR), and a constrained simplex-weighted ensemble, reporting Random Forest empirical tree dispersion ($\pm 1.96\sigma_{tree}$) as a measure of ensemble disagreement. For binary activity, it deploys Logistic Regression, Random Forest Classifier, XGBoost Classifier, Support Vector Classifier (SVC), and Ridge Classifier with operating thresholds calibrated via Youden's $J$ statistic on development out-of-fold cross-validation predictions. All preprocessing, scaling, and hyperparameter selections are restricted strictly to development data under an 80/20 stratified partitioning protocol. The system is delivered as a production-grade full-stack application featuring a high-throughput FastAPI backend and a modern React/TypeScript web interface.

### Results
The software framework was evaluated across four benchmark datasets comprising 16,749 experimental records: HeLa ($N = 8,101$), HCT116 ($N = 4,239$), HL60 ($N = 2,076$), and A375 ($N = 2,333$, from Doench et al. 2016). Across 100-fold stratified cross-validation on the development pool, the strongest regression performance was achieved in HCT116 (Ridge Pearson $r = 0.3864 \pm 0.1423$; Simplex Ensemble $r = 0.4019 \pm 0.1427$), HL60 (Ridge $r = 0.3874 \pm 0.2080$; Simplex Ensemble $r = 0.3957 \pm 0.2117$), and HeLa (Ridge $r = 0.3652 \pm 0.0963$; Simplex Ensemble $r = 0.3763 \pm 0.0927$). For binary classification, HL60 yielded the strongest discrimination (SVC $\text{ROC-AUC} = 0.8238 \pm 0.1284$, XGBoost $\text{ROC-AUC} = 0.8211 \pm 0.1179$, $\text{PR-AUC} = 0.9186 \pm 0.0640$). A 50-kb maximum-gap connected genomic locus cross-validation confirmed that genome-wide models generalize to unseen genomic loci without locus leakage ($r = 0.3475$ vs $0.3507$ in HeLa; $r = 0.3600$ vs $0.3562$ in HCT116). In contrast, the A375 dataset exhibited lower predictive performance ($r = 0.2180 \pm 0.2521$ in 100-fold CV; $r = 0.1748$ in 10-fold CV holdout; $\text{ROC-AUC} = 0.5578$), which corresponds to its distinct dataset composition as a dense tiling screen across only 8 genes / 12 loci rather than a genome-wide screen. Cross-cell-line transfer experiments revealed that HeLa and HCT116 share 4,181 guide sequences (98.6% of HCT116) from a common library; transfer correlation on shared sequences was $r = 0.4470$, while transfer to novel unseen sequences achieved $r = 0.2470$. Software profiling confirmed a 600-feature extraction throughput of 1,817.4 sequences/second and linear model inference at 1,042.9 guides/second.

### Conclusions
CRISPR Guide AI provides a transparent, methodologically audited, and reproducible software platform for SpCas9 guide design. By separating continuous efficacy scoring from binary active/inactive calls, reporting empirical ensemble disagreement, establishing cross-locus generalization, and providing high-throughput software interfaces, the framework establishes a reliable foundation for experimental guide RNA selection and design automation. The complete source code, trained model artifacts, test suites, and documentation are freely available under the MIT License at https://github.com/ofcitsaman/CRISPR-Guide-Ai.

**Keywords:** CRISPR-Cas9, sgRNA on-target efficacy, Machine learning, Dual-task prediction, Software engineering, Reproducible bioinformatics, Cross-validation integrity.

---

## Background

The clustered regularly interspaced short palindromic repeats (CRISPR)-Cas9 system has transformed functional genomics and molecular therapeutics by enabling programmable, RNA-guided double-strand DNA cleavage [1, 2]. For the canonical *Streptococcus pyogenes* Cas9 (SpCas9), target recognition is mediated by a 20-nucleotide (nt) spacer sequence within a single-guide RNA (sgRNA), which hybridizes to complementary genomic target DNA adjacent to a 5'-NGG-3' protospacer adjacent motif (PAM) [3]. Cleavage occurs between nucleotide positions 17 and 18 (3 nucleotides upstream of the PAM), initiating cellular DNA repair through non-homologous end joining (NHEJ) or homology-directed repair (HDR) [4].

Despite the apparent simplicity of the Watson-Crick base-pairing mechanism, experimental on-target cleavage efficacy varies widely among sgRNAs, ranging from near-complete target gene disruption to undetectable cleavage [5, 6]. Biophysical studies have demonstrated that efficacy is governed by an intricate combination of primary sequence grammar (e.g., GC content, nucleotide preferences along the 20-nt protospacer), thermodynamic stability in the proximal "seed" region (positions 11-20 adjacent to the PAM), steric accessibility, chromatin structure, and cellular context [7, 8].

Over the past decade, numerous computational algorithms have been developed to predict sgRNA efficacy. Seminal early tools, such as Rule Set 2 / Azimuth [9] and CRISPRpred(SEQ) [10], employed hand-crafted sequence features paired with gradient-boosted regression trees or support vector machines. Subsequently, deep learning architectures—most notably DeepCRISPR [11], CRISPRon [12], and Attn-SpCas9 [13]—demonstrated the utility of convolutional neural networks (CNNs) and attention mechanisms. However, several methodological and software limitations persist across existing bioinformatics solutions:

1. **Conflation of Continuous and Binary Objectives**: Many tools evaluate either continuous efficacy or binary active/inactive status, or convert continuous regression outputs into pseudo-probabilities without calibration, obscuring the distinct statistical semantics of continuous cutting rates versus binary experimental classifications.
2. **Absence of Uncertainty Quantification**: Most publicly available guide design tools return point estimates without confidence intervals, standard errors, or ensemble dispersion metrics, depriving experimentalists of information regarding prediction reliability.
3. **Data Leakage and Cross-Locus Generalization Risks**: Random cross-validation protocols frequently place sgRNAs targeting the same genomic locus or gene into both training and test partitions, risking optimistic performance estimates driven by locus memorization rather than sequence grammar.
4. **Ambiguous Benchmark Provenance**: Benchmark datasets are frequently reused without investigating original experimental conditions, leading to mislabeled cell lines and conflation of genome-wide screens with dense tiling screens.
5. **Software Usability and Deployment Deficits**: Many academic prediction scripts exist solely as unmaintained, container-dependent CLI scripts that lack modern web interfaces, automated unit testing, or high-throughput batch APIs suitable for integration into automated laboratory pipelines.

To resolve these challenges, we introduce **CRISPR Guide AI**, a robust, reproducible, and full-stack computational software framework designed for cell-line-specific sgRNA on-target efficacy prediction. CRISPR Guide AI combines an exhaustive 600-dimensional sequence-derived biophysical feature representation, an audited dual-task machine-learning engine (regression and classification across five algorithms per cell line), empirical uncertainty estimation via Random Forest ensemble tree dispersion, automated rule-based biophysical checks, and a production-grade full-stack architecture with a FastAPI REST backend and a modern React/TypeScript user interface.

---

## Implementation

### Architecture Overview
CRISPR Guide AI is architected as a modular, decoupled full-stack application comprising four core components (**Figure 1**):
1. **Biological Validation and Normalization Engine**: Validates raw nucleotide strings, flags non-canonical characters, processes 20-nt guide spacers or 23-nt protospacer+PAM targets, and detects Pol-III termination motifs.
2. **600-Dimensional Biophysical Feature Pipeline**: Computes 28 compositional parameters, 444 positional 1-mers and contiguous 2-mers, 80 global $k$-mers, and 48 $n$-gapped dinucleotide frequencies.
3. **Dual-Task Predictive Engine**: Hosts cell-line-specific models for continuous efficacy regression and binary activity classification across four benchmark datasets (HeLa, HCT116, A375, and HL60).
4. **Full-Stack Application Layer**: A high-performance asynchronous REST API (FastAPI) and an interactive single-page web application (React, TypeScript, Tailwind CSS) providing real-time single and batch prediction interfaces, positional saliency maps, and metric exploration.

---

### Sequence Input Validation & Normalization
The input validation module enforces strict biological constraints:
- Input sequences are validated to contain exclusively standard DNA characters ($\{A, C, G, T\}$).
- Valid sequence lengths are strictly 20 nucleotides (protospacer without PAM) or 23 nucleotides (protospacer + PAM).
- When a 20-nt protospacer is provided, the engine appends the canonical SpCas9 PAM context (`TGG`), marks `was_extended_with_pam: true`, and explicitly appends a warning to `bio_flags`: *"20-nt protospacer input: canonical 'TGG' PAM appended for model evaluation (assumed context)."*
- Cleavage site geometry is mapped explicitly between positions 17 and 18 (3-nt upstream of PAM).

---

### 600-Dimensional Sequence-Derived Feature Engineering
The feature extraction engine transforms each 23-mer sequence into an exact 600-dimensional numeric vector:
1. **Compositional & Transcriptional Features ($d = 28$, Indices 0–27)**:
   - Sequence length ($1$).
   - Absolute single nucleotide counts ($A, C, G, T$) ($4$).
   - Relative nucleotide frequencies ($4$).
   - Overall GC content and AT content ($2$).
   - Distal region GC content (positions 1–10) ($1$).
   - Proximal seed region GC content (positions 11–20) ($1$).
   - Seed purine percentage ($A+G$ in seed) ($1$).
   - RNA Polymerase III termination penalty: length of maximum poly-T tract (`poly_T_max_run`) and binary indicator for canonical terminator `TTTT` (`has_terminator_TTTT`) ($2$).
   - Canonical PAM indicator (`pam_is_NGG`) and one-hot PAM nucleotide indicators (`pam_base21`, `pam_base22`, `pam_base23`) ($4$).
   - Cleavage flank GC content (positions 16–18) ($1$).
   - Core seed GC content (positions 15–20) ($1$).
   - Seed-distal GC differential ($\text{GC}_{seed} - \text{GC}_{distal}$) ($1$).
   - Cleavage-seed GC product ($\text{GC}_{cleavage} \times \text{GC}_{seed}$) ($1$).
   - Maximum homopolymer tract lengths for G (G-quadruplex potential), C, and A ($3$).
   - Shannon sequence entropy ($H = -\sum p_i \log_2 p_i$) ($1$).
2. **Position-Specific 1-mer Features ($d = 92$, Indices 28–119)**:
   - One-hot indicator for each base $b \in \{A, C, G, T\}$ across all 23 positions ($23 \times 4 = 92$).
3. **Position-Specific Contiguous 2-mer Features ($d = 352$, Indices 120–471)**:
   - One-hot indicator for each dinucleotide pair across all 22 contiguous adjacent positions ($22 \times 16 = 352$).
4. **Global $k$-mer Frequency Counts ($d = 80$, Indices 472–551)**:
   - Position-independent counts of all 16 dinucleotides ($16$) and all 64 trinucleotides ($64$).
5. **$n$-Gapped Dinucleotide Frequencies ($d = 48$, Indices 552–599)**:
   - Frequencies of non-adjacent pairs separated by 1, 2, or 3 intervening nucleotides ($3 \times 16 = 48$).

---

### Machine-Learning Engine: Dual-Task Portfolio
CRISPR Guide AI deploys separate, dedicated models for continuous regression and binary classification:

#### Continuous Regression Models
- **Ridge Linear Regression**: L2-regularized linear model ($\alpha = 100.0$) providing convex, interpretable baselines with closed-form solutions.
- **Random Forest Regressor**: Ensemble of 100–200 de-correlated regression trees with maximum depth regularization.
- **XGBoost Regressor**: Gradient-boosted regression trees optimizing mean squared error with shrinkage ($\eta = 0.05$) and tree pruning.
- **Support Vector Regression (SVR)**: Epsilon-insensitive SVR with Radial Basis Function (RBF) kernel mapping non-linear feature interactions.
- **Constrained Weighted Ensemble**: A non-negative, simplex-constrained convex ensemble ($\sum w_i = 1, w_i \ge 0$) optimized strictly on development out-of-fold predictions.
- **Baselines**: `DummyRegressor(strategy="mean")` and `DummyRegressor(strategy="median")`.

#### Binary Classification Models
- **Logistic Regression**: L2-regularized probabilistic classifier ($C = 0.1$).
- **Random Forest Classifier**: Ensemble of balanced classification trees optimizing Gini impurity.
- **XGBoost Classifier**: Gradient-boosted decision trees minimizing binary cross-entropy.
- **Support Vector Classifier (SVC)**: RBF kernel classifier with Platt scaling for class probabilities.
- **Ridge Classifier**: Linear classifier minimizing squared error on $\{-1, +1\}$ targets.
- **Decision Threshold Optimization**: The operating threshold $\theta$ is calibrated using Youden's $J$ statistic ($J = \text{Sensitivity} + \text{Specificity} - 1$) on out-of-fold continuous prediction scores, maximizing balanced sensitivity and specificity.
- **Baseline**: `DummyClassifier(strategy="most_frequent")`.

---

### Uncertainty Quantification: Ensemble Disagreement
In Random Forest and Weighted Ensemble regression, individual decision trees $t \in \{1, \dots, T\}$ yield individual predictions $y_t(x)$. CRISPR Guide AI computes the standard deviation of tree predictions:
$$\sigma_{tree}(x) = \sqrt{\frac{1}{T-1} \sum_{t=1}^T (y_t(x) - \bar{y}(x))^2}$$
The platform reports the empirical dispersion bounds $[\bar{y}(x) - 1.96\sigma_{tree}, \bar{y}(x) + 1.96\sigma_{tree}]$. It accompanies the output with an explicit disclaimer clarifying that this represents **ensemble disagreement / empirical tree dispersion**, rather than a formally calibrated conformal prediction interval or Bayesian posterior distribution.

---

### Partitioning Integrity and Fixed Historical Holdout
To evaluate models without data leakage:
- Each dataset is partitioned into an **80% Development Pool** and a **20% Fixed Historical Holdout Partition** using stratified sampling (random seed 42).
- Zero sequence overlap ($\text{dev} \cap \text{holdout} = \emptyset$) was verified across all datasets.
- Feature scaling (`StandardScaler`), feature selection, inner hyperparameter optimization, and Youden's $J$ threshold selection are performed strictly inside the development pool.
- **Disclosure of Holdout Status:** The 20% holdout partition was generated with random seed 42 during earlier exploratory phases of the project history. Although preserved untouched during recent model training and parameter freezing, it is formally designated as a **fixed historical holdout partition** rather than a newly quarantined prospective test set.

---

## Results & Discussion

### Dataset Provenance & Resolution of the A375 (Doench 2016) Benchmark
The software was benchmarked across four curated datasets comprising 16,749 experimental sgRNA records (**Table 1**):
1. **HeLa** ($N = 8,101$): Cervical adenocarcinoma; lentiviral dropout screen from Hart et al. (*Cell* 2015) [5]; mean continuous efficiency $0.2569 \pm 0.1821$; positive prevalence 62.46% (5,060 / 8,101).
2. **HCT116** ($N = 4,239$): Colorectal carcinoma; lentiviral dropout screen from Hart et al. (*Cell* 2015) [5]; mean continuous efficiency $0.2543 \pm 0.1717$; positive prevalence 64.12% (2,718 / 4,239).
3. **HL60** ($N = 2,076$): Acute promyelocytic leukemia; essentiality screen from Xu et al. (*Genome Res* 2015) [6] (incorporating screens from Wang et al., *Science* 2014); mean continuous efficiency $0.2322 \pm 0.1135$; positive prevalence 71.00% (1,474 / 2,076).
4. **A375 (Doench 2016)** ($N = 2,333$): Malignant melanoma; drug-resistance and viability tiling screen across 8 genes from Doench et al. (*Nat Biotechnol* 2016) [9]; mean continuous efficiency $0.2376 \pm 0.1198$; positive prevalence 58.89% (1,374 / 2,333).

#### Forensic Resolution of Legacy Mislabeling
In published benchmark repositories, this 2,333-guide dataset was distributed under filenames `hek293t.repisgt` (DeepCRISPR [11]) and `hek293.csv` (CRISPRpredSEQ [10]). However, forensic coordinate and sequence tracing confirmed 100% concordance with the 8-gene A375 melanoma screen of Doench et al. (2016) [9]. DeepCRISTL (Elkayam et al., 2024) [14] independently cataloged these 2,333 guides as A375, contrasting them with true HEK293T data (1,234 guides from Chari et al. 2015). We definitively reclassify this dataset as **A375 (Doench 2016)**, withdraw all claims of embryonic kidney biology, and map all legacy paths to this verified identity.

#### Authoritative Genomic Gene Annotations
Authoritative GRCh37/hg19 gene mapping (GENCODE v19 / Ensembl 75) confirms exact gene boundaries for all eight target genes in A375: *TADA1* (chr1: 166.8 Mb, $N=103$), *CUL3* (chr2: 225.3 Mb, $N=146$), *TADA2B* (chr4: 7.04 Mb, $N=198$), *CCDC101* / *SGF29* (chr16: 28.59 Mb, $N=100$), *NF1* (chr17: 29.42 Mb, $N=598$), *NF2* (chr22: 29.99 Mb, $N=214$), *MED12* (chrX: 70.33 Mb, $N=910$), and *HPRT1* (chrX: 133.59 Mb, $N=63$). This corrects earlier inverted chromosomal assignments (*CUL3* on chr1, *TADA1* on chr4, *TADA2B* on chr17) and accounts for one single-character upstream typo (chr16 instead of chr17 for coordinate 29476816 in *NF1*) and one unmapped legacy artifact (chr21:15373878).

---

### 100-Fold Cross-Validation Performance
Across 100-fold stratified cross-validation on the development pool (**Table 2A**, **Table 2B**, **Figure 2**, **Figure 3**):
- **Continuous Regression**:
  - In **HCT116**, Ridge Linear Regression achieved a mean Pearson $r = 0.3864 \pm 0.1423$ (Spearman $\rho = 0.3783 \pm 0.1482$, $R^2 = 0.1147 \pm 0.1673$, $\text{MAE} = 0.1261 \pm 0.0163$). The Constrained Simplex Ensemble achieved $r = 0.4019 \pm 0.1427$ ($\rho = 0.3949 \pm 0.1480$).
  - In **HL60**, Ridge achieved Pearson $r = 0.3874 \pm 0.2080$ ($\rho = 0.3791 \pm 0.2155$, $\text{MAE} = 0.0805 \pm 0.0149$), and the Ensemble reached $r = 0.3957 \pm 0.2117$.
  - In **HeLa**, Ridge achieved Pearson $r = 0.3652 \pm 0.0963$ ($\rho = 0.3560 \pm 0.1010$, $R^2 = 0.1205 \pm 0.0809$, $\text{MAE} = 0.1352 \pm 0.0124$), and the Ensemble achieved $r = 0.3763 \pm 0.0927$.
  - In **A375**, Ridge yielded $r = 0.2180 \pm 0.2521$ ($\rho = 0.1665 \pm 0.2322$, $\text{MAE} = 0.0951 \pm 0.0171$), with the Ensemble reaching $r = 0.2562 \pm 0.2805$. All models significantly outperformed DummyRegressor ($r = 0.0, R^2 \le 0.0$).
- **Binary Classification**:
  - In **HL60**, Support Vector Classifier (SVC) achieved mean $\text{ROC-AUC} = 0.8238 \pm 0.1284$ ($\text{PR-AUC} = 0.9167 \pm 0.0687$, balanced accuracy $= 0.7548 \pm 0.1166$), and XGBoost achieved $\text{ROC-AUC} = 0.8211 \pm 0.1179$ ($\text{PR-AUC} = 0.9186 \pm 0.0640$, balanced accuracy $= 0.7544 \pm 0.1043$, $\text{F1} = 0.8455 \pm 0.0701$).
  - In **HeLa**, Ridge Classifier and Logistic Regression achieved mean $\text{ROC-AUC} = 0.7143 \pm 0.0599$ and $0.7141 \pm 0.0603$ ($\text{PR-AUC} = 0.7951 \pm 0.0515$, balanced accuracy $= 0.6617 \pm 0.0580$).
  - In **HCT116**, SVC achieved $\text{ROC-AUC} = 0.7182 \pm 0.0889$, Random Forest achieved $\text{ROC-AUC} = 0.7149 \pm 0.0912$, and Logistic Regression achieved $\text{ROC-AUC} = 0.7127 \pm 0.0873$ ($\text{PR-AUC} = 0.8128 \pm 0.0704$, balanced accuracy $= 0.6578 \pm 0.0768$).
  - In **A375**, Random Forest achieved $\text{ROC-AUC} = 0.5643 \pm 0.1536$ and Ridge Classifier achieved $\text{ROC-AUC} = 0.5618 \pm 0.1284$ ($\text{PR-AUC} = 0.6892 \pm 0.1002$, balanced accuracy $= 0.5495 \pm 0.1064$).

---

### 50-kb Genomic-Window-Grouped Cross-Validation & Statistical Feasibility
To evaluate whether models memorize genomic locus characteristics or learn sequence grammar, we clustered sgRNAs using a 50,000 bp maximum-gap connected genomic locus algorithm and performed 10-fold **GroupKFold** cross-validation:
- **HeLa** ($771$ development locus clusters across 23 chromosomes): Window-grouped CV Pearson was $r = 0.3475 \pm 0.0434$ compared to standard random K-fold $r = 0.3507 \pm 0.0291$ (a difference of $\Delta r = -0.0032$). Classification ROC-AUC was $0.7075 \pm 0.0195$ vs. $0.7063 \pm 0.0264$.
- **HCT116** ($761$ development locus clusters): Window-grouped CV Pearson was $r = 0.3600 \pm 0.0403$ compared to random K-fold $r = 0.3562 \pm 0.0398$ ($\Delta r = +0.0038$).
- **HL60** ($236$ development locus clusters): Window-grouped CV Pearson was $r = 0.3470 \pm 0.0597$ compared to random K-fold $r = 0.3373 \pm 0.0706$ ($\Delta r = +0.0097$).

These empirical results show that for genome-wide screening datasets, model performance does not experience substantial degradation when evaluated on unseen 50-kb genomic windows. However, 50-kb window grouping is distinct from true gene grouping: large genes such as *NF1* (279 kb) span multiple 50-kb windows. In A375, all 2,333 guides target only eight genes across 11 development locus clusters; standard 10-fold gene-grouped cross-validation is not statistically supportable ($N_{\text{genes}} = 8 < 10$ folds). Even an 8-fold Leave-One-Gene-Out evaluation leaves $N=1$ gene per test fold, where gene-specific phenotypic variance dominates. Grouped validation on A375 is therefore documented as **Incomplete / Statistically Limited**.

---

### Clustered Statistical Inference
To account for non-independence among guides targeting the same locus, we performed paired cluster bootstrap resampling ($B=1,000$) using 50-kb maximum-gap connected locus clusters as the independent resampling unit, evaluating both the 95% bootstrap confidence interval on model Pearson $r$ and the paired $\Delta\text{MAE}$ improvement over a constant DummyRegressor baseline:
- **HeLa** ($G = 771$ clusters, $N = 6,480$ dev guides): Ridge observed Pearson $r = 0.3619$ (95% CI $[0.3411, 0.3832]$), observed paired $\Delta\text{MAE} = -0.0134$ (95% CI $[-0.0150, -0.0120]$), cluster-level Wilcoxon signed-rank test $p_{\text{adj}} = 2.74 \times 10^{-37}$.
- **HCT116** ($G = 756$ clusters, $N = 3,391$ dev guides): Ridge observed Pearson $r = 0.3833$ (95% CI $[0.3540, 0.4103]$), observed paired $\Delta\text{MAE} = -0.0137$ (95% CI $[-0.0159, -0.0114]$), cluster-level Wilcoxon $p_{\text{adj}} = 3.53 \times 10^{-25}$.
- **HL60** ($G = 236$ clusters, $N = 1,660$ dev guides): Ridge observed Pearson $r = 0.3769$ (95% CI $[0.3371, 0.4153]$), observed paired $\Delta\text{MAE} = -0.0067$ (95% CI $[-0.0084, -0.0049]$), cluster-level Wilcoxon $p_{\text{adj}} = 2.21 \times 10^{-6}$.
- **A375** ($G = 11$ development locus clusters across 8 genes, $N = 1,866$ dev guides): Ridge observed Pearson $r = 0.2205$ (95% CI $[0.1572, 0.3180]$), observed paired $\Delta\text{MAE} = -0.0023$ (95% CI $[-0.0071, 0.0007]$), cluster-level Wilcoxon raw $p = 0.0615$ ($p_{\text{adj}} = 0.123$).

Because A375 comprises only 11 development locus clusters across 8 genes, cluster-level asymptotic tests have low statistical power, and universal cross-gene generalizability cannot be statistically claimed from an 8-gene screen alone.

---

### Dataset Composition Differences: Genome-Wide vs. Tiled Screens
Our coordinate analysis clarifies why A375 exhibits lower predictive correlation ($r \approx 0.175 - 0.22$) than genome-wide screens ($r \approx 0.36 - 0.39$):
- **HeLa**, **HCT116**, and **HL60** are genome-wide knockout screens distributed across 238 to 776 genomic loci spanning all human chromosomes. Guides target thousands of diverse genes, providing broad biological and sequence variability.
- **A375** is a dense tiling screen concentrating 2,333 guides across just eight specific genes. In dense tiling screens, dozens of adjacent guides interrogate the same exon under drug-selection pressure, where chromatin accessibility and phenotype selection dynamics differ substantially from genome-wide screens.

---

### Cross-Cell-Line Transferability & Library Overlap
Evaluating models trained on one cell line against the holdouts of other cell lines revealed that cross-cell transfer is strongly governed by shared library content (**Figure 4**):
- **HeLa** and **HCT116** share **4,181 guide sequences** (98.6% of HCT116) because both screens utilized the Toronto KnockOut (TKOv1) library from Hart et al. [5].
- When the HeLa-trained Ridge model was evaluated on the HCT116 holdout ($N = 848$):
  - For the 667 guides previously seen in the HeLa development pool (78.7%), transfer correlation was $r = 0.4470$.
  - For the 181 guides completely novel and unseen in HeLa (21.3%), transfer correlation was $r = 0.2470$.
  - The overall aggregate transfer correlation was $r = 0.4069$.
- In contrast, transfer between genome-wide screens and A375 yielded $r = 0.12 - 0.17$, reflecting zero library overlap and divergent screening modalities.

Regarding the human erythroleukemia (HEL) cell line referenced in DeepCRISPR [11], raw HEL benchmark records were not included in the public repository release. Independent external validation on HEL was therefore **Not Executed**, and cross-cell-line transfer is not conflated with external validation.

---

### Controlled Feature Ablation Study
We conducted an ablation study across 7 representation subsets (**Figure 5**, **Table 5**):
- **Position-Specific Features (444 Features):** In genome-wide screening datasets, position-specific 1-mer and contiguous 2-mer features achieved predictive performance closely matching the full 600-feature space within $\Delta r \le 0.01$ (HeLa 10-fold CV $r = 0.3444 \pm 0.0304$ vs. $0.3507$, $\Delta r = -0.0063$; HCT116 $r = 0.3471$ vs. $0.3562$, $\Delta r = -0.0091$; HL60 $r = 0.3445$ vs. $0.3373$, $\Delta r = +0.0072$). In A375, position-specific features yielded $r = 0.1769 \pm 0.0533$ vs. $0.1971 \pm 0.0681$ ($\Delta r = -0.0202$), reflecting greater sensitivity to global sequence patterns in this dense 8-gene tiling screen.
- **Basic Composition (28 Features):** Provided lower baseline predictive capacity (HeLa $r = 0.1965$, $\Delta r = -0.1542$; HCT116 $r = 0.1918$; HL60 $r = 0.2446$; A375 $r = 0.1917$).
- **PAM-Only Features (Positions 21–23, 48 Features):** Yielded near-zero predictive correlation ($r \le 0.094$, $\Delta r \approx -0.14$ to $-0.31$, $\text{ROC-AUC} \approx 0.50 - 0.55$), confirming that once sgRNAs are pre-filtered for canonical SpCas9 NGG PAMs, PAM variation alone provides negligible on-target discrimination.

---

### Positional Saliency Profile
Analysis of empirical feature weights along the 23-mer target region revealed strong biological alignment with known Cas9 structural biology (**Figure 6**):
- **Cleavage Flank (Positions 16–17):** Demonstrated the highest attribution weights (peak weight $\approx 0.115$), corresponding to the phosphodiester cleavage site (3 nt upstream of the PAM).
- **Proximal Seed Region (Positions 11–15):** Showed elevated weights ($\approx 0.052 - 0.082$), reflecting the thermodynamic importance of RNA-DNA R-loop nucleation.
- **Distal Heteroduplex Region (Positions 1–10):** Exhibited lower, uniform weights ($\approx 0.022 - 0.045$).
- **Position 20 Thymine Penalty:** A thymine immediately adjacent to the PAM (position 20) consistently incurred negative weights, matching empirical observations of Cas9 binding or transcription inhibition.

---

### Literature-Reported Baseline Comparisons
CRISPR Guide AI was compared descriptively against published guide design tools (**Table 3**):
- Compared to literature-reported benchmarks for **Rule Set 2 / Azimuth** [9] and **CRISPRpred(SEQ)** [10], CRISPR Guide AI demonstrates competitive predictive performance on HeLa and HCT116, while providing dual-task outputs, empirical ensemble disagreement intervals, and an interactive web interface.
- Compared to deep learning models such as **DeepCRISPR** [11] and **CRISPRon** [12], CRISPR Guide AI provides fully interpretable linear and tree-based models that execute in milliseconds on standard CPUs without requiring GPU acceleration. We explicitly note that external tool scores represent literature-reported values and were not re-executed on our specific partitions.

---

### Software Engineering & Runtime Performance
Runtime profiling was conducted on an AMD Ryzen 5 5600H workstation (6 cores, 12 threads, 3.3 GHz base, 16.0 GB RAM, Windows 11 Home, Python 3.13.14) (**Table 4**):
- **Feature Extraction Throughput**: The 600-feature pipeline processed **1,817.4 sequences per second** (1,000 guides in 0.55 seconds).
- **Linear Model Inference (Ridge)**: Operated at **1,042.9 guides per second** (median latency: 0.96 ms end-to-end; estimator alone: 0.09 ms).
- **Tree-Based Estimators**: Median estimator inference alone required **0.38 ms for XGBoost** (2,605.9 guides/s) and **35.98 ms for Random Forest** (27.8 guides/s).
- **Ensemble Runtime Trade-off**: The Weighted Ensemble executes at 11.6 guides/second (~86 ms/guide) because it sequentially loops through all 100 individual trees in the Random Forest to extract tree-level predictions for empirical tree dispersion ($\pm 1.96\sigma_{tree}$).
- **Automated Test Suite**: 33 backend unit, API, provenance, and scientific rigor tests pass in ~8 seconds (`pytest`), and the frontend TypeScript client builds with zero errors.

---

## Conclusions

CRISPR Guide AI delivers a transparent, methodologically audited, and reproducible software platform for CRISPR-Cas9 sgRNA efficacy prediction. By separating continuous efficacy scoring from binary active/inactive calls, reporting empirical ensemble disagreement, demonstrating cross-window generalization on genome-wide screens, uncovering the compositional reason for divergent predictability in tiled screens, resolving the A375 (Doench 2016) benchmark identity, and providing high-throughput software interfaces, the framework establishes a reliable foundation for experimental guide RNA selection.

---

## Declarations

### Ethics Approval and Consent to Participate
Not applicable. The study utilized exclusively published, anonymized, publicly accessible benchmark datasets.

### Consent for Publication
Not applicable.

### Availability of Data and Materials
All source code, documentation, test suites, and model training scripts are deposited in an open-source GitHub repository under the MIT License at:  
https://github.com/ofcitsaman/CRISPR-Guide-Ai  
All raw and processed datasets, feature dictionaries, cross-validation metrics, out-of-fold predictions, and reproducibility manifests are available in the repository `publication_audit/` directory.

### Competing Interests
The authors declare that they have no competing interests.

### Funding
No specific grant from any funding agency in the public, commercial, or not-for-profit sectors was received for this software development.

### Authors' Contributions
Conceptualization: [Author Initials]. Software Architecture and Development: [Author Initials]. Data Curation and Machine Learning: [Author Initials]. Statistical Analysis and Validation: [Author Initials]. Writing – Original Draft: [Author Initials]. Writing – Review & Editing: [Author Initials]. All authors read and approved the final manuscript.

### Acknowledgements
We thank the open-source bioinformatics and machine-learning communities for maintaining scikit-learn, XGBoost, FastAPI, and React.

---

## References

1. Jinek M, Chylinski K, Fonfara I, et al. A programmable dual-RNA-guided DNA endonuclease in adaptive bacterial immunity. *Science*. 2012;337(6096):816–821.
2. Cong L, Ran FA, Cox D, et al. Multiplex genome engineering using CRISPR/Cas systems. *Science*. 2013;339(6121):819–823.
3. Sternberg SH, Redding S, Jinek M, Greene EC, Doudna JA. DNA interrogation by the CRISPR RNA-guided endonuclease Cas9. *Nature*. 2014;507(7490):62–67.
4. Richardson CD, Ray GJ, DeWitt MA, et al. Enhancing homology-directed genome editing by asymmetric donor DNA. *Nat Biotechnol*. 2016;34(3):339–344.
5. Hart T, Chandrashekhar M, Morgan M, et al. High-resolution CRISPR screens reveal fitness genes and genotype-specific cancer liabilities. *Cell*. 2015;163(6):1515–1526.
6. Xu H, Xiao T, Chen CH, et al. Sequence determinants of improved CRISPR knockout library design. *Genome Res*. 2015;25(8):1147–1157. doi:10.1101/gr.191452.115 (incorporating essentiality screens from Wang T, et al. *Science*. 2014;343(6166):80–84).
7. Moreno-Mateos MA, Vejnar CE, Beaudoin JD, et al. CRISPRscan: designing highly efficient sgRNAs for CRISPR-Cas9 targeting in vivo. *Nat Methods*. 2015;12(10):982–988.
8. Labun K, Guo X, Chavez A, et al. Accurate analysis of genuine CRISPR editing events with CRISPRMatch. *Bioinformatics*. 2019;35(14):i297–i304.
9. Doench JG, Fusi N, Sullender M, et al. Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9. *Nat Biotechnol*. 2016;34(2):184–191.
10. Rafid AHM, Toufikuzzaman M, Rahman MS, Rahman MS. CRISPRpred(SEQ): a sequence-based method for sgRNA on target activity prediction using traditional machine learning. *BMC Bioinformatics*. 2020;21(1):223. doi:10.1186/s12859-020-3531-9.
11. Chuai G, Ma H, Yan J, et al. DeepCRISPR: optimized CRISPR guide RNA design by deep learning. *Genome Biol*. 2018;19(1):80.
12. Xiang X, Corsi GI, Anthon C, et al. Enhancing CRISPR-Cas9 gRNA efficiency prediction by data integration and deep learning. *Nat Commun*. 2021;12(1):3238.
13. Zhang Y, Yin L, Cheng Y, Chen W. Attn-SpCas9: an attention-based deep learning model for SpCas9 sgRNA on-target activity prediction. *Brief Bioinform*. 2022;23(4):bbac228.
14. Elkayam E, Tziony R, Orenstein Y. DeepCRISTL: deep learning for CRISPR/Cas9 on-target editing efficiency prediction based on transfer learning. *Bioinformatics*. 2024;40(8):btae481. doi:10.1093/bioinformatics/btae481.
15. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. *J Mach Learn Res*. 2011;12:2825–2830.
16. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. *Proc 22nd ACM SIGKDD*. 2016;785–794.
