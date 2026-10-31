# CRISPR Guide AI: Dataset Documentation & Provenance

## 1. Overview & Provenance
The benchmark dataset used in CRISPR Guide AI comprises **16,749 experimentally validated human sgRNA target sites** across 1,071 human genes. The dataset unifies the benchmark curated by:
- **DeepCRISPR** (*Chuai et al., Genome Biology 2018; 19:80*)
- **CRISPRpred(SEQ)** (*Rafid et al., BMC Bioinformatics 2020; 21:223*)
- **CRISPOR Benchmark Study** (*Haeussler et al., Genome Biology 2016; 17:148*)

## 2. Component Cell Lines & Experimental Screens
The dataset aggregates experimental readouts from negative-selection fitness screens in 4 human cancer cell lines:

| Cell Line | Primary Study | Assayed Guides | Organism & Tissue Origin | Experimental Assay |
| :--- | :--- | :--- | :--- | :--- |
| **HeLa** | Hart et al., *Cell* 2015 | 8,101 | *Homo sapiens*, Cervical adenocarcinoma | Lentiviral negative selection knockout screen |
| **HCT116** | Hart et al., *Cell* 2015 | 4,239 | *Homo sapiens*, Colorectal carcinoma | Lentiviral negative selection knockout screen |
| **A375** | Doench et al., *Nat Biotechnol* 2016 | 2,333 | *Homo sapiens*, Malignant melanoma | High-throughput drug-resistance / viability 8-gene tiling screen |
| **HL60** | Xu et al., *Genome Research* 2015 | 2,076 | *Homo sapiens*, Acute promyelocytic leukemia | High-throughput functional screen |
| **Total** | Multi-center benchmark | **16,749** | Human genome (GRCh38/hg38) | Standardized log2 fold-change |

## 3. Target Variables & Normalization
1. **Continuous Efficiency Score (`efficiency_score`)**:
   - Normalized continuous score in the interval $[0.0, 1.0]$.
   - Higher values correspond to greater targeted gene depletion / functional knockout efficiency in cellular assays.
   - Derived via collaborative filtering and rank-based normalization as described by Chuai et al. (2018) to remove cross-batch experimental baseline differences.
2. **Binary Activity Label (`binary_label`)**:
   - Cutoff applied at $\log_2 \text{fold-change} \ge 1.0$.
   - Label `1`: High-efficiency guide (active knockout).
   - Label `0`: Moderate to low-efficiency guide.

## 4. Sequence Format
- Every record corresponds to a 23-nucleotide sequence:
  - Bases 1–20: Protospacer single-guide RNA sequence (5' to 3').
  - Bases 21–23: Protospacer Adjacent Motif (canonical SpCas9 PAM: `5'-NGG-3'`).
- Characters are strictly standard deoxyribonucleotides: `{A, C, G, T}`.

## 5. Dataset Limitations & Caveats
- **Assay Specificity**: Experimental measurements reflect cellular fitness readouts (loss of guide abundance in proliferating populations), which strongly correlates with, but is not exclusively driven by, DNA cleavage rates.
- **Cell-Type Differences**: Chromatin accessibility and epigenetic states vary between cell types. A375 data displays notable divergence from the other three lines due to distinct screening conditions (dense 8-gene tiling under drug selection vs genome-wide essentiality dropout).
- **Computational Nature**: Computational model predictions represent estimated statistical activity distributions; empirical wet-lab validation remains necessary for gene therapy or precision applications.
