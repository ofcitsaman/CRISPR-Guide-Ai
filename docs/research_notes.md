# CRISPR Guide AI: Research Notes & Scientific Insights

## 1. Sequence-Only vs. Epigenetic Representation
A central debate in computational CRISPR guide design is whether primary sequence features alone are sufficient to predict cleavage efficiency, or whether cell-specific epigenomics (DNase I accessibility, H3K4me3 methylation, CTCF binding) are indispensable.

Our empirical findings corroborate those of **CRISPRpred(SEQ)** (*Rafid et al., 2020*):
- Primary sequence features intrinsically encode a substantial proportion of Cas9 binding thermodynamics, R-loop stability, and cleavage kinetics.
- High-order sequence representations (position-specific dinucleotides, n-gapped dinucleotides) achieved strong predictive correlations (Pearson $r \approx 0.50$ in Random Forest) on multi-cell benchmarks without requiring costly experimental epigenetic sequencing tracks.

## 2. Recovery of Known Biophysical Cleavage Motifs
Our automated feature importance ranking (extracted from XGBoost and Random Forest on 16,749 samples) independently recovered key biophysical principles documented in CRISPR structural biology:
1. **Position 18 Cytosine (`pos_18_C`)**: Identified as the single most predictive positional feature. In SpCas9, cleavage occurs precisely between positions 17 and 18. Cytosine at position 18 favors optimal catalytic domain engagement.
2. **PAM-Proximal Thymine Penalty (`pos_18_T`, `pos_19_T`, `pos_20_T`)**: Poly-T tracts and individual Ts in the PAM-proximal seed region consistently received negative predictive weight. In addition to acting as an RNA Polymerase III terminator, consecutive Ts disrupt the electrostatic interaction between the Cas9 bridge helix and the RNA:DNA hybrid.
3. **Position 20 Guanine (`pos_20_G`)**: Guanine at position 20 (immediately adjacent to the PAM) promotes stable transition from initial PAM interrogation to seed sequence unwinding.

## 3. The A375 Assay Discrepancy & Dense Tiling Screen Variance (Doench 2016)
When comparing model performance across the four independently trained benchmarks:
- Models trained on **HL60** ($r = 0.388$, ROC-AUC = 0.782), **HCT116** ($r = 0.347$, ROC-AUC = 0.710), and **HeLa** ($r = 0.339$, ROC-AUC = 0.722) evaluated on holdout data achieved strong correlation with viability depletion.
- However, models trained on **A375** exhibited lower correlation ($r = 0.175$, ROC-AUC = 0.558).

**Scientific Explanation**:
Forensic coordinate mapping and published benchmark re-evaluations (DeepCRISTL, Zheng et al. 2022) confirmed that the 2,333 records previously misattributed in DeepCRISPR as HEK293T represent the A375 human malignant melanoma drug-resistance and viability screen from Doench et al. (Nature Biotechnology 2016). Unlike the genome-wide essentiality dropout screens in HeLa, HCT116 (Hart et al. 2015), and HL60 (Wang et al. 2014), the A375 dataset is a dense targeted tiling screen across exactly eight human genes (*MED12, NF1, NF2, TADA2B, CUL3, TADA1, CCDC101, HPRT1*). The restricted biological diversity of 8 target genes, combined with positive selection readouts (e.g., vemurafenib resistance), generates a distinct dynamic range and lower sequence-alone generalization than genome-wide essential gene screens. Transparently reporting this variance per dataset is essential for scientific integrity.
