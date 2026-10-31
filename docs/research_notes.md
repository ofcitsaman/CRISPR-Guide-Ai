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

## 3. The HEK293 Assay Discrepancy & Cross-Cell-Line Divergence
When comparing model performance across the four independently trained benchmarks:
- Models trained on **HL60** ($r = 0.447$, ROC-AUC = 0.822), **HCT116** ($r = 0.390$, ROC-AUC = 0.724), and **HeLa** ($r = 0.344$, ROC-AUC = 0.734) achieved strong correlation with viability depletion.
- However, models trained on **HEK293** exhibited lower correlation ($r = 0.258$, ROC-AUC = 0.592).

**Scientific Explanation**:
This phenomenon was extensively analyzed by Haeussler et al. (2016) and Rafid et al. (2020). The HEK293 dataset from Doench et al. (2016) evaluated cell-surface marker knockouts (CD33, CD15) by flow cytometry under transient delivery, whereas the Hart et al. (HeLa, HCT116) and Xu et al. (HL60) datasets evaluated viability-based lentiviral screens targeting essential ribosomal and spliceosomal genes over multi-week growth assays. Different selection pressures, promoter contexts, and screening kinetics explain the distinct performance profile of HEK293. Transparently reporting this variance per dataset is essential for scientific integrity.
