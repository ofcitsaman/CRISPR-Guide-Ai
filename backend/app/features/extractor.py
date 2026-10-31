"""
CRISPR-Cas9 sgRNA Feature Extraction Engine.
Extracts biologically meaningful sequence-based features:
- Basic sequence composition (length, base counts, base percentages, GC/AT ratios)
- Subregion GC contents (PAM-distal bases 1-10 vs. PAM-proximal seed bases 11-20)
- Biophysical markers (poly-T transcription termination motifs, self-complementarity indicators)
- Position-Specific Features (PSF: 1-mer at positions 1-23, 2-mer adjacent pairs)
- Position-Independent k-mers (PIF: 2-mer frequencies, 3-mer frequencies)
- n-Gapped Di-nucleotides (nGD: pairs at gap distances 1, 2, 3)
"""

import re
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd

NUCLEOTIDES = ["A", "C", "G", "T"]
DINUCLEOTIDES = [f"{b1}{b2}" for b1 in NUCLEOTIDES for b2 in NUCLEOTIDES]
TRINUCLEOTIDES = [f"{b1}{b2}{b3}" for b1 in NUCLEOTIDES for b2 in NUCLEOTIDES for b3 in NUCLEOTIDES]


class SequenceValidationError(ValueError):
    """Raised when an sgRNA sequence fails biological or format validation."""
    pass


def validate_sgrna_sequence(sequence: str) -> Tuple[str, bool]:
    """
    Validates and normalizes sgRNA sequence.
    
    Returns:
        tuple of (normalized_23mer_sequence, was_extended_with_pam)
    
    Raises:
        SequenceValidationError if sequence contains invalid characters or has unacceptable length.
    """
    if not isinstance(sequence, str):
        raise SequenceValidationError("Sequence must be a string.")

    cleaned = sequence.strip().upper()
    if not cleaned:
        raise SequenceValidationError("Sequence cannot be empty.")

    # Check for invalid characters
    invalid_chars = set(cleaned) - set(NUCLEOTIDES)
    if invalid_chars:
        invalid_list = ", ".join(sorted(list(invalid_chars)))
        raise SequenceValidationError(
            f"Invalid nucleotide characters detected: {invalid_list}. "
            f"CRISPR-Cas9 sgRNA sequences must contain only standard DNA nucleotides: A, C, G, T."
        )

    extended = False
    if len(cleaned) == 20:
        # Standard 20-nt protospacer without PAM provided: extend with canonical NGG PAM (e.g. CGG)
        # Note: We append 'TGG' as standard canonical SpCas9 NGG PAM context and flag it.
        cleaned = cleaned + "TGG"
        extended = True
    elif len(cleaned) == 23:
        pass
    else:
        raise SequenceValidationError(
            f"Invalid sequence length ({len(cleaned)} nt). "
            f"Expected either a 20-nucleotide guide spacer or a 23-nucleotide sequence (20-nt guide + 3-nt PAM, e.g. NGG). "
            f"Predictions outside the training distribution (23 nt) may be unreliable."
        )

    return cleaned, extended


class CRISPRFeatureExtractor:
    """
    Extracts sequence-based features for machine learning models.
    Can be instantiated once and reused for single sequences or high-throughput batch arrays.
    """

    def __init__(self):
        self._feature_names = self._build_feature_names()

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def _build_feature_names(self) -> List[str]:
        names = []

        # 1. Basic Composition
        names.extend([
            "seq_length",
            "count_A", "count_C", "count_G", "count_T",
            "pct_A", "pct_C", "pct_G", "pct_T",
            "gc_content", "at_content",
            "distal_gc",  # positions 1-10
            "seed_gc",    # positions 11-20
            "seed_purine_pct", # A+G in seed
            "poly_T_max_run",  # Pol-III termination penalty
            "has_terminator_TTTT",
            "pam_is_NGG",
            "pam_base21", "pam_base22", "pam_base23",
            "cleavage_flank_gc", # positions 16-18
            "seed_distal_gc_diff", # seed GC minus distal GC
            "cleavage_seed_gc_prod", # cleavage GC * seed GC
            "core_seed_gc", # positions 15-20 core seed GC
            "homopolymer_G_max_run", # G-quadruplex indicator
            "homopolymer_C_max_run",
            "homopolymer_A_max_run",
            "shannon_entropy", # sequence complexity
        ])

        # 2. Position-Specific 1-mer Features (positions 1 to 23, 1-indexed)
        for pos in range(1, 24):
            for base in NUCLEOTIDES:
                names.append(f"pos_{pos}_{base}")

        # 3. Position-Specific 2-mer Features (adjacent pairs across 22 positions)
        for pos in range(1, 23):
            for di in DINUCLEOTIDES:
                names.append(f"pos_{pos}_{di}")

        # 4. Position-Independent 2-mer Counts (16 features)
        for di in DINUCLEOTIDES:
            names.append(f"count_{di}")

        # 5. Position-Independent 3-mer Counts (64 features)
        for tri in TRINUCLEOTIDES:
            names.append(f"count_{tri}")

        # 6. n-Gapped Di-nucleotides (gap distances 1, 2, 3)
        for gap in [1, 2, 3]:
            for di in DINUCLEOTIDES:
                names.append(f"gap{gap}_{di}")

        return names

    def extract_features_single(self, sequence: str) -> Dict[str, float]:
        """Extracts named features for a single 23-mer sequence."""
        seq, _ = validate_sgrna_sequence(sequence)
        length = len(seq)  # 23

        features: Dict[str, float] = {}

        # 1. Basic counts
        cnt_a = seq.count("A")
        cnt_c = seq.count("C")
        cnt_g = seq.count("G")
        cnt_t = seq.count("T")

        features["seq_length"] = float(length)
        features["count_A"] = float(cnt_a)
        features["count_C"] = float(cnt_c)
        features["count_G"] = float(cnt_g)
        features["count_T"] = float(cnt_t)

        features["pct_A"] = cnt_a / length
        features["pct_C"] = cnt_c / length
        features["pct_G"] = cnt_g / length
        features["pct_T"] = cnt_t / length

        features["gc_content"] = (cnt_g + cnt_c) / length
        features["at_content"] = (cnt_a + cnt_t) / length

        # Distal (1-10) and Seed (11-20)
        distal_seq = seq[0:10]
        seed_seq = seq[10:20]
        features["distal_gc"] = (distal_seq.count("G") + distal_seq.count("C")) / 10.0
        features["seed_gc"] = (seed_seq.count("G") + seed_seq.count("C")) / 10.0
        features["seed_purine_pct"] = (seed_seq.count("A") + seed_seq.count("G")) / 10.0

        # Poly-T run
        t_runs = re.findall(r"T+", seq)
        max_t = max([len(r) for r in t_runs]) if t_runs else 0
        features["poly_T_max_run"] = float(max_t)
        features["has_terminator_TTTT"] = 1.0 if "TTTT" in seq else 0.0

        # PAM & Cleavage
        pam = seq[20:23]
        features["pam_is_NGG"] = 1.0 if pam[1:] == "GG" else 0.0
        features["pam_base21"] = 1.0 if pam[0] == "G" else 0.0
        features["pam_base22"] = 1.0 if pam[1] == "G" else 0.0
        features["pam_base23"] = 1.0 if pam[2] == "G" else 0.0

        cleav_site = seq[15:18] # positions 16, 17, 18
        cleavage_gc = (cleav_site.count("G") + cleav_site.count("C")) / 3.0
        features["cleavage_flank_gc"] = cleavage_gc

        features["seed_distal_gc_diff"] = float(features["seed_gc"] - features["distal_gc"])
        features["cleavage_seed_gc_prod"] = float(cleavage_gc * features["seed_gc"])

        core_seed_seq = seq[14:20] # positions 15-20 (6 nt core seed)
        features["core_seed_gc"] = (core_seed_seq.count("G") + core_seed_seq.count("C")) / 6.0

        g_runs = re.findall(r"G+", seq)
        c_runs = re.findall(r"C+", seq)
        a_runs = re.findall(r"A+", seq)
        features["homopolymer_G_max_run"] = float(max([len(r) for r in g_runs])) if g_runs else 0.0
        features["homopolymer_C_max_run"] = float(max([len(r) for r in c_runs])) if c_runs else 0.0
        features["homopolymer_A_max_run"] = float(max([len(r) for r in a_runs])) if a_runs else 0.0

        probs = [c / length for c in [cnt_a, cnt_c, cnt_g, cnt_t] if c > 0]
        features["shannon_entropy"] = float(-sum(p * np.log2(p) for p in probs))

        # 2. Position-Specific 1-mer Features
        for idx, base in enumerate(seq):
            pos = idx + 1
            for b in NUCLEOTIDES:
                features[f"pos_{pos}_{b}"] = 1.0 if base == b else 0.0

        # 3. Position-Specific 2-mer Features
        for idx in range(len(seq) - 1):
            pos = idx + 1
            di = seq[idx : idx + 2]
            for target_di in DINUCLEOTIDES:
                features[f"pos_{pos}_{target_di}"] = 1.0 if di == target_di else 0.0

        # 4. Position-Independent 2-mer counts
        for di in DINUCLEOTIDES:
            # count occurrences (including overlapping)
            cnt = sum(1 for i in range(len(seq) - 1) if seq[i : i + 2] == di)
            features[f"count_{di}"] = float(cnt)

        # 5. Position-Independent 3-mer counts
        for tri in TRINUCLEOTIDES:
            cnt = sum(1 for i in range(len(seq) - 2) if seq[i : i + 3] == tri)
            features[f"count_{tri}"] = float(cnt)

        # 6. n-Gapped Di-nucleotides (gap distances 1, 2, 3)
        for gap in [1, 2, 3]:
            for di in DINUCLEOTIDES:
                b1, b2 = di[0], di[1]
                cnt = 0
                step = gap + 1
                for i in range(len(seq) - step):
                    if seq[i] == b1 and seq[i + step] == b2:
                        cnt += 1
                features[f"gap{gap}_{di}"] = float(cnt)

        return features

    def extract_features_df(self, sequences: List[str]) -> pd.DataFrame:
        """Extracts features for a list of sequences and returns a formatted DataFrame."""
        rows = []
        for s in sequences:
            rows.append(self.extract_features_single(s))
        df = pd.DataFrame(rows, columns=self.feature_names)
        return df

    extract_features = extract_features_single

    def extract_features_array(self, sequences: List[str]) -> np.ndarray:
        """Extracts features and returns a 2D NumPy array of shape (N, num_features)."""
        df = self.extract_features_df(sequences)
        return df.values.astype(np.float32)

    def sequence_to_onehot_tensor(self, sequence: str) -> np.ndarray:
        """
        Converts a 23-mer sequence into a 4x23 one-hot matrix.
        Shape: (4, 23) where rows are A, C, G, T channels.
        """
        seq, _ = validate_sgrna_sequence(sequence)
        mat = np.zeros((4, 23), dtype=np.float32)
        base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        for i, b in enumerate(seq):
            mat[base_map[b], i] = 1.0
        return mat

    def batch_to_onehot_tensor(self, sequences: List[str]) -> np.ndarray:
        """
        Converts a list of N sequences into a (N, 4, 23) batch tensor.
        """
        return np.stack([self.sequence_to_onehot_tensor(s) for s in sequences], axis=0)


if __name__ == "__main__":
    extractor = CRISPRFeatureExtractor()
    print(f"Total engineered features per sequence: {len(extractor.feature_names)}")
    test_seq = "CTTGCTCGCGCAGGACGAGGCGG"
    feats = extractor.extract_features_single(test_seq)
    print(f"Sample features for {test_seq}:")
    print(f"  GC content: {feats['gc_content']:.2%}")
    print(f"  Distal GC:  {feats['distal_gc']:.2%}")
    print(f"  Seed GC:    {feats['seed_gc']:.2%}")
    print(f"  PAM NGG?:   {feats['pam_is_NGG']}")
    print(f"  Poly-T max: {feats['poly_T_max_run']}")
    print(f"  Cleavage GC:{feats['cleavage_flank_gc']:.2%}")
    onehot = extractor.sequence_to_onehot_tensor(test_seq)
    print(f"  One-hot tensor shape: {onehot.shape}")
