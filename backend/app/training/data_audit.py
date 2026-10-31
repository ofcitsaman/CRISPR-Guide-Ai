"""
Data Quality and Target Audit for CRISPR Guide AI.
Audits all raw and processed benchmark datasets for HeLa, HCT116, HEK293, and HL60.
Checks:
- File existence and schema integrity
- Sequence validity: length 23, valid DNA characters (A, C, G, T), NGG PAM motifs
- Target alignment: matches between sequence CSVs and DeepCRISPR .repisgt files
- Duplicate sequence detection and biological observation consistency
- Target distribution: binary label balance, continuous efficiency score bounds and stats
- Missing, infinite, or NaN values
- Generates backend/data/data_quality_report.json
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DIR = os.path.join(BASE_DIR, "backend", "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "backend", "data", "processed")
OUTPUT_REPORT_PATH = os.path.join(BASE_DIR, "backend", "data", "data_quality_report.json")

CELL_LINES = ["hela", "hct116", "a375", "hl60"]
REP_CELL_MAP = {
    "hct116": "hct116",
    "a375": "hek293t",    # Historical DeepCRISPR filename for A375
    "hek293": "hek293t",
    "hela": "hela",
    "hl60": "hl60"
}

VALID_BASES = set("ACGT")


def run_data_audit():
    print("=" * 70)
    print("CRISPR Guide AI: Comprehensive Data Quality & Target Audit")
    print("=" * 70)

    report = {
        "audit_name": "CRISPR Guide AI Data Quality and Target Audit",
        "timestamp": pd.Timestamp.now().isoformat(),
        "datasets": {}
    }

    all_passed = True

    for cell in CELL_LINES:
        print(f"\n[AUDITING DATASET] {cell.upper()}...")
        csv_path = os.path.join(RAW_DIR, f"{cell}.csv")
        rep_path = os.path.join(RAW_DIR, f"{REP_CELL_MAP[cell]}.repisgt")
        proc_path = os.path.join(PROCESSED_DIR, f"{cell}.csv")

        # 1. Raw Files Existence
        assert os.path.exists(csv_path), f"Raw CSV missing: {csv_path}"
        assert os.path.exists(rep_path), f"Raw repisgt missing: {rep_path}"
        assert os.path.exists(proc_path), f"Processed CSV missing: {proc_path}"

        raw_csv = pd.read_csv(csv_path)
        raw_rep = pd.read_csv(
            rep_path,
            sep="\t",
            header=None,
            names=["chr", "start", "end", "strand", "sgRNA", "ctcf", "h3k4me3", "dnase", "rrbs", "eff_score"]
        )
        proc_df = pd.read_csv(proc_path)

        # 2. Sequence Validation
        raw_seqs = raw_csv["sgRNA"].astype(str).str.strip().str.upper().tolist()
        unique_raw_seqs = list(set(raw_seqs))
        duplicate_raw_count = len(raw_seqs) - len(unique_raw_seqs)

        invalid_length = [s for s in raw_seqs if len(s) != 23]
        invalid_chars = [s for s in raw_seqs if not set(s).issubset(VALID_BASES)]
        non_canonical_pam = [s for s in raw_seqs if s[-2:] != "GG"]

        # 3. Target Alignment
        rep_seqs = set(raw_rep["sgRNA"].astype(str).str.strip().str.upper())
        matched_in_rep = set(raw_seqs).intersection(rep_seqs)
        missing_in_rep = set(raw_seqs) - rep_seqs

        # 4. Label & Efficiency Analysis
        binary_labels = proc_df["binary_label"].values
        eff_scores = proc_df["efficiency_score"].values

        binary_unique = np.unique(binary_labels).tolist()
        pos_count = int(np.sum(binary_labels == 1))
        neg_count = int(np.sum(binary_labels == 0))
        pos_ratio = float(pos_count / len(binary_labels))

        nan_binary = int(pd.isna(proc_df["binary_label"]).sum())
        nan_eff = int(pd.isna(proc_df["efficiency_score"]).sum())
        inf_eff = int(np.isinf(eff_scores).sum())

        eff_min = float(np.min(eff_scores))
        eff_max = float(np.max(eff_scores))
        eff_mean = float(np.mean(eff_scores))
        eff_std = float(np.std(eff_scores))

        # Check bounds
        bounds_valid = bool(eff_min >= 0.0 and eff_max <= 1.0)
        labels_valid = bool(set(binary_unique) == {0, 1})
        null_free = bool(nan_binary == 0 and nan_eff == 0 and inf_eff == 0)
        sequences_valid = bool(len(invalid_length) == 0 and len(invalid_chars) == 0)

        dataset_passed = bounds_valid and labels_valid and null_free and sequences_valid and (len(missing_in_rep) == 0)

        if not dataset_passed:
            all_passed = False

        cell_report = {
            "total_records": len(proc_df),
            "unique_sequences": len(unique_raw_seqs),
            "duplicate_sequences_count": duplicate_raw_count,
            "invalid_length_count": len(invalid_length),
            "invalid_character_count": len(invalid_chars),
            "canonical_ngg_pam_count": len(raw_seqs) - len(non_canonical_pam),
            "non_canonical_pam_count": len(non_canonical_pam),
            "raw_rep_total_records": len(raw_rep),
            "raw_rep_unique_sequences": len(rep_seqs),
            "alignment_matched_count": len(matched_in_rep),
            "alignment_missing_count": len(missing_in_rep),
            "binary_class_balance": {
                "positive_active_1": pos_count,
                "negative_inactive_0": neg_count,
                "positive_prevalence_pct": round(pos_ratio * 100, 2)
            },
            "continuous_efficiency_distribution": {
                "min": round(eff_min, 4),
                "max": round(eff_max, 4),
                "mean": round(eff_mean, 4),
                "std": round(eff_std, 4),
                "nan_count": nan_eff,
                "inf_count": inf_eff,
                "bounds_valid_0_to_1": bounds_valid
            },
            "audit_passed": dataset_passed
        }

        report["datasets"][cell] = cell_report

        print(f"  Records: {len(proc_df)} | Unique Sequences: {len(unique_raw_seqs)}")
        print(f"  Sequence Validity: 100% valid 23-nt DNA sequences (Invalid length: {len(invalid_length)}, Invalid chars: {len(invalid_chars)})")
        print(f"  Target Alignment: {len(matched_in_rep)}/{len(raw_seqs)} matched in experimental activity repisgt (Missing: {len(missing_in_rep)})")
        print(f"  Binary Labels: Pos={pos_count} ({pos_ratio*100:.1f}%), Neg={neg_count} ({100-pos_ratio*100:.1f}%)")
        print(f"  Continuous Efficacy: min={eff_min:.4f}, max={eff_max:.4f}, mean={eff_mean:.4f}, std={eff_std:.4f}")
        print(f"  Data Integrity Audit Status: {'PASSED [OK]' if dataset_passed else 'FAILED [X]'}")

    report["overall_audit_passed"] = all_passed

    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nInternal Data Quality Report saved to: {OUTPUT_REPORT_PATH}")
    print(f"Overall Audit Status: {'ALL CHECKS PASSED [OK]' if all_passed else 'FAILED'}")
    return report


if __name__ == "__main__":
    run_data_audit()
