"""
CRISPR Guide AI: Final Sample Count Reconciliation and Checksum Manifest
Reconciles all sample counts across raw, processed, split assignments, and OOF predictions.
Explains the origin of historical 4,242 vs 4,239, 3,394 vs 3,391, and 1,664 vs 1,660 discrepancies.
Outputs:
- publication_audit/FINAL_SAMPLE_COUNT_RECONCILIATION.csv
"""

import os
import hashlib
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DIR = os.path.join(BASE_DIR, "publication_audit")
DATA_DIR = os.path.join(BASE_DIR, "backend", "data")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "backend", "saved_models")

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

datasets = [
    {
        "dataset_key": "hela",
        "verified_identity": "HeLa (Hart et al. 2015)",
        "raw_csv_name": "hela.csv",
        "proc_csv_name": "hela.csv",
        "repisgt_name": "hela.repisgt",
        "reported_discrepancy_note": "No count discrepancy. Consistent across all records."
    },
    {
        "dataset_key": "hct116",
        "verified_identity": "HCT116 (Hart et al. 2015)",
        "raw_csv_name": "hct116.csv",
        "proc_csv_name": "hct116.csv",
        "repisgt_name": "hct116.repisgt",
        "reported_discrepancy_note": "Early draft texts cited 4,242 total and 3,394 dev (an approximate calculation 4242*0.8=3393.6). Actual ground-truth file contains exactly 4,239 rows and 4,239 unique sequences, with exactly 3,391 dev and 848 holdout."
    },
    {
        "dataset_key": "hl60",
        "verified_identity": "HL60 (Wang et al. 2014 / Xu et al. 2015)",
        "raw_csv_name": "hl60.csv",
        "proc_csv_name": "hl60.csv",
        "repisgt_name": "hl60.repisgt",
        "reported_discrepancy_note": "Early draft texts cited 1,664 dev records (based on 2080*0.8=1664). Actual ground-truth file contains exactly 2,076 rows and 2,076 unique sequences, with exactly 1,660 dev and 416 holdout (1660 + 416 = 2076)."
    },
    {
        "dataset_key": "a375",
        "verified_identity": "A375 (Doench et al. 2016)",
        "raw_csv_name": "hek293.csv",
        "proc_csv_name": "hek293.csv",
        "repisgt_name": "hek293t.repisgt",
        "reported_discrepancy_note": "Legacy filenames hek293.csv and hek293t.repisgt inherited from DeepCRISPR / CRISPRpred misattribution. Repisgt had 4,566 rows due to replicate pair duplications (2,233 duplicate rows); deduplicated unique guide count is exactly 2,333, with 1,866 dev and 467 holdout."
    }
]

rows = []
for d in datasets:
    raw_p = os.path.join(DATA_DIR, "raw", d["raw_csv_name"])
    proc_p = os.path.join(DATA_DIR, "processed", d["proc_csv_name"])
    split_p = os.path.join(SAVED_MODELS_DIR, "hek293" if d["dataset_key"] == "a375" else d["dataset_key"], "split_assignments.json")
    oof_p = os.path.join(SAVED_MODELS_DIR, "hek293" if d["dataset_key"] == "a375" else d["dataset_key"], "oof_predictions.csv")
    
    df_raw = pd.read_csv(raw_p)
    df_proc = pd.read_csv(proc_p)
    with open(split_p) as f:
        splits = json.load(f)
    df_oof = pd.read_csv(oof_p)
    
    dev_n = len(splits["dev_indices"])
    hold_n = len(splits["holdout_indices"])
    models_n = df_oof["model"].nunique()
    oof_unique_seqs = df_oof["sequence"].nunique()
    
    dev_seqs = set(df_proc.iloc[splits["dev_indices"]]["sgRNA"])
    hold_seqs = set(df_proc.iloc[splits["holdout_indices"]]["sgRNA"])
    overlap_n = len(dev_seqs.intersection(hold_seqs))
    
    rows.append({
        "dataset_key": d["dataset_key"],
        "verified_cell_line": d["verified_identity"],
        "raw_csv_file": d["raw_csv_name"],
        "raw_csv_sha256": get_sha256(raw_p),
        "raw_csv_row_count": len(df_raw),
        "raw_csv_unique_sequences": df_raw["sgRNA"].nunique(),
        "processed_csv_file": d["proc_csv_name"],
        "processed_csv_sha256": get_sha256(proc_p),
        "processed_csv_row_count": len(df_proc),
        "processed_csv_unique_sequences": df_proc["sgRNA"].nunique(),
        "duplicate_sequence_count": df_proc["sgRNA"].duplicated().sum(),
        "development_pool_N": dev_n,
        "historical_holdout_N": hold_n,
        "dev_plus_holdout_sum": dev_n + hold_n,
        "sum_equals_total_rows": (dev_n + hold_n) == len(df_proc),
        "dev_holdout_sequence_overlap": overlap_n,
        "oof_prediction_rows_total": len(df_oof),
        "oof_models_evaluated": models_n,
        "oof_rows_per_model": len(df_oof) // models_n,
        "oof_unique_sequences": oof_unique_seqs,
        "oof_matches_dev_pool_exactly": oof_unique_seqs == dev_n,
        "reconciliation_notes": d["reported_discrepancy_note"]
    })

df_recon = pd.DataFrame(rows)
out_csv = os.path.join(AUDIT_DIR, "FINAL_SAMPLE_COUNT_RECONCILIATION.csv")
df_recon.to_csv(out_csv, index=False)
print(f"Saved FINAL_SAMPLE_COUNT_RECONCILIATION.csv to: {out_csv}")
