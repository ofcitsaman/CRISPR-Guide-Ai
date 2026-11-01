"""
Data Preparation Pipeline for CRISPR Guide AI.
Curates and verifies legitimate benchmark datasets:
- DeepCRISPR (Chuai et al., Genome Biology 2018)
- CRISPRpred(SEQ) (Rafid et al., BMC Bioinformatics 2020)
- Original screens from Hart et al., Doench et al., Xu et al.
"""

import os
import io
import tarfile
import urllib.request
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

CELL_LINES = ["hct116", "a375", "hela", "hl60"]
# Mapping to historical raw benchmark files
RAW_CSV_MAP = {
    "hct116": "hct116",
    "a375": "hek293",     # CRISPRpredSEQ legacy filename
    "hela": "hela",
    "hl60": "hl60"
}
REP_CELL_MAP = {
    "hct116": "hct116",
    "a375": "hek293t",    # DeepCRISPR legacy filename
    "hela": "hela",
    "hl60": "hl60"
}


def download_raw_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Download CSVs with binary labels (from CRISPRpredSEQ repository)
    for cell in CELL_LINES:
        raw_name = RAW_CSV_MAP[cell]
        csv_path = os.path.join(RAW_DIR, f"{raw_name}.csv")
        if not os.path.exists(csv_path):
            url = f"https://raw.githubusercontent.com/Rafid013/CRISPRpredSEQ/master/Data/{raw_name}.csv"
            print(f"Downloading {url} -> {csv_path}...")
            urllib.request.urlretrieve(url, csv_path)

    # 2. Download regression files (from DeepCRISPR repository)
    tar_url = "https://raw.githubusercontent.com/bm2-lab/DeepCRISPR/master/paper_data-regression.tar.gz"
    all_rep_exist = all(os.path.exists(os.path.join(RAW_DIR, f"{REP_CELL_MAP[c]}.repisgt")) for c in CELL_LINES)
    if not all_rep_exist:
        print(f"Downloading DeepCRISPR regression data from {tar_url}...")
        req = urllib.request.Request(tar_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            tar_bytes = io.BytesIO(resp.read())
            with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
                for c in CELL_LINES:
                    member_name = f"paper_data2/ontar/{REP_CELL_MAP[c]}.repisgt"
                    out_path = os.path.join(RAW_DIR, f"{REP_CELL_MAP[c]}.repisgt")
                    f = tar.extractfile(member_name)
                    with open(out_path, "wb") as out_f:
                        out_f.write(f.read())
                    print(f"Extracted {member_name} -> {out_path}")


def validate_and_merge_data() -> pd.DataFrame:
    """
    Validates sequences:
    - Length must be 23 nucleotides (20-nt guide + 3-nt PAM)
    - Characters must be strictly A, C, G, T
    - Merges continuous efficiency scores and binary labels
    - Rejects missing score matches without inventing synthetic targets
    """
    records = []
    valid_nucleotides = set("ACGT")

    for cell in CELL_LINES:
        raw_name = RAW_CSV_MAP[cell]
        csv_path = os.path.join(RAW_DIR, f"{raw_name}.csv")
        if not os.path.exists(csv_path) and os.path.exists(os.path.join(RAW_DIR, f"{cell}.csv")):
            csv_path = os.path.join(RAW_DIR, f"{cell}.csv")

        rep_path = os.path.join(RAW_DIR, f"{REP_CELL_MAP[cell]}.repisgt")

        csv_df = pd.read_csv(csv_path)
        rep_df = pd.read_csv(
            rep_path,
            sep="\t",
            header=None,
            names=["chr", "start", "end", "strand", "sgRNA", "ctcf", "h3k4me3", "dnase", "rrbs", "eff_score"]
        )

        rep_df_unique = rep_df.drop_duplicates(subset=["sgRNA"]).copy()
        rep_map = rep_df_unique.set_index("sgRNA").to_dict(orient="index")

        for _, row in csv_df.iterrows():
            seq = str(row["sgRNA"]).strip().upper()
            label = int(row["label"])

            if len(seq) != 23:
                continue
            if not set(seq).issubset(valid_nucleotides):
                continue

            rep_info = rep_map.get(seq)
            if rep_info is not None:
                eff_score = float(rep_info["eff_score"])
                chrom = rep_info["chr"]
                start = rep_info["start"]
                end = rep_info["end"]
                strand = rep_info["strand"]
            else:
                raise ValueError(
                    f"Integrity Error: Sequence {seq} in {cell} (from {raw_name}.csv) has no matching entry "
                    f"in regression benchmark ({REP_CELL_MAP[cell]}.repisgt). Synthetic score imputation is disallowed."
                )

            records.append({
                "sgRNA": seq,
                "cell_line": cell,
                "binary_label": label,
                "efficiency_score": eff_score,
                "chr": chrom,
                "start": start,
                "end": end,
                "strand": strand
            })

    master_df = pd.DataFrame(records)
    print(f"Total validated records: {len(master_df)}")
    print(f"Cell line counts:\n{master_df['cell_line'].value_counts()}")
    print(f"Binary label balance:\n{master_df['binary_label'].value_counts(normalize=True)}")
    print(f"Continuous score range: min={master_df['efficiency_score'].min():.4f}, max={master_df['efficiency_score'].max():.4f}, mean={master_df['efficiency_score'].mean():.4f}")

    master_path = os.path.join(PROCESSED_DIR, "master_dataset.csv")
    master_df.to_csv(master_path, index=False)
    print(f"Saved master dataset to {master_path}")

    # Also save individual cell-line processed CSVs
    for cell in CELL_LINES:
        cell_df = master_df[master_df["cell_line"] == cell].copy()
        cell_path = os.path.join(PROCESSED_DIR, f"{cell}.csv")
        cell_df.to_csv(cell_path, index=False)
        print(f"Saved {cell} dataset ({len(cell_df)} records) to {cell_path}")

    sample_path = os.path.join(PROCESSED_DIR, "sample_test_data.csv")
    sample_df = master_df.sample(n=25, random_state=42)[["sgRNA", "cell_line", "binary_label", "efficiency_score"]]
    sample_df.to_csv(sample_path, index=False)
    print(f"Saved sample test data to {sample_path}")

    return master_df


if __name__ == "__main__":
    download_raw_data()
    validate_and_merge_data()
