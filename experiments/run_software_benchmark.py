"""
CRISPR Guide AI: Software Performance and Latency Benchmark
Measures:
1. Single-sequence prediction latency (mean, median, p95, p99 across 100 iterations)
2. High-throughput batch inference latency across N = 10, 100, 500, 1,000 guides
3. Feature extraction throughput (sequences / second)
4. Model inference throughput across Ridge, RF, XGB, SVM, Weighted Ensemble
5. System memory footprint (RAM in MB)
Outputs:
- publication_audit/software_benchmark_results.json
"""

import os
import sys
import time
import json
import psutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app.features.extractor import CRISPRFeatureExtractor
from backend.app.models.predictor import CRISPRPredictor

AUDIT_DIR = os.path.join(BASE_DIR, "publication_audit")
os.makedirs(AUDIT_DIR, exist_ok=True)

print("=" * 80)
print("RUNNING SOFTWARE PERFORMANCE AND LATENCY BENCHMARK")
print("=" * 80)

# Track memory before initialization
process = psutil.Process(os.getpid())
mem_initial_mb = process.memory_info().rss / (1024 * 1024)

# Initialize predictor and extractor
t0 = time.perf_counter()
predictor = CRISPRPredictor()
t_init = time.perf_counter() - t0

mem_loaded_mb = process.memory_info().rss / (1024 * 1024)
extractor = predictor.extractor

# Sample sequences
TEST_SEQS = [
    "GAGTCCGAGCAGAAGAAGAAGGG",
    "ACATCAGGTTACCTCTACCAAGG",
    "CTTGCTCGCGCAGGACGAGGCGG",
    "GGCTGCTTTACCCGCTGTGGGGG",
    "TCCGGGTTGGCCTTCCACTGGGG",
    "CTGATGCCAGCTAGTGGGCGAGG",
    "GAAGGGCGGCGAGAAGGAGAAGG",
    "GAGAACGGAAAGGAGAAGGGCGG",
    "ATCGATCGATCGATCGATCGTGG",
    "CCGGCCGGCCGGCCGGCCGGCGG"
]

print(f"Memory before load: {mem_initial_mb:.1f} MB, after load: {mem_loaded_mb:.1f} MB (Delta: {mem_loaded_mb - mem_initial_mb:.1f} MB)")
print(f"Model initialization time: {t_init:.2f} s")

# 1. Single sequence latency benchmark (100 runs)
print("\n[1] Benchmarking single-sequence prediction latency (100 trials)...")
latencies_ms = []
for i in range(100):
    seq = TEST_SEQS[i % len(TEST_SEQS)]
    t_start = time.perf_counter()
    res = predictor.predict_single(seq, model_name="Weighted Ensemble", dataset_name="HeLa")
    t_end = time.perf_counter()
    latencies_ms.append((t_end - t_start) * 1000.0)

latencies_ms = np.array(latencies_ms)
single_bench = {
    "num_trials": 100,
    "model_tested": "Weighted Ensemble (HeLa)",
    "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
    "median_latency_ms": round(float(np.median(latencies_ms)), 2),
    "std_latency_ms": round(float(np.std(latencies_ms)), 2),
    "min_latency_ms": round(float(np.min(latencies_ms)), 2),
    "max_latency_ms": round(float(np.max(latencies_ms)), 2),
    "p90_latency_ms": round(float(np.percentile(latencies_ms, 90)), 2),
    "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
    "p99_latency_ms": round(float(np.percentile(latencies_ms, 99)), 2)
}
print(f"  Single prediction: Mean={single_bench['mean_latency_ms']} ms, Median={single_bench['median_latency_ms']} ms, p95={single_bench['p95_latency_ms']} ms")

# 2. Feature Extraction Throughput Benchmark
print("\n[2] Benchmarking 600-feature extraction throughput...")
n_guides_feat = 1000
feat_seqs = [TEST_SEQS[i % len(TEST_SEQS)] for i in range(n_guides_feat)]
t_feat_start = time.perf_counter()
X_extracted = extractor.extract_features_df(feat_seqs)
t_feat_end = time.perf_counter()
feat_duration_s = t_feat_end - t_feat_start
feat_throughput = n_guides_feat / feat_duration_s

feature_bench = {
    "features_extracted": 600,
    "batch_size": n_guides_feat,
    "total_duration_seconds": round(feat_duration_s, 3),
    "throughput_sequences_per_second": round(feat_throughput, 1)
}
print(f"  Feature extraction throughput: {feat_throughput:.1f} sequences/sec ({feat_duration_s:.2f} s for {n_guides_feat} sequences)")

# 3. High-throughput Batch Inference Latencies
print("\n[3] Benchmarking batch inference latencies across batch sizes N = 10, 100, 500, 1,000...")
batch_sizes = [10, 100, 500, 1000]
batch_results = {}

for bs in batch_sizes:
    items = [{"id": f"guide_{i}", "sequence": TEST_SEQS[i % len(TEST_SEQS)]} for i in range(bs)]
    t_b_start = time.perf_counter()
    batch_res = predictor.predict_batch(items, model_name="Weighted Ensemble", dataset_name="HeLa")
    t_b_end = time.perf_counter()
    b_duration = t_b_end - t_b_start
    b_throughput = bs / b_duration
    batch_results[f"batch_size_{bs}"] = {
        "batch_size_N": bs,
        "total_latency_seconds": round(b_duration, 3),
        "total_latency_ms": round(b_duration * 1000.0, 1),
        "per_sequence_latency_ms": round((b_duration * 1000.0) / bs, 2),
        "throughput_sequences_per_second": round(b_throughput, 1)
    }
    print(f"  Batch N={bs:>4}: Total={b_duration:.3f} s ({b_duration*1000.0/bs:.2f} ms/guide), Throughput={b_throughput:.1f} guides/sec")

# 4. Latency by Model Architecture
print("\n[4] Benchmarking latency by model architecture (N=100)...")
model_names = ["Ridge Linear Regression", "Random Forest", "XGBoost", "SVM", "Weighted Ensemble"]
arch_results = {}
test_100_items = [{"id": f"guide_{i}", "sequence": TEST_SEQS[i % len(TEST_SEQS)]} for i in range(100)]

for m in model_names:
    t_m0 = time.perf_counter()
    res_m = predictor.predict_batch(test_100_items, model_name=m, dataset_name="HeLa")
    t_m1 = time.perf_counter()
    dur = t_m1 - t_m0
    arch_results[m] = {
        "model_name": m,
        "batch_size_100_duration_ms": round(dur * 1000.0, 1),
        "per_guide_latency_ms": round(dur * 10.0, 2),
        "throughput_guides_per_sec": round(100.0 / dur, 1)
    }
    print(f"  Model: {m:>24}: {dur*10.0:.2f} ms/guide ({100.0/dur:.1f} guides/sec)")

benchmark_payload = {
    "benchmark_name": "CRISPR Guide AI Software Runtime Benchmark",
    "timestamp": "2026-10-31T15:30:00Z",
    "environment": {
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "memory_initial_rss_mb": round(mem_initial_mb, 1),
        "memory_post_load_rss_mb": round(mem_loaded_mb, 1),
        "memory_footprint_mb": round(mem_loaded_mb - mem_initial_mb, 1),
        "model_initialization_seconds": round(t_init, 2)
    },
    "single_sequence_latency": single_bench,
    "feature_extraction_throughput": feature_bench,
    "batch_inference_scaling": batch_results,
    "model_architecture_latencies": arch_results
}

out_bench = os.path.join(AUDIT_DIR, "software_benchmark_results.json")
with open(out_bench, "w") as f:
    json.dump(benchmark_payload, f, indent=2)

print(f"\nSaved software benchmark results to: {out_bench}")
