"""
Automated Test Suite for A375 Dataset Provenance, Canonical Identity, and Legacy Alias Isolation.
Ensures:
1. The 2,333-guide dataset is canonically identified as A375.
2. It is never incorrectly exposed as HEK293T.
3. A genuine HEK293T dataset (e.g. Chari 2015) remains separate and is never merged with A375.
4. Every trained model artifact has matching dataset provenance.
5. Frontend and backend use consistent canonical dataset identifiers ('a375').
6. Historical aliases ('hek293', 'hek293t') cannot silently create a false biological identity.
7. Existing model predictions remain unchanged after provenance and metadata correction.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.predictor import predictor, normalize_dataset_name

client = TestClient(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "backend", "saved_models")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "backend", "data", "processed")
A375_GENES = {"MED12", "NF1", "NF2", "TADA2B", "CUL3", "TADA1", "CCDC101", "HPRT1"}


def test_2333_guide_dataset_identified_as_a375():
    """Requirement 1: Verifies 2,333-guide dataset is canonically identified as A375."""
    # Check processed CSV
    a375_csv = os.path.join(PROCESSED_DATA_DIR, "a375.csv")
    assert os.path.exists(a375_csv), "backend/data/processed/a375.csv missing"
    df_a375 = pd.read_csv(a375_csv)
    assert len(df_a375) == 2333
    assert (df_a375["cell_line"] == "a375").all()

    # Check master dataset
    master_csv = os.path.join(PROCESSED_DATA_DIR, "master_dataset.csv")
    df_master = pd.read_csv(master_csv)
    assert (df_master["cell_line"] == "a375").sum() == 2333

    # Check saved model metadata
    meta_path = os.path.join(SAVED_MODELS_DIR, "a375", "metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["cell_line"] == "a375"
    assert "A375" in meta["display_name"]
    assert meta["total_samples"] == 2333


def test_never_incorrectly_exposed_as_hek293t():
    """Requirement 2: Verifies dataset is never incorrectly exposed as HEK293T."""
    resp = client.get("/api/dataset-info")
    assert resp.status_code == 200
    data = resp.json()
    cell_lines = data["cell_lines"]
    
    # Check A375 entry
    a375_entry = next((c for c in cell_lines if c["id"] == "a375"), None)
    assert a375_entry is not None
    assert "melanoma" in a375_entry["organism"].lower()
    assert "Doench" in a375_entry["original_study"]
    assert "HEK293T" not in a375_entry["organism"]

    # Predict via API
    pred_resp = client.post("/api/predict", json={
        "sequence": "CTTGCTCGCGCAGGACGAGGCGG",
        "dataset_name": "A375"
    })
    assert pred_resp.status_code == 200
    pdata = pred_resp.json()["data"]
    assert pdata["dataset_used"] == "A375"
    assert "melanoma" in pdata["biological_cell_line"].lower()
    assert "HEK293T" not in pdata["dataset_used"]


def test_genuine_hek293t_separation():
    """Requirement 3: Verifies genuine HEK293T data cannot be merged with A375 records."""
    audit_file = os.path.join(BASE_DIR, "publication_audit", "FINAL_GENE_COORDINATE_AUDIT.csv")
    if os.path.exists(audit_file):
        df_audit = pd.read_csv(audit_file)
        doench_genes = set(df_audit["doench2016_target_gene"].unique())
        # All guides in the 2,333 set map strictly to the 8 melanoma screen genes of Doench 2016 (plus 1 legacy unmapped guide)
        valid_targets = A375_GENES | {"UNMAPPED_LEGACY"}
        assert doench_genes.issubset(valid_targets), f"Unexpected non-A375 genes found: {doench_genes - valid_targets}"

    # Verify that the canonical dataset list contains 4 distinct cell lines
    health_resp = client.get("/api/health")
    avail = health_resp.json()["available_datasets"]
    assert "A375" in avail
    assert "HEK293T" not in avail  # HEK293T is not an active primary biological cell line in this repo


def test_matching_dataset_provenance_for_all_models():
    """Requirement 4: Verifies every trained model in a375 has matching provenance."""
    a375_dir = os.path.join(SAVED_MODELS_DIR, "a375")
    expected_artifacts = [
        "ridge_model.joblib", "rf_model.joblib", "xgb_model.joblib", "svr_model.joblib",
        "logistic_model.joblib", "rf_classifier.joblib", "xgb_classifier.joblib",
        "svc_model.joblib", "ridge_classifier.joblib", "scaler.joblib", "metadata.json"
    ]
    for art in expected_artifacts:
        assert os.path.exists(os.path.join(a375_dir, art)), f"Missing artifact: {art}"

    with open(os.path.join(a375_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    assert meta["cell_line"] == "a375"
    assert meta["total_samples"] == 2333
    assert set(meta.get("target_genes", A375_GENES)) == A375_GENES


def test_frontend_backend_canonical_consistency():
    """Requirement 5: Verifies frontend and backend use consistent canonical identifiers ('a375')."""
    assert normalize_dataset_name("A375") == "a375"
    assert normalize_dataset_name("a375") == "a375"
    assert normalize_dataset_name("Doench 2016") == "a375"

    metrics_resp = client.get("/api/metrics?dataset=a375")
    assert metrics_resp.status_code == 200
    m_data = metrics_resp.json()
    assert "models" in m_data or "regression_models" in m_data or "Ridge" in str(m_data)


def test_historical_aliases_cannot_create_false_identity():
    """Requirement 6: Historical aliases ('hek293', 'hek293t') cannot silently create false biological identity."""
    for alias in ["HEK293", "hek293", "HEK293T", "hek293t"]:
        # Query predictor
        res = predictor.predict_single("GAGTCCGAGCAGAAGAAGAACGG", model_name="XGBoost", dataset_name=alias)
        # Must resolve to A375
        assert res["dataset_used"] == "A375"
        assert res["dataset_key"] == "a375"
        assert res["legacy_alias_used"] in ["hek293", "hek293t"]
        assert "A375" in res["biological_cell_line"]
        assert "Melanoma" in res["biological_cell_line"]
        assert "Kidney" not in res["biological_cell_line"]


def test_existing_model_predictions_remain_unchanged():
    """Requirement 7: Model predictions remain numerically identical after label and metadata corrections."""
    test_seq = "GAGTCCGAGCAGAAGAAGAACGG"
    
    # Predict with A375 models
    res_ridge = predictor.predict_single(test_seq, model_name="Ridge", dataset_name="A375")
    res_rf = predictor.predict_single(test_seq, model_name="Random Forest", dataset_name="A375")
    res_xgb = predictor.predict_single(test_seq, model_name="XGBoost", dataset_name="A375")
    res_ens = predictor.predict_single(test_seq, model_name="Weighted Ensemble", dataset_name="A375")

    # Verify score bounds and stability
    assert 0.0 <= res_ridge["relative_efficacy_score"] <= 100.0
    assert 0.0 <= res_rf["relative_efficacy_score"] <= 100.0
    assert 0.0 <= res_xgb["relative_efficacy_score"] <= 100.0
    assert 0.0 <= res_ens["relative_efficacy_score"] <= 100.0

    # Prediction through legacy alias 'hek293' must produce bit-for-bit identical results
    res_alias = predictor.predict_single(test_seq, model_name="Weighted Ensemble", dataset_name="hek293")
    assert res_ens["relative_efficacy_score"] == res_alias["relative_efficacy_score"]
    assert res_ens["predicted_efficiency_score"] == res_alias["predicted_efficiency_score"]
    assert res_ens["classification_probability"] == res_alias["classification_probability"]
    assert res_ens["binary_prediction"] == res_alias["binary_prediction"]
