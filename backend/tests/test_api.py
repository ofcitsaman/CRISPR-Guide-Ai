"""
Unit and Integration Tests for CRISPR Guide AI FastAPI Service.
Tests:
- /api/health
- /api/predict (dataset-specific, valid 23-nt, valid 20-nt, invalid characters, invalid length)
- /api/batch-predict (dataset selection, ranking)
- /api/metrics (dataset-specific metrics & ROC curves)
- /api/model-info (Ridge, Random Forest, XGBoost, SVM/SVR)
- /api/dataset-info (HeLa, HCT116, HEK293, HL60)
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.models.predictor import predictor

# Ensure models are loaded
predictor.load_artifacts()
client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert "XGBoost" in data["available_models"]
    assert "Random Forest" in data["available_models"]
    assert "SVM" in data["available_models"]
    assert "Ridge Linear Regression" in data["available_models"]
    assert "1D CNN" not in data["available_models"]
    assert "HeLa" in data["available_datasets"]


def test_predict_dataset_specific():
    for dataset in ["HeLa", "HCT116", "HEK293", "HL60"]:
        payload = {
            "sequence": "CTTGCTCGCGCAGGACGAGGCGG",
            "model_name": "XGBoost",
            "dataset_name": dataset
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        res = response.json()["data"]
        assert res["sequence"] == "CTTGCTCGCGCAGGACGAGGCGG"
        assert res["dataset_used"] == dataset
        assert 0.0 <= res["predicted_efficiency_score"] <= 1.0
        assert 0.0 <= res["relative_efficacy_score"] <= 100.0
        assert "verification_metrics" in res
        assert "roc_curve" in res["verification_metrics"]


def test_predict_svm():
    payload = {
        "sequence": "CTTGCTCGCGCAGGACGAGGCGG",
        "model_name": "SVM",
        "dataset_name": "HeLa"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    res = response.json()["data"]
    assert res["model_used"] == "SVM"
    assert 0.0 <= res["relative_efficacy_score"] <= 100.0


def test_predict_weighted_ensemble():
    payload = {
        "sequence": "CTTGCTCGCGCAGGACGAGGCGG",
        "model_name": "Weighted Ensemble",
        "dataset_name": "HeLa"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    res = response.json()["data"]
    assert res["model_used"] == "Weighted Ensemble"
    assert 0.0 <= res["relative_efficacy_score"] <= 100.0
    assert res["ensemble_dispersion"] is not None


def test_predict_valid_20nt_auto_pam():
    payload = {
        "sequence": "GACCTGATCGATCGATCGTA",
        "model_name": "Random Forest",
        "dataset_name": "HeLa"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    res = response.json()["data"]
    assert res["was_extended_with_pam"] is True
    assert len(res["sequence"]) == 23
    assert res["ensemble_dispersion"] is not None


def test_predict_invalid_nucleotide():
    payload = {"sequence": "CTTGCTCGCGCAGGACGAGGCGX"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "Invalid nucleotide characters detected" in response.json()["detail"]


def test_predict_invalid_length():
    payload = {"sequence": "ACGT"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "Invalid sequence length" in response.json()["detail"]


def test_batch_predict_with_dataset():
    payload = {
        "guides": [
            {"id": "g1", "sequence": "CTTGCTCGCGCAGGACGAGGCGG"},
            {"id": "g2", "sequence": "ACATCAGGTTACCTCTACCAAGG"},
            {"id": "invalid_g", "sequence": "NON_NUCLEOTIDE_SEQ"}
        ],
        "model_name": "XGBoost",
        "dataset_name": "HCT116"
    }
    response = client.post("/api/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_submitted"] == 3
    assert data["total_successful"] == 2
    assert data["dataset_used"] == "HCT116"
    results = data["results"]
    assert any(r.get("rank") == 1 for r in results)
    assert any(r.get("rank") == 2 for r in results)


def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert "hela" in data["datasets"]
    assert "hct116" in data["datasets"]
    assert "hek293" in data["datasets"]
    assert "hl60" in data["datasets"]

    # Check model presence in HeLa
    hela_models = data["datasets"]["hela"]["models"]
    assert "XGBoost" in hela_models
    assert "Random Forest" in hela_models
    assert "SVM" in hela_models
    assert "Ridge Linear Regression" in hela_models
    assert "1D CNN" not in hela_models


def test_model_info_endpoint():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    models = response.json()["models"]
    names = [m["name"] for m in models]
    assert "XGBoost" in names
    assert "Random Forest" in names
    assert any("SVM" in n or "Support Vector" in n for n in names)
    assert any("Ridge" in n for n in names)
    assert not any("CNN" in n for n in names)


def test_dataset_info_endpoint():
    response = client.get("/api/dataset-info")
    assert response.status_code == 200
    data = response.json()
    assert "cell_lines" in data
    cell_ids = [c["id"] for c in data["cell_lines"]]
    assert "hela" in cell_ids
    assert "hct116" in cell_ids
    assert "hek293" in cell_ids
    assert "hl60" in cell_ids
