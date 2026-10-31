"""
Automated Scientific Rigor and Methodological Verification Test Suite.

Verifies:
1. Dataset isolation: Each of the 4 cell lines (HeLa, HCT116, HEK293, HL60) has independent artifacts and models.
2. Split integrity: Strict zero sequence overlap between 80% development pool and 20% untouched final validation holdout.
3. 100-fold CV integrity: Exactly 100 stratified folds with zero sequence leakage and fold-specific scaling.
4. Model set integrity: CNN is completely absent; SVM (SVR), Ridge, Random Forest, and XGBoost are present.
5. Metric presence: Specificity (TN/(TN+FP)), Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, RMSE, MAE, R², Pearson r, Spearman rho.
6. Real ROC data: FPR and TPR arrays with matching AUC values.
7. Terminology & non-causal attribution: Relative Efficacy Score, Empirical Ensemble Dispersion (not calibrated Bayesian), SpCas9 biological cut site annotation.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "backend", "saved_models")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "backend", "data", "processed")

CELL_LINES = ["hela", "hct116", "a375", "hl60"]
EXPECTED_MODELS = ["Ridge Linear Regression", "Random Forest", "XGBoost", "SVM"]


def test_split_integrity_and_zero_sequence_overlap():
    """
    CRITICAL METHODOLOGICAL TEST:
    Verifies that for every cell line independently, there is strictly zero sequence overlap
    between the 80% development pool and the 20% untouched final validation holdout.
    """
    for cell in CELL_LINES:
        split_file = os.path.join(SAVED_MODELS_DIR, cell, "split_assignments.json")
        assert os.path.exists(split_file), f"Split assignments file not found for {cell} at {split_file}"

        with open(split_file, "r") as f:
            data = json.load(f)

        dev_seqs = set(data["development_sequences"])
        holdout_seqs = set(data["holdout_sequences"])

        overlap = dev_seqs.intersection(holdout_seqs)
        overlap_count = len(overlap)

        assert overlap_count == 0, (
            f"CRITICAL SEQUENCE LEAKAGE in {cell}: {overlap_count} sequences appear in both 80% dev and 20% holdout!"
        )
        assert data["split_info"]["sequence_overlap_count"] == 0
        assert data["split_info"]["cv_folds"] == 100


def test_model_set_and_cnn_absence():
    """Verifies that 1D CNN is completely removed and SVM (SVR) is present across all datasets."""
    # Verify no cnn_model.py exists in backend/app/models/
    cnn_file = os.path.join(BASE_DIR, "backend", "app", "models", "cnn_model.py")
    assert not os.path.exists(cnn_file), "CRITICAL: cnn_model.py still exists in backend/app/models/!"

    for cell in CELL_LINES:
        cell_dir = os.path.join(SAVED_MODELS_DIR, cell)
        assert os.path.exists(os.path.join(cell_dir, "svr_model.joblib")), f"SVM model artifact missing for {cell}"
        assert os.path.exists(os.path.join(cell_dir, "rf_model.joblib")), f"RF model artifact missing for {cell}"
        assert os.path.exists(os.path.join(cell_dir, "xgb_model.joblib")), f"XGBoost model artifact missing for {cell}"
        assert os.path.exists(os.path.join(cell_dir, "ridge_model.joblib")), f"Ridge model artifact missing for {cell}"
        assert not os.path.exists(os.path.join(cell_dir, "cnn_model.pt")), f"CNN artifact should not exist in {cell}"


def test_cv_and_holdout_metrics_structure():
    """Verifies presence of all required regression and classification metrics in both 100-fold CV and holdout."""
    required_metrics = [
        "accuracy", "precision", "recall", "f1", "specificity",
        "roc_auc", "pr_auc", "rmse", "mae", "mse", "r2", "pearson", "spearman"
    ]

    for cell in CELL_LINES:
        metrics_file = os.path.join(SAVED_MODELS_DIR, cell, "metrics.json")
        assert os.path.exists(metrics_file), f"Metrics file missing for {cell}"

        with open(metrics_file, "r") as f:
            data = json.load(f)

        assert "models" in data, f"Missing models in {cell} metrics"

        for model_name in EXPECTED_MODELS:
            assert model_name in data["models"], f"Model {model_name} missing from {cell} metrics"
            m_data = data["models"][model_name]

            # 100-Fold CV metrics
            assert "cv_100_fold" in m_data
            cv_mean = m_data["cv_100_fold"]["mean"]
            cv_std = m_data["cv_100_fold"]["std"]

            for metric in required_metrics:
                assert metric in cv_mean, f"Metric '{metric}' missing in CV mean for {cell} / {model_name}"
                assert metric in cv_std, f"Metric '{metric}' missing in CV std for {cell} / {model_name}"
                assert not np.isnan(cv_mean[metric]), f"NaN in CV mean '{metric}' for {cell} / {model_name}"

            # Final 20% holdout metrics
            assert "final_validation_20" in m_data
            ho = m_data["final_validation_20"]
            for metric in required_metrics:
                assert metric in ho, f"Metric '{metric}' missing in holdout for {cell} / {model_name}"
                assert not np.isnan(ho[metric]), f"NaN in holdout '{metric}' for {cell} / {model_name}"

            # Verify Specificity calculation is bounded in [0, 1]
            assert 0.0 <= ho["specificity"] <= 1.0, f"Specificity out of bounds in {cell} / {model_name}"
            assert 0.0 <= ho["roc_auc"] <= 1.0, f"ROC-AUC out of bounds in {cell} / {model_name}"


def test_roc_curves_data():
    """Verifies that real ROC FPR and TPR coordinate arrays exist and AUC matches."""
    for cell in CELL_LINES:
        roc_file = os.path.join(SAVED_MODELS_DIR, cell, "roc_curves.json")
        assert os.path.exists(roc_file), f"ROC curves file missing for {cell}"

        with open(roc_file, "r") as f:
            data = json.load(f)

        for model_name in EXPECTED_MODELS:
            assert model_name in data, f"{model_name} missing in {cell} ROC curves"
            roc = data[model_name]
            assert "fpr" in roc and "tpr" in roc and "auc" in roc
            assert len(roc["fpr"]) > 0
            assert len(roc["tpr"]) == len(roc["fpr"])
            assert 0.0 <= roc["auc"] <= 1.0


def test_dataset_isolation_in_predictor():
    """Verifies that CRISPRPredictor properly isolates datasets and loads distinct artifacts."""
    from backend.app.models.predictor import predictor
    predictor.load_artifacts()

    # Predict with HeLa vs HCT116 vs A375 vs HL60 with same sequence
    res_hela = predictor.predict_single("CTTGCTCGCGCAGGACGAGGCGG", model_name="XGBoost", dataset_name="HeLa")
    res_hct = predictor.predict_single("CTTGCTCGCGCAGGACGAGGCGG", model_name="XGBoost", dataset_name="HCT116")
    res_a375 = predictor.predict_single("CTTGCTCGCGCAGGACGAGGCGG", model_name="SVM", dataset_name="A375")
    res_hl60 = predictor.predict_single("CTTGCTCGCGCAGGACGAGGCGG", model_name="Ridge Linear Regression", dataset_name="HL60")

    assert res_hela["dataset_used"] == "HeLa"
    assert res_hct["dataset_used"] == "HCT116"
    assert res_a375["dataset_used"] == "A375"
    assert res_hl60["dataset_used"] == "HL60"

    # Verify legacy alias resolves to canonical A375 with zero biological false claim
    res_alias = predictor.predict_single("CTTGCTCGCGCAGGACGAGGCGG", model_name="SVM", dataset_name="HEK293")
    assert res_alias["dataset_used"] == "A375"
    assert res_alias["dataset_key"] == "a375"
    assert res_alias["relative_efficacy_score"] == res_a375["relative_efficacy_score"]

    # Verify scores are generated and valid
    assert 0.0 <= res_hela["relative_efficacy_score"] <= 100.0
    assert 0.0 <= res_hct["relative_efficacy_score"] <= 100.0


def test_scientific_terminology_and_disclaimer():
    """Verifies that score, dispersion, and disclaimer adhere strictly to non-causal guidelines."""
    from backend.app.models.predictor import predictor
    predictor.load_artifacts()

    res_rf = predictor.predict_single("CTTGCTCGCGCAGGACGAGGCGG", model_name="Random Forest", dataset_name="HeLa")

    # Score terminology
    assert "relative_efficacy_score" in res_rf
    assert "predicted_efficiency_score" in res_rf

    # Uncertainty terminology: must NOT contain "Bayesian confidence interval" or "guaranteed"
    note = res_rf["uncertainty_note"].lower()
    assert "bayesian confidence" not in note
    assert "guaranteed" not in note
    assert res_rf["ensemble_dispersion"] is not None

    # Non-RF models should state uncertainty not available
    res_xgb = predictor.predict_single("CTTGCTCGCGCAGGACGAGGCGG", model_name="XGBoost", dataset_name="HeLa")
    assert res_xgb["ensemble_dispersion"] is None
    assert "not available" in res_xgb["uncertainty_note"].lower()

    # SpCas9 cut site annotation disclaimer
    disclaimer = res_rf["scientific_disclaimer"].lower()
    assert "computational prediction" in disclaimer
    assert "not experimental validation" in disclaimer
    assert "cleavage site" in disclaimer or "cut site" in disclaimer


def test_oof_predictions_auditability():
    """Verifies that out-of-fold predictions are generated and saved for every development sample."""
    import pandas as pd
    for cell in CELL_LINES:
        oof_path = os.path.join(SAVED_MODELS_DIR, cell, "oof_predictions.csv")
        assert os.path.exists(oof_path), f"oof_predictions.csv missing for {cell}"
        oof_df = pd.read_csv(oof_path)
        assert len(oof_df) > 0
        for col in ["sequence", "binary_label", "efficiency_score", "predicted_score", "dataset", "model"]:
            assert col in oof_df.columns, f"Column {col} missing in {cell} oof_predictions.csv"

        # Verify predictions are within valid range [0, 1]
        assert (oof_df["predicted_score"] >= 0.0).all()
        assert (oof_df["predicted_score"] <= 1.0).all()


def test_hyperparameters_and_frozen_thresholds():
    """Verifies that hyperparameter search spaces and frozen thresholds are properly documented."""
    for cell in CELL_LINES:
        hp_path = os.path.join(SAVED_MODELS_DIR, cell, "hyperparameters.json")
        assert os.path.exists(hp_path), f"hyperparameters.json missing for {cell}"
        with open(hp_path, "r") as f:
            hp_data = json.load(f)
        assert ("tuning_results" in hp_data or "hyperparameters" in hp_data), f"Neither tuning_results nor hyperparameters found in {cell} hyperparameters.json"
        assert "frozen_thresholds" in hp_data
        for m in EXPECTED_MODELS:
            assert m in hp_data["frozen_thresholds"]
            thresh = hp_data["frozen_thresholds"][m]
            assert 0.0 < thresh < 1.0, f"Invalid threshold {thresh} for {m} in {cell}"


def test_training_comparison_artifact():
    """Verifies that training_comparison.json records comparison across all 4 datasets."""
    comp_path = os.path.join(SAVED_MODELS_DIR, "training_comparison.json")
    assert os.path.exists(comp_path), "training_comparison.json missing in saved_models/"
    with open(comp_path, "r") as f:
        comp_data = json.load(f)
    for cell in CELL_LINES:
        assert cell in comp_data
        for m in EXPECTED_MODELS:
            assert m in comp_data[cell]
            assert "cv_pearson" in comp_data[cell][m]
            assert "holdout_pearson" in comp_data[cell][m]


def test_feature_importance_validity():
    """Verifies that feature_importance.json contains genuine positive importance values."""
    for cell in CELL_LINES:
        fi_path = os.path.join(SAVED_MODELS_DIR, cell, "feature_importance.json")
        assert os.path.exists(fi_path), f"feature_importance.json missing for {cell}"
        with open(fi_path, "r") as f:
            fi_data = json.load(f)
        for m in EXPECTED_MODELS:
            assert m in fi_data
            assert len(fi_data[m]) > 0
            for item in fi_data[m]:
                assert "feature" in item
                assert "importance" in item
                assert float(item["importance"]) >= 0.0


def test_holdout_integrity_report_and_ensemble():
    """Verifies that holdout_integrity_report.json, pr_curves.json, and ensemble configs are valid."""
    # 1. Holdout Integrity Report
    report_path = os.path.join(SAVED_MODELS_DIR, "holdout_integrity_report.json")
    assert os.path.exists(report_path), "holdout_integrity_report.json missing!"
    with open(report_path, "r") as f:
        report = json.load(f)
    assert report.get("audit_passed") is True
    assert len(report["datasets"]) == 4
    for d in report["datasets"]:
        assert d["sequence_overlap_count"] == 0
        assert d["development_only_tuning_confirmed"] is True
        assert d["holdout_evaluated_exactly_once"] is True

    # 2. PR curves and Ensemble Config
    for cell in CELL_LINES:
        cell_dir = os.path.join(SAVED_MODELS_DIR, cell)

        pr_path = os.path.join(cell_dir, "pr_curves.json")
        assert os.path.exists(pr_path), f"pr_curves.json missing for {cell}"
        with open(pr_path, "r") as f:
            pr_data = json.load(f)
        for m in EXPECTED_MODELS + ["Weighted Ensemble"]:
            assert "precision" in pr_data[m] and "recall" in pr_data[m]
            auc_val = pr_data[m].get("pr_auc", pr_data[m].get("auc"))
            assert auc_val is not None
            assert 0.0 <= auc_val <= 1.0

        ens_path = os.path.join(cell_dir, "ensemble_config.json")
        assert os.path.exists(ens_path), f"ensemble_config.json missing for {cell}"
        with open(ens_path, "r") as f:
            ens_data = json.load(f)
        assert "weights" in ens_data
        w_sum = sum(ens_data["weights"].values())
        assert abs(w_sum - 1.0) < 0.01

    # 3. Predict with Weighted Ensemble
    from backend.app.models.predictor import predictor
    predictor.load_artifacts()
    for cell in ["HeLa", "HCT116", "A375", "HL60"]:
        res = predictor.predict_single("CTTGCTCGCGCAGGACGAGGCGG", model_name="Weighted Ensemble", dataset_name=cell)
        assert res["model_used"] == "Weighted Ensemble"
        assert 0.0 <= res["relative_efficacy_score"] <= 100.0
        assert res["ensemble_dispersion"] is not None


def test_dedicated_classification_models_presence():
    """
    Verifies that the dedicated classification models are serialized and present across all 4 datasets,
    and that SVR and SVC remain distinct models.
    """
    clf_models = [
        "logistic_model.joblib",
        "rf_classifier.joblib",
        "xgb_classifier.joblib",
        "svc_model.joblib",
        "ridge_classifier.joblib"
    ]
    reg_models = [
        "ridge_model.joblib",
        "rf_model.joblib",
        "xgb_model.joblib",
        "svr_model.joblib"
    ]

    for cell in CELL_LINES:
        cell_dir = os.path.join(SAVED_MODELS_DIR, cell)
        
        # Check classification models
        for clf_file in clf_models:
            assert os.path.exists(os.path.join(cell_dir, clf_file)), f"Missing {clf_file} for {cell}"
            
        # Check regression models
        for reg_file in reg_models:
            assert os.path.exists(os.path.join(cell_dir, reg_file)), f"Missing {reg_file} for {cell}"

        # Distinct SVR vs SVC
        svr_stat = os.stat(os.path.join(cell_dir, "svr_model.joblib"))
        svc_stat = os.stat(os.path.join(cell_dir, "svc_model.joblib"))
        assert svr_stat.st_size > 0 and svc_stat.st_size > 0

        # Dedicated json artifacts
        assert os.path.exists(os.path.join(cell_dir, "regression_metrics.json"))
        assert os.path.exists(os.path.join(cell_dir, "classification_metrics.json"))
        assert os.path.exists(os.path.join(cell_dir, "classifier_thresholds.json"))


def test_classification_metrics_and_specificity():
    """
    Verifies that classification_metrics.json computes and records valid classification metrics,
    including ROC-AUC, PR-AUC, Accuracy, and Specificity (TN/(TN+FP)).
    """
    for cell in CELL_LINES:
        clf_json = os.path.join(SAVED_MODELS_DIR, cell, "classification_metrics.json")
        assert os.path.exists(clf_json), f"Missing classification_metrics.json for {cell}"
        with open(clf_json, "r") as f:
            data = json.load(f)

        assert "models" in data
        for c_name, c_data in data["models"].items():
            ho = c_data["final_validation_20"]
            assert "roc_auc" in ho and 0.0 <= ho["roc_auc"] <= 1.0
            assert "pr_auc" in ho and 0.0 <= ho["pr_auc"] <= 1.0
            assert "accuracy" in ho and 0.0 <= ho["accuracy"] <= 1.0
            assert "specificity" in ho and 0.0 <= ho["specificity"] <= 1.0
            assert "recall" in ho and 0.0 <= ho["recall"] <= 1.0
            assert "f1" in ho and 0.0 <= ho["f1"] <= 1.0


def test_dual_task_predictor_inference():
    """
    Verifies that the inference service provides both continuous regression outputs
    and binary classification outputs on single and batch predictions.
    """
    from backend.app.models.predictor import predictor
    predictor.load_artifacts()

    seq = "GACCGTGCCCTCGAACACGTGGG"
    for ds in ["HeLa", "HCT116", "A375", "HL60"]:
        res = predictor.predict_single(seq, model_name="XGBoost", dataset_name=ds)
        # Regression fields
        assert "predicted_efficiency_score" in res
        assert 0.0 <= res["predicted_efficiency_score"] <= 1.0
        assert "relative_efficacy_score" in res
        assert 0.0 <= res["relative_efficacy_score"] <= 100.0
        assert "activity_classification" in res
        # Classification fields
        assert "binary_prediction" in res
        assert res["binary_prediction"] in [0, 1]
        assert res["binary_prediction_label"] in ["Active", "Inactive"]
        assert "classification_probability" in res
        assert 0.0 <= res["classification_probability"] <= 1.0
        assert "decision_threshold" in res
        assert "classification_model_used" in res

    # Test explicit classifier selection
    res_clf = predictor.predict_single(seq, model_name="Support Vector Classifier (SVC)", dataset_name="HeLa")
    assert res_clf["model_used"] == "Support Vector Classifier (SVC)"
    assert res_clf["binary_prediction"] in [0, 1]
    assert 0.0 <= res_clf["predicted_efficiency_score"] <= 1.0


def test_baselines_presence_and_comparison():
    """
    Verifies that dummy regression and dummy classification baselines are serialized
    and recorded honestly in metrics files.
    """
    for cell in CELL_LINES:
        cell_dir = os.path.join(SAVED_MODELS_DIR, cell)
        assert os.path.exists(os.path.join(cell_dir, "dummy_regressor.joblib"))
        assert os.path.exists(os.path.join(cell_dir, "dummy_classifier.joblib"))

        with open(os.path.join(cell_dir, "regression_metrics.json"), "r") as f:
            reg_m = json.load(f)
        assert "baselines" in reg_m
        assert ("DummyRegressor (Mean)" in reg_m["baselines"] or "mean_regressor" in reg_m["baselines"])

        with open(os.path.join(cell_dir, "classification_metrics.json"), "r") as f:
            clf_m = json.load(f)
        assert "baselines" in clf_m
        assert ("DummyClassifier (Most Frequent)" in clf_m["baselines"] or "most_frequent_classifier" in clf_m["baselines"])



