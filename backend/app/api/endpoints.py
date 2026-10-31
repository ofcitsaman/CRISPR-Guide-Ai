"""
FastAPI REST API routes for CRISPR Guide AI.
Exposes dataset-specific endpoints for:
- Single guide prediction with verification metrics & ROC points
- Batch guide prediction & ranking (JSON and CSV upload)
- Model descriptions (Ridge, Random Forest, XGBoost, SVM/SVR)
- Dataset provenance (HeLa, HCT116, HEK293, HL60)
- 100-Fold Cross-Validation & Final 20% Holdout evaluation metrics
- Feature importance attribution
"""

import io
import csv
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field

from backend.app.models.predictor import (
    predictor,
    DATASET_DISPLAY_NAMES,
    normalize_dataset_name,
    normalize_model_name
)
from backend.app.features.extractor import SequenceValidationError

router = APIRouter()


class PredictRequest(BaseModel):
    sequence: str = Field(..., description="sgRNA nucleotide sequence (20 nt guide or 23 nt guide+PAM)")
    model_name: Optional[str] = Field("XGBoost", description="Model: 'Random Forest', 'XGBoost', 'SVM', or 'Ridge Linear Regression'")
    dataset_name: Optional[str] = Field("HeLa", description="Cell line dataset: 'HeLa', 'HCT116', 'HEK293', or 'HL60'")


class BatchGuideItem(BaseModel):
    id: Optional[str] = Field(None, description="Unique guide identifier")
    sequence: str = Field(..., description="sgRNA nucleotide sequence")


class BatchPredictRequest(BaseModel):
    guides: List[BatchGuideItem]
    model_name: Optional[str] = Field("XGBoost", description="Model architecture")
    dataset_name: Optional[str] = Field("HeLa", description="Cell line dataset")


@router.get("/health")
def health_check():
    """Returns system status, loaded datasets, and available models."""
    if not predictor._is_loaded:
        try:
            predictor.load_artifacts()
        except Exception:
            pass

    return {
        "status": "healthy",
        "service": "CRISPR Guide AI",
        "models_loaded": predictor._is_loaded,
        "available_datasets": ["HeLa", "HCT116", "A375", "HL60", "HEK293"],
        "available_models": ["Ridge Linear Regression", "Random Forest", "XGBoost", "SVM", "Weighted Ensemble"],
        "available_regression_models": ["Ridge Linear Regression", "Random Forest", "XGBoost", "SVM", "Weighted Ensemble"],
        "available_classification_models": [
            "Logistic Regression",
            "Random Forest Classifier",
            "XGBoost Classifier",
            "Support Vector Classifier (SVC)",
            "Ridge Classifier"
        ],
        "loaded_datasets": [DATASET_DISPLAY_NAMES.get(k, k) for k in predictor.dataset_artifacts.keys()]
    }


@router.post("/predict")
def predict_sgrna(request: PredictRequest):
    """Predicts knockout efficiency for a single sgRNA sequence using dataset-specific models."""
    try:
        result = predictor.predict_single(
            sequence=request.sequence,
            model_name=request.model_name or "XGBoost",
            dataset_name=request.dataset_name or "HeLa"
        )
        return {"status": "success", "data": result}
    except SequenceValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/batch-predict")
def predict_batch_sgrna(request: BatchPredictRequest):
    """Predicts knockout efficiency for multiple sgRNAs and ranks them by activity."""
    if len(request.guides) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1,000 guides per request.")

    items = [{"id": g.id or f"guide_{i+1}", "sequence": g.sequence} for i, g in enumerate(request.guides)]
    model_name = request.model_name or "XGBoost"
    dataset_name = request.dataset_name or "HeLa"

    results = predictor.predict_batch(items, model_name=model_name, dataset_name=dataset_name)
    return {
        "status": "success",
        "dataset_used": DATASET_DISPLAY_NAMES.get(normalize_dataset_name(dataset_name), dataset_name),
        "model_used": normalize_model_name(model_name),
        "total_submitted": len(items),
        "total_successful": sum(1 for r in results if r.get("status") == "success"),
        "results": results
    }


@router.post("/upload-csv")
async def upload_csv_predict(
    file: UploadFile = File(...),
    model_name: str = Form("XGBoost"),
    dataset_name: str = Form("HeLa")
):
    """Uploads a CSV file with columns (id, sequence) for batch prediction."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv file.")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:  # 5MB limit
        raise HTTPException(status_code=400, detail="CSV file size exceeds 5MB limit.")

    try:
        text = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))

        # Check column headers
        fieldnames = [f.strip().lower() for f in (reader.fieldnames or [])]
        seq_col = None
        id_col = None

        for original_col in (reader.fieldnames or []):
            low = original_col.strip().lower()
            if low in ("sequence", "sgrna", "guide", "spacer", "seq"):
                seq_col = original_col
            elif low in ("id", "name", "guide_id", "gene"):
                id_col = original_col

        if not seq_col:
            raise HTTPException(
                status_code=400,
                detail=f"CSV missing required sequence column. Found columns: {reader.fieldnames}. Please name the column 'sequence' or 'sgRNA'."
            )

        items = []
        for idx, row in enumerate(reader):
            seq = row[seq_col].strip()
            guide_id = row[id_col].strip() if (id_col and row.get(id_col)) else f"guide_{idx+1}"
            items.append({"id": guide_id, "sequence": seq})
            if len(items) >= 1000:
                break

        results = predictor.predict_batch(items, model_name=model_name, dataset_name=dataset_name)
        return {
            "status": "success",
            "dataset_used": DATASET_DISPLAY_NAMES.get(normalize_dataset_name(dataset_name), dataset_name),
            "model_used": normalize_model_name(model_name),
            "filename": file.filename,
            "total_submitted": len(items),
            "total_successful": sum(1 for r in results if r.get("status") == "success"),
            "results": results
        }
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV file must be UTF-8 encoded text.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")


@router.get("/model-info")
def get_model_info():
    """Returns educational and technical descriptions of all 4 production models."""
    return {
        "models": [
            {
                "name": "Ridge Linear Regression",
                "type": "Parametric Linear Model with L2 Regularization",
                "features_used": "Engineered sequence & biophysical features (scaled + SelectKBest)",
                "description": (
                    "Analytical linear regression model regularized with L2 penalty (alpha optimized per dataset in [100.0, 300.0]). "
                    "Establishes a transparent baseline evaluating whether linear positional frequencies suffice for efficacy ranking."
                ),
                "strengths": "Ultra-fast inference, non-black-box signed coefficients, robust against overfitting small partitions.",
                "limitations": "Cannot learn synergistic non-linear epistatic interactions between distant nucleotide positions."
            },
            {
                "name": "Random Forest",
                "type": "Non-Parametric Ensemble (Bagged Decision Trees)",
                "features_used": "Engineered sequence & biophysical features (SelectKBest)",
                "description": (
                    "Ensemble of decorrelated decision trees (tuned n_estimators=150-300, max_depth=10-14). "
                    "Variance among individual tree predictions provides empirical ensemble dispersion (±1.96σ)."
                ),
                "strengths": "Captures non-linear feature splits; provides empirical ensemble dispersion reflecting tree disagreement.",
                "limitations": "Dispersion reflects tree disagreement and is not a formally calibrated Bayesian confidence interval."
            },
            {
                "name": "XGBoost",
                "type": "Gradient-Boosted Decision Trees",
                "features_used": "Engineered sequence & biophysical features (SelectKBest)",
                "description": (
                    "Gradient-boosted decision trees (tuned max_depth=3-5, learning_rate=0.03-0.06, subsample=0.8-0.9). "
                    "Sequentially optimizes pseudo-residuals with second-order gradient approximations and column subsampling."
                ),
                "strengths": "Strong performance on tabular genomic feature sets; captures complex non-linear nucleotide dependencies.",
                "limitations": "Point prediction model without built-in dispersion estimation."
            },
            {
                "name": "Support Vector Regression (SVM/SVR)",
                "type": "Kernel Support Vector Machine (RBF Kernel)",
                "features_used": "Engineered sequence & biophysical features (scaled + SelectKBest)",
                "description": (
                    "Support Vector Regression with Radial Basis Function (RBF) kernel (tuned C=1.0-2.0, epsilon=0.05-0.10, gamma=scale/0.005). "
                    "Maps standardized sequence features into high-dimensional Hilbert space to fit an optimal margin hyperplane."
                ),
                "strengths": "Effective in high-dimensional feature spaces; margin epsilon provides robustness against label noise; RBF kernel captures non-linear base interactions.",
                "limitations": "Requires strict feature scaling and is computationally intensive on large cross-validation splits."
            },
            {
                "name": "Weighted Ensemble",
                "type": "Constrained Optimal Weighted Model Stacking",
                "features_used": "Optimal combination of Ridge, Random Forest, XGBoost, and SVR predictions",
                "description": (
                    "Non-negative constrained ensemble whose model weights are optimized via SLSQP strictly on development out-of-fold predictions. "
                    "Integrates linear stability, non-linear tree partitions, gradient boosting, and kernel margins."
                ),
                "strengths": "Top overall Pearson correlation, lowest continuous prediction error (RMSE/MAE), and maximum AUC stability.",
                "limitations": "Requires evaluating all constituent base models for inference."
            }
        ],
        "biophysical_context": {
            "target_system": "SpCas9 (Streptococcus pyogenes Cas9)",
            "pam_requirement": "5'-NGG-3' at positions 21-23",
            "seed_region": "Positions 11-20 (PAM-proximal 10-12 nucleotides, most sensitive to mismatches)",
            "cleavage_plane": "Between positions 17 and 18 (3 nt upstream of PAM on non-target strand)",
            "transcription_terminator": "Poly-T run (>= 4 consecutive Ts causes early termination in U6/H1 promoters)"
        }
    }


@router.get("/dataset-info")
def get_dataset_info():
    """Returns verified dataset origin, citations, sample sizes, and screening details for all 4 cell lines."""
    return {
        "dataset_name": "DeepCRISPR & CRISPRpred(SEQ) Multi-Cell Human sgRNA Benchmark",
        "description": (
            "Four independent cell-line benchmark datasets compiled from published high-throughput CRISPR knockout screens. "
            "Each dataset is evaluated independently under a strict 80% development + 20% untouched final holdout protocol "
            "with 100-fold cross-validation."
        ),
        "cell_lines": [
            {
                "id": "hela",
                "name": "HeLa",
                "samples": 8101,
                "unique_sgrnas": 8101,
                "positive_count": 5060,
                "negative_count": 3041,
                "organism": "Homo sapiens (Cervical carcinoma)",
                "assay_type": "Viability dropout screen (essential gene knockout)",
                "original_study": "Hart et al., Cell 2015",
                "doi": "10.1016/j.cell.2015.11.015"
            },
            {
                "id": "hct116",
                "name": "HCT116",
                "samples": 4239,
                "unique_sgrnas": 4239,
                "positive_count": 2718,
                "negative_count": 1521,
                "organism": "Homo sapiens (Colorectal carcinoma)",
                "assay_type": "Viability dropout screen (essential gene knockout)",
                "original_study": "Hart et al., Cell 2015",
                "doi": "10.1016/j.cell.2015.11.015"
            },
            {
                "id": "a375",
                "legacy_id": "hek293",
                "name": "A375",
                "display_name": "A375 (Doench 2016)",
                "samples": 2333,
                "unique_sgrnas": 2333,
                "positive_count": 1374,
                "negative_count": 959,
                "organism": "Homo sapiens (Malignant melanoma)",
                "assay_type": "Viability and drug-resistance screen tiling 8 human genes (MED12, NF1, NF2, TADA2B, CUL3, TADA1, CCDC101, HPRT1)",
                "original_study": "Doench et al., Nature Biotechnology 2016",
                "doi": "10.1038/nbt.3437",
                "provenance_note": "Legacy benchmark files misattributed these 2,333 records as HEK293T; definitively reclassified to A375."
            },
            {
                "id": "hek293",
                "name": "HEK293 (Legacy Alias for A375)",
                "samples": 2333,
                "unique_sgrnas": 2333,
                "positive_count": 1374,
                "negative_count": 959,
                "organism": "Homo sapiens (Malignant melanoma)",
                "assay_type": "Legacy alias mapping to verified A375 (Doench 2016) models",
                "original_study": "Doench et al., Nature Biotechnology 2016",
                "doi": "10.1038/nbt.3437"
            },
            {
                "id": "hl60",
                "name": "HL60",
                "samples": 2076,
                "unique_sgrnas": 2076,
                "positive_count": 1474,
                "negative_count": 602,
                "organism": "Homo sapiens (Acute promyelocytic leukemia)",
                "assay_type": "Viability dropout screen (essential gene knockout)",
                "original_study": "Xu et al., Genome Research 2015",
                "doi": "10.1101/gr.184531.114"
            }
        ],
        "target_variables": {
            "continuous_target": "efficiency_score: continuous normalized knockout activity score [0.0 - 1.0]",
            "binary_target": "binary_label: experimental activity classification (log2 fold-change >= 1.0)"
        },
        "citations": [
            {
                "title": "DeepCRISPR: optimized CRISPR guide RNA design by deep learning",
                "authors": "Guohui Chuai, Hanhui Ma, Jifang Yan, Ming Chen, Qi Liu, et al.",
                "journal": "Genome Biology",
                "year": 2018,
                "doi": "10.1186/s13059-018-1459-4",
                "url": "https://doi.org/10.1186/s13059-018-1459-4"
            },
            {
                "title": "CRISPRpred(SEQ): a sequence-based method for sgRNA on target activity prediction using traditional machine learning",
                "authors": "Ali Haisam Muhammad Rafid, Md. Toufikuzzaman, Mohammad Saifur Rahman, M. Sohel Rahman",
                "journal": "BMC Bioinformatics",
                "year": 2020,
                "doi": "10.1186/s12859-020-3531-9",
                "url": "https://doi.org/10.1186/s12859-020-3531-9"
            },
            {
                "title": "High-Resolution CRISPR Screens Reveal Fitness Genes and Genotype-Specific Cancer Liabilities",
                "authors": "Traver Hart, Megha Chandrashekhar, Michael Aregger, et al.",
                "journal": "Cell",
                "year": 2015,
                "doi": "10.1016/j.cell.2015.11.015",
                "url": "https://doi.org/10.1016/j.cell.2015.11.015"
            },
            {
                "title": "Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9",
                "authors": "John G Doench, Nicolo Fusi, Meagan Sullender, et al.",
                "journal": "Nature Biotechnology",
                "year": 2016,
                "doi": "10.1038/nbt.3437",
                "url": "https://doi.org/10.1038/nbt.3437"
            }
        ]
    }


@router.get("/metrics")
def get_metrics(
    dataset: Optional[str] = Query(None, description="Optional cell line filter: 'hela', 'hct116', 'hek293', 'hl60'"),
    task: Optional[str] = Query(None, description="Optional task filter: 'regression', 'classification'")
):
    """Returns dataset-specific 100-fold CV and final 20% holdout metrics."""
    if not predictor._is_loaded:
        predictor.load_artifacts()

    if dataset:
        key = normalize_dataset_name(dataset)
        if key in predictor.dataset_artifacts:
            art = predictor.dataset_artifacts[key]
            if task == "regression" and "regression_metrics" in art and art["regression_metrics"]:
                return art["regression_metrics"]
            elif task == "classification" and "classification_metrics" in art and art["classification_metrics"]:
                return art["classification_metrics"]
            return art.get("metrics", {})
        if "datasets" in predictor.global_metrics and key in predictor.global_metrics["datasets"]:
            cell_metrics = predictor.global_metrics["datasets"][key]
            if task == "regression":
                return cell_metrics.get("regression_models", cell_metrics.get("models", {}))
            elif task == "classification":
                return cell_metrics.get("classification_models", {})
            return cell_metrics

    return predictor.global_metrics


@router.get("/feature-importance")
def get_feature_importance(dataset: Optional[str] = Query("HeLa")):
    """Returns top globally important sequence features for the selected cell line."""
    if not predictor._is_loaded:
        predictor.load_artifacts()

    key = normalize_dataset_name(dataset)
    if key in predictor.dataset_artifacts:
        return {
            "dataset": DATASET_DISPLAY_NAMES.get(key, key),
            "top_features": predictor.dataset_artifacts[key].get("feature_importance", []),
            "note": "Feature importance represents statistical predictive contribution across training sequences, not proven biological causality."
        }
    return {
        "dataset": "HeLa",
        "top_features": [],
        "note": "No feature importance available."
    }
