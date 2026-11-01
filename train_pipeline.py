"""
Master pipeline runner for CRISPR Guide AI.
Executes:
1. Data downloading & verification from official DeepCRISPR / CRISPRpred(SEQ) repositories
2. Dataset preprocessing, sequence validation & master dataset generation
3. Feature engineering (600 biophysical sequence features)
4. Model training (Ridge, Random Forest, XGBoost, Support Vector Regression SVR)
5. Evaluation under dataset-specific 80/20 holdout + 100-fold cross-validation
6. Serialization of dataset-specific model artifacts, metrics, and real ROC curves
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app.training.prepare_data import download_raw_data, validate_and_merge_data
from backend.app.training.train import run_training_pipeline

if __name__ == "__main__":
    print("Step 1: Downloading & verifying benchmark datasets...")
    download_raw_data()
    validate_and_merge_data()

    print("\nStep 2: Running feature extraction & model training pipeline...")
    run_training_pipeline()

    print("\nMaster pipeline execution complete!")
