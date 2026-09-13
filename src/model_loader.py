import joblib
import json
import os

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

class Models:
    def __init__(self):
        lgbm_path = os.path.join(ARTIFACT_DIR, "lgbm_baseline_binary.joblib")
        rf_path = os.path.join(ARTIFACT_DIR, "rf_baseline_binary.joblib")
        self.binary_model = joblib.load(lgbm_path) if os.path.exists(lgbm_path) else joblib.load(rf_path)
        self.binary_scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_binary.joblib"))

        self.multiclass_model = joblib.load(os.path.join(ARTIFACT_DIR, "lgbm_multiclass.joblib"))
        self.multiclass_scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_multiclass.joblib"))
        self.label_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder_multiclass.joblib"))

        with open(os.path.join(ARTIFACT_DIR, "feature_columns.json")) as f:
            self.feature_columns = json.load(f)

models = Models()
