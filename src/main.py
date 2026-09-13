from fastapi import FastAPI, HTTPException
import numpy as np

from src.schemas import PredictionRequest, PredictionResponse, HealthResponse, ModelInfoResponse
from src.model_loader import models

app = FastAPI(title="Network Anomaly Detection API", version="1.0")

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        binary_model=type(models.binary_model).__name__,
        multiclass_model=type(models.multiclass_model).__name__,
        num_features=len(models.feature_columns),
        multiclass_classes=list(models.label_encoder.classes_)
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.features) != len(models.feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(models.feature_columns)} features, got {len(request.features)}"
        )

    X = np.array(request.features).reshape(1, -1)

    X_binary_scaled = models.binary_scaler.transform(X)
    binary_pred = models.binary_model.predict(X_binary_scaled)[0]
    binary_proba = models.binary_model.predict_proba(X_binary_scaled)[0]

    is_attack = (binary_pred == "ATTACK") or (binary_pred == 1)
    attack_prob = binary_proba[list(models.binary_model.classes_).index("ATTACK")] \
        if "ATTACK" in models.binary_model.classes_ else binary_proba[1]

    response = PredictionResponse(
        is_attack=bool(is_attack),
        attack_probability=float(attack_prob)
    )

    if is_attack:
        X_multi_scaled = models.multiclass_scaler.transform(X)
        multi_pred = models.multiclass_model.predict(X_multi_scaled)[0]
        multi_proba = models.multiclass_model.predict_proba(X_multi_scaled)[0]

        response.attack_type = models.label_encoder.inverse_transform([multi_pred])[0]
        response.attack_type_confidence = float(multi_proba[multi_pred])

    return response
