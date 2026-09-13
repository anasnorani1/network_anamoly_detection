from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    is_attack: bool
    attack_probability: float
    attack_type: str | None = None
    attack_type_confidence: float | None = None

class HealthResponse(BaseModel):
    status: str

class ModelInfoResponse(BaseModel):
    binary_model: str
    multiclass_model: str
    num_features: int
    multiclass_classes: List[str]
