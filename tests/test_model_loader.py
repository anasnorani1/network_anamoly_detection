import pytest
import numpy as np
from src.model_loader import models


def test_models_loaded():
    """Confirm all model artifacts loaded without error at import time."""
    assert models.binary_model is not None
    assert models.multiclass_model is not None
    assert models.binary_scaler is not None
    assert models.multiclass_scaler is not None
    assert models.label_encoder is not None


def test_feature_columns_loaded():
    """Confirm the feature schema matches what the models expect."""
    assert len(models.feature_columns) == 78
    assert isinstance(models.feature_columns, list)


def test_binary_model_predicts_valid_labels():
    """A dummy input should produce a valid ATTACK/BENIGN-style prediction."""
    dummy_input = np.zeros((1, len(models.feature_columns)))
    scaled = models.binary_scaler.transform(dummy_input)
    pred = models.binary_model.predict(scaled)[0]
    assert pred in models.binary_model.classes_


def test_multiclass_model_predicts_known_class():
    """A dummy input should produce a class the label encoder recognizes."""
    dummy_input = np.zeros((1, len(models.feature_columns)))
    scaled = models.multiclass_scaler.transform(dummy_input)
    pred = models.multiclass_model.predict(scaled)[0]
    decoded = models.label_encoder.inverse_transform([pred])[0]
    assert decoded in models.label_encoder.classes_
