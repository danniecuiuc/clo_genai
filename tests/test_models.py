from __future__ import annotations

import numpy as np
import pandas as pd

from pricing.models import build_candidate_models, evaluate_regression, train_candidate_models


def test_build_candidate_models_contains_core_models() -> None:
    models = build_candidate_models(random_state=123)

    assert "linear" in models
    assert "ridge" in models
    assert all(hasattr(model, "fit") for model in models.values())
    assert all(hasattr(model, "predict") for model in models.values())


def test_evaluate_regression_perfect_fit() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    metrics = evaluate_regression(y_true, y_pred)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_train_candidate_models_returns_fitted_outputs() -> None:
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(40, 4))
    y_train = 2.0 * x_train[:, 0] - 0.5 * x_train[:, 1] + 1.0
    x_valid = rng.normal(size=(12, 4))
    y_valid = 2.0 * x_valid[:, 0] - 0.5 * x_valid[:, 1] + 1.0

    outputs = train_candidate_models(
        X_train=x_train,
        y_train=y_train,
        X_valid=x_valid,
        y_valid=y_valid,
        random_state=42,
    )

    assert outputs.best_model_name in outputs.fitted_models
    assert outputs.best_model_name in outputs.metrics
    assert outputs.residual_std >= 0.0
    assert set(outputs.validation_predictions.columns) == {"y_true", "y_pred", "residual"}
    assert len(outputs.validation_predictions) == len(y_valid)

    for model_metrics in outputs.metrics.values():
        assert set(model_metrics) == {"rmse", "mae", "r2"}
        assert np.isfinite(model_metrics["rmse"])
        assert np.isfinite(model_metrics["mae"])
        assert np.isfinite(model_metrics["r2"])
