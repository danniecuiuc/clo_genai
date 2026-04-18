from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


@dataclass
class PricingTrainingOutputs:
    fitted_models: dict[str, Any]
    metrics: dict[str, dict[str, float]]
    best_model_name: str
    validation_predictions: pd.DataFrame
    residual_std: float


def build_candidate_models(random_state: int = 42) -> dict[str, Any]:
    return {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'xgboost': XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=1,
            random_state=random_state,
        ),
    }


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        'rmse': rmse,
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def train_candidate_models(X_train, y_train, X_valid, y_valid, random_state: int = 42) -> PricingTrainingOutputs:
    models = build_candidate_models(random_state=random_state)
    fitted_models: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}

    best_name = ''
    best_rmse = float('inf')
    best_predictions = None
    residual_std = 0.0

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        stats = evaluate_regression(y_valid, preds)
        fitted_models[model_name] = model
        metrics[model_name] = stats

        if stats['rmse'] < best_rmse:
            best_name = model_name
            best_rmse = stats['rmse']
            best_predictions = preds
            residual_std = float(np.std(y_valid - preds))

    validation_predictions = pd.DataFrame({
        'y_true': np.asarray(y_valid),
        'y_pred': np.asarray(best_predictions),
        'residual': np.asarray(y_valid) - np.asarray(best_predictions),
    })

    return PricingTrainingOutputs(
        fitted_models=fitted_models,
        metrics=metrics,
        best_model_name=best_name,
        validation_predictions=validation_predictions,
        residual_std=residual_std,
    )
