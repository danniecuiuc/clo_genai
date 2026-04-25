from __future__ import annotations

import numpy as np
import pandas as pd

from common.bundle import save_bundle
from common.config import ID_COL, NUMERIC_COLUMNS, TARGET_COL
from preprocessing.cleaning import clean_raw_dataframe
from preprocessing.pipeline import build_preprocessor, get_transformed_feature_names, prepare_dataset
from pricing.explainability import build_feature_importance_table
from pricing.models import train_candidate_models
from product_pipeline.predict import predict_for_existing_id, predict_for_new_payload


def _training_df(n: int = 16) -> pd.DataFrame:
    data = {
        ID_COL: [f"BID-{i}" for i in range(n)],
        "Collateral manager": ["Manager A", "Manager B"] * (n // 2),
        TARGET_COL: np.linspace(300.0, 450.0, n),
        "Trade Date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "Closing Date": pd.date_range("2023-01-01", periods=n, freq="D"),
    }
    for idx, col in enumerate(NUMERIC_COLUMNS):
        if col not in data:
            data[col] = np.linspace(10.0 + idx, 20.0 + idx, n)
    return clean_raw_dataframe(pd.DataFrame(data))


def _save_tiny_bundle(path) -> tuple[pd.DataFrame, dict]:
    df = _training_df()
    prepared = prepare_dataset(df, target_col=TARGET_COL)
    assert prepared.target is not None

    preprocessor = build_preprocessor(prepared.features)
    x = preprocessor.fit_transform(prepared.features)
    feature_names = get_transformed_feature_names(preprocessor)

    outputs = train_candidate_models(
        X_train=x,
        y_train=prepared.target,
        X_valid=x,
        y_valid=prepared.target,
        random_state=42,
    )

    best_model = outputs.fitted_models[outputs.best_model_name]
    feature_importance = build_feature_importance_table(best_model, feature_names)

    bundle = {
        "config": {},
        "target_col": TARGET_COL,
        "preprocessor": preprocessor,
        "pricing_models": outputs.fitted_models,
        "best_model_name": outputs.best_model_name,
        "residual_std": outputs.residual_std,
        "training_reference_df": df.reset_index(drop=True),
        "reference_features_raw": prepared.features.reset_index(drop=True),
        "reference_features_transformed": preprocessor.transform(prepared.features),
        "transformed_feature_names": feature_names,
        "feature_importance": feature_importance,
    }
    save_bundle(bundle, path)
    return df, bundle


def test_predict_for_existing_id_returns_price_bounds_and_similar_deals(tmp_path) -> None:
    bundle_path = tmp_path / "model_bundle.joblib"
    df, _ = _save_tiny_bundle(bundle_path)

    result = predict_for_existing_id(df.iloc[0][ID_COL], model_bundle_path=str(bundle_path), k=3)

    assert isinstance(result.prediction, float)
    assert result.lower_bound <= result.prediction <= result.upper_bound
    assert result.uncertainty_flag in {"low", "medium", "high"}
    assert len(result.similar_deals) == 3
    assert len(result.top_feature_importance) > 0
    assert result.best_model_name in {"linear", "ridge", "xgboost"}


def test_predict_for_new_payload_returns_prediction_and_neighbors(tmp_path) -> None:
    bundle_path = tmp_path / "model_bundle.joblib"
    df, _ = _save_tiny_bundle(bundle_path)
    payload = df.iloc[[1]].copy()

    result = predict_for_new_payload(payload, model_bundle_path=str(bundle_path), k=4)

    assert isinstance(result.prediction, float)
    assert result.lower_bound <= result.prediction <= result.upper_bound
    assert len(result.similar_deals) == 4


def test_predict_for_existing_id_raises_clear_error_for_missing_id(tmp_path) -> None:
    import pytest

    bundle_path = tmp_path / "model_bundle.joblib"
    _save_tiny_bundle(bundle_path)

    with pytest.raises(ValueError, match="Bloomberg ID not found"):
        predict_for_existing_id("DOES-NOT-EXIST", model_bundle_path=str(bundle_path), k=3)
