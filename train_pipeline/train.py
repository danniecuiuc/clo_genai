from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from common.bundle import save_bundle
from common.config import (
    DEFAULT_NEIGHBORS,
    FEATURE_IMPORTANCE_PNG,
    MODEL_BUNDLE_PATH,
    PROCESSED_TRAINING_DATA_CSV,
    TARGET_COL,
    TRAINING_METRICS_PATH,
    TRAINING_PREDICTIONS_CSV,
)
from common.io import load_dataframe, save_json
from preprocessing.cleaning import clean_raw_dataframe
from preprocessing.pipeline import build_preprocessor, get_transformed_feature_names, prepare_dataset
from pricing.explainability import build_feature_importance_table, save_feature_importance_plot
from pricing.models import train_candidate_models
from synthetic.bootstrap_generator import combine_real_and_synthetic


@dataclass
class TrainConfig:
    raw_input: str
    synthetic_input: str | None = None
    target_col: str = TARGET_COL
    test_size: float = 0.2
    random_state: int = 42
    neighbors: int = DEFAULT_NEIGHBORS


def run_training_pipeline(config: TrainConfig) -> dict:
    raw_df = clean_raw_dataframe(load_dataframe(config.raw_input))
    synthetic_df = None
    if config.synthetic_input:
        synthetic_df = clean_raw_dataframe(load_dataframe(config.synthetic_input))

    training_df = combine_real_and_synthetic(
        real_df=raw_df,
        synthetic_df=synthetic_df,
        random_state=config.random_state,
    )

    prepared = prepare_dataset(training_df, target_col=config.target_col)
    if prepared.target is None:
        raise ValueError(f'Target column {config.target_col} not found in training data.')

    x_train, x_valid, y_train, y_valid = train_test_split(
        prepared.features,
        prepared.target,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    preprocessor = build_preprocessor(x_train)
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_valid_transformed = preprocessor.transform(x_valid)
    transformed_feature_names = get_transformed_feature_names(preprocessor)

    pricing_outputs = train_candidate_models(
        X_train=x_train_transformed,
        y_train=y_train,
        X_valid=x_valid_transformed,
        y_valid=y_valid,
        random_state=config.random_state,
    )

    best_model = pricing_outputs.fitted_models[pricing_outputs.best_model_name]
    importance_df = build_feature_importance_table(best_model, transformed_feature_names)
    save_feature_importance_plot(importance_df, FEATURE_IMPORTANCE_PNG)

    pricing_outputs.validation_predictions.to_csv(TRAINING_PREDICTIONS_CSV, index=False)
    prepared.features.assign(**{config.target_col: prepared.target}).to_csv(PROCESSED_TRAINING_DATA_CSV, index=False)

    bundle = {
        'config': asdict(config),
        'target_col': config.target_col,
        'preprocessor': preprocessor,
        'pricing_models': pricing_outputs.fitted_models,
        'best_model_name': pricing_outputs.best_model_name,
        'residual_std': pricing_outputs.residual_std,
        'training_reference_df': training_df.reset_index(drop=True),
        'reference_features_raw': prepared.features.reset_index(drop=True),
        'reference_features_transformed': preprocessor.transform(prepared.features),
        'transformed_feature_names': transformed_feature_names,
        'feature_importance': importance_df,
    }
    save_bundle(bundle, MODEL_BUNDLE_PATH)
    save_json(pricing_outputs.metrics, TRAINING_METRICS_PATH)

    return {
        'model_bundle_path': str(MODEL_BUNDLE_PATH),
        'best_model_name': pricing_outputs.best_model_name,
        'metrics': pricing_outputs.metrics,
        'feature_importance_path': str(FEATURE_IMPORTANCE_PNG),
        'validation_predictions_path': str(TRAINING_PREDICTIONS_CSV),
        'processed_training_data_path': str(PROCESSED_TRAINING_DATA_CSV),
        'row_count': int(len(training_df)),
        'feature_count': int(prepared.features.shape[1]),
    }
