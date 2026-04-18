from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from common.bundle import load_bundle
from common.config import DEFAULT_NEIGHBORS, ID_COL, MODEL_BUNDLE_PATH
from features.builder import build_feature_frame
from pricing.uncertainty import distance_based_uncertainty_flag, prediction_interval
from similarity.knn_engine import find_neighbors_for_existing_row, find_neighbors_for_new_payload


@dataclass
class ProductResult:
    prediction: float
    lower_bound: float
    upper_bound: float
    uncertainty_flag: str
    similar_deals: list[dict[str, Any]]
    top_feature_importance: list[dict[str, Any]]
    best_model_name: str


def _predict_from_transformed_row(bundle: dict, transformed_row, similarity_distances) -> ProductResult:
    best_model_name = bundle['best_model_name']
    best_model = bundle['pricing_models'][best_model_name]
    prediction = float(best_model.predict(transformed_row)[0])
    lower_bound, upper_bound = prediction_interval(prediction, bundle['residual_std'])
    uncertainty_flag = distance_based_uncertainty_flag(similarity_distances)

    top_feature_importance = bundle['feature_importance'].head(10).to_dict(orient='records')
    return ProductResult(
        prediction=prediction,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        uncertainty_flag=uncertainty_flag,
        similar_deals=[],
        top_feature_importance=top_feature_importance,
        best_model_name=best_model_name,
    )


def predict_for_existing_id(
    bloomberg_id: str,
    model_bundle_path: str = str(MODEL_BUNDLE_PATH),
    k: int = DEFAULT_NEIGHBORS,
) -> ProductResult:
    bundle = load_bundle(model_bundle_path)
    reference_df = bundle['training_reference_df']

    matches = reference_df[reference_df[ID_COL] == bloomberg_id]
    if matches.empty:
        raise ValueError(f'Bloomberg ID not found: {bloomberg_id}')

    target_row_idx = int(matches.index[0])
    similarity_outputs = find_neighbors_for_existing_row(
        reference_df=reference_df,
        transformed_reference=bundle['reference_features_transformed'],
        transformed_feature_names=bundle['transformed_feature_names'],
        target_row_idx=target_row_idx,
        k=k,
    )

    transformed_row = bundle['reference_features_transformed'][target_row_idx]
    result = _predict_from_transformed_row(
        bundle=bundle,
        transformed_row=transformed_row,
        similarity_distances=similarity_outputs.ranked_clean['distance'].values,
    )
    result.similar_deals = similarity_outputs.ranked_clean.to_dict(orient='records')
    return result


def predict_for_new_payload(
    payload_df: pd.DataFrame,
    model_bundle_path: str = str(MODEL_BUNDLE_PATH),
    k: int = DEFAULT_NEIGHBORS,
) -> ProductResult:
    bundle = load_bundle(model_bundle_path)
    feature_frame = build_feature_frame(payload_df)
    feature_frame = feature_frame[bundle['reference_features_raw'].columns].copy()
    transformed_new = bundle['preprocessor'].transform(feature_frame)

    similarity_outputs = find_neighbors_for_new_payload(
        reference_df=bundle['training_reference_df'],
        transformed_reference=bundle['reference_features_transformed'],
        transformed_new=transformed_new,
        transformed_feature_names=bundle['transformed_feature_names'],
        k=k,
    )
    result = _predict_from_transformed_row(
        bundle=bundle,
        transformed_row=transformed_new,
        similarity_distances=similarity_outputs.ranked_clean['distance'].values,
    )
    result.similar_deals = similarity_outputs.ranked_clean.to_dict(orient='records')
    return result
