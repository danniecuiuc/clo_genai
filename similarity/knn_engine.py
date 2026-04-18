from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

from common.config import ID_COL, MANAGER_COL, TOP_SIMILARITY_EXPLANATION_COUNT


@dataclass
class SimilarityOutputs:
    ranked_raw: pd.DataFrame
    ranked_clean: pd.DataFrame


def fit_similarity_model(reference_matrix, n_neighbors: int) -> NearestNeighbors:
    model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    model.fit(reference_matrix)
    return model


def _to_dense_row(vector) -> np.ndarray:
    if hasattr(vector, 'toarray'):
        return np.asarray(vector.toarray()).reshape(1, -1)
    return np.asarray(vector).reshape(1, -1)


def explain_top_distance_contributors(
    full_matrix,
    target_vector,
    neighbor_indices: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.10,
) -> dict[int, list[tuple[str, float]]]:
    target = _to_dense_row(target_vector)
    explanations: dict[int, list[tuple[str, float]]] = {}

    for idx in neighbor_indices:
        neighbor = _to_dense_row(full_matrix[idx])
        diff = ((neighbor - target) ** 2).ravel()
        total = float(diff.sum()) + 1e-12
        ranked = sorted(zip(feature_names, diff / total), key=lambda x: x[1], reverse=True)
        explanations[int(idx)] = [
            (feature, float(weight))
            for feature, weight in ranked
            if weight > threshold
        ][:TOP_SIMILARITY_EXPLANATION_COUNT]

    return explanations


def build_ranked_similarity_tables(
    reference_df: pd.DataFrame,
    distances: np.ndarray,
    indices: np.ndarray,
    explanations: dict[int, list[tuple[str, float]]],
) -> SimilarityOutputs:
    ranked_raw = reference_df.iloc[indices].copy().reset_index(drop=False).rename(columns={'index': 'source_index'})
    ranked_raw['distance'] = distances

    keep_cols = [col for col in [ID_COL, MANAGER_COL, 'distance', 'source_index'] if col in ranked_raw.columns]
    ranked_clean = ranked_raw[keep_cols].copy()
    ranked_clean['top_features'] = ranked_clean['source_index'].map(explanations)
    return SimilarityOutputs(ranked_raw=ranked_raw, ranked_clean=ranked_clean)


def find_neighbors_for_existing_row(
    reference_df: pd.DataFrame,
    transformed_reference,
    transformed_feature_names: list[str],
    target_row_idx: int,
    k: int,
) -> SimilarityOutputs:
    model = fit_similarity_model(transformed_reference, n_neighbors=k)
    target_vector = transformed_reference[target_row_idx]
    distances, indices = model.kneighbors(target_vector)
    explanations = explain_top_distance_contributors(
        full_matrix=transformed_reference,
        target_vector=target_vector,
        neighbor_indices=indices[0],
        feature_names=transformed_feature_names,
    )
    return build_ranked_similarity_tables(reference_df, distances[0], indices[0], explanations)


def find_neighbors_for_new_payload(
    reference_df: pd.DataFrame,
    transformed_reference,
    transformed_new,
    transformed_feature_names: list[str],
    k: int,
) -> SimilarityOutputs:
    distances = euclidean_distances(transformed_new, transformed_reference)[0]
    indices = np.argsort(distances)[:k]
    explanations = explain_top_distance_contributors(
        full_matrix=transformed_reference,
        target_vector=transformed_new[0],
        neighbor_indices=indices,
        feature_names=transformed_feature_names,
    )
    return build_ranked_similarity_tables(reference_df, distances[indices], indices, explanations)
