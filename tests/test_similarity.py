from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import ID_COL, MANAGER_COL
from similarity.knn_engine import (
    explain_top_distance_contributors,
    find_neighbors_for_existing_row,
    find_neighbors_for_new_payload,
)


def _reference_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COL: ["A", "B", "C"],
            MANAGER_COL: ["M1", "M2", "M3"],
            "Spread": [100, 110, 130],
        }
    )


def test_find_neighbors_for_existing_row_returns_self_first() -> None:
    matrix = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])

    outputs = find_neighbors_for_existing_row(
        reference_df=_reference_df(),
        transformed_reference=matrix,
        transformed_feature_names=["feature_1", "feature_2"],
        target_row_idx=0,
        k=2,
    )

    assert len(outputs.ranked_clean) == 2
    assert outputs.ranked_clean.iloc[0][ID_COL] == "A"
    assert outputs.ranked_clean.iloc[0]["distance"] == 0.0
    assert "top_features" in outputs.ranked_clean.columns


def test_find_neighbors_for_new_payload_orders_by_distance() -> None:
    matrix = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
    new_row = np.array([[0.9, 0.0]])

    outputs = find_neighbors_for_new_payload(
        reference_df=_reference_df(),
        transformed_reference=matrix,
        transformed_new=new_row,
        transformed_feature_names=["feature_1", "feature_2"],
        k=2,
    )

    assert outputs.ranked_clean[ID_COL].tolist() == ["B", "A"]
    assert outputs.ranked_clean["distance"].is_monotonic_increasing


def test_explain_top_distance_contributors_returns_weighted_features() -> None:
    matrix = np.array([[0.0, 0.0], [3.0, 4.0]])
    target = np.array([0.0, 0.0])

    explanations = explain_top_distance_contributors(
        full_matrix=matrix,
        target_vector=target,
        neighbor_indices=np.array([1]),
        feature_names=["x", "y"],
        threshold=0.0,
    )

    assert 1 in explanations
    assert explanations[1][0][0] == "y"
    assert explanations[1][0][1] > explanations[1][1][1]
