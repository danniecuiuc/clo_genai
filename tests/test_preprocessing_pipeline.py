from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import CATEGORICAL_COLUMNS, ID_COL, NUMERIC_COLUMNS, TARGET_COL
from features.builder import build_feature_frame
from preprocessing.cleaning import clean_raw_dataframe
from preprocessing.pipeline import build_preprocessor, get_transformed_feature_names, prepare_dataset


def _sample_raw_df(n: int = 8) -> pd.DataFrame:
    data = {
        ID_COL: [f"BID {i}" for i in range(n)],
        "Collateral manager": ["Manager A", "Manager B"] * (n // 2) + ["Manager A"] * (n % 2),
        TARGET_COL: np.linspace(400, 520, n),
        "Trade Date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "Closing Date": pd.date_range("2023-01-01", periods=n, freq="D"),
    }

    for idx, col in enumerate(NUMERIC_COLUMNS):
        if col not in data:
            data[col] = np.linspace(1.0 + idx, 2.0 + idx, n)

    return pd.DataFrame(data)


def test_clean_raw_dataframe_strips_columns_coerces_types_and_drops_invalid_rows() -> None:
    raw = _sample_raw_df(4)
    raw = raw.rename(columns={ID_COL: f" {ID_COL} "})
    raw.loc[1, TARGET_COL] = "N/A"
    raw.loc[2, f" {ID_COL} "] = ""
    raw.loc[3, "Manager Discount"] = "12.5"

    clean = clean_raw_dataframe(raw)

    assert ID_COL in clean.columns
    assert len(clean) == 2
    assert pd.api.types.is_numeric_dtype(clean[TARGET_COL])
    assert pd.api.types.is_numeric_dtype(clean["Manager Discount"])
    assert clean[ID_COL].isna().sum() == 0
    assert clean[TARGET_COL].isna().sum() == 0


def test_build_feature_frame_adds_date_and_ratio_features() -> None:
    raw = _sample_raw_df(3)
    features = build_feature_frame(raw)

    assert "Trade Date_year" in features.columns
    assert "Closing Date_month" in features.columns
    assert "seasoning_days" in features.columns
    assert "attach_to_thickness" in features.columns
    assert "mvoc_gap_ratio" in features.columns
    assert "excess_spread_minus_wacc" in features.columns
    assert "excess_spread_to_wacc" in features.columns


def test_prepare_dataset_excludes_target_from_features_and_builds_transformer() -> None:
    clean = clean_raw_dataframe(_sample_raw_df(10))
    prepared = prepare_dataset(clean, target_col=TARGET_COL)

    assert prepared.target is not None
    assert TARGET_COL not in prepared.features.columns
    assert len(prepared.features) == len(clean)
    assert len(prepared.model_columns) == prepared.features.shape[1]

    preprocessor = build_preprocessor(prepared.features)
    transformed = preprocessor.fit_transform(prepared.features)
    feature_names = get_transformed_feature_names(preprocessor)

    assert transformed.shape[0] == len(clean)
    assert transformed.shape[1] == len(feature_names)
    assert len(feature_names) > 0
