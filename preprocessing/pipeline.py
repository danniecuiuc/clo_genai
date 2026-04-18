from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COL
from features.builder import build_feature_frame


@dataclass
class PreparedDataset:
    raw: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series | None
    model_columns: list[str]


def infer_model_columns(feature_df: pd.DataFrame) -> list[str]:
    numeric_candidates = [col for col in feature_df.columns if col in NUMERIC_COLUMNS]
    categorical_candidates = [col for col in feature_df.columns if col in CATEGORICAL_COLUMNS]
    engineered_candidates = [
        col for col in feature_df.columns
        if col.endswith('_year')
        or col.endswith('_month')
        or col.endswith('_dayofyear')
        or col in {
            'seasoning_days',
            'attach_to_thickness',
            'mvoc_gap_ratio',
            'cover_price_to_coupon',
            'excess_spread_minus_wacc',
            'excess_spread_to_wacc',
        }
    ]
    return numeric_candidates + engineered_candidates + categorical_candidates


def prepare_dataset(df: pd.DataFrame, target_col: str = TARGET_COL) -> PreparedDataset:
    feature_df = build_feature_frame(df)
    model_columns = infer_model_columns(feature_df)
    x = feature_df[model_columns].copy()
    y = feature_df[target_col].copy() if target_col in feature_df.columns else None
    return PreparedDataset(raw=df.copy(), features=x, target=y, model_columns=model_columns)


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = feature_df.select_dtypes(include=['number', 'datetime64[ns]']).columns.tolist()
    categorical_cols = [col for col in feature_df.columns if col not in numeric_cols]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    return ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols),
    ])


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names: list[str] = []
        for _, _, columns in preprocessor.transformers_:
            if isinstance(columns, list):
                names.extend(columns)
        return names
