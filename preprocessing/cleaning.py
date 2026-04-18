from __future__ import annotations

import pandas as pd

from common.config import CATEGORICAL_COLUMNS, DATE_COLUMNS, ID_COL, NUMERIC_COLUMNS, TARGET_COL


PLACEHOLDER_NULLS = {'', 'NA', 'N/A', 'na', 'n/a', 'None', 'none', '-'}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.copy()
    for placeholder in PLACEHOLDER_NULLS:
        out = out.replace(placeholder, pd.NA)

    for col in NUMERIC_COLUMNS + [TARGET_COL]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors='coerce')

    for col in CATEGORICAL_COLUMNS + [ID_COL]:
        if col in out.columns:
            out[col] = out[col].astype('string').str.strip()

    return out


def drop_empty_or_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if ID_COL in out.columns:
        out = out[out[ID_COL].notna()]
    if TARGET_COL in out.columns:
        out = out[out[TARGET_COL].notna()]
    out = out.dropna(how='all').reset_index(drop=True)
    return out


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = standardize_column_names(df)
    out = coerce_types(out)
    out = drop_empty_or_invalid_rows(out)
    return out
