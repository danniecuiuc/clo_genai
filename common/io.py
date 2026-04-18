from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from common.config import ID_COL


def _normalize_header_row(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw worksheet whose first non-empty data row contains headers."""
    header_row_idx = 0
    for idx in range(min(len(raw_df), 10)):
        row_values = raw_df.iloc[idx].astype(str).tolist()
        if ID_COL in row_values:
            header_row_idx = idx
            break

    headers = raw_df.iloc[header_row_idx].fillna('').astype(str).tolist()
    data = raw_df.iloc[header_row_idx + 1 :].copy()
    data.columns = headers
    data = data.loc[:, [col for col in data.columns if str(col).strip() != '']]
    data = data.dropna(how='all').reset_index(drop=True)
    return data


def load_excel_dataframe(path: str | Path) -> pd.DataFrame:
    """Load an Excel file and normalize the CLO raw workbook format when needed."""
    path = Path(path)
    raw = pd.read_excel(path, header=None)
    raw = raw.dropna(how='all').reset_index(drop=True)

    first_rows = raw.head(5).astype(str).fillna('')
    if first_rows.isin([ID_COL]).any().any():
        return _normalize_header_row(raw)

    return pd.read_excel(path)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        return load_excel_dataframe(path)
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix == '.parquet':
        return pd.read_parquet(path)
    raise ValueError(f'Unsupported file type: {suffix}')


def save_json(data: dict[str, Any], path: str | Path) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
