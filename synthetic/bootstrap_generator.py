from __future__ import annotations

import pandas as pd

from common.config import ROW_SOURCE_COL


def bootstrap_resample(df: pd.DataFrame, n_samples: int, random_state: int = 42) -> pd.DataFrame:
    if n_samples <= 0:
        return df.iloc[0:0].copy()
    return df.sample(n=n_samples, replace=True, random_state=random_state).reset_index(drop=True)


def combine_real_and_synthetic(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame | None = None,
    multiplier: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Use provided bootstrap file when available, otherwise generate one by resampling."""
    if synthetic_df is not None and not synthetic_df.empty:
        return synthetic_df.copy().reset_index(drop=True)

    synthetic = bootstrap_resample(real_df, n_samples=len(real_df) * multiplier, random_state=random_state)
    synthetic[ROW_SOURCE_COL] = 'synthetic_bootstrap_generated'

    real = real_df.copy()
    if ROW_SOURCE_COL not in real.columns:
        real[ROW_SOURCE_COL] = 'real'

    return pd.concat([real, synthetic], ignore_index=True)
