"""Compatibility wrappers around pricing.clo_pricing synthetic generation.

`pricing/clo_pricing.py` is now the source of truth.  Keep this module only so
older imports continue to work while avoiding a second, contradictory synthetic
implementation.
"""

from __future__ import annotations

import pandas as pd

from common.config import ROW_SOURCE_COL
from pricing.clo_pricing import generate_synthetic


def bootstrap_resample(df: pd.DataFrame, n_samples: int, random_state: int = 42) -> pd.DataFrame:
    """Generate bootstrap synthetic rows using clo_pricing.generate_synthetic."""
    if n_samples <= 0:
        return df.iloc[0:0].copy()
    return generate_synthetic(df, n_rows=n_samples, method="bootstrap").reset_index(drop=True)


def combine_real_and_synthetic(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame | None = None,
    multiplier: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return real rows plus synthetic rows, using clo_pricing as source of truth."""
    real = real_df.copy().reset_index(drop=True)
    if ROW_SOURCE_COL not in real.columns:
        real[ROW_SOURCE_COL] = "real"

    if synthetic_df is not None and not synthetic_df.empty:
        synthetic = synthetic_df.copy().reset_index(drop=True)
    else:
        synthetic = bootstrap_resample(real_df, n_samples=len(real_df) * multiplier, random_state=random_state)

    if not synthetic.empty and ROW_SOURCE_COL not in synthetic.columns:
        synthetic[ROW_SOURCE_COL] = "synthetic_bootstrap_generated"

    return pd.concat([real, synthetic], ignore_index=True)
