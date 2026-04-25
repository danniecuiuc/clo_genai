from __future__ import annotations

import pandas as pd

from common.config import ROW_SOURCE_COL
from synthetic.bootstrap_generator import bootstrap_resample, combine_real_and_synthetic


def test_bootstrap_resample_respects_requested_size() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

    sampled = bootstrap_resample(df, n_samples=7, random_state=42)
    empty = bootstrap_resample(df, n_samples=0, random_state=42)

    assert len(sampled) == 7
    assert list(sampled.columns) == ["x", "y"]
    assert empty.empty
    assert list(empty.columns) == ["x", "y"]


def test_combine_real_and_synthetic_tags_rows_without_mutating_input() -> None:
    real = pd.DataFrame({"x": [1, 2], "y": [10, 20]})

    combined = combine_real_and_synthetic(real_df=real, multiplier=2, random_state=42)

    assert len(combined) == 6
    assert ROW_SOURCE_COL not in real.columns
    assert combined[ROW_SOURCE_COL].value_counts().to_dict() == {
        "real": 2,
        "synthetic_bootstrap_generated": 4,
    }


def test_combine_uses_provided_synthetic_dataframe_when_available() -> None:
    real = pd.DataFrame({"x": [1, 2]})
    provided = pd.DataFrame({"x": [99], ROW_SOURCE_COL: ["provided"]})

    combined = combine_real_and_synthetic(real_df=real, synthetic_df=provided, multiplier=10)

    assert len(combined) == 1
    assert combined.iloc[0]["x"] == 99
    assert combined.iloc[0][ROW_SOURCE_COL] == "provided"
