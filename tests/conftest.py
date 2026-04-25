from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pricing.clo_pricing import ALL_FEATURES, TARGET


@pytest.fixture
def tiny_clo_df() -> pd.DataFrame:
    n = 24
    data = {
        "Bloomberg ID": [f"TEST {i:03d}" for i in range(n)],
        "Collateral manager": ["Manager A", "Manager B"] * (n // 2),
        TARGET: np.linspace(90.0, 104.0, n),
    }
    for idx, col in enumerate(ALL_FEATURES):
        data[col] = np.linspace(1.0 + idx, 2.0 + idx, n)
    return pd.DataFrame(data)
