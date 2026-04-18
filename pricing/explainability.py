from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import FEATURE_IMPORTANCE_PNG, TOP_IMPORTANCE_COUNT


def build_feature_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, 'feature_importances_'):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        values = np.abs(np.asarray(model.coef_).ravel())
    else:
        values = np.zeros(len(feature_names))

    out = pd.DataFrame({'feature': feature_names, 'importance': values})
    out = out.sort_values('importance', ascending=False).reset_index(drop=True)
    return out


def save_feature_importance_plot(
    importance_df: pd.DataFrame,
    output_path: str | Path = FEATURE_IMPORTANCE_PNG,
    top_n: int = TOP_IMPORTANCE_COUNT,
) -> None:
    top = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top['feature'], top['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top Feature Importance')
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
