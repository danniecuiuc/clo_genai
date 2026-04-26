"""
Explainability utilities for CLO pricing models.

This module owns:
- feature importance table construction
- Ridge vs XGBoost feature-importance comparison
- actual-vs-predicted plot
- optional SHAP explanation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def build_feature_importance_table(
    baseline_model,
    final_model,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Build feature-importance table for the active pricing model.

    If XGBoost final_model exists, use XGBoost feature_importances_.
    Otherwise use absolute Ridge coefficients.
    """
    if final_model is not None:
        importance = getattr(final_model, "feature_importances_", None)

        if importance is None:
            importance = np.zeros(len(feature_names))

        fi = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        )

    else:
        coef = getattr(baseline_model, "coef_", np.zeros(len(feature_names)))
        fi = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.abs(coef),
            }
        )

    return (
        fi.sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_feature_importance_comparison(
    baseline_model,
    final_model,
    feature_names: list[str],
    output_dir: Path,
    write_artifacts: bool,
    top_n: int = 15,
) -> Optional[pd.DataFrame]:
    """
    Side-by-side feature importance:
      - Ridge: absolute coefficient magnitude, normalized
      - XGBoost: gain-based importance, normalized

    Returns the comparison table.
    """
    coef = getattr(baseline_model, "coef_", np.zeros(len(feature_names)))
    ridge_raw = np.abs(coef)
    ridge_sum = ridge_raw.sum()
    ridge_norm = ridge_raw / ridge_sum if ridge_sum > 0 else ridge_raw

    ridge_fi = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": ridge_norm,
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    if final_model is not None and hasattr(final_model, "feature_importances_"):
        xgb_raw = final_model.feature_importances_
        xgb_sum = xgb_raw.sum()
        xgb_norm = xgb_raw / xgb_sum if xgb_sum > 0 else xgb_raw

        xgb_fi = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": xgb_norm,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
        )
    else:
        xgb_fi = ridge_fi.copy()

    merged = (
        ridge_fi.rename(columns={"importance": "Ridge (norm coef)"})
        .merge(
            xgb_fi.rename(columns={"importance": "XGBoost (gain)"}),
            on="feature",
            how="outer",
        )
        .fillna(0)
        .sort_values("XGBoost (gain)", ascending=False)
        .reset_index(drop=True)
    )

    merged["Ridge (norm coef)"] = merged["Ridge (norm coef)"].round(4)
    merged["XGBoost (gain)"] = merged["XGBoost (gain)"].round(4)

    ridge_rank = merged["Ridge (norm coef)"].rank(ascending=False)
    xgb_rank = merged["XGBoost (gain)"].rank(ascending=False)

    merged["Agreement"] = [
        "✓" if abs(r - x) <= 3 else "~"
        for r, x in zip(ridge_rank, xgb_rank)
    ]

    print("\n── Feature Importance: Ridge vs XGBoost ──")
    print("  Ridge coef  — linear marginal effect; suitable for model validation")
    print("  XGBoost gain — nonlinear split contribution; stronger predictive signal")
    print("  Agreement ✓ — feature ranks similarly in both models")
    print("  Agreement ~ — feature rank differs\n")
    print(merged.to_string(index=False))

    if write_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_dir / "feature_importance_comparison.csv", index=False)
        log.info("Saved feature_importance_comparison.csv")

    if HAS_PLOT:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ridge_plot = ridge_fi.sort_values("importance")
        axes[0].barh(ridge_plot["feature"], ridge_plot["importance"])
        axes[0].set_title("Ridge — Normalized |Coefficient|")
        axes[0].set_xlabel("Normalized Importance")

        xgb_plot = xgb_fi.sort_values("importance")
        axes[1].barh(xgb_plot["feature"], xgb_plot["importance"])
        axes[1].set_title("XGBoost — Normalized Gain")
        axes[1].set_xlabel("Normalized Importance")

        plt.suptitle("Feature Importance Comparison: Ridge vs XGBoost")
        plt.tight_layout()

        if write_artifacts:
            plt.savefig(
                output_dir / "feature_importance_comparison.png",
                dpi=150,
                bbox_inches="tight",
            )
            log.info("Saved feature_importance_comparison.png")

        plt.show()

    return merged


def plot_actuals_vs_predicted(
    model,
    preprocessor,
    X: pd.DataFrame,
    y: pd.Series,
    all_features: list[str],
    target_name: str,
    output_dir: Path,
    write_artifacts: bool,
) -> None:
    """Plot actual target vs model prediction on a test set."""
    if not HAS_PLOT:
        return

    X_t = preprocessor.transform(X[all_features])
    preds = model.predict(X_t)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, preds, alpha=0.5, s=20)

    mn = min(y.min(), preds.min())
    mx = max(y.max(), preds.max())

    ax.plot([mn, mx], [mn, mx], "r--", lw=1)
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title("Actuals vs Predicted – Test Set")

    plt.tight_layout()

    if write_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_dir / "actuals_vs_predicted.png",
            dpi=150,
            bbox_inches="tight",
        )
        log.info("Saved actuals_vs_predicted.png")

    plt.show()


def shap_explain(
    final_model,
    preprocessor,
    X: pd.DataFrame,
    all_features: list[str],
    feature_names: list[str],
    max_display: int = 15,
):
    """Optional SHAP summary plot for XGBoost final model."""
    if not HAS_SHAP or final_model is None:
        log.warning("SHAP unavailable or XGBoost not trained.")
        return None

    X_t = preprocessor.transform(X[all_features])

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_t)

    if HAS_PLOT:
        shap.summary_plot(
            shap_values,
            X_t,
            feature_names=feature_names,
            max_display=max_display,
            show=True,
        )

    return shap_values