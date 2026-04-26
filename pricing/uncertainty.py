"""
Uncertainty helpers for CLO pricing.

This module owns:
- residual-based confidence intervals
- optional quantile-model confidence intervals
"""

from __future__ import annotations

from typing import Optional


def residual_confidence_band(
    point_estimate: float,
    residual_std: float,
    z_score: float = 1.96,
) -> tuple[float, float]:
    """
    Residual-based confidence band.

    This is the fallback when quantile models are unavailable.
    """
    lower = point_estimate - z_score * residual_std
    upper = point_estimate + z_score * residual_std
    return float(lower), float(upper)


def quantile_confidence_band(
    q_low_model,
    q_high_model,
    transformed_x,
) -> tuple[float, float]:
    """
    Confidence band from trained quantile regressors.
    """
    lower = float(q_low_model.predict(transformed_x)[0])
    upper = float(q_high_model.predict(transformed_x)[0])
    return lower, upper


def prediction_interval(
    point_estimate: float,
    residual_std: float,
    transformed_x=None,
    q_low_model=None,
    q_high_model=None,
    z_score: float = 1.96,
) -> tuple[float, float]:
    """
    Return confidence band.

    Prefer quantile models if both are available.
    Otherwise fall back to point ± z_score * residual_std.
    """
    if q_low_model is not None and q_high_model is not None and transformed_x is not None:
        return quantile_confidence_band(
            q_low_model=q_low_model,
            q_high_model=q_high_model,
            transformed_x=transformed_x,
        )

    return residual_confidence_band(
        point_estimate=point_estimate,
        residual_std=residual_std,
        z_score=z_score,
    )