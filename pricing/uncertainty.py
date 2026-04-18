from __future__ import annotations

import numpy as np


def prediction_interval(prediction: float, residual_std: float, z_value: float = 1.96) -> tuple[float, float]:
    width = z_value * residual_std
    return float(prediction - width), float(prediction + width)


def distance_based_uncertainty_flag(distances) -> str:
    distances = np.asarray(distances)
    mean_distance = float(np.mean(distances))
    if mean_distance < 1.0:
        return 'low'
    if mean_distance < 2.0:
        return 'medium'
    return 'high'
