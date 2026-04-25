from __future__ import annotations

import pandas as pd

from pricing.clo_pricing import ALL_FEATURES, PricingModel


def test_pricing_model_fit_predict_evaluate(tiny_clo_df) -> None:
    model = PricingModel()
    x = tiny_clo_df[ALL_FEATURES]
    y = tiny_clo_df["Price"]

    model.fit(x, y)
    prediction = model.predict(x.iloc[[0]])
    metrics = model.evaluate(x, y)

    assert set(["point_estimate", "lower_bound", "upper_bound", "baseline_estimate", "feature_importance"]).issubset(prediction)
    assert prediction["lower_bound"] <= prediction["point_estimate"] <= prediction["upper_bound"]
    assert isinstance(prediction["feature_importance"], pd.DataFrame)
    assert "Ridge baseline" in metrics
