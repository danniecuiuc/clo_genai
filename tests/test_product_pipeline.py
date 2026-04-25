from __future__ import annotations

from pricing.clo_pricing import predict_from_bundle, train_and_save_bundle


def test_clo_pricing_replaces_old_product_pipeline(tmp_path, tiny_clo_df) -> None:
    bundle_path = tmp_path / "model_bundle.joblib"
    train_and_save_bundle(tiny_clo_df, bundle_path=str(bundle_path), synth_rows=0)

    result = predict_from_bundle(tiny_clo_df.iloc[[1]], bundle_path=str(bundle_path))

    assert "point_estimate" in result
    assert "comparables" in result
