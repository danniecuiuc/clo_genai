from __future__ import annotations

from pricing.clo_pricing import ALL_FEATURES, build_preprocessor


def test_clo_pricing_preprocessor_uses_authoritative_feature_list(tiny_clo_df) -> None:
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(tiny_clo_df[ALL_FEATURES])

    assert transformed.shape[0] == len(tiny_clo_df)
    assert transformed.shape[1] == len(ALL_FEATURES)
