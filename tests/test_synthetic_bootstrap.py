from __future__ import annotations

from pricing.clo_pricing import ALL_FEATURES, TARGET, generate_synthetic


def test_generate_synthetic_bootstrap_shape_and_columns(tiny_clo_df) -> None:
    synth = generate_synthetic(tiny_clo_df, n_rows=7, method="bootstrap")

    assert len(synth) == 7
    assert list(synth.columns) == ALL_FEATURES + [TARGET]


def test_generate_synthetic_zero_rows(tiny_clo_df) -> None:
    synth = generate_synthetic(tiny_clo_df, n_rows=0, method="bootstrap")

    assert synth.empty
    assert list(synth.columns) == ALL_FEATURES + [TARGET]
