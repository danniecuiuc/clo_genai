from __future__ import annotations

from pricing.clo_pricing import SimilarityModule


def test_similarity_module_returns_comparable_deals(tiny_clo_df) -> None:
    model = SimilarityModule(n_neighbors=3)
    model.fit(tiny_clo_df)

    comps = model.query(tiny_clo_df.iloc[[0]])

    assert len(comps) == 3
    assert "similarity_score" in comps.columns
    assert "euclidean_distance" in comps.columns
    assert comps["euclidean_distance"].is_monotonic_increasing
