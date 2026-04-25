from __future__ import annotations

from pathlib import Path

from common.bundle import load_bundle
from common.config import MODEL_BUNDLE_PATH
from product_pipeline.predict import predict_for_existing_id
from train_pipeline.train import TrainConfig, run_training_pipeline


def test_training_and_prediction_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_input = repo_root / 'input' / 'raw' / 'CLO_Tranche_Data.xlsx'
    synthetic_input = repo_root / 'input' / 'synthetic' / 'bootstrap_with_real_on_top.xlsx'

    result = run_training_pipeline(
        TrainConfig(
            raw_input=str(raw_input),
            synthetic_input=str(synthetic_input),
        )
    )

    assert result['best_model_name'] in {'linear', 'ridge', 'xgboost'}
    bundle = load_bundle(MODEL_BUNDLE_PATH)
    assert bundle['best_model_name'] == result['best_model_name']

    prediction = predict_for_existing_id('RRAM 2025-37A D', str(MODEL_BUNDLE_PATH), k=5)
    assert isinstance(prediction.prediction, float)
    assert len(prediction.similar_deals) > 0
