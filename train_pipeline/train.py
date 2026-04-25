"""Training entry point backed by pricing.clo_pricing.

The older modular pipeline trained a separate model bundle format.  That created
contradictions with pricing/clo_pricing.py.  This wrapper preserves the public
`run_training_pipeline` entry point but delegates all training/model/similarity
logic to `pricing.clo_pricing.train_and_save_bundle`.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.config import MODEL_BUNDLE_PATH, TARGET_COL
from pricing.clo_pricing import load_data, train_and_save_bundle


@dataclass
class TrainConfig:
    raw_input: str
    synthetic_input: str | None = None  # retained for compatibility; not used
    target_col: str = TARGET_COL        # retained for compatibility; clo_pricing.TARGET is authoritative
    test_size: float = 0.15             # retained for compatibility; clo_pricing.TEST_SIZE is authoritative
    random_state: int = 42
    neighbors: int = 5
    synth_rows: int = 500
    similarity_real_only: bool = True


def run_training_pipeline(config: TrainConfig) -> dict:
    df = load_data(config.raw_input)
    result = train_and_save_bundle(
        df=df,
        bundle_path=str(MODEL_BUNDLE_PATH),
        synth_rows=int(config.synth_rows),
        random_state=int(config.random_state),
        similarity_on_real_only=bool(config.similarity_real_only),
    )
    return {
        "model_bundle_path": str(MODEL_BUNDLE_PATH),
        "bundle_path": result["bundle_path"],
        "metrics": result["metrics"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
    }
