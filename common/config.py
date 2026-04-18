from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / 'input'
RAW_INPUT_DIR = INPUT_DIR / 'raw'
SYNTHETIC_INPUT_DIR = INPUT_DIR / 'synthetic'
MODELS_DIR = PROJECT_ROOT / 'models'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
DOCS_DIR = PROJECT_ROOT / 'docs'

MODEL_BUNDLE_PATH = MODELS_DIR / 'model_bundle.joblib'
TRAINING_METRICS_PATH = ARTIFACTS_DIR / 'training_metrics.json'
FEATURE_IMPORTANCE_PNG = ARTIFACTS_DIR / 'feature_importance.png'
TRAINING_PREDICTIONS_CSV = ARTIFACTS_DIR / 'validation_predictions.csv'
PROCESSED_TRAINING_DATA_CSV = ARTIFACTS_DIR / 'processed_training_data.csv'

TARGET_COL = 'Spread'
ID_COL = 'Bloomberg ID'
MANAGER_COL = 'Collateral manager'
ROW_SOURCE_COL = 'row_source'

DATE_COLUMNS = ['Trade Date', 'Closing Date']
NUMERIC_COLUMNS = [
    'Manager Discount',
    'Cover Price',
    'NC Yrs left',
    'RP Yrs left',
    'DM TC',
    'DM to RP+24',
    'MVOC',
    'MVOC (w/o X)',
    'Attach',
    'Thickness',
    'Diversity',
    'WAS %ile',
    '% <80',
    '% <50',
    '% CCC',
    'Equity NAV',
    'Eq Last Payment',
    'Junior OC',
    'Excess Spread',
    'Deal WACC',
    'AAA Coupon',
    'LSTA',
    'CLOIE',
    'USD 3M',
    'HY CDX',
    'VIX',
]
CATEGORICAL_COLUMNS = [MANAGER_COL]

SIMILARITY_FEATURE_COLUMNS = [
    'MVOC', 'MVOC (w/o X)', 'Attach', 'Thickness', 'Diversity',
    'WAS %ile', '% <80', '% <50', '% CCC', 'Equity NAV',
    'Eq Last Payment', 'Junior OC', 'Excess Spread',
    'Deal WACC', 'AAA Coupon', 'NC Yrs left', 'RP Yrs left',
]

TOP_IMPORTANCE_COUNT = 15
TOP_SIMILARITY_EXPLANATION_COUNT = 5
DEFAULT_NEIGHBORS = 10

for folder in [INPUT_DIR, RAW_INPUT_DIR, SYNTHETIC_INPUT_DIR, MODELS_DIR, ARTIFACTS_DIR, DOCS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
