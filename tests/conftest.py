from __future__ import annotations

import sys
from pathlib import Path

# Avoid importing heavy optional xgboost during unit tests. The production code
# already supports a linear/ridge fallback when xgboost is unavailable.
sys.modules.setdefault("xgboost", None)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
