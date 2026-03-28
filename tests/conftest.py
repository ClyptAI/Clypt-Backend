from __future__ import annotations

import sys
from pathlib import Path


# Keep the repo root importable when pytest is invoked via different entrypoints.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
