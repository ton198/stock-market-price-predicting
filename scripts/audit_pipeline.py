#!/usr/bin/env python3
"""Run the canonical leakage-aware data audit from the repository root."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
