#!/usr/bin/env python3
"""Train canonical Stage 1 Hybrid LSTM seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_pipeline.data import prepare_dataset
from quant_pipeline.stage1 import fit_stage1_seed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--train-stride", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    args = parser.parse_args(argv)

    dataset = prepare_dataset(args.data)
    summaries: dict[str, object] = {}
    for seed in args.seeds:
        seed_directory = args.run_dir / f"seed-{seed:03d}"
        summaries[str(seed)] = fit_stage1_seed(
            dataset,
            seed_directory,
            seed=seed,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            window_size=args.window_size,
            train_stride=args.train_stride,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.early_stopping_patience,
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
