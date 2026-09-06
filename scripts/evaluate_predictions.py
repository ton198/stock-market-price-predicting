#!/usr/bin/env python3
"""Evaluate saved probabilities under the repository's canonical contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_pipeline.data import prepare_dataset
from quant_pipeline.evaluator import evaluate_probability_stream, load_prediction_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="CSV with exactly the canonical test dates and Date,probability columns",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "nasdaq_multivariate.csv",
    )
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = prepare_dataset(args.data)
    dates, probabilities = load_prediction_csv(args.predictions)
    report = evaluate_probability_stream(
        dataset,
        dates,
        probabilities,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nWrote evaluation report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
