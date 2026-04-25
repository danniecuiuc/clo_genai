from __future__ import annotations

import argparse
import json

from common.config import MODEL_BUNDLE_PATH
from pricing.clo_pricing import (
    DATA_PATH,
    main as run_clo_pricing,
    load_data,
    train_and_save_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLO pricing CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_save = sub.add_parser("train-save", help="Train pricing/similarity and save model bundle.")
    train_save.add_argument("--data-path", default=DATA_PATH, help="Input data file (xlsx/csv).")
    train_save.add_argument("--bundle-path", default=str(MODEL_BUNDLE_PATH), help="Output bundle path.")
    train_save.add_argument("--synth-rows", type=int, default=500, help="Number of synthetic rows.")
    train_save.add_argument(
        "--similarity-real-only",
        action="store_true",
        help="Fit similarity model on real rows only.",
    )

    run = sub.add_parser("run-analysis", help="Run full clo_pricing analysis flow.")
    run.add_argument("--run-batch", action="store_true", help="Run batch pricing flow.")
    run.add_argument("--run-query", action="store_true", help="Run single query example.")
    run.add_argument("--run-compare", action="store_true", help="Run synthesizer comparison.")
    run.add_argument("--run-ablation", action="store_true", help="Run feature ablation.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train-save":
        df = load_data(args.data_path)
        result = train_and_save_bundle(
            df=df,
            bundle_path=args.bundle_path,
            synth_rows=args.synth_rows,
            similarity_on_real_only=args.similarity_real_only,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    if args.command == "run-analysis":
        run_clo_pricing(
            run_batch=args.run_batch,
            run_query=args.run_query,
            run_compare=args.run_compare,
            run_ablation_flag=args.run_ablation,
        )


if __name__ == '__main__':
    main()
