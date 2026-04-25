from __future__ import annotations

import argparse
from pricing.clo_pricing import main as run_clo_pricing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CLO pricing CLI (single entrypoint)')
    parser.add_argument('--run-batch', action='store_true', help='Run batch pricing flow.')
    parser.add_argument('--run-query', action='store_true', help='Run single query example.')
    parser.add_argument('--run-compare', action='store_true', help='Run synthesizer comparison.')
    parser.add_argument('--run-ablation', action='store_true', help='Run feature ablation.')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_clo_pricing(
        run_batch=args.run_batch,
        run_query=args.run_query,
        run_compare=args.run_compare,
        run_ablation_flag=args.run_ablation,
    )


if __name__ == '__main__':
    main()
