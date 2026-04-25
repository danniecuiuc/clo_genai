from __future__ import annotations

import argparse
import json

from train_pipeline.train import TrainConfig, run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CLO tranche pricing MVP CLI')
    sub = parser.add_subparsers(dest='command', required=True)

    train_cmd = sub.add_parser('train', help='Train preprocessing, pricing, and product bundle.')
    train_cmd.add_argument('--raw-input', required=True)
    train_cmd.add_argument('--synthetic-input', default=None)
    train_cmd.add_argument('--target', default='Price')
    train_cmd.add_argument('--test-size', type=float, default=0.2)
    train_cmd.add_argument('--random-state', type=int, default=42)
    train_cmd.add_argument('--neighbors', type=int, default=10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'train':
        config = TrainConfig(
            raw_input=args.raw_input,
            synthetic_input=args.synthetic_input,
            target_col=args.target,
            test_size=args.test_size,
            random_state=args.random_state,
            neighbors=args.neighbors,
        )
        result = run_training_pipeline(config)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
