"""Command-line entry point for feature export, evaluation, and prediction."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_test, load_training
from .evaluation import evaluate
from .model import predict_log_prices, verify_checkpoint
from .pipeline import prepare_features


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    features = commands.add_parser("features", help="Rebuild features from the raw inputs.")
    validation = commands.add_parser("evaluate", help="Run the recorded cross-validation protocol.")
    prediction = commands.add_parser("predict", help="Create the Kaggle submission CSV.")

    for command in (features, validation, prediction):
        command.add_argument("--config", type=Path, default=Path("configs/final.json"))
        command.add_argument("--data-dir", type=Path, default=Path("data/raw"))
        command.add_argument("--output-dir", type=Path, default=Path("outputs"))
    for command in (validation, prediction):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--device", choices=("mps", "cuda", "cpu"), required=True)
    return parser.parse_args()


def export_features(train, test, train_ids, test_ids, sale_prices, output_dir):
    folder = output_dir / "features"
    folder.mkdir(parents=True, exist_ok=True)
    train.insert(0, "Id", train_ids)
    train["SalePrice"] = sale_prices
    test.insert(0, "Id", test_ids)
    train.to_csv(folder / "train.csv", index=False)
    test.to_csv(folder / "test.csv", index=False)
    print(f"Wrote {len(train)} training rows and {len(test)} test rows to {folder}.")


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command != "features":
        verify_checkpoint(args.checkpoint, config["checkpoint"]["sha256"])

    train, log_prices, train_ids, sale_prices = load_training(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate":
        evaluate(train, log_prices, config, args.checkpoint, args.device, args.output_dir)
        return

    test, test_ids = load_test(args.data_dir, train.columns)
    train, test = prepare_features(train, log_prices, test, config)
    if args.command == "features":
        export_features(train, test, train_ids, test_ids, sale_prices, args.output_dir)
        return

    predicted = predict_log_prices(train, log_prices, test, args.checkpoint, args.device, config["model"])
    prices = np.exp(predicted)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Submission sale prices must be finite and strictly positive.")
    output = args.output_dir / "submission.csv"
    pd.DataFrame({"Id": test_ids, "SalePrice": prices}).to_csv(
        output, index=False, float_format="%.10f"
    )
    print(f"Wrote {len(prices)} predictions to {output}.")


if __name__ == "__main__":
    main()
