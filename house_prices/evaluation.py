"""Evaluate the complete training pipeline using shuffled five-fold CV."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .model import predict_log_prices
from .pipeline import prepare_features


def evaluate(raw_features, log_prices, config, checkpoint, device, output_dir):
    rows = []
    fold_count = config["evaluation"]["n_splits"]
    for seed in config["evaluation"]["split_seeds"]:
        splitter = KFold(n_splits=fold_count, shuffle=True, random_state=seed)
        for fold, (train_indices, valid_indices) in enumerate(splitter.split(raw_features), start=1):
            train_labels = log_prices[train_indices]
            train, valid = prepare_features(
                raw_features.iloc[train_indices], train_labels,
                raw_features.iloc[valid_indices], config,
            )
            predicted = predict_log_prices(
                train, train_labels, valid, checkpoint, device, config["model"]
            )
            log_rmse = float(np.sqrt(np.mean((log_prices[valid_indices] - predicted) ** 2)))
            rows.append({"split_seed": seed, "fold": fold, "log_rmse": log_rmse})
            print(f"Seed {seed}, fold {fold}: log RMSE {log_rmse:.8f}", flush=True)

    folds = pd.DataFrame(rows)
    summary = folds.groupby("split_seed", as_index=False)["log_rmse"].mean()
    summary = summary.rename(columns={"log_rmse": "mean_fold_log_rmse"})
    folds.to_csv(output_dir / "cv_folds.csv", index=False)
    summary.to_csv(output_dir / "cv_summary.csv", index=False)
    print(f"Mean across split seeds: {summary['mean_fold_log_rmse'].mean():.8f}")
