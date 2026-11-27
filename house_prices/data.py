"""Load and validate the original competition inputs."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_training(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv", keep_default_na=False, na_values=["NA", ""])
    if train["Id"].isna().any() or not train["Id"].is_unique:
        raise ValueError("train.csv must contain unique, nonmissing IDs.")
    prices = train["SalePrice"].to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Training sale prices must be finite and strictly positive.")
    return train.drop(columns=["Id", "SalePrice"]), np.log(prices), train["Id"], train["SalePrice"]


def load_test(data_dir: Path, train_columns: pd.Index):
    test = pd.read_csv(data_dir / "test.csv", keep_default_na=False, na_values=["NA", ""])
    if test["Id"].isna().any() or not test["Id"].is_unique:
        raise ValueError("test.csv must contain unique, nonmissing IDs.")
    features = test.drop(columns="Id")
    if set(features.columns) != set(train_columns):
        raise ValueError("The training and test CSV files must contain the same predictor columns.")
    return features.loc[:, train_columns], test["Id"]
