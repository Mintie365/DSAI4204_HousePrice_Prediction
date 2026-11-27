"""Build the C0 feature representation used in the final submission."""

import numpy as np

from .legacy.feature_selection import SelectedFeatures


NEIGHBORHOOD_FIELDS = ["OverallQual", "LotArea", "YearBuilt"]


class C0Pipeline:
    def __init__(self, selection_settings):
        self.selector = SelectedFeatures(**selection_settings)

    def fit(self, features, log_prices):
        self.selector.fit(features, log_prices)
        self.neighborhood_medians = features.groupby("Neighborhood")[NEIGHBORHOOD_FIELDS].median()
        self.global_medians = features[NEIGHBORHOOD_FIELDS].median()
        return self

    def transform(self, features):
        frame = self.selector.transform(features)
        for column in NEIGHBORHOOD_FIELDS:
            medians = features["Neighborhood"].map(self.neighborhood_medians[column])
            frame[f"CommunityMedian_{column}"] = medians.fillna(
                self.global_medians[column]
            ).to_numpy()
        frame["RelativeQuality"] = features["OverallQual"].to_numpy() / frame[
            "CommunityMedian_OverallQual"
        ].to_numpy()
        frame["RelativeLotArea"] = features["LotArea"].to_numpy() / frame[
            "CommunityMedian_LotArea"
        ].to_numpy()
        frame["CommunityYearDifference"] = features["YearBuilt"].to_numpy() - frame[
            "CommunityMedian_YearBuilt"
        ].to_numpy()
        return frame.replace([np.inf, -np.inf], np.nan)


def prepare_features(train, log_prices, other, config):
    pipeline = C0Pipeline(config["feature_selection"]).fit(train, log_prices)
    train_features = pipeline.transform(train)
    other_features = pipeline.transform(other)
    for frame in (train_features, other_features):
        if frame.shape[1] != config["expected_predictors"] or frame.isna().any().any():
            raise ValueError("The C0 pipeline produced an unexpected feature count or missing values.")
    return train_features, other_features
