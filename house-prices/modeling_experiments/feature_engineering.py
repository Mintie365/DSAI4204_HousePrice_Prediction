"""Fold-fitted feature engineering for the supplied Ames Housing data.

No target values or external data are used by this transformer.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCHEMES = {
    'F0_baseline': '原始79特征；数值中位数填补，类别Unknown，one-hot，数值标准化',
    'F1_semantic': 'F0 + 设施语义填补、缺失标记、社区临街长度填补、异常年份/饰面标记',
    'F2_age': 'F1 + 房龄、装修年限、是否装修、车库年龄',
    'F3_totals': 'F2 + 总面积、总卫浴、室外平台面积、设施存在标记',
    'F4_interaction': 'F3 + 质量×居住面积、质量×总面积',
    'F5_log_ordinal': 'F4 + 偏态非负字段log1p、质量有序编码（替代相应原始表示）',
}

QUALITY_COLUMNS = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
]


class HouseFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, scheme='F0_baseline'):
        self.scheme = scheme

    @staticmethod
    def _base(X):
        out = X.copy().drop(columns=['Id', 'SalePrice'], errors='ignore')
        # MSSubClass is nominal, even though the raw file uses integer codes.
        out['MSSubClass'] = out['MSSubClass'].astype(str)
        return out

    def fit(self, X, y=None):
        if self.scheme not in SCHEMES:
            raise ValueError(self.scheme)
        out = self._base(X)
        self.missing_columns_ = out.columns[out.isna().any()].tolist()
        self.frontage_by_neighborhood_ = out.groupby('Neighborhood')['LotFrontage'].median()
        self.frontage_median_ = out['LotFrontage'].median()
        return self

    def transform(self, X):
        out = self._base(X)
        level = list(SCHEMES).index(self.scheme)
        if level == 0:
            return out

        # Indicators are defined using training-fold columns only, never test statistics.
        for col in self.missing_columns_:
            out[col + '_Missing'] = out[col].isna().astype(float)
        for col, fill in [('Alley', 'NoAlley'), ('Fence', 'NoFence'),
                          ('MiscFeature', 'NoMiscFeature')]:
            out[col] = out[col].fillna(fill)

        def fill_absent(columns, absent):
            for col in columns:
                out[col] = out[col].where(~(out[col].isna() & absent), 'Absent')

        # Unknown capacity remains unknown: fill "Absent" only with evidence of absence.
        fill_absent(['PoolQC'], out['PoolArea'].eq(0))
        fill_absent(['FireplaceQu'], out['Fireplaces'].eq(0))
        no_basement = out['TotalBsmtSF'].eq(0)
        fill_absent(['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], no_basement)
        no_garage = out['GarageCars'].eq(0) & out['GarageArea'].eq(0)
        fill_absent(['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'], no_garage)
        out['LotFrontage'] = out['LotFrontage'].fillna(
            out['Neighborhood'].map(self.frontage_by_neighborhood_)
        ).fillna(self.frontage_median_)

        invalid_garage_year = ((out['GarageYrBlt'] > out['YrSold'] + 1)
                               | (out['GarageYrBlt'] < 1800))
        invalid_house_year = out['YearBuilt'] > out['YrSold']
        out['GarageYearInvalid'] = invalid_garage_year.astype(float)
        out['HouseYearInvalid'] = invalid_house_year.astype(float)
        out.loc[invalid_garage_year, 'GarageYrBlt'] = np.nan
        out.loc[invalid_house_year, 'YearBuilt'] = np.nan
        masonry_conflict = out['MasVnrType'].eq('None') & out['MasVnrArea'].gt(0)
        out['MasonryConflict'] = masonry_conflict.astype(float)
        out.loc[masonry_conflict, 'MasVnrType'] = 'Unknown'

        if level >= 2:
            out['HouseAge'] = out['YrSold'] - out['YearBuilt']
            # Do not silently change unusual observations to zero years.
            out['YearsSinceRemodel'] = (out['YrSold'] - out['YearRemodAdd']).where(
                out['YrSold'] >= out['YearRemodAdd'])
            out['HasRemodel'] = (out['YearRemodAdd'] > out['YearBuilt']).astype(float)
            out['GarageAge'] = (out['YrSold'] - out['GarageYrBlt']).where(
                ~no_garage & (out['YrSold'] >= out['GarageYrBlt']))

        if level >= 3:
            # min_count prevents a partly unknown total from being silently understated.
            out['TotalSF'] = out[['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']].sum(axis=1, min_count=3)
            out['TotalBath'] = (out['FullBath'] + .5 * out['HalfBath']
                                + out['BsmtFullBath'] + .5 * out['BsmtHalfBath'])
            porch = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
            out['TotalOutdoorSF'] = out[porch].sum(axis=1, min_count=len(porch))
            for name, col in [('HasGarage', 'GarageCars'), ('HasBasement', 'TotalBsmtSF'),
                              ('HasFireplace', 'Fireplaces'), ('HasPool', 'PoolArea'),
                              ('HasSecondFloor', '2ndFlrSF'), ('HasMasonry', 'MasVnrArea')]:
                out[name] = out[col].gt(0).astype(float).where(out[col].notna())

        if level >= 4:
            out['QualityArea'] = out['OverallQual'] * out['GrLivArea']
            out['QualityTotalSF'] = out['OverallQual'] * out['TotalSF']

        if level >= 5:
            mapping = {'Absent': 0., 'Po': 1., 'Fa': 2., 'TA': 3., 'Gd': 4., 'Ex': 5.}
            for col in QUALITY_COLUMNS:
                out[col + '_Unknown'] = (~out[col].isin(mapping)).astype(float)
                out[col] = out[col].map(mapping)
            skew_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
                            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
                            '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea',
                            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                            'ScreenPorch', 'PoolArea', 'MiscVal', 'TotalSF',
                            'TotalOutdoorSF', 'QualityArea', 'QualityTotalSF']
            for col in skew_columns:
                out[col] = np.log1p(out[col].where(out[col] >= 0))
        return out


def make_preprocessor():
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('categorical', categorical, make_column_selector(dtype_exclude=np.number)),
    ], sparse_threshold=0)


def make_pipeline(scheme, estimator):
    return Pipeline([
        ('features', HouseFeatures(scheme)),
        ('preprocess', make_preprocessor()),
        ('model', estimator),
    ])
