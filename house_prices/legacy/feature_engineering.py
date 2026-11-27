"""Preserve legacy F0--F5; add six raw-unit F6 ratios/interactions and two flags."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from .legacy_feature_engineering import HouseFeatures

F6_FIELDS=['LivLotRatio','BsmtFinishRatio','FirstFloorRatio','QualityCondition',
           'GarageAreaPerCar','RelativeLivingArea','BsmtFinishRatio_Unknown','GarageAreaPerCar_Unknown']


class Engineering(TransformerMixin,BaseEstimator):
    def __init__(self,scheme='F5_log_ordinal'):self.scheme=scheme

    def fit(self,X,y=None):
        base='F5_log_ordinal' if self.scheme=='F6_enhanced' else self.scheme
        self.base_=HouseFeatures(base).fit(X)
        self.raw_semantics_=HouseFeatures('F4_interaction').fit(X)
        self.community_area_=X.groupby('Neighborhood').GrLivArea.median()
        self.global_area_=float(X.GrLivArea.median())
        return self

    def transform(self,X):
        z=self.base_.transform(X)
        if self.scheme!='F6_enhanced':return z
        z = z.copy()
        raw=self.raw_semantics_.transform(X)  # F6 ratios use the original measurement units.
        def ratio(a,b):return (a/b.where(b>0)).replace([np.inf,-np.inf],np.nan)
        z['LivLotRatio']=ratio(raw.GrLivArea,raw.LotArea)
        finished=raw.BsmtFinSF1+raw.BsmtFinSF2
        basement=ratio(finished,raw.TotalBsmtSF)
        absent_bsmt=raw.TotalBsmtSF.eq(0)&raw.BsmtFinSF1.eq(0)&raw.BsmtFinSF2.eq(0)
        basement=basement.mask(absent_bsmt,0.)
        z['BsmtFinishRatio']=basement
        z['BsmtFinishRatio_Unknown']=basement.isna().astype(float)
        z['FirstFloorRatio']=ratio(raw['1stFlrSF'],raw.GrLivArea)
        z['QualityCondition']=raw.OverallQual*raw.OverallCond
        garage=ratio(raw.GarageArea,raw.GarageCars)
        garage=garage.mask(raw.GarageArea.eq(0)&raw.GarageCars.eq(0),0.)
        z['GarageAreaPerCar']=garage
        z['GarageAreaPerCar_Unknown']=garage.isna().astype(float)
        median=X.Neighborhood.map(self.community_area_).fillna(self.global_area_)
        z['RelativeLivingArea']=ratio(raw.GrLivArea,median)
        return z.replace([np.inf,-np.inf],np.nan)


class FrameImputer(TransformerMixin,BaseEstimator):
    """Fold-fitted, model-neutral filling; retain engineering field names and types."""
    def fit(self,X,y=None):
        self.columns_=X.columns.tolist()
        self.numeric_=X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_=[c for c in X if c not in self.numeric_]
        self.medians_=X[self.numeric_].median().fillna(0.)
        # For these two ratios, 0 means proven absence; -1 explicitly means unknown.
        for c in ['BsmtFinishRatio','GarageAreaPerCar']:
            if c in self.medians_:self.medians_.loc[c]=-1.
        return self
    def transform(self,X):
        z=X[self.columns_].copy()
        z[self.numeric_]=z[self.numeric_].fillna(self.medians_)
        for c in self.categorical_:z[c]=z[c].fillna('Unknown').astype(str)
        if z.isna().any().any() or not np.isfinite(z[self.numeric_].to_numpy()).all():
            raise ValueError('Invalid engineered frame')
        return z
