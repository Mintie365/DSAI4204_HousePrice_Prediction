"""Field-level nested importance, guarded filters, redundancy and stability selection."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin,clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from .feature_engineering import Engineering, FrameImputer


def encoding(frame):
    nums=frame.select_dtypes(include=np.number).columns.tolist()
    cats=[c for c in frame if c not in nums]
    return ColumnTransformer([('numeric',StandardScaler(),nums),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)


def auxiliary(name):
    if name=='SVR':return SVR(C=5.,gamma=.0005,epsilon=.01)
    if name=='Ridge':return Ridge(alpha=30.)
    raise ValueError(name)


def corrected_v(x,y):
    a,_=pd.factorize(x);b,_=pd.factorize(y)
    r=int(a.max()+1);c=int(b.max()+1);n=len(a)
    if min(r,c)<2:return 0.
    table=np.bincount(a*c+b,minlength=r*c).reshape(r,c).astype(float)
    expected=table.sum(axis=1,keepdims=True)*table.sum(axis=0,keepdims=True)/n
    chi=np.sum((table-expected)**2/expected)
    phi=max(0.,chi/n-(r-1)*(c-1)/(n-1))
    den=min(r-1-(r-1)**2/(n-1),c-1-(c-1)**2/(n-1))
    return float(np.sqrt(phi/den)) if den>0 else 0.


def field_importances(a,b,yt,yv,seed):
    names=a.columns.tolist();cat=a.select_dtypes(exclude=np.number).columns.tolist()
    matrix=[];discrete=[]
    for c in names:
        if c in cat:
            matrix.append(pd.factorize(a[c],sort=True)[0]);discrete.append(True)
        else:
            matrix.append(a[c].to_numpy());discrete.append(a[c].nunique()<=15)
    mi=mutual_info_regression(np.array(matrix,dtype=float).T,yt,
                              discrete_features=discrete,n_neighbors=3,random_state=seed)
    enc=encoding(a);train=enc.fit_transform(a);valid=enc.transform(b)
    forest=RandomForestRegressor(n_estimators=120,min_samples_leaf=3,max_features=.8,n_jobs=4,random_state=seed)
    forest.fit(train,yt)
    groups={};offset=0
    for c in enc.transformers_[0][2]:groups[c]=[offset];offset+=1
    onehot=enc.named_transformers_['category']
    for c,levels in zip(enc.transformers_[1][2],onehot.categories_):
        groups[c]=list(range(offset,offset+len(levels)));offset+=len(levels)
    mdi=np.array([forest.feature_importances_[groups[c]].sum() for c in names])
    ridge=Ridge(alpha=30.).fit(train,yt)
    base=float(np.mean((yv-ridge.predict(valid))**2))
    rng=np.random.default_rng(seed);perm=[];perm_sd=[]
    # Permute the entire engineered field, retaining all its one-hot levels together.
    for c in names:
        scores=[]
        for _ in range(2):
            z=valid.copy();order=rng.permutation(len(z));z[:,groups[c]]=valid[order][:,groups[c]]
            scores.append(float(np.mean((yv-ridge.predict(z))**2))-base)
        perm.append(np.mean(scores));perm_sd.append(np.std(scores,ddof=1))
    out=pd.DataFrame({'field':names,'mi':mi,'forest_importance':mdi,
                       'permutation_mse_increase':perm,'permutation_sd':perm_sd}).set_index('field')
    for c in ['mi','forest_importance','permutation_mse_increase']:
        positive=out[c].clip(lower=0)
        out[c+'_rank_score']=positive.rank(pct=True,method='average').where(positive>0,0.)
    out['score']=out[[c+'_rank_score' for c in ['mi','forest_importance','permutation_mse_increase']]].mean(axis=1)
    out['inner_top40']=out.score.rank(ascending=False,method='first').le(40).astype(float)
    return out


class SelectedFeatures(TransformerMixin,BaseEstimator):
    def __init__(self,scheme='F5_log_ordinal',k=60,near_constant=.99,compact=False,seed=42):
        self.scheme=scheme;self.k=k;self.near_constant=near_constant;self.compact=compact;self.seed=seed

    def fit(self,X,y):
        if 'Id' in X or 'SalePrice' in X:raise ValueError('Exclude Id and SalePrice before fitting')
        y=np.asarray(y);parts=[]
        for fold,(tr,v) in enumerate(KFold(3,shuffle=True,random_state=314).split(X),1):
            eng=Engineering(self.scheme).fit(X.iloc[tr]);fill=FrameImputer().fit(eng.transform(X.iloc[tr]))
            a=fill.transform(eng.transform(X.iloc[tr]));b=fill.transform(eng.transform(X.iloc[v]))
            p=field_importances(a,b,y[tr],y[v],self.seed+fold);p['inner_fold']=fold;parts.append(p)
        self.inner_importances_=pd.concat(parts).reset_index()
        self.engineer_=Engineering(self.scheme).fit(X)
        self.imputer_=FrameImputer().fit(self.engineer_.transform(X))
        frame=self.imputer_.transform(self.engineer_.transform(X))
        joined=self.inner_importances_.groupby('field').mean(numeric_only=True).reindex(frame.columns).fillna(0.)
        joined=joined.rename(columns={'inner_top40':'inner_top40_frequency'})
        self.statistics_=joined
        self.statistics_['dominant_share']=[frame[c].value_counts(normalize=True,dropna=False).iloc[0] for c in frame]
        self.statistics_['unique_values']=[frame[c].nunique(dropna=False) for c in frame]
        self.statistics_['dtype']=['numeric' if pd.api.types.is_numeric_dtype(frame[c]) else 'categorical' for c in frame]
        self.pairs_=[];nums=frame.select_dtypes(include=np.number).columns.tolist()
        corr=frame[nums].corr()
        special={frozenset(p) for p in [('GarageCars','GarageArea'),('GrLivArea','TotRmsAbvGrd'),
                 ('YearBuilt','HouseAge'),('QualityArea','QualityTotalSF')]}
        for i,a in enumerate(nums):
            for b in nums[i+1:]:
                value=corr.loc[a,b]
                if abs(value)>=.8 or frozenset([a,b]) in special:
                    self.pairs_.append({'field_a':a,'field_b':b,'metric':'Pearson','association':float(value)})
        cats=frame.select_dtypes(exclude=np.number).columns.tolist()
        for i,a in enumerate(cats):
            for b in cats[i+1:]:
                value=corrected_v(frame[a],frame[b])
                if value>=.8:self.pairs_.append({'field_a':a,'field_b':b,'metric':'corrected_Cramers_V','association':value})
        self.redundancy_={}
        for p in self.pairs_:
            if abs(p['association'])>=.95:
                self.redundancy_.setdefault(p['field_a'],set()).add(p['field_b'])
                self.redundancy_.setdefault(p['field_b'],set()).add(p['field_a'])
        self.selected_fields_,self.decisions_=self.choose(self.k,self.near_constant,self.compact)
        return self

    def choose(self,k,threshold,compact=False):
        s=self.statistics_.copy()
        s['selection_score']=.7*s.score+.3*s.inner_top40_frequency if compact else s.score
        ranked=s.sort_values(['selection_score','mi'],ascending=False,kind='stable')
        selected=[];reasons={}
        for c,row in ranked.iterrows():
            if row.unique_values<=1:reasons[c]='constant';continue
            rescue=(row.mi_rank_score>=.75 or row.forest_importance_rank_score>=.75 or
                    row.permutation_mse_increase>2*row.permutation_sd)
            if row.dominant_share>=threshold and not rescue:reasons[c]='near_constant_weak_signal';continue
            conflicts=[p for p in selected if p in self.redundancy_.get(c,set())]
            if conflicts:reasons[c]='redundant_with:'+conflicts[0];continue
            if len(selected)>=k:reasons[c]='outside_selected_budget';continue
            selected.append(c);reasons[c]='retained_rare_signal' if row.dominant_share>=threshold else 'retained_rank_and_nonredundancy'
        return selected,reasons

    def transform(self,X):
        frame=self.imputer_.transform(self.engineer_.transform(X))
        return frame[self.selected_fields_].copy()


def make_delivery_pipeline(config,estimator=None):
    """Fit on RAW house predictors and log(SalePrice), inside the teammate's CV."""
    from sklearn.compose import make_column_selector
    preprocess=ColumnTransformer([('numeric',StandardScaler(),make_column_selector(dtype_include=np.number)),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),make_column_selector(dtype_exclude=np.number))],sparse_threshold=0)
    steps=[('selected_features',SelectedFeatures(**config))]
    if estimator is not None:steps.extend([('encode',preprocess),('model',clone(estimator))])
    return Pipeline(steps)
