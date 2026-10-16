"""Run reproducible feature/model comparisons, then evaluate the locked winner once.

Usage: python3 run_experiments.py
Full details and Chinese explanations are written to report.html and README.md.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dsai4204-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import hashlib
import html
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from feature_engineering import HouseFeatures, SCHEMES, make_pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
MODELS = ROOT / 'models'
SEED = 42


def metrics(y, pred):
    price, price_pred = np.exp(y), np.exp(pred)
    return {'log_rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'price_rmse_usd': float(np.sqrt(mean_squared_error(price, price_pred))),
            'price_mae_usd': float(mean_absolute_error(price, price_pred)),
            'price_r2': float(r2_score(price, price_pred))}


def models():
    return {
        'Ridge': Ridge(alpha=30.),
        'ElasticNet': ElasticNet(alpha=.001, l1_ratio=.15, max_iter=20000, tol=1e-5),
        'SVR': SVR(C=20., epsilon=.02, gamma='scale'),
        'RandomForest': RandomForestRegressor(n_estimators=220, max_features=.8, min_samples_leaf=2,
                                               n_jobs=2, random_state=SEED),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=450, learning_rate=.035,
                                                      max_depth=2, min_samples_leaf=8,
                                                      loss='huber', random_state=SEED),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=350, learning_rate=.045,
                                                            max_leaf_nodes=15, l2_regularization=5.,
                                                            min_samples_leaf=15, early_stopping=False,
                                                            random_state=SEED),
    }


def tuning_options(family):
    return {
        'Ridge': [{'alpha': a} for a in [3., 10., 60., 100.]],
        'ElasticNet': [{'alpha': a, 'l1_ratio': r} for a, r in
                       [(.0003, .15), (.003, .15), (.001, .5), (.003, .5)]],
        'SVR': [{'C': c, 'epsilon': e} for c, e in [(5., .02), (60., .02), (20., .01), (20., .04)]],
        'RandomForest': [{'min_samples_leaf': leaf, 'max_features': f}
                         for leaf, f in [(1, .6), (1, 1.), (3, .6), (3, 1.)]],
        'GradientBoosting': [{'n_estimators': n, 'max_depth': d, 'learning_rate': .035}
                             for n, d in [(700, 2), (1000, 2), (450, 3), (700, 3)]],
        'HistGradientBoosting': [{'max_iter': n, 'max_leaf_nodes': leaves, 'l2_regularization': reg}
                                 for n, leaves, reg in [(550, 15, 5.), (350, 7, 5.),
                                                        (550, 7, 10.), (350, 25, 10.)]],
    }[family]


def main():
    start = time.perf_counter()
    for path in [RESULTS, FIGURES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)
    opts = dict(keep_default_na=False, na_values=['NA', ''])
    train, test = (pd.read_csv(DATA / name, **opts) for name in ['train.csv', 'test.csv'])
    X, y = train.drop(columns=['SalePrice', 'Id']), np.log(train['SalePrice'].to_numpy())
    assert X.shape == (1460, 79) and len(test) == 1459
    development, holdout = train_test_split(np.arange(len(train)), test_size=.2, random_state=SEED)
    Xdev, ydev = X.iloc[development], y[development]
    cv = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(Xdev))
    fold_assignment = np.empty(len(Xdev), dtype=int)
    for fold, (_, val) in enumerate(cv, 1):
        fold_assignment[val] = fold
    split_table = pd.DataFrame({'Id': train['Id'], 'split': 'holdout', 'cv_fold': 0})
    split_table.loc[development, 'split'] = 'development'
    split_table.loc[development, 'cv_fold'] = fold_assignment
    split_table.to_csv(RESULTS / 'data_splits.csv', index=False)

    registry, scores, fold_rows, oof = {}, [], [], {}
    base_models = models()

    def flush():
        table = pd.DataFrame(scores).sort_values('cv_log_rmse_mean')
        table.to_csv(RESULTS / 'cv_results.csv', index=False)
        pd.DataFrame(fold_rows).to_csv(RESULTS / 'cv_fold_metrics.csv', index=False)
        return table

    def evaluate(name, family, scheme, estimator, stage):
        tic = time.perf_counter()
        pred, fold_scores, widths = np.empty(len(Xdev)), [], []
        for fold, (tr, val) in enumerate(cv, 1):
            pipe = make_pipeline(scheme, clone(estimator))
            pipe.fit(Xdev.iloc[tr], ydev[tr])
            pred[val] = pipe.predict(Xdev.iloc[val])
            result = metrics(ydev[val], pred[val])
            fold_rows.append({'candidate': name, 'fold': fold, **result})
            fold_scores.append(result['log_rmse'])
            widths.append(pipe.named_steps['preprocess'].get_feature_names_out().size)
        registry[name] = {'family': family, 'scheme': scheme, 'estimator': clone(estimator)}
        oof[name] = pred
        scores.append({'candidate': name, 'stage': stage, 'model': family, 'features': scheme,
                       'cv_log_rmse_mean': np.mean(fold_scores),
                       'cv_log_rmse_std': np.std(fold_scores, ddof=1),
                       'oof_log_rmse': metrics(ydev, pred)['log_rmse'],
                       'oof_price_rmse_usd': metrics(ydev, pred)['price_rmse_usd'],
                       'oof_price_mae_usd': metrics(ydev, pred)['price_mae_usd'],
                       'mean_encoded_features': np.mean(widths),
                       'seconds': time.perf_counter() - tic,
                       'parameters': json.dumps(estimator.get_params(), ensure_ascii=False)})
        flush()
        print(f'{len(scores):02d} {name:46s} CV={np.mean(fold_scores):.6f} +/- {np.std(fold_scores, ddof=1):.6f} '
              f'({time.perf_counter()-tic:.1f}s)', flush=True)

    evaluate('Dummy__F0_baseline', 'Dummy', 'F0_baseline', DummyRegressor(strategy='mean'), 'reference')
    for scheme in SCHEMES:
        for family, estimator in base_models.items():
            evaluate(f'{family}__{scheme}', family, scheme, estimator, 'feature_model_grid')

    # Fixed budget: tune only the two families with the best development CV result.
    grid = pd.DataFrame(scores).query("stage == 'feature_model_grid'").sort_values('cv_log_rmse_mean')
    top_families = grid.drop_duplicates('model').head(2)
    for _, row in top_families.iterrows():
        family, scheme = row['model'], row['features']
        for i, params in enumerate(tuning_options(family), 1):
            estimator = clone(base_models[family]).set_params(**params)
            evaluate(f'{family}__{scheme}__tune{i}', family, scheme, estimator, 'tuning')

    # Two equal-weight log-price blends, using best CV candidate per model family.
    ranked = pd.DataFrame(scores).query("model != 'Dummy'").sort_values('cv_log_rmse_mean')
    per_family = ranked.drop_duplicates('model')
    blends = {}
    for count in [2, 3]:
        names = per_family.head(count)['candidate'].tolist()
        name = f'BlendTop{count}'
        blends[name] = names
        pred = np.mean([oof[n] for n in names], axis=0)
        oof[name] = pred
        f_scores = []
        for fold, (_, val) in enumerate(cv, 1):
            m = metrics(ydev[val], pred[val])
            fold_rows.append({'candidate': name, 'fold': fold, **m})
            f_scores.append(m['log_rmse'])
        m = metrics(ydev, pred)
        scores.append({'candidate': name, 'stage': 'blend', 'model': 'Blend', 'features': 'mixed',
                       'cv_log_rmse_mean': np.mean(f_scores), 'cv_log_rmse_std': np.std(f_scores, ddof=1),
                       'oof_log_rmse': m['log_rmse'], 'oof_price_rmse_usd': m['price_rmse_usd'],
                       'oof_price_mae_usd': m['price_mae_usd'], 'mean_encoded_features': np.nan,
                       'seconds': 0., 'parameters': json.dumps({'members': names, 'weights': [1/count]*count})})
        print(f'{name}: CV={np.mean(f_scores):.6f}; members={names}', flush=True)

    table = flush()
    winner = table.iloc[0]['candidate']
    # Winner is locked before any holdout prediction/metric is computed.
    (RESULTS / 'selection_lock.json').write_text(json.dumps({
        'selected_candidate': winner, 'selection_metric': 'mean 5-fold RMSE of natural-log prices',
        'selection_uses_holdout': False, 'blend_members': blends,
    }, ensure_ascii=False, indent=2))
    print(f'LOCKED WINNER: {winner}; now evaluating held-out observations.', flush=True)

    def fit_candidate(name, frame, target):
        names = blends.get(name, [name])
        fitted = []
        for member in names:
            spec = registry[member]
            pipe = make_pipeline(spec['scheme'], clone(spec['estimator']))
            pipe.fit(frame, target)
            fitted.append((member, pipe))
        return fitted

    def predict_fitted(fitted, frame):
        return np.mean([pipe.predict(frame) for _, pipe in fitted], axis=0)

    holdout_rows = []
    # Only these predeclared references and the selected winner are evaluated here.
    check_names = list(dict.fromkeys(['Dummy__F0_baseline', 'Ridge__F0_baseline', winner]))
    holdout_preds = {}
    for name in check_names:
        fitted = fit_candidate(name, Xdev, ydev)
        pred = predict_fitted(fitted, X.iloc[holdout])
        holdout_preds[name] = pred
        holdout_rows.append({'candidate': name, 'n': len(holdout), **metrics(y[holdout], pred)})
    holdout_table = pd.DataFrame(holdout_rows)
    holdout_table.to_csv(RESULTS / 'holdout_metrics.csv', index=False)
    selected_holdout = holdout_preds[winner]
    errors = pd.DataFrame({'Id': train.iloc[holdout]['Id'].to_numpy(),
                           'actual_price': np.exp(y[holdout]),
                           'predicted_price': np.exp(selected_holdout),
                           'actual_log_price': y[holdout], 'predicted_log_price': selected_holdout})
    errors['log_residual'] = errors['actual_log_price'] - errors['predicted_log_price']
    errors['absolute_error_usd'] = abs(errors['actual_price'] - errors['predicted_price'])
    errors['absolute_percentage_error'] = errors['absolute_error_usd'] / errors['actual_price'] * 100
    errors.to_csv(RESULTS / 'holdout_predictions.csv', index=False)
    errors.nlargest(25, 'absolute_error_usd').to_csv(RESULTS / 'largest_holdout_errors.csv', index=False)

    # Bootstrap quantifies holdout sampling uncertainty; it is not a tuning step.
    rng = np.random.default_rng(SEED)
    squared = (y[holdout] - selected_holdout) ** 2
    boot = np.sqrt(np.mean(squared[rng.integers(0, len(squared), (3000, len(squared)))], axis=1))
    ci = np.quantile(boot, [.025, .975]).tolist()
    holdout_table['selected'] = holdout_table['candidate'].eq(winner)

    oof_table = pd.DataFrame({'Id': train.iloc[development]['Id'].to_numpy(), 'cv_fold': fold_assignment,
                              'actual_log_price': ydev})
    for name, pred in oof.items():
        oof_table[name] = pred
    oof_table.to_csv(RESULTS / 'development_oof_predictions.csv', index=False)
    grid.pivot(index='features', columns='model', values='cv_log_rmse_mean').to_csv(
        RESULTS / 'feature_model_matrix.csv')
    effects = []
    for family in base_models:
        rows = grid.loc[grid['model'].eq(family)].set_index('features')
        previous = None
        for scheme in SCHEMES:
            current = rows.loc[scheme, 'cv_log_rmse_mean']
            effects.append({'model': family, 'features': scheme, 'cv_log_rmse': current,
                            'delta_vs_previous': np.nan if previous is None else current-previous,
                            'delta_vs_baseline': current-rows.loc['F0_baseline', 'cv_log_rmse_mean']})
            previous = current
    pd.DataFrame(effects).to_csv(RESULTS / 'feature_engineering_effects.csv', index=False)

    final_fitted = fit_candidate(winner, X, y)
    test_log_pred = predict_fitted(final_fitted, test.drop(columns='Id'))
    submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': np.exp(test_log_pred)})
    sample = pd.read_csv(DATA / 'sample_submission.csv')
    assert submission.columns.tolist() == sample.columns.tolist()
    assert submission['Id'].equals(sample['Id'])
    assert len(submission) == 1459 and np.isfinite(submission['SalePrice']).all()
    assert submission['SalePrice'].gt(0).all()
    submission.to_csv(RESULTS / 'submission_best.csv', index=False)
    joblib.dump({'candidate': winner, 'target_transform': 'natural log; inverse exp',
                 'models': final_fitted, 'weights': [1/len(final_fitted)]*len(final_fitted),
                 'sklearn_version': sklearn.__version__}, MODELS / 'best_model.joblib', compress=3)
    for name, pipe in final_fitted:
        engineered = pipe.named_steps['features'].transform(X)
        engineered.insert(0, 'Id', train['Id'])
        engineered.to_csv(RESULTS / f'engineered_train__{name}.csv', index=False)
        encoded_names = pipe.named_steps['preprocess'].get_feature_names_out()
        pd.Series(encoded_names, name='encoded_feature').to_csv(
            RESULTS / f'encoded_feature_names__{name}.csv', index=False)

    best_cv = table.iloc[0].to_dict()
    best_holdout = next(r for r in holdout_rows if r['candidate'] == winner)
    ridge_cv = table.loc[table['candidate'].eq('Ridge__F0_baseline'), 'cv_log_rmse_mean'].iloc[0]
    summary = {'best_candidate': winner, 'cv': best_cv, 'holdout': best_holdout,
               'holdout_log_rmse_bootstrap_95ci': ci,
               'cv_log_rmse_reduction_vs_baseline_ridge_pct': float(100*(ridge_cv-best_cv['cv_log_rmse_mean'])/ridge_cv),
               'development_rows': len(development), 'holdout_rows': len(holdout), 'training_rows': len(train),
               'test_rows': len(test), 'candidates_evaluated': len(table), 'folds': 5, 'random_state': SEED,
               'feature_schemes': SCHEMES,
               'final_model_refitted_on_all_training_rows': True,
               'kaggle_test_error_available': False,
               'limitations': ['CV scores are used for selection and may be optimistic.',
                              'The earlier EDA examined all training labels; holdout is isolated from this modeling search, not from prior EDA.',
                              'Holdout estimates performance of a model fit on 1168 rows, not the final refit on 1460 rows.',
                              'Cumulative feature differences are conditional on the fixed model settings; F5 combines log and ordinal changes.',
                              'No validation observations or large-area outliers are removed.',
                              'Best means lowest CV error among the tested candidates, not a global optimum.'],
               'source_sha256': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                                 for name in ['train.csv', 'test.csv', 'sample_submission.csv']},
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                            'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__,
                            'matplotlib': matplotlib.__version__},
               'elapsed_seconds': time.perf_counter()-start,
               'sources': ['https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview',
                           'https://scikit-learn.org/stable/modules/compose.html']}
    (RESULTS / 'best_result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    make_figures(table, grid, errors, ydev, oof[winner])
    write_report(summary, table, grid, holdout_table, pd.DataFrame(effects))
    print(json.dumps({'winner': winner, 'cv_log_rmse': best_cv['cv_log_rmse_mean'],
                      'holdout': best_holdout, 'elapsed_seconds': time.perf_counter()-start}, indent=2), flush=True)


def make_figures(table, grid, errors, ydev, oof):
    plt.rcParams.update({'figure.dpi': 140, 'axes.spines.top': False, 'axes.spines.right': False})
    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top['candidate'], top['cv_log_rmse_mean'], xerr=top['cv_log_rmse_std'], color='#276c85', alpha=.85)
    ax.set_xlabel('Mean 5-fold log RMSE (lower is better); bars: fold SD')
    ax.set_title('Top 15 development CV candidates')
    fig.tight_layout(); fig.savefig(FIGURES/'01_model_comparison.png'); plt.close(fig)

    matrix = grid.pivot(index='features', columns='model', values='cv_log_rmse_mean').reindex(SCHEMES)
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(matrix, cmap='YlGnBu_r', aspect='auto')
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f'{matrix.iloc[i,j]:.4f}', ha='center', va='center', color='black')
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=20, ha='right')
    ax.set_yticks(range(len(matrix)), matrix.index)
    ax.set_title('Feature engineering x fixed model: mean CV log RMSE')
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(FIGURES/'02_feature_model_matrix.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(errors['actual_log_price'], errors['predicted_log_price'], alpha=.6, s=18)
    lo, hi = errors['actual_log_price'].min(), errors['actual_log_price'].max()
    axes[0].plot([lo, hi], [lo, hi], '--', color='black')
    axes[0].set(xlabel='Actual log price', ylabel='Predicted log price', title='Locked winner: holdout predictions')
    axes[1].scatter(errors['predicted_log_price'], errors['log_residual'], alpha=.6, s=18)
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set(xlabel='Predicted log price', ylabel='Actual - predicted log price', title='Holdout residuals')
    fig.tight_layout(); fig.savefig(FIGURES/'03_holdout_diagnostics.png'); plt.close(fig)


def write_report(summary, table, grid, holdout, effects):
    best, h = summary['cv'], summary['holdout']
    per_model = grid.sort_values('cv_log_rmse_mean').drop_duplicates('model')
    columns = ['model', 'features', 'cv_log_rmse_mean', 'cv_log_rmse_std']
    best_effect = effects.loc[effects['features'].ne('F0_baseline')].sort_values('delta_vs_previous').iloc[0]
    css = '''body{font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#183343;background:#f4f7f8;margin:0}
    main{max-width:1140px;margin:40px auto;background:white;padding:40px;border-radius:16px}h1,h2{line-height:1.35}h2{margin-top:36px}
    table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow:auto}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}
    th{background:#e9f2f5}img{width:100%;margin:15px 0}.cards{display:flex;gap:16px;flex-wrap:wrap}.card{background:#e9f2f5;padding:18px;border-radius:9px;flex:1}
    .value{font-size:29px;font-weight:700}code{background:#edf2f4;padding:2px 5px}.note{border-left:4px solid #b77b21;padding:12px;background:#fff8eb}a{color:#126982}'''
    def fmt(frame):
        return frame.to_html(index=False, float_format=lambda x: f'{x:,.6f}', escape=True, border=0)
    schemes = pd.DataFrame({'方案': list(SCHEMES), '处理': list(SCHEMES.values())})
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>House Prices 特征工程与建模比较</title><style>{css}</style><main>
    <h1>House Prices：特征工程与建模比较</h1><p>依据现有 EDA 建立的可复现实验。训练数据 1,460 条，Kaggle 测试数据 1,459 条。所有价格单位均为美元。</p>
    <div class="cards"><div class="card">本次最优方案<div class="value" style="font-size:20px">{html.escape(summary['best_candidate'])}</div></div>
    <div class="card">五折 CV 平均 log RMSE<div class="value">{best['cv_log_rmse_mean']:.6f}</div>折间标准差 {best['cv_log_rmse_std']:.6f}</div>
    <div class="card">留出集 log RMSE<div class="value">{h['log_rmse']:.6f}</div>仅在方案锁定后评估</div></div>
    <h2>1. 如何比较，误差表示什么</h2><p>固定随机种子 42，将 1,168 条分为开发集，292 条留作最终检查。
    在开发集使用相同的五折划分比较 {summary['candidates_evaluated']} 个候选；按各折 log RMSE 的算术平均值选择方案。
    数值填补、社区临街长度中位数、缺失标记列、one-hot 词表和标准化参数都仅从各折训练数据学习。
    锁定方案后，只评估获胜方案和预先指定的 Dummy、原始特征 Ridge 两个参照，不再根据留出集调参。</p>
    <p><code>log RMSE = sqrt(mean((ln(真实房价) - ln(预测房价))²))</code>，越低越好，不能直接解释为百分比错误率。
    CV 平均值与汇总所有折外预测计算的 OOF RMSE 略有区别，排行榜使用前者。美元 RMSE/MAE 是对 <code>exp(预测对数)</code> 计算的辅助指标。</p>
    <p class="note">先前 EDA 已查看全部训练标签，因此该留出集隔离了本次建模搜索，但不是从未参与过任何分析的全新数据。
    选模使用的 CV 最低值可能乐观；留出集只有 292 条，也存在抽样波动。“最优”仅指本次测试的候选范围。</p>
    <h2>2. 特征工程方案</h2>{fmt(schemes)}<p>各方案累积增加处理；F5 同时改变偏态字段表示与质量编码，不能把其变化单独归因于其中一项。
    不使用 Id 预测房价，不进行目标编码，不根据整个训练集的 MI 排名筛选特征，不删除大面积离群点。
    只有在面积/容量明确为零时才将设施类别缺失填为 Absent；有设施但质量缺失仍保留为未知。
    明显异常年份标为未知，并保留标记；不会将 2207 猜改为某个年份。总面积为地下室＋一楼＋二楼，避免重复计入地上面积。</p>
    <h2>3. 固定参数下的模型与特征比较</h2>{fmt(per_model[columns])}<img src="figures/02_feature_model_matrix.png">
    <p>Ridge 与 ElasticNet 为正则化线性模型，SVR 使用 RBF 核，其余为随机森林与两类梯度提升模型。
    所有模型预测自然对数房价；相同模型比较不同特征时参数固定。</p>
    <p>固定模型的相邻步骤中，观察到最大改善出现在 <b>{html.escape(best_effect['model'])}</b> 的
    <b>{html.escape(best_effect['features'])}</b>，log RMSE 变化为 <b>{best_effect['delta_vs_previous']:.6f}</b>。
    这是该模型与前置处理条件下的结果，不代表所有模型都会同样改善。详见 feature_engineering_effects.csv。</p>
    <h2>4. 调参与融合后的排名</h2><p>先比较六种模型×六组特征，再对开发集表现最好的两个模型家族各尝试四组参数。
    最后尝试前两/前三个不同家族最优模型的等权对数预测融合（对应价格的几何平均），权重未通过留出集优化。</p>
    {fmt(table[['candidate','stage','cv_log_rmse_mean','cv_log_rmse_std','oof_price_rmse_usd','oof_price_mae_usd']].head(15))}
    <img src="figures/01_model_comparison.png"><p>最优方案相对原始特征 Ridge 的 CV log RMSE 降低
    <b>{summary['cv_log_rmse_reduction_vs_baseline_ridge_pct']:.2f}%</b>。这表示误差指标的相对下降，不是准确率提升。</p>
    <h2>5. 方案锁定后的留出集结果</h2>{fmt(holdout)}
    <p>最优方案留出集美元 RMSE：<b>${h['price_rmse_usd']:,.2f}</b>；美元 MAE：<b>${h['price_mae_usd']:,.2f}</b>；R²：<b>{h['price_r2']:.4f}</b>。
    通过对固定留出误差重采样 3,000 次，log RMSE 的近似 95% bootstrap 区间为
    <b>[{summary['holdout_log_rmse_bootstrap_95ci'][0]:.6f}, {summary['holdout_log_rmse_bootstrap_95ci'][1]:.6f}]</b>。
    该区间不包含重新划分、重新训练及选模的不确定性。</p><img src="figures/03_holdout_diagnostics.png">
    <p>留出集每套房屋的真实值、预测值、对数残差和美元误差见 holdout_predictions.csv；最大美元误差的 25 条记录见 largest_holdout_errors.csv。</p>
    <h2>6. 最终交付与复现</h2><p>最终模型在全部 1,460 条训练数据上重新拟合，保存在 models/best_model.joblib。
    <a href="results/submission_best.csv">submission_best.csv</a> 含 1,459 条测试集预测，列为 Id、SalePrice，可用于 Kaggle 提交。
    没有测试集真实房价，因此这里没有计算 Kaggle 测试误差，也没有自动上传提交。</p>
    <p>最终重训模型使用的数据量不同于留出评估模型，留出误差是泛化估计，不是最终重训模型的已知测试成绩。
    最终模型的工程特征样表和编码后字段名已导出；缺失值可能保留在工程样表中，后续由模型内的预处理器填补。</p>
    <p>运行 <code>python3 run_experiments.py</code> 重现实验；运行 <code>python3 predict.py --output results/submission_reproduced.csv</code> 从保存模型预测。
    原始文件 SHA-256、依赖版本、所有参数、折划分、折外预测和详细结果均已保存。总运行时间约 {summary['elapsed_seconds']/60:.1f} 分钟。</p>
    <p>指标依据：<a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview">Kaggle 官方比赛页</a>；
    折内预处理依据：<a href="https://scikit-learn.org/stable/modules/compose.html">scikit-learn Pipeline 官方文档</a>。</p>
    </main></html>'''
    (ROOT / 'report.html').write_text(body, encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
