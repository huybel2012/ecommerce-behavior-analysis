"""W10: forecast gross GBP for known customers; fixed temporal protocol, no test tuning."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.pipeline import build_sklearn_preprocessor
from src.classification_evaluation import temporal_datasets, CUTOFFS

TARGET = 'target_spend_90d'
# Prespecified compact numeric subset of W11 features. No geography or ID;
# no simultaneous gross = frequency * AOV or near-redundant basket measures.
REG_FEATURES = {
    'log_num_cols': ['recency_days', 'observation_span_days', 'frequency',
                     'avg_order_value', 'unique_products'],
    'linear_num_cols': ['peak_month_ratio', 'cancel_rate'],
    'cat_cols': [],
}
FEATURES = sum(REG_FEATURES.values(), [])


def regression_datasets(df, observed_until, cutoffs=CUTOFFS):
    return temporal_datasets(df, observed_until, cutoffs, include_spend_target=True)


def split_xy(data):
    if data['customer_id'].duplicated().any():
        raise ValueError('One row per customer per cutoff required.')
    y = data[TARGET].to_numpy(dtype=float)
    if len(y) < 2 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Need at least two finite nonnegative targets.')
    return data[FEATURES].copy(), y


def candidate_models():
    models = {'Dummy median': DummyRegressor(strategy='median'),
              'Dummy mean': DummyRegressor(strategy='mean'), 'OLS': LinearRegression()}
    for alpha in (.01, .1, 1., 10., 100., 1000.):
        models[f'Ridge alpha={alpha:g}'] = Ridge(alpha=alpha)
    for alpha in (.1, 1., 10., 100., 1000.):
        models[f'Lasso alpha={alpha:g}'] = Lasso(alpha=alpha, max_iter=50000, tol=1e-5)
    return models


def make_pipeline(model):
    return Pipeline([('prep', build_sklearn_preprocessor(**REG_FEATURES)), ('reg', clone(model))])


def predict_gbp(model, X):
    """Project negative predictions to zero; same rule in every evaluation."""
    pred = np.asarray(model.predict(X), dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite predictions.')
    return np.maximum(pred, 0.)


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if y.ndim != 1 or y.shape != pred.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Invalid metric inputs.')
    return {'mae_gbp': mean_absolute_error(y, pred), 'rmse_gbp': float(np.sqrt(mean_squared_error(y, pred))),
            'r2': r2_score(y, pred) if len(y) >= 2 and np.ptp(y) > 0 else np.nan}


def neg_mae(model, X, y):
    return -mean_absolute_error(y, predict_gbp(model, X))


def neg_rmse(model, X, y):
    return -float(np.sqrt(mean_squared_error(y, predict_gbp(model, X))))


def select_model(X_train, y_train, models=None):
    """Lock winner by train CV MAE; tie by RMSE then name. Preprocessing inside folds."""
    if len(y_train) < 4:
        raise ValueError('Need at least four train customers for CV.')
    models = candidate_models() if models is None else models
    folds = min(5, len(y_train)//2)
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        scores = cross_validate(make_pipeline(model), X_train, y_train, cv=cv,
                                scoring={'mae': neg_mae, 'rmse': neg_rmse}, n_jobs=1, error_score='raise')
        rows.append({'model': name, 'cv_mae_gbp': -scores['test_mae'].mean(),
                     'cv_mae_sd': scores['test_mae'].std(ddof=1),
                     'cv_rmse_gbp': -scores['test_rmse'].mean(), 'folds': folds})
    table = pd.DataFrame(rows).sort_values(['cv_mae_gbp', 'cv_rmse_gbp', 'model']).reset_index(drop=True)
    return table, table.loc[0, 'model']


def fit_candidates(X_train, y_train, cv_table):
    """Fit one CV-best alpha per family + OLS + both baselines; all fixed before test."""
    names = cv_table['model'].tolist()
    keep = [name for name in names if name in ('Dummy median', 'Dummy mean', 'OLS')]
    keep += [next(name for name in names if name.startswith(family)) for family in ('Ridge ', 'Lasso ')]
    models = candidate_models()
    return {name: make_pipeline(models[name]).fit(X_train, y_train) for name in keep}


def evaluate_models(models, X, y):
    rows = []
    for name, pipe in models.items():
        raw = pipe.predict(X)
        pred = predict_gbp(pipe, X)
        rows.append({'model': name, 'n': len(y), **metrics(y, pred),
                     'negative_raw_fraction': np.mean(raw < 0), 'mean_error_gbp': np.mean(y-pred)})
    return pd.DataFrame(rows).set_index('model')


def ols_diagnostics(pipe, X, y):
    """Unclipped OLS train fit: SVD/pseudoinverse diagnostics, HC3 95% intervals.

    If rank deficient or saturated, report no individual coefficient intervals;
    HC3 addresses unequal variance, not dependence/model misspecification.
    """
    z = np.asarray(pipe.named_steps['prep'].transform(X), dtype=float)
    names = ['Intercept'] + pipe.named_steps['prep'].get_feature_names_out().tolist()
    a = np.column_stack([np.ones(len(z)), z])
    n, p = a.shape
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    rank = int((singular > np.finfo(float).eps * max(a.shape) * singular[0]).sum())
    pinv = (vt[:rank].T / singular[:rank]) @ u[:, :rank].T
    beta = np.r_[pipe.named_steps['reg'].intercept_, pipe.named_steps['reg'].coef_]
    fitted = np.asarray(pipe.predict(X))
    residual = np.asarray(y)-fitted
    h = np.clip(np.sum(u[:, :rank]**2, axis=1), 0., 1.)
    df = n-rank
    mse = np.sum(residual**2)/df if df > 0 else np.nan
    cook = np.full(n, np.nan)
    ok = (1-h) > 1e-10
    if df > 0 and mse > 0:
        cook[ok] = residual[ok]**2 * h[ok] / (rank*mse*(1-h[ok])**2)
    se = np.full(p, np.nan)
    if rank == p and df > 0 and ok.all():
        # diagonal of (X'X)^-1 X' diag(e²/(1-h)²) X (X'X)^-1
        se = np.sqrt(np.sum((pinv * (residual/(1-h)))**2, axis=1))
    margin = stats.t.ppf(.975, df)*se if df > 0 else se
    coef = pd.DataFrame({'feature': names, 'coefficient_gbp': beta, 'hc3_se': se,
                         'ci95_low': beta-margin, 'ci95_high': beta+margin})
    vif = []
    for j, name in enumerate(names[1:]):
        col = z[:, j]
        if np.ptp(col) < 1e-12:
            value = np.nan  # constant predictor, undefined VIF
        else:
            other = np.column_stack([np.ones(n), np.delete(z, j, axis=1)])
            err = col - other @ np.linalg.lstsq(other, col, rcond=None)[0]
            ratio = np.sum(err**2)/np.sum((col-col.mean())**2)
            value = 1/ratio if ratio > 1e-12 else np.inf
        vif.append({'feature': name, 'vif': value})
    standardized = np.full(n, np.nan)
    if mse > 0:
        standardized[ok] = residual[ok]/np.sqrt(mse*(1-h[ok]))
    points = pd.DataFrame({'fitted_gbp': fitted, 'residual_gbp': residual, 'leverage': h,
                           'standardized_residual': standardized, 'cooks_distance': cook})
    corr = stats.spearmanr(fitted, np.abs(residual)).statistic if np.ptp(fitted)>0 and np.ptp(np.abs(residual))>0 else np.nan
    # Descriptive heteroscedasticity and normality tests, not selection rules.
    e2 = residual**2
    bp_p = np.nan
    if rank > 1 and df > 0 and np.ptp(e2) > 0:
        aux_residual = e2 - u[:, :rank] @ (u[:, :rank].T @ e2)
        aux_r2 = np.clip(1 - np.sum(aux_residual**2)/np.sum((e2-e2.mean())**2), 0., 1.)
        bp_p = stats.chi2.sf(n*aux_r2, rank-1)  # studentized Breusch-Pagan/Koenker
    jb_p = stats.jarque_bera(residual).pvalue if n >= 8 and np.ptp(residual)>0 else np.nan
    summary = {'n': n, 'rank': rank, 'parameters': p, 'residual_df': df,
               'rank_deficient': rank < p, 'hc3_intervals_available': np.isfinite(se).all(),
               'residual_mean_gbp': residual.mean(), 'residual_skew': stats.skew(residual) if np.ptp(residual)>0 else np.nan,
               'spearman_abs_residual_fitted': corr, 'koenker_bp_p': bp_p, 'jarque_bera_p': jb_p, 'cook_above_4_over_n': int(np.sum(cook > 4/n)),
               'leverage_above_2rank_over_n': int(np.sum(h > 2*rank/n)),
               'abs_standardized_residual_above_3': int(np.sum(np.abs(standardized)>3))}
    return coef, pd.DataFrame(vif), points, summary
