import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from src.pipeline import RAW_COLUMNS, clean_transactions, unknown_country_mask, missing_description_mask
from src.utils import audit, check_disguised_missing
from src.classification_features import build_cutoff_customer_dataset
from src.regression_evaluation import (TARGET, FEATURES, regression_datasets, split_xy, select_model,
    fit_candidates, make_pipeline, predict_gbp, metrics, ols_diagnostics, evaluate_models)
from src.robustness import sensitivity_analysis


def raw_fixture():
    rng = np.random.default_rng(17)
    rows = []
    for i in range(1,61):
        for k, month in enumerate(['2010-11-01','2011-01-05','2011-04-01','2011-07-01','2011-10-01']):
            if k == 0 or rng.random() < .68:
                when = pd.Timestamp(month)+pd.Timedelta(days=int(rng.integers(0,20)))
                rows.append([f'{i}-{k}', f'ITEM{i%7}', 'Item', int(rng.integers(1,20)), when,
                             float(rng.integers(1,12)), i, 'United Kingdom' if i%3 else 'France'])
                if k==0 and i%4==0:
                    rows.append([f'C{i}-{k}', f'ITEM{i%7}', 'Item', -1, when+pd.Timedelta(days=1), 2.,i,'France'])
    # coverage row (cancel-only)
    rows.append(['C-end','ITEM','Item',-1,pd.Timestamp('2011-12-09'),1.,999,'France'])
    return pd.DataFrame(rows,columns=RAW_COLUMNS)


def test_audit_does_not_expose_samples():
    frame = pd.DataFrame({'customer_id':['private_id'], 'invoice':['private_invoice']})
    report = audit(frame)
    assert 'sample' not in report and 'private_' not in report.to_string()


def test_disguised_missing_consistent_with_pipeline():
    values = pd.Series(['European Community',' NULL ','UNKNOWN','check','damaged','',None,'France'])
    frame = pd.DataFrame({'Country':values,'description':values,'country_original':values})
    counts = check_disguised_missing(frame).set_index('column')['n_disguised']
    assert counts['Country'] == counts['country_original'] == unknown_country_mask(values.dropna()).sum()
    assert counts['description'] == missing_description_mask(values.dropna()).sum()
    assert check_disguised_missing(pd.DataFrame({'Country':['France',None]})).empty


def test_sensitivity_counts_cancel_only_consistently():
    raw = raw_fixture()
    clean,_ = clean_transactions(raw)
    result = sensitivity_analysis(raw)
    buys = clean.loc[~clean.is_cancelled].groupby('customer_id').total_amount.sum()
    excluded = buys[buys > buys.quantile(.99)].index
    assert result.loc['baseline','clean_lines'] == len(clean)
    assert result.loc['exclude_above_p99','clean_lines'] == len(clean[~clean.customer_id.isin(excluded)])
    assert '999' not in excluded


def test_spend_boundaries_cancellations_and_old_invoice():
    rows = [['A','ITEM','X',1,'2011-08-31 23:59:00',2.,1,'United Kingdom'],
            ['A','ITEM','X',5,'2011-09-01 00:01:00',2.,1,'United Kingdom'],
            ['B','ITEM','X',3,'2011-09-01 00:00:00',4.,1,'France'],
            ['B','ITEM2','X',2,'2011-09-01 00:02:00',5.,1,'France'],
            ['C1','ITEM','X',-1,'2011-09-02 00:00:00',4.,1,'France'],
            ['D','ITEM','X',5,'2011-11-30 00:00:00',1.,1,'France'],
            ['E','ITEM','X',1,'2011-08-01 00:00:00',2.,2,'France'],
            ['F','ITEM','X',4,'2011-09-02 00:00:00',2.,3,'France']]
    raw=pd.DataFrame(rows,columns=RAW_COLUMNS);clean,_=clean_transactions(raw)
    plain=build_cutoff_customer_dataset(clean, observed_until='2011-12-09')
    enriched=build_cutoff_customer_dataset(clean, observed_until='2011-12-09',include_spend_target=True)
    pd.testing.assert_frame_equal(plain,enriched.drop(columns=TARGET))
    assert enriched.set_index('customer_id')[TARGET].to_dict()=={'1':22.,'2':0.}
    assert enriched.set_index('customer_id').loc['1','gross_valid_revenue']==2.


def test_temporal_regression_contract():
    raw=raw_fixture(); clean,_=clean_transactions(raw)
    snapshots=regression_datasets(clean, '2011-12-09')
    for data in snapshots.values():
        assert data[TARGET].gt(0).eq(data.target_repeat_90d.eq(1)).all()
        X,y=split_xy(data)
        assert list(X.columns)==FEATURES and (y>=0).all()
        assert len(X)==60
    with pytest.raises(ValueError):
        regression_datasets(clean,'2011-12-09',('2011-05-01','2011-06-01','2011-09-01'))
    with pytest.raises(ValueError):
        regression_datasets(clean,'2011-11-29')


def test_cv_and_metrics_and_diagnostics():
    clean,_=clean_transactions(raw_fixture()); snaps=regression_datasets(clean,'2011-12-09')
    X,y=split_xy(snaps['train']); cv,winner=select_model(X,y); models=fit_candidates(X,y,cv)
    assert len(models)==5 and winner in models and len(cv)==14
    assert cv.cv_mae_gbp.is_monotonic_increasing
    test_X,test_y=split_xy(snaps['test']); values=evaluate_models(models,test_X,test_y)
    assert (values.mae_gbp>=0).all() and (values.rmse_gbp>=values.mae_gbp-1e-8).all()
    assert (predict_gbp(models[winner],test_X)>=0).all()
    coef,vif,points,summary=ols_diagnostics(models['OLS'],X,y)
    assert len(coef)==len(FEATURES)+1 and len(vif)==len(FEATURES) and len(points)==len(y)
    assert points.leverage.between(0,1).all()
    assert points.leverage.sum()==pytest.approx(summary['rank'])
    assert np.isnan(metrics([0,0],[1,2])['r2'])
    assert metrics([0,2],[1,1])['mae_gbp']==1


def test_ols_hc3_matches_direct_formula_full_rank():
    rng=np.random.default_rng(21); n=120
    X=pd.DataFrame({c:rng.uniform(.01,5,n) for c in FEATURES})
    y=10+X.frequency.to_numpy()*3+rng.normal(0,2,n)
    model=make_pipeline(LinearRegression()).fit(X,y)
    coef,_,points,summary=ols_diagnostics(model,X,y)
    assert summary['hc3_intervals_available']
    z=model.named_steps['prep'].transform(X); a=np.column_stack([np.ones(n),z]); b=np.linalg.inv(a.T@a)
    assert coef.coefficient_gbp.to_numpy()==pytest.approx(b@a.T@y)
    weighted=a*(points.residual_gbp/(1-points.leverage)).to_numpy()[:,None]
    var=b@weighted.T@weighted@b
    assert coef.hc3_se.to_numpy()==pytest.approx(np.sqrt(np.diag(var)))
    X['frequency']=1.;model=make_pipeline(LinearRegression()).fit(X,y)
    coef,_,_,summary=ols_diagnostics(model,X,y)
    assert summary['rank_deficient'] and coef.ci95_low.isna().all()
