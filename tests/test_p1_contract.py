import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from src.pipeline import RAW_COLUMNS, clean_transactions, integrate_external_sources
from src.data_quality import quality_dimensions, missing_id_comparison, invoice_timestamp_audit, describe_numeric
from src.classification_evaluation import (temporal_datasets, select_model, choose_threshold,
                                            evaluate_probabilities, FEATURES, split_xy)


def raw_rows(rows):
    return pd.DataFrame([[inv,'ITEM','Product',qty,pd.Timestamp(ts),price,cid,country]
                         for inv,ts,cid,qty,price,country in rows],columns=RAW_COLUMNS)


def test_geo_log_counts_and_unknown_fallback():
    raw = raw_rows([('A','2011-01-01',1,1,2,'Unspecified'),('B','2011-01-02',1,1,2,'France'),
                    ('Q','2011-01-03',2,1,2,'Unspecified'),('D','2011-01-04',2,1,2,'European Community'),
                    ('E','2011-01-05',3,1,2,'Channel Islands')])
    cleaned,log = clean_transactions(raw)
    assert log.set_index('Step').loc['1.5','Rows']==1
    assert log.set_index('Step').loc['1.5a','Rows']==2
    merged = integrate_external_sources(cleaned).set_index('invoice')
    assert merged.loc['A','country_original']=='Unspecified'
    assert merged.loc['A','region']=='Western_Europe'
    assert merged.loc['Q','region']==merged.loc['D','region']=='Unknown'
    assert merged.loc['E','region']=='Other_International'
    assert len(merged)==len(raw)


def test_missing_id_comparison_denominator():
    raw=raw_rows([('A','2011-01-01',1,2,5,'France'),('B','2011-01-01',None,3,10,'France'),
                  ('B','2011-01-01',None,3,10,'France'),('C1','2011-01-01',None,-1,10,'France')])
    summary,monthly,country = missing_id_comparison(raw)
    assert summary.loc['Missing_ID','all_lines']==2
    assert summary.loc['Missing_ID','gross_value_gbp']==30
    assert summary.loc['Missing_ID','gross_value_gbp_share']==.75
    assert monthly.iloc[0]['missing_share']==.5
    assert country.iloc[0]['total']==2
    quality=quality_dimensions(raw)
    assert len(quality)==6 and quality['Score_1_5'].between(1,5).all()


def test_invoice_crossing_and_shared_key():
    raw=raw_rows([('A','2011-03-01 00:01',1,1,2,'France'),('A','2011-02-28 23:59',1,1,2,'France'),
                  ('A','2011-02-28 23:55',2,1,2,'France')])
    cleaned,_=clean_transactions(raw)
    counts,multi,bounds=invoice_timestamp_audit(cleaned)
    assert counts['multi_timestamp_invoices']==1
    assert counts['invoice_codes_shared_by_customers']==1
    assert multi.iloc[0]['span_minutes']==2
    assert bounds.iloc[0]['crossing_invoices']==1


def test_descriptives():
    out=describe_numeric(pd.DataFrame({'x':[1,2,3,4],'y':[-1,0,1,0]}),['x','y'])
    assert out.loc['x','median']==2.5
    assert out.loc['x','IQR']==1.5
    assert np.isnan(out.loc['y','CV'])


def test_temporal_snapshots_and_boundaries():
    raw=raw_rows([('A','2011-01-01',1,1,2,'France'),('B','2011-04-01',1,1,2,'France'),
                  ('Q','2011-07-01',1,1,2,'France'),('D','2011-10-01',1,1,2,'France'),
                  ('E','2011-01-01',2,1,2,'France')])
    cleaned,_=clean_transactions(raw)
    snap=temporal_datasets(cleaned,'2011-12-09')
    assert [snap[r].set_index('customer_id').loc['1','frequency'] for r in snap]==[1,2,3]
    assert [snap[r].set_index('customer_id').loc['1','target_repeat_90d'] for r in snap]==[1,1,1]
    for d in snap.values():
        assert d.set_index('customer_id').loc['2','target_repeat_90d']==0
        assert list(split_xy(d)[0].columns)==FEATURES
    with pytest.raises(ValueError,match='chồng lấn'):
        temporal_datasets(cleaned,'2011-12-09',('2011-03-01','2011-04-01','2011-09-01'))
    with pytest.raises(ValueError,match='Thiếu cửa sổ'):
        temporal_datasets(cleaned,'2011-11-01')


def test_select_and_threshold():
    y=np.array([0,1]*30)
    X=pd.DataFrame({c:y*10+1 for c in FEATURES if c not in ['region','shipping_zone']})
    X['region']='Domestic_UK';X['shipping_zone']='Domestic'
    table,name,pipe=select_model(X,y,{'Dummy':DummyClassifier(strategy='most_frequent'),'Signal':LogisticRegression()})
    assert name=='Signal' and table.iloc[0]['roc_auc_mean']==1
    pipe.fit(X,y)
    p=pipe.predict_proba(X)[:,1]
    threshold,f1=choose_threshold(y,p)
    assert f1==1 and evaluate_probabilities(y,p,threshold)['f1']==1
    assert 'customer_id' not in pipe.feature_names_in_
