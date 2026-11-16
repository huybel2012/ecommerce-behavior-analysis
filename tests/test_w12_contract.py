import numpy as np
import pandas as pd
import pytest
from src.fairness_evaluation import (market_groups, classification_by_group, regression_by_group,
    privacy_summary, country_selection, signed_gap, update_card)


def test_historical_market_groups():
    assert market_groups(pd.DataFrame({'region':['Domestic_UK','Other_International','Unknown',None]})).tolist()==['UK','International','Unknown','Unknown']


def test_classification_support_and_gap():
    y = np.tile([0,1], 40)
    g = ['UK']*40 + ['International']*38 + ['Unknown']*2
    p = np.r_[y[:40]*.8+.1, 1-y[40:]*.8-.1]
    t = classification_by_group(y,p,g,.5)
    assert t.loc['UK','recall']==1 and t.loc['International','recall']==0
    assert t.loc['International','fpr']==1 and signed_gap(t,'recall')==-1
    assert t.loc['Unknown','n']==2 and np.isnan(t.loc['Unknown','roc_auc'])
    assert t.loc['Overall','n']==sum(t.loc[x,'n'] for x in ['UK','International','Unknown'])


def test_classification_degenerate_denominator():
    t=classification_by_group(np.zeros(40),np.zeros(40),['UK']*40,.5)
    assert t.loc['UK','accuracy']==1 and t.loc['UK','fpr']==0
    assert np.isnan(t.loc['UK','recall']) and np.isnan(t.loc['UK','precision']) and np.isnan(t.loc['UK','roc_auc'])


def test_regression_underprediction_and_constant_target():
    t=regression_by_group(np.ones(40)*15,np.ones(40)*5,['UK']*40)
    assert t.loc['UK','mae_gbp']==t.loc['UK','rmse_gbp']==t.loc['UK','mean_error_gbp']==10
    assert np.isnan(t.loc['UK','r2']) and np.isnan(signed_gap(t,'mae_gbp'))


def test_privacy_k_after_suppression_no_microdata():
    d=pd.DataFrame({'customer_id':range(13),'region':['Domestic_UK']*12+['Unknown'],
                    'recency_days':range(13),'frequency':[1]*13,'gross_valid_revenue':[50]*13})
    t=privacy_summary(d)
    assert t.loc['exact','k_min']==1 and t.loc['generalized','k_min']==1
    assert t.loc['generalized_suppressed','customers']==12 and t.loc['generalized_suppressed','k_min']==12
    assert t.loc['generalized_suppressed','suppressed']==1 and 'customer_id' not in t.columns
    with pytest.raises(ValueError): privacy_summary(pd.concat([d,d]))


def test_privacy_all_suppressed():
    d=pd.DataFrame({'customer_id':[1,2],'region':['Domestic_UK','Unknown'],'recency_days':[1,1], 'frequency':[1,1], 'gross_valid_revenue':[10,10]})
    t=privacy_summary(d)
    assert t.loc['generalized_suppressed','customers']==0 and np.isnan(t.loc['generalized_suppressed','k_min'])


def test_country_selection_unknown_and_small_lines():
    t=country_selection(pd.DataFrame({'Known_ID':[1,12,0],'Missing_ID':[4,0,20]},index=pd.Index(['NULL','Unspecified','HK'],name='Country')))
    assert t.loc['Unknown','total']==17 and t.loc['Unknown','missing_share']==pytest.approx(4/17)
    assert t.loc['HK','missing_share']==1


def test_reject_invalid_inputs():
    with pytest.raises(ValueError): classification_by_group([0,2],[.1,.9],['UK','UK'],.5)
    with pytest.raises(ValueError): regression_by_group([1],[-1],['UK'])
    with pytest.raises(ValueError): regression_by_group([1],[1],['bad'])


def test_card_update_idempotent(tmp_path):
    p=tmp_path/'model.md';p.write_text('start\n<!-- W12_RESULTS_START -->\nold\n<!-- W12_RESULTS_END -->\nend',encoding='utf8')
    t=pd.DataFrame({'n':[50],'rate':[np.nan]}, index=['Unknown'])
    update_card(p,t,['n','rate'],'note'); first=p.read_text(encoding='utf8')
    update_card(p,t,['n','rate'],'note')
    assert p.read_text(encoding='utf8')==first and 'N/A' in first and 'old' not in first
