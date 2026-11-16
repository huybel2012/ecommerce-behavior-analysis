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
    t = classification_by_group(y,p,g,.5, public=False)
    assert t.loc['UK','recall']==1 and t.loc['International','recall']==0
    assert t.loc['International','fpr']==1 and signed_gap(t,'recall')==-1
    assert t.loc['Unknown','n']==2 and np.isnan(t.loc['Unknown','roc_auc'])
    assert t.loc['Overall','n']==sum(t.loc[x,'n'] for x in ['UK','International','Unknown'])


def test_classification_degenerate_denominator():
    t=classification_by_group(np.zeros(40),np.zeros(40),['UK']*40,.5, public=False)
    assert t.loc['UK','accuracy']==1 and t.loc['UK','fpr']==0
    assert np.isnan(t.loc['UK','recall']) and np.isnan(t.loc['UK','precision']) and np.isnan(t.loc['UK','roc_auc'])


def test_regression_underprediction_and_constant_target():
    t=regression_by_group(np.ones(40)*15,np.ones(40)*5,['UK']*40, public=False)
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


# Regression tests for public release, keyed observations and locked provenance.
def test_public_partition_combines_unknown_no_separate_gap():
    y = np.tile([0,1], 45)
    g = ['UK']*40 + ['International']*44 + ['Unknown']*6
    p = y*.8+.1
    c = classification_by_group(y,p,g,.5)
    r = regression_by_group(y*10,p*10,g)
    for t in (c,r):
        assert list(t.index)==['Overall','UK','Non-UK / Unknown']
        assert t.loc['Non-UK / Unknown','n']==50
    assert c.loc['Non-UK / Unknown','recall']==1
    assert r.loc['Non-UK / Unknown','mae_gbp']==pytest.approx(1)


def test_public_complementary_suppression_hides_both_groups():
    y = np.tile([0,1], 25)
    g = ['UK']*48 + ['Unknown']*2
    for t in (classification_by_group(y,y,g,.5), regression_by_group(y,y,g)):
        assert t.loc['Overall','n']==50
        assert t.loc[['UK','Non-UK / Unknown']].drop(columns='status').isna().all().all()


def test_public_suppression_small_class_or_predicted_support():
    y = np.r_[np.tile([0,1],20),np.ones(40)]
    g = ['UK']*40+['International']*40
    t = classification_by_group(y,y,g,.5)
    assert t.loc[['UK','Non-UK / Unknown'],'accuracy'].isna().all()
    assert t.loc['Overall','accuracy']==1
    assert t.loc['UK','n']==40
    y = np.tile([0,1],40)
    p = np.r_[y[:40],np.zeros(40)]
    t = classification_by_group(y,p,g,.5)
    assert t.loc[['UK','Non-UK / Unknown'],'recall'].isna().all()


@pytest.mark.parametrize('kind', ['classification','regression'])
def test_keyed_vectors_reject_permutation_mixed_and_duplicate(kind):
    y = pd.Series(np.tile([0,1],40),index=range(100,180))
    p = y.astype(float)
    g = pd.Series(['UK']*40+['International']*40,index=y.index)
    def run(a,b,c):
        return classification_by_group(a,b,c,.5) if kind=='classification' else regression_by_group(a,b,c)
    assert run(y,p,g).loc['Overall','n']==80
    for bad in [(y,p,g.iloc[::-1]), (y,p.iloc[::-1],g), (y,p.to_numpy(),g),
                (y.set_axis([0]*80),p.set_axis([0]*80),g.set_axis([0]*80)),
                (y.set_axis([np.nan]+list(range(79))),p,g)]:
        with pytest.raises(ValueError, match='index|mixture'):
            run(*bad)


def test_public_representation_no_small_group_complement():
    from src.fairness_evaluation import public_representation
    snapshots = {'train': pd.DataFrame({'region':['Domestic_UK']*40+['Unknown']*2}),
                 'test': pd.DataFrame({'region':['Domestic_UK']*40+['Other_International']*40+['Unknown']*2})}
    t = public_representation(snapshots)
    assert t.train.isna().all()
    assert t.loc['Non-UK / Unknown','test']==42
    assert 'Unknown' not in t.index and 'International' not in t.index


def test_reference_metrics_fail_on_missing_extra_or_changed_value():
    from src.fairness_reference import metric_issues, CLF_EXPECTED, REG_EXPECTED
    assert not metric_issues(CLF_EXPECTED.copy(),CLF_EXPECTED,'clf')
    assert not metric_issues(REG_EXPECTED.copy(),REG_EXPECTED,'reg')
    for key in CLF_EXPECTED:
        altered = CLF_EXPECTED.copy(); altered[key] += .001
        assert metric_issues(altered,CLF_EXPECTED,'clf')
    for key in REG_EXPECTED:
        altered = REG_EXPECTED.copy(); altered[key] += .001
        assert metric_issues(altered,REG_EXPECTED,'reg')
    assert metric_issues({},CLF_EXPECTED,'clf')
    assert metric_issues(dict(CLF_EXPECTED, unexpected=1),CLF_EXPECTED,'clf')


def test_source_lock_matches_shipped_source():
    from pathlib import Path
    from src.fairness_reference import SOURCE_LOCK, source_digest
    root = Path(__file__).resolve().parents[1]
    assert all(source_digest(root/name)==digest for name,digest in SOURCE_LOCK.items())


def test_reference_gate_blocks_source_version_metric_and_privacy(tmp_path,monkeypatch):
    import hashlib
    from src import fairness_reference as fr
    raw = tmp_path/'data/raw/online_retail_II.csv'
    raw.parent.mkdir(parents=True); raw.write_bytes(b'test fixture')
    source = tmp_path/'model.py'; source.write_text('original\n')
    monkeypatch.setattr(fr,'RAW_SHA256',hashlib.sha256(raw.read_bytes()).hexdigest())
    monkeypatch.setattr(fr,'SOURCE_LOCK',{'model.py':fr.source_digest(source)})
    monkeypatch.setattr(fr,'version',lambda package:fr.LOCKED_VERSIONS[package])
    monkeypatch.setattr(fr.sys,'version_info',(3,11,9))
    privacy = pd.DataFrame.from_dict(fr.PRIVACY_EXPECTED,orient='index',columns=fr.PRIVACY_COLUMNS)
    def check(c=fr.CLF_EXPECTED,r=fr.REG_EXPECTED,p=privacy):
        return fr.reference_issues(tmp_path,c,r,p)
    assert check()==[]
    source.write_text('changed\n')
    assert any('source changed' in e for e in check())
    source.write_text('original\n')
    bad = dict(fr.CLF_EXPECTED, threshold=.4805)
    assert any('threshold' in e for e in check(c=bad))
    p = privacy.copy();p.loc['generalized','customers']+=1
    assert any('privacy:' in e for e in check(p=p))
    raw.write_bytes(b'different fixture')
    assert 'CSV SHA-256' in check()
    monkeypatch.setattr(fr,'version',lambda package:'0')
    assert any('version' in e for e in check())


def test_validate_all_markers_before_writes(tmp_path):
    from src.fairness_evaluation import update_cards
    a,b = tmp_path/'a.md',tmp_path/'b.md'
    original='<!-- W12_RESULTS_START -->\nold\n<!-- W12_RESULTS_END -->'
    a.write_text(original); b.write_text('invalid')
    t = pd.DataFrame({'n':[40]},index=['UK'])
    with pytest.raises(ValueError):
        update_cards([(a,t,['n'],'new'),(b,t,['n'],'new')])
    assert a.read_text()==original and b.read_text()=='invalid'


def test_publication_gate_requires_partition_and_reference_rows():
    from src import fairness_reference as fr
    representation = pd.DataFrame(fr.PUBLIC_COUNTS,index=fr.PUBLIC_GROUPS[1:])
    def make_table(expected, keys):
        table = pd.DataFrame.from_dict(expected,orient='index',columns=keys)
        table.loc['Non-UK / Unknown'] = 1.
        table.loc['Non-UK / Unknown','n'] = 452
        table['status'] = 'descriptive'
        return table
    c = make_table(fr.CLF_PUBLIC_EXPECTED,fr.CLF_PUBLIC_KEYS)
    r = make_table(fr.REG_PUBLIC_EXPECTED,fr.REG_PUBLIC_KEYS)
    assert not fr.publication_issues(representation,c,r)
    assert fr.publication_issues(representation,c.rename(index={'Non-UK / Unknown':'International'}),r)
    c.loc['UK','recall'] -= .01
    assert fr.publication_issues(representation,c,r)
    r.loc['Non-UK / Unknown','mae_gbp'] = np.nan
    assert fr.publication_issues(representation,c,r)
