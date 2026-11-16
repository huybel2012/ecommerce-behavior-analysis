import numpy as np
import pandas as pd
import pytest
from scipy import stats
from src.pipeline import RAW_COLUMNS, clean_transactions, integrate_external_sources, unknown_country_mask
from src.customer_features import classify_market
from src.classification_features import build_cutoff_customer_dataset
from src.robustness import safe_market_test, sensitivity_analysis


def raw_sample():
    rows = [["A", "ITEM", None, 1, "2011-01-01", 2., 1, "United Kingdom"],
            ["B", "ITEM", "Product", 2, "2011-09-01", 2., 1, "United Kingdom"],
            ["D", "ITEM", " N/A ", 4, "2011-02-01", 2., 2, "France"],
            ["C1", "ITEM", "Product", -1, "2011-03-01", 2., 2, "France"],
            ["E", "ITEM", " null ", 1, "2011-02-01", 2., 3, "Unknown"]]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


@pytest.mark.parametrize('token', ['n/a', ' NULL ', 'European Community', None, pd.NA])
def test_unknown_shared(token):
    assert unknown_country_mask(pd.Series([token])).iloc[0]
    frame = pd.DataFrame({'customer_id':['1'], 'country_original':[token]})
    assert classify_market(frame).iloc[0]['is_uk'] == 'Unknown'
    clean, _ = clean_transactions(raw_sample())
    clean.loc[clean.customer_id.eq('3'), 'country_original'] = token
    model = build_cutoff_customer_dataset(clean, observed_until='2011-12-09')
    assert model.set_index('customer_id').loc['3','region'] == 'Unknown'


def test_missing_description_and_duplicate_policy():
    raw = raw_sample()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    main, log = clean_transactions(raw)
    alternate, alt_log = clean_transactions(raw, drop_exact_duplicates=False)
    assert main['description'].eq('UNKNOWN_PRODUCT').sum() == 3
    assert log.set_index('Step').loc['1.2','Rows'] == 1
    assert alt_log.set_index('Step').loc['1.2','Rows'] == 0
    assert len(alternate) == len(main) + 1


@pytest.mark.parametrize('column,value', [('quantity',-1), ('quantity',1.5), ('price',0), ('price',np.inf),
                                           ('total_amount',999), ('is_cancelled',True), ('stockcode',''), ('invoice','')])
def test_temporal_rejects_invalid_transaction(column,value):
    clean, _ = clean_transactions(raw_sample())
    clean[column] = clean[column].astype(object)
    clean.loc[clean.invoice.eq('B'), column] = value
    with pytest.raises(ValueError):
        build_cutoff_customer_dataset(clean, observed_until='2011-12-09')


@pytest.mark.parametrize('table', [[[10,0],[2,0]], [[0,0],[2,3]], [[0,0],[0,0]]])
def test_degenerate_contingency(table):
    chi,p,_,expected,method = safe_market_test(table)
    assert np.isnan(chi) and np.isnan(p) and np.isnan(expected).all()
    assert method == 'Không đủ điều kiện'


def test_valid_contingency():
    for table in [[[100,50],[80,70]], [[1,2],[3,4]]]:
        chi,p,_,_,method = safe_market_test(table)
        assert chi == pytest.approx(stats.chi2_contingency(table, correction=False).statistic)
        expected_p = stats.fisher_exact(table).pvalue if method == 'Fisher exact' else stats.chi2_contingency(table, correction=False).pvalue
        assert p == pytest.approx(expected_p)


def test_sensitivity_preserves_baseline_and_input():
    raw = raw_sample(); raw = pd.concat([raw,raw.iloc[[0]]],ignore_index=True)
    original = raw.copy(deep=True)
    result = sensitivity_analysis(raw)
    pd.testing.assert_frame_equal(raw, original)
    assert len(result) == 5
    assert result.loc['keep_exact_duplicates','total_gross_gbp'] == result.loc['baseline','total_gross_gbp'] + 2
    assert result.loc['exclude_above_p99','customers'] < result.loc['baseline','customers']
    assert result.loc['peak_threshold_40pct','peak_n'] >= result.loc['peak_threshold_60pct','peak_n']
