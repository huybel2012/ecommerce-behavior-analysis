from pathlib import Path
from html.parser import HTMLParser
import json
import numpy as np
import pandas as pd
import pytest
from src import fairness_reference as fr
from src.fairness_evaluation import country_selection
from src.publication import public_labels, public_region_summary, POOLED

ROOT = Path(__file__).resolve().parents[1]


def reference_tables():
    return (pd.DataFrame(fr.PUBLIC_COUNTS,index=fr.PUBLIC_GROUPS[1:]),
            pd.DataFrame.from_dict(fr.CLF_PUBLIC_EXPECTED,orient='index',columns=fr.CLF_PUBLIC_KEYS).assign(status='descriptive'),
            pd.DataFrame.from_dict(fr.REG_PUBLIC_EXPECTED,orient='index',columns=fr.REG_PUBLIC_KEYS).assign(status='descriptive'))


@pytest.mark.parametrize('kind,key,value',[
    ('clf','accuracy',1.5),('clf','accuracy','0.7301'),('clf','precision',True),
    ('clf','positive_support',211),('clf','recall',.6800),('clf','roc_auc',.8010),
    ('clf','selection_rate',np.inf),('reg','mae_gbp',-100),('reg','rmse_gbp',500),
    ('reg','r2',1.5),('reg','zero_share',.5),('reg','mean_error_gbp',1),('reg','mean_predicted_gbp',1)])
def test_invalid_pooled_metric(kind,key,value):
    rep,c,r=reference_tables()
    assert fr.publication_issues(rep,c,r)==[]
    target = c if kind=='clf' else r
    target[key]=target[key].astype(object)
    target.loc['Non-UK / Unknown',key]=value
    assert fr.publication_issues(rep,c,r)


def test_publication_shape_and_status():
    rep,c,r=reference_tables()
    assert fr.publication_issues(rep,pd.concat([c,c[['n']]],axis=1),r)
    assert fr.publication_issues(pd.concat([rep,rep[['test']]],axis=1),c,r)
    r.loc['UK','status']='suppressed'
    assert fr.publication_issues(rep,c,r)


@pytest.mark.parametrize('values,expected', [({'UK':100,'A':9,'B':24},{'UK':'UK','A':POOLED,'B':POOLED}),({'UK':100,'A':9,'B':40},{'UK':'UK','A':POOLED,'B':POOLED}),({'A':9},None),({'A':30},{'A':'A'}),({'A':12,'B':12},None)])
def test_shared_pool(values,expected):
    assert public_labels(pd.Series(values))==expected


def test_country_pool_preserves_totals_without_singleton_complement():
    d=pd.DataFrame({'Known_ID':[100,7,25],'Missing_ID':[50,2,5]},index=pd.Index(['UK','Small','Other'],name='Country'))
    result=country_selection(d)
    assert set(result.index)=={'UK',POOLED}
    assert result.loc[POOLED,'total']==39
    assert result.Known_ID.sum()==d.Known_ID.sum() and result.Missing_ID.sum()==d.Missing_ID.sum()
    assert country_selection(d.iloc[1:2]).empty


def test_region_pool_uses_distinct_customer_median_not_average_of_medians():
    n=80; d=pd.DataFrame({'customer_id':range(n),'region':['UK']*40+['A']*10+['B']*30,'monetary':np.arange(n,dtype=float),'avg_order_value':10,'frequency':1,'observation_days':10,'is_repeat_buyer':False})
    table=public_region_summary(d)
    assert set(table.index)=={'UK',POOLED} and table.n.sum()==80
    assert table.loc[POOLED,'median_spend_gbp']==59.5
    assert public_region_summary(d.iloc[:5]).empty
    with pytest.raises(ValueError): public_region_summary(pd.concat([d,d]))


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.images=[]; self.resources=[]; self.ids=set(); self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='img': self.images.append(a.get('src')); assert a.get('alt')
        if tag in ['img','script','link','iframe']: self.resources.extend(a[k] for k in ('src','href') if k in a)
        if 'id' in a: assert a['id'] not in self.ids; self.ids.add(a['id'])
        if tag=='a': self.links.append(a['href'])


def test_html_offline_assets_and_report_values():
    page=(ROOT/'index.html').read_text(encoding='utf8'); parser=ReportParser();parser.feed(page)
    expected={p.relative_to(ROOT).as_posix() for p in (ROOT/'reports/figures').glob('*.png')}
    assert len(parser.images)==9 and set(parser.images)==expected
    assert all(not x.startswith(('http:', 'https:', '//','data:')) and (ROOT/x).is_file() for x in parser.resources)
    for link in parser.links:
        assert link[1:] in parser.ids if link.startswith('#') else (ROOT/link).is_file()
    assert 'generateFallbackSVGDataURI' not in page and 'GITHUB_RAW_BASE' not in page
    for token in ['73,01%','0,7936','0,6579','Random Forest','0,0211','5.853','5.876','0,1945','−0,3156','−0,0276','1016,9378','Không tải được biểu đồ']:
        assert token in page


def test_notebooks_use_shared_release():
    def source(prefix):
        p=next((ROOT/'notebooks').glob(prefix+'*.ipynb')); return '\n'.join(''.join(c['source']) for c in json.loads(p.read_text(encoding='utf8'))['cells'] if c['cell_type']=='code')
    assert 'country_selection(missing_by_country)' in source('01')
    assert 'country = country_selection(country)' in source('06')
    assert 'public_region_summary(customer_df)' in source('02')
    assert "grouped('is_uk')" not in source('02')


@pytest.mark.parametrize('dtype', ['int64', 'Int64', 'int64[pyarrow]'])
def test_public_labels_count_dtypes(dtype):
    if dtype.endswith('[pyarrow]'):
        pytest.importorskip('pyarrow')
    counts = pd.Series([100, 9, 24], index=['UK', 'A', 'B'], dtype=dtype)
    assert public_labels(counts) == {'UK': 'UK', 'A': POOLED, 'B': POOLED}


@pytest.mark.parametrize('value', [pd.NA, np.nan, np.inf, -1, 2.5, True, '9'])
def test_public_labels_reject_invalid_counts(value):
    with pytest.raises(ValueError, match='Invalid group counts'):
        public_labels(pd.Series([100, value], index=['UK', 'A'], dtype=object))


def test_region_pool_arrow_backed_strings():
    pytest.importorskip('pyarrow')
    n = 80
    customers = pd.DataFrame({
        'customer_id': range(n),
        'region': ['UK'] * 40 + ['A'] * 10 + ['B'] * 30,
        'monetary': np.arange(n, dtype=float), 'avg_order_value': 10,
        'frequency': 1, 'observation_days': 10, 'is_repeat_buyer': False,
    })
    with pd.option_context('mode.string_storage', 'pyarrow'):
        result = public_region_summary(customers)
    assert result.n.sum() == n
    assert result.loc[POOLED, 'median_spend_gbp'] == 59.5
