"""W12: aggregate-only retrospective audit; never tune on the test snapshot."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.regression_evaluation import metrics
from src.pipeline import unknown_country_mask

MIN_GROUP = 30
MIN_SUPPORT = 10
MARKETS = ('UK', 'International', 'Unknown')


def market_groups(snapshot):
    """Use pre-cutoff modal geography from W11, not full-period W9 labels."""
    region = snapshot['region'].astype('string')
    return pd.Series(np.where(region.eq('Domestic_UK').fillna(False), 'UK',
                             np.where(region.isna() | region.eq('Unknown').fillna(False), 'Unknown', 'International')),
                     index=snapshot.index, name='market')


def _inputs(y, pred, groups):
    y, pred, groups = np.asarray(y, dtype=float), np.asarray(pred, dtype=float), np.asarray(groups)
    if y.ndim != 1 or y.shape != pred.shape or y.shape != groups.shape or not len(y):
        raise ValueError('Require aligned nonempty one-dimensional vectors.')
    if not np.isfinite(y).all() or not np.isfinite(pred).all() or not np.isin(groups, MARKETS).all():
        raise ValueError('Invalid targets, predictions or market labels.')
    return y, pred, groups


def _slices(groups):
    yield 'Overall', np.ones(len(groups), dtype=bool)
    for name in MARKETS:
        yield name, groups == name


def classification_by_group(y, prob, groups, threshold):
    """Rates withheld for n<30 or relevant denominator<10; no threshold fitting."""
    y, prob, groups = _inputs(y, prob, groups)
    if not np.isin(y, [0, 1]).all() or ((prob < 0) | (prob > 1)).any() or not 0 <= threshold <= 1:
        raise ValueError('Invalid binary evaluation inputs.')
    rows = []
    for name, mask in _slices(groups):
        a, p = y[mask], prob[mask]
        n = len(a)
        pred = p >= threshold
        tp, fp = int(((a == 1) & pred).sum()), int(((a == 0) & pred).sum())
        pos, neg, selected = int(a.sum()), int(n-a.sum()), int(pred.sum())
        enough = n >= MIN_GROUP
        both = enough and min(pos, neg) >= MIN_SUPPORT
        def rate(num, den):
            return num/den if enough and den >= MIN_SUPPORT else np.nan
        rows.append(dict(market=name, n=n, share=n/len(y),
                         positive_support=pos if enough else np.nan,
                         negative_support=neg if enough else np.nan,
                         prevalence=rate(pos, n), selection_rate=rate(selected, n),
                         accuracy=rate(tp+neg-fp, n), recall=rate(tp, pos),
                         fpr=rate(fp, neg), precision=rate(tp, selected),
                         f1=2*tp/(pos+selected) if both and selected >= MIN_SUPPORT else np.nan,
                         roc_auc=roc_auc_score(a,p) if both else np.nan,
                         average_precision=average_precision_score(a,p) if both else np.nan,
                         status='n<30: suppressed' if not enough else ('limited class support' if not both else 'descriptive')))
    return pd.DataFrame(rows).set_index('market')


def regression_by_group(y, pred, groups):
    """Positive mean_error_gbp = underprediction (actual minus predicted)."""
    y, pred, groups = _inputs(y, pred, groups)
    if (y < 0).any() or (pred < 0).any():
        raise ValueError('Expected nonnegative gross GBP.')
    rows = []
    for name, mask in _slices(groups):
        a, p = y[mask], pred[mask]
        n = len(a)
        enough = n >= MIN_GROUP
        row = metrics(a, p) if enough else dict(mae_gbp=np.nan, rmse_gbp=np.nan, r2=np.nan)
        rows.append(dict(market=name, n=n, share=n/len(y), **row,
                         mean_error_gbp=float(np.mean(a-p)) if enough else np.nan,
                         mean_actual_gbp=float(np.mean(a)) if enough else np.nan,
                         mean_predicted_gbp=float(np.mean(p)) if enough else np.nan,
                         zero_share=float(np.mean(a==0)) if enough else np.nan,
                         status='descriptive' if enough else 'n<30: suppressed'))
    return pd.DataFrame(rows).set_index('market')


def signed_gap(table, metric):
    """International minus UK; NaN when either metric is unavailable."""
    return float(table.loc['International', metric] - table.loc['UK', metric])


def country_selection(country_table):
    """From W3 purchase-line counts, not inferred numbers of anonymous people."""
    out = country_table.copy().reset_index()
    country = out.columns[0]
    out[country] = out[country].astype('string').str.strip()
    out.loc[unknown_country_mask(out[country]), country] = 'Unknown'
    out = out.groupby(country, dropna=False)[['Known_ID', 'Missing_ID']].sum()
    out['total'] = out.Known_ID + out.Missing_ID
    out['missing_share'] = out.Missing_ID / out.total.where(out.total > 0)
    out['known_line_share'] = out.Known_ID / (out.Known_ID.sum() if out.Known_ID.sum() else np.nan)
    # No country-level rates/ID split for very small line denominators.
    out.loc[out.total < MIN_SUPPORT, ['Known_ID','Missing_ID','missing_share','known_line_share']] = np.nan
    return out.sort_values('total', ascending=False)


def privacy_summary(snapshot, k_target=10):
    """Illustrative k on customer quasi-identifiers: no microdata leaves function.

    Exact tuple is a conservative linkage example, not a proof of anonymity.
    Generalize to market, recency/frequency/spend bands; suppress classes < k.
    k counts distinct customers, not transaction lines. Publish only summaries.
    """
    if k_target < 2 or snapshot.empty or snapshot.customer_id.isna().any() or snapshot.customer_id.duplicated().any():
        raise ValueError('Require distinct customer rows and k>=2.')
    cols = ['recency_days', 'frequency', 'gross_valid_revenue']
    if not np.isfinite(snapshot[cols].to_numpy(dtype=float)).all() or (snapshot[cols] < 0).any().any():
        raise ValueError('Invalid quasi-identifiers.')
    exact = snapshot[['region', *cols]].copy()
    generalized = pd.DataFrame({'market': market_groups(snapshot),
        'recency_band':pd.cut(snapshot.recency_days, [-np.inf,30,90,180,np.inf]),
        'frequency_band':pd.cut(snapshot.frequency, [-np.inf,1,3,10,np.inf]),
        'gross_band':pd.cut(snapshot.gross_valid_revenue, [-np.inf,100,500,2000,np.inf])})
    rows = []
    for name, q in [('exact', exact), ('generalized', generalized)]:
        counts = q.groupby(list(q.columns), observed=True, dropna=False).size()
        for label, retained in [(name, counts)]+([('generalized_suppressed', counts[counts>=k_target])] if name=='generalized' else []):
            n = int(retained.sum())
            rows.append(dict(scenario=label, customers=n, suppressed=len(snapshot)-n,
                             k_min=int(retained.min()) if n else np.nan,
                             singleton_customers=int(retained[retained==1].sum()),
                             customers_below_k=int(retained[retained<k_target].sum()), k_target=k_target))
    return pd.DataFrame(rows).set_index('scenario')


def markdown_table(frame, columns):
    """Render small aggregate tables without optional tabulate dependency."""
    def fmt(x):
        if pd.isna(x): return 'N/A'
        if isinstance(x, (float, np.floating)): return f'{x:.4f}'
        return str(x)
    rows = [['Nhóm', *columns], ['---']*(len(columns)+1)]
    rows += [[str(i), *(fmt(row[c]) for c in columns)] for i,row in frame.iterrows()]
    return '\n'.join('| '+' | '.join(r)+' |' for r in rows)


def update_card(path, table, columns, note):
    """Idempotently replace only the aggregate results block in a model card."""
    start, end = '<!-- W12_RESULTS_START -->', '<!-- W12_RESULTS_END -->'
    text = path.read_text(encoding='utf-8')
    if text.count(start)!=1 or text.count(end)!=1 or text.index(start)>text.index(end):
        raise ValueError('Invalid model card markers.')
    before, rest = text.split(start)
    _, after = rest.split(end)
    path.write_text(before+start+'\n'+note+'\n\n'+markdown_table(table, columns)+'\n'+end+after, encoding='utf-8')
