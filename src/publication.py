"""Shared minimum-denominator policy for descriptive aggregate releases.

Pooling is a reporting safeguard, not anonymity or a change to training data.
"""
import math
from numbers import Real

import numpy as np
import pandas as pd

MIN_GROUP = 30
MIN_SUPPORT = 10
POOLED = 'Small / pooled markets'


def public_labels(counts):
    """Map groups to a disjoint release partition; return None for tiny totals.

    Counts must be for the same unit (lines or distinct customers). A small
    pool absorbs the smallest remaining groups until it has >=30 observations
    and at least two constituent groups. Never publish a singleton pool.
    """
    if not counts.index.is_unique or POOLED in counts.index:
        raise ValueError('Duplicate or reserved group label.')
    # Validate scalar values before arithmetic: Arrow-backed Series do not
    # implement modulo. Use Python integers to preserve exact group counts.
    values = counts.to_numpy(dtype=object)
    if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_))
           or not math.isfinite(v) or v < 0 or v != int(v) for v in values):
        raise ValueError('Invalid group counts.')
    counts = pd.Series([int(v) for v in values], index=counts.index, dtype=object)
    counts = counts[counts > 0]
    if counts.sum() < MIN_GROUP:
        return None
    order = sorted(counts.index, key=lambda label: (counts[label], str(label)))
    pool = [label for label in order if counts[label] < MIN_GROUP]
    if pool:
        for label in order:
            if len(pool) >= 2 and counts.loc[pool].sum() >= MIN_GROUP:
                break
            if label not in pool:
                pool.append(label)
    return {label: POOLED if label in pool else label for label in counts.index}


def public_region_summary(customers):
    """Summarize disjoint customer groups, recomputing medians after pooling."""
    if customers.customer_id.isna().any() or customers.customer_id.duplicated().any():
        raise ValueError('Require distinct customers.')
    region = customers.region.astype('string').fillna('Unknown')
    mapping = public_labels(region.value_counts())
    columns = ['n','gross_gbp','median_spend_gbp','mean_spend_gbp','median_aov_gbp',
               'median_frequency','median_observation_days','repeat_share']
    if mapping is None:
        return pd.DataFrame(columns=columns).rename_axis('region')
    return customers.assign(region=region.map(mapping)).groupby('region').agg(
        n=('customer_id','size'), gross_gbp=('monetary','sum'),
        median_spend_gbp=('monetary','median'), mean_spend_gbp=('monetary','mean'),
        median_aov_gbp=('avg_order_value','median'), median_frequency=('frequency','median'),
        median_observation_days=('observation_days','median'), repeat_share=('is_repeat_buyer','mean')
    ).sort_values('n',ascending=False)


def plot_public_regions(table, destination):
    """Plot only named groups eligible for release, without individual outliers."""
    import matplotlib.pyplot as plt
    named = table.drop(index=POOLED, errors='ignore').head(5)
    fig, ax = plt.subplots(figsize=(11,5.5))
    if named.empty:
        ax.text(.5,.5,'Không có nhóm khu vực đủ điều kiện công bố',ha='center',va='center',transform=ax.transAxes)
        ax.set_axis_off()
    else:
        ax.bar(range(len(named)), named.median_spend_gbp, color='#2b5c8f')
        ax.set_xticks(range(len(named)), [f'{r}\n(n={int(v):,})' for r,v in named.n.items()],rotation=12)
        ax.set_ylabel('Trung vị giá trị mua hợp lệ / khách (£)')
        ax.set_ylim(bottom=0)
        for i,v in enumerate(named.median_spend_gbp):
            ax.annotate(f'£{v:,.2f}',(i,v),xytext=(0,4),textcoords='offset points',ha='center')
        ax.margins(y=.15)
    ax.set_title('Chi tiêu trung vị theo khu vực đủ điều kiện công bố')
    fig.text(.5,.015,'Nguồn: Online Retail II; n ≥ 30 khách; tối đa 5 nhóm lớn nhất, không vẽ nhóm gộp.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1])
    fig.savefig(destination,dpi=300,bbox_inches='tight')
    return fig
