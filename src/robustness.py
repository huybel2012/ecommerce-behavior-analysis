"""Kiểm định an toàn và độ nhạy mô tả (không dùng để chọn mô hình)."""
import numpy as np
import pandas as pd
from scipy import stats
from src.pipeline import clean_transactions, integrate_external_sources
from src.customer_features import build_customer_profiles


def safe_market_test(table):
    """Pearson không Yates, Fisher nếu tần số kỳ vọng nhỏ; NaN nếu suy biến."""
    a = np.asarray(table, dtype=float)
    if a.shape != (2, 2) or not np.isfinite(a).all() or (a < 0).any() or (a != np.floor(a)).any():
        raise ValueError("Cần bảng đếm 2x2 không âm, nguyên và hữu hạn.")
    if (a.sum(axis=0) == 0).any() or (a.sum(axis=1) == 0).any():
        return np.nan, np.nan, np.nan, np.full((2, 2), np.nan), "Không đủ điều kiện"
    chi, p, dof, expected = stats.chi2_contingency(a, correction=False)
    if (expected < 5).any():
        return chi, stats.fisher_exact(a, alternative="two-sided").pvalue, dof, expected, "Fisher exact"
    return chi, p, dof, expected, "Pearson Chi-square"


def exploratory_effects(profiles, threshold=0.5):
    """Hiệu ứng mô tả theo dấu H1/H2/H3; NaN khi thiếu nhóm."""
    geo = profiles[profiles['is_uk'].isin(['UK', 'International'])]
    uk = geo[geo['is_uk'].eq('UK')]
    intl = geo[geo['is_uk'].eq('International')]
    peak = profiles['peak_month_ratio'].ge(threshold)
    log = np.log1p(profiles['monetary'])
    ct = pd.crosstab(geo['is_uk'], geo['is_repeat_buyer']).reindex(index=['UK', 'International'], columns=[0, 1], fill_value=0)
    _, p, _, _, method = safe_market_test(ct)
    return {'customers': len(profiles), 'uk_n': len(uk), 'intl_n': len(intl),
            'peak_n': int(peak.sum()), 'other_n': int((~peak).sum()),
            'total_gross_gbp': profiles['monetary'].sum(), 'median_gross_gbp': profiles['monetary'].median(),
            'mean_gross_gbp': profiles['monetary'].mean(),
            'h1_median_aov_diff_gbp': intl['avg_order_value'].median() - uk['avg_order_value'].median(),
            'h2_mean_log_diff': log[peak].mean() - log[~peak].mean(),
            'h3_repeat_diff_pp': 100 * (uk['is_repeat_buyer'].mean() - intl['is_repeat_buyer'].mean()),
            'h3_exploratory_p': p, 'h3_method': method}


def sensitivity_analysis(raw):
    """Mỗi kịch bản thay một giả định; không thay dữ liệu chính hoặc feature mô hình."""
    base_clean, _ = clean_transactions(raw)
    base = build_customer_profiles(integrate_external_sources(base_clean))
    keep_clean, _ = clean_transactions(raw, drop_exact_duplicates=False)
    keep = build_customer_profiles(integrate_external_sources(keep_clean))
    p99 = base['monetary'].quantile(.99)
    trimmed = base[base['monetary'].le(p99)]
    cases = [('baseline', base, .5, len(base_clean)), ('keep_exact_duplicates', keep, .5, len(keep_clean)),
             ('exclude_above_p99', trimmed, .5, len(base_clean[base_clean['customer_id'].isin(trimmed['customer_id'])])),
             ('peak_threshold_40pct', base, .4, len(base_clean)), ('peak_threshold_60pct', base, .6, len(base_clean))]
    rows = []
    for name, profiles, threshold, n_lines in cases:
        rows.append({'scenario': name, 'peak_threshold': threshold, 'clean_lines': n_lines,
                     **exploratory_effects(profiles, threshold)})
    result = pd.DataFrame(rows).set_index('scenario')
    result.attrs['p99_gbp'] = float(p99)
    return result
