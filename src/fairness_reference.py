"""Fail-closed provenance gate for the accepted historical models.

Reference values are from the saved, real-data outputs of notebooks 04/05/06
at 1589fb3. This gate is not a signature or a new test set. Do not regenerate
the lock just to make a failed check pass. Review changes explicitly.
"""
import hashlib
import json
import sys
from importlib.metadata import version, PackageNotFoundError
import numpy as np
from numbers import Real

RAW_SHA256 = '32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9'
LOCKED_VERSIONS = {'numpy': '1.26.4', 'pandas': '2.3.3',
                   'scipy': '1.17.1', 'scikit-learn': '1.9.1'}
SOURCE_LOCK = {
 'src/publication.py': 'eda40f9828f8eef926fa025f704f471ecb288c128bdff5b6248e4e24191b9fb3',
 'notebooks/01_data_audit_and_cleaning.ipynb': '173fd04e4e25b5db632e29184ec8994dcb77227a17052e93792733b0f9b42689',
 'notebooks/02_exploratory_data_analysis.ipynb': '03da3bfe1bd53674e6cdd016f0bbe163337d1f78333fb50b81c6f134e6e8eee7',
 'notebooks/04_classification_repeat_purchase.ipynb': '165d1e7208d3fd761f7a427ee70d936543720cdfea236dfafd5c0ec8a7d9bf27',
 'notebooks/05_regression_90day_spend.ipynb': '6b7ba9a5a30333c9f9c7b229161075e8d9e769107d719b5d9fdfc0e2a60692bb',
 'notebooks/06_ethics_and_bias.ipynb': '72663eaf3d9234a2084cc291a36bdbce0a3ec3fbda36535c21b2714d1b9b9724',
 'requirements.txt': '8fb91a14599748135966d761099a57ab89a6f6102d3623595feb21a77b90a5b3',
 'src/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'src/classification_evaluation.py': 'c2460337b54c215297a874ff03b47ad3d8af4bb760c8cadeab8ed2f08031beb5',
 'src/classification_features.py': 'a416e0d4062e144db3ed3c85b439d3edd5e73552c33dfc3e9c7a267724fac9e8',
 'src/customer_features.py': '6b17ece91b67d7b47b2edb216682907a3ef11e74bcf30814e3c6804949b46b04',
 'src/data_quality.py': 'd8aa5ef1dd139ec4ebc1d86a2640694cd35f87fe5a3fcebccd5a2ffa4805dff3',
 'src/fairness_evaluation.py': 'a4761a57ac532851ad40968a3f98526cba332f77fa9cbbc990b2cd7b0c8f8516',
 'src/pipeline.py': 'd85051cc48b94979ebb8d80a977f2fc59caa67268fd4c90c3e67e86462866507',
 'src/regression_evaluation.py': '776362fb9051a179a42ba44f2c60bc427581df2001a377b5464e861607399dbc',
 'src/robustness.py': 'b6097870cd65f0c9af6408199e70a18bcdc3aeba2f0b28502cdc0f9c6826695a',
 'src/utils.py': '7c665937ad0287d794d3adf2a0c1d21a7bc92763a0abad2ffd03f55dc7a61bd2'}

# Rounded aggregate metrics, not individually identifying records.
CLF_EXPECTED = {
    'train_n': 4513, 'validation_n': 4909, 'test_n': 5225,
    'train_positive': 1585, 'validation_positive': 1617, 'test_positive': 2236,
    'threshold': 0.480468678468, 'validation_f1': 0.6602,
}
for prefix, values in {
    'validation': [.7659, .6327, .6902, .6602, .8204, .7257, .3294],
    'test_default': [.7319, .7342, .5854, .6514, .7936, .7639, .4279],
    'test_tuned': [.7301, .7190, .6064, .6579, .7936, .7639, .4279],
}.items():
    CLF_EXPECTED.update({f'{prefix}.{k}': v for k, v in zip(
        ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision', 'prevalence'], values)})

REG_EXPECTED = {
    'train_n': 4513, 'validation.n': 4909, 'test.n': 5225,
    'train_zero_share': .6488, 'validation_zero_share': .6706, 'test_zero_share': .5721,
    'train_all_zero_predictions': 1,
    'validation_all_zero_predictions': 1, 'test_all_zero_predictions': 1,
}
for prefix, values in {
    'validation': [338.1545, 2086.5166, -.0270, 0., 338.1545],
    'test': [515.3416, 3144.9414, -.0276, 0., 515.3416],
}.items():
    REG_EXPECTED.update({f'{prefix}.{k}': v for k, v in zip(
        ['mae_gbp', 'rmse_gbp', 'r2', 'negative_raw_fraction', 'mean_error_gbp'], values)})

PRIVACY_EXPECTED = {
    'exact': [5225, 0, 1, 5225, 5225, 10],
    'generalized': [5225, 0, 1, 11, 172, 10],
    'generalized_suppressed': [5053, 172, 10, 0, 0, 10],
}
PRIVACY_COLUMNS = ['customers', 'suppressed', 'k_min', 'singleton_customers', 'customers_below_k', 'k_target']


def source_digest(path):
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.ipynb':
        cells = json.loads(text)['cells']
        text = json.dumps([''.join(c['source']).replace('\\r\\n', '\\n') for c in cells
                           if c['cell_type'] == 'code'], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(text.replace('\\r\\n', '\\n').encode('utf-8')).hexdigest()


def classification_reference(ce, y_train, y_val, val_prob, y_test, prob, threshold, val_f1):
    actual = {'threshold': threshold, 'validation_f1': val_f1}
    for prefix, y in [('train', y_train), ('validation', y_val), ('test', y_test)]:
        actual[f'{prefix}_n'] = len(y)
        actual[f'{prefix}_positive'] = int(np.sum(y))
    for prefix, y, p, t in [('validation', y_val, val_prob, threshold),
                             ('test_default', y_test, prob, .5), ('test_tuned', y_test, prob, threshold)]:
        actual.update({f'{prefix}.{k}': v for k, v in ce.evaluate_probabilities(y, p, t).items()
                       if k != 'threshold'})
    return actual


def regression_reference(re, model, X_train, y_train, X_val, y_val, X_test, y_test):
    actual = {'train_n': len(y_train)}
    for prefix, X, y in [('train', X_train, y_train), ('validation', X_val, y_val), ('test', X_test, y_test)]:
        p = re.predict_gbp(model, X)
        actual[f'{prefix}_all_zero_predictions'] = int(np.all(p == 0))
        actual[f'{prefix}_zero_share'] = float(np.mean(y == 0))
        if prefix != 'train':
            row = re.evaluate_models({'locked': model}, X, y).loc['locked']
            actual.update({f'{prefix}.{k}': v for k, v in row.items()})
    return actual


def metric_issues(actual, expected, label):
    issues = []
    if set(actual) != set(expected):
        issues.append(f'{label}: metric keys differ')
    for key, value in expected.items():
        # Rounded published values: absolute +/- half a final decimal place.
        # Counts/flags exact; threshold stored to 12 significant digits.
        tol = 1e-12 if key == 'threshold' else (0 if isinstance(value, int) else 5.00001e-5)
        if (key not in actual or not isinstance(actual[key], Real)
                or isinstance(actual[key], (bool, np.bool_))
                or not np.isfinite(actual[key])
                or not np.isclose(actual[key], value, rtol=0, atol=tol)):
            issues.append(f'{label}: {key}')
    return issues


def reference_issues(root, clf_actual, reg_actual, privacy):
    """Only the accepted source/configuration and complete aggregate checks may publish."""
    issues = []
    raw = root/'data/raw/online_retail_II.csv'
    if raw.is_file():
        with raw.open('rb') as fh:
            if hashlib.file_digest(fh, 'sha256').hexdigest() != RAW_SHA256:
                issues.append('CSV SHA-256')
    else:
        issues.append('CSV missing')
    if sys.version_info[:2] != (3, 11):
        issues.append('Python 3.11 required')
    for package, expected in LOCKED_VERSIONS.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = None
        if installed != expected:
            issues.append(f'{package} version: {installed}, expected {expected}')
    for name, expected in SOURCE_LOCK.items():
        try:
            if source_digest(root/name) != expected:
                issues.append(f'source changed: {name}')
        except (OSError, ValueError, KeyError, TypeError):
            issues.append(f'invalid source: {name}')
    issues.extend(metric_issues(clf_actual, CLF_EXPECTED, 'classification'))
    issues.extend(metric_issues(reg_actual, REG_EXPECTED, 'regression'))
    if set(privacy.index) != set(PRIVACY_EXPECTED) or not set(PRIVACY_COLUMNS).issubset(privacy.columns):
        issues.append('privacy schema')
    else:
        for name, values in PRIVACY_EXPECTED.items():
            if not np.array_equal(privacy.loc[name, PRIVACY_COLUMNS].to_numpy(), values):
                issues.append(f'privacy: {name}')
    return issues

# The new pool is computed from the original individual-level predictions, not
# by averaging the old published group metrics. Pool reference rows below are
# from the saved 26fe794 notebook 06 (real data), rounded to four decimals.
PUBLIC_COUNTS = {'train': [4149, 364], 'validation': [4503, 406], 'test': [4773, 452]}
PUBLIC_GROUPS = ['Overall', 'UK', 'Non-UK / Unknown']
CLF_PUBLIC_KEYS = ['n', 'share', 'positive_support', 'negative_support', 'prevalence',
                   'selection_rate', 'accuracy', 'recall', 'fpr', 'precision', 'f1',
                   'roc_auc', 'average_precision']
REG_PUBLIC_KEYS = ['n', 'share', 'mae_gbp', 'rmse_gbp', 'r2', 'mean_error_gbp',
                   'mean_actual_gbp', 'mean_predicted_gbp', 'zero_share']
CLF_PUBLIC_EXPECTED = {
    'Overall': [5225, 1., 2236, 2989, .4279, .3610, .7301, .6064, .1773, .7190, .6579, .7936, .7639],
    'UK': [4773, .9135, 2026, 2747, .4245, .3537, .7301, .5987, .1729, .7186, .6532, .7927, .7606],
    'Non-UK / Unknown': [452, .0865, 210, 242, .4646, .4381, .7301, .6810, .2273, .7222, .7010, .8004, .7943],
}
REG_PUBLIC_EXPECTED = {
    'Overall': [5225, 1., 515.3416, 3144.9414, -.0276, 515.3416, 515.3416, 0., .5721],
    'UK': [4773, .9135, 467.8408, 2787.4855, -.0290, 467.8408, 467.8408, 0., .5755],
    'Non-UK / Unknown': [452, .0865, 1016.9378, 5681.8507, -.0331, 1016.9378, 1016.9378, 0., .5354],
}


def publication_issues(representation, clf, reg):
    """Fail closed on any deviation from the accepted historical release.

    All three rows are locked, including non-additive AUC/AP/R². Additive
    statistics are also reconciled; rounded inputs carry half-unit tolerance.
    """
    issues = []
    if (list(representation.index) != PUBLIC_GROUPS[1:]
            or not representation.columns.is_unique
            or set(representation.columns) != set(PUBLIC_COUNTS)):
        issues.append('representation schema')
    else:
        for role, counts in PUBLIC_COUNTS.items():
            if not np.array_equal(representation[role].to_numpy(), counts):
                issues.append(f'representation: {role}')
    tables = [('classification', clf, CLF_PUBLIC_KEYS, CLF_PUBLIC_EXPECTED),
              ('regression', reg, REG_PUBLIC_KEYS, REG_PUBLIC_EXPECTED)]
    for label, table, keys, expected in tables:
        if (list(table.index) != PUBLIC_GROUPS or not table.columns.is_unique
                or set(table.columns) != set(keys + ['status'])):
            issues.append(f'{label}: schema')
            continue
        if not table.status.eq('descriptive').fillna(False).all():
            issues.append(f'{label}: status')
        for group, values in expected.items():
            issues.extend(metric_issues(table.loc[group, keys].to_dict(),
                                        dict(zip(keys, values)), f'{label}: {group}'))
    if issues:
        return issues
    tol = 5.00001e-5
    def close(a, b, atol=tol):
        return np.isclose(a, b, rtol=0, atol=atol)
    confusion = []
    for group in PUBLIC_GROUPS:
        c, r = clf.loc[group], reg.loc[group]
        n, pos, neg = int(c.n), int(c.positive_support), int(c.negative_support)
        if (not all(0 <= c[k] <= 1 for k in CLF_PUBLIC_KEYS[1:] if k not in ('positive_support','negative_support'))
                or not all(0 <= r[k] <= 1 for k in ['share','zero_share'])
                or r.r2 > 1 or min(r.mae_gbp,r.rmse_gbp,r.mean_actual_gbp,r.mean_predicted_gbp) < 0
                or r.rmse_gbp + tol < r.mae_gbp or abs(r.mean_error_gbp) > r.mae_gbp + tol
                or pos + neg != n or r.n != n or not close(c.share,n/clf.loc['Overall','n'])
                or not close(r.share,c.share) or not close(r.zero_share,neg/n)
                or not close(r.mean_error_gbp,r.mean_actual_gbp-r.mean_predicted_gbp,3*tol)):
            issues.append(f'{group}: domain or consistency')
        tp, fp = round(c.recall*pos), round(c.fpr*neg)
        sel = tp+fp
        confusion.append(np.array([tp,fp,pos,neg]))
        rates = dict(prevalence=pos/n,selection_rate=sel/n,accuracy=(tp+neg-fp)/n,
                     recall=tp/pos,fpr=fp/neg,precision=tp/sel,f1=2*tp/(pos+sel))
        if not all(close(c[k],v) for k,v in rates.items()):
            issues.append(f'{group}: confusion metrics')
    if not np.array_equal(confusion[0],confusion[1]+confusion[2]):
        issues.append('classification: partition counts')
    sub = reg.iloc[1:]; overall = reg.iloc[0]; weights = sub.n
    for key in ['mae_gbp','mean_error_gbp','mean_actual_gbp','mean_predicted_gbp','zero_share']:
        if not close(overall[key],np.average(sub[key],weights=weights),2*tol):
            issues.append(f'regression: partition {key}')
    if not close(overall.rmse_gbp,np.sqrt(np.average(sub.rmse_gbp**2,weights=weights)),2*tol):
        issues.append('regression: partition RMSE')
    return issues
