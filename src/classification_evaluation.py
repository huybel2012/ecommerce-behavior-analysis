"""W11: chọn mô hình trên train, ngưỡng trên validation, báo cáo trên cutoff sau."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, precision_recall_curve)
from src.pipeline import build_sklearn_preprocessor
from src.classification_features import (FEATURE_WHITELIST, build_cutoff_customer_dataset,
                                         PREDICTION_HORIZON_DAYS)

CUTOFFS = ('2011-03-01', '2011-06-01', '2011-09-01')
FEATURES = sum(FEATURE_WHITELIST.values(), [])
TARGET = 'target_repeat_90d'


def temporal_datasets(df, observed_until, cutoffs=CUTOFFS, *, include_spend_target=False):
    """Ba snapshot riêng, nhãn train/validation đã quan sát xong trước mốc tiếp theo."""
    dates = tuple(pd.Timestamp(c) for c in cutoffs)
    if len(dates) != 3 or any(pd.isna(c) for c in dates):
        raise ValueError('Cần đúng ba cutoff hợp lệ.')
    delta = pd.Timedelta(days=PREDICTION_HORIZON_DAYS)
    if any(a+delta > b for a,b in zip(dates, dates[1:])):
        raise ValueError('Cửa sổ nhãn không được chồng lấn, phải theo thứ tự thời gian.')
    return {name: build_cutoff_customer_dataset(df, c, observed_until=observed_until,
                                                include_spend_target=include_spend_target)
            for name,c in zip(('train','validation','test'), dates)}


def split_xy(data):
    if data['customer_id'].duplicated().any():
        raise ValueError('Mỗi snapshot phải có đúng một dòng trên mỗi khách hàng.')
    y = data[TARGET]
    if set(y.unique()) != {0, 1}:
        raise ValueError('Mỗi snapshot cần có cả hai lớp để đánh giá ROC-AUC.')
    return data[FEATURES].copy(), y.to_numpy(dtype=int)


def candidate_models():
    return {
        'Dummy Baseline': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'k-NN': KNeighborsClassifier(n_neighbors=15, weights='distance'),
        'Gaussian Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                                class_weight='balanced', random_state=42, n_jobs=1),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=200, max_depth=5,
                                                              learning_rate=.05, random_state=42),
    }


def make_pipeline(model):
    return Pipeline([('prep', build_sklearn_preprocessor(**FEATURE_WHITELIST)), ('clf', clone(model))])


def select_model(X_train, y_train, models=None):
    """Primary: mean ROC-AUC CV. Tie: AP, rồi tên. Không dùng validation/test."""
    if models is None:
        models = candidate_models()
    counts = pd.Series(y_train).value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError('Train cần ít nhất hai mẫu mỗi lớp để CV.')
    folds = int(min(5, counts.min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    # Tránh nested parallelism quá tải RAM trên laptop Windows.
    rows = []
    for name, model in models.items():
        pipe = make_pipeline(model)
        # KNN không thể dùng k > kích thước fold train.
        if isinstance(model, KNeighborsClassifier):
            min_train = len(y_train) - int(np.ceil(len(y_train)/folds))
            pipe.set_params(clf__n_neighbors=min(model.n_neighbors, min_train))
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring={'roc_auc':'roc_auc','ap':'average_precision',
                                         'f1':'f1','accuracy':'accuracy'},
                                n_jobs=1, error_score='raise')
        rows.append({'model':name,'roc_auc_mean':scores['test_roc_auc'].mean(),
                     'roc_auc_sd':scores['test_roc_auc'].std(ddof=1),
                     'ap_mean':scores['test_ap'].mean(),'f1_mean':scores['test_f1'].mean(),
                     'accuracy_mean':scores['test_accuracy'].mean(),'folds':folds})
    table = pd.DataFrame(rows).sort_values(['roc_auc_mean','ap_mean','model'], ascending=[False,False,True]).reset_index(drop=True)
    winner = table.iloc[0]['model']
    pipe = make_pipeline(models[winner])
    if isinstance(models[winner], KNeighborsClassifier):
        pipe.set_params(clf__n_neighbors=min(models[winner].n_neighbors, len(y_train)-int(np.ceil(len(y_train)/folds))))
    return table, winner, pipe


def choose_threshold(y_validation, probabilities):
    """F1 trên validation; hòa chọn threshold cao hơn để hạn chế FP."""
    if set(np.unique(y_validation)) != {0,1}:
        raise ValueError('Validation cần cả hai lớp.')
    precision, recall, thresholds = precision_recall_curve(y_validation, probabilities)
    f1 = np.divide(2*precision[:-1]*recall[:-1], precision[:-1]+recall[:-1],
                   out=np.zeros_like(thresholds), where=(precision[:-1]+recall[:-1]) > 0)
    idx = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[idx]), float(f1[idx])


def evaluate_probabilities(y, p, threshold):
    pred = (np.asarray(p) >= threshold).astype(int)
    return {'threshold':threshold,'accuracy':accuracy_score(y,pred),
            'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),
            'f1':f1_score(y,pred,zero_division=0),'roc_auc':roc_auc_score(y,p),
            'average_precision':average_precision_score(y,p), 'prevalence':np.mean(y)}
