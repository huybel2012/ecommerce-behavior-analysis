import numpy as np
import pandas as pd
import pytest

from src.pipeline import clean_transactions, build_sklearn_preprocessor
from src.classification_features import (
    FEATURE_WHITELIST,
    build_cutoff_customer_dataset,
)


def _make_sample_raw_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "Invoice": ["500001", "500001", "500002", "C500003", "500004", "500005"],
        "StockCode": ["85123A", "71053", "84406B", "85123A", "85123A", "71053"],
        "Description": ["ITEM A", "ITEM B", "ITEM C", "ITEM A", "ITEM A", "ITEM B"],
        "Quantity": [6, 4, 10, -2, 12, 8],
        "InvoiceDate": [
            "2011-06-01 10:00:00",
            "2011-06-01 10:02:00",  # Cùng hóa đơn nhưng lệch 2 phút
            "2011-08-15 14:30:00",
            "2011-08-20 09:15:00",
            "2011-09-10 11:00:00",  # Sau cutoff 2011-09-01
            "2011-07-10 16:00:00",
        ],
        "Price": [2.5, 3.0, 1.5, 2.5, 2.5, 3.0],
        "Customer ID": [10001.0, 10001.0, 10001.0, 10001.0, 10001.0, 10002.0],
        "Country": ["United Kingdom", "United Kingdom", "Unspecified", "United Kingdom", "France", "Germany"],
    })


def test_clean_transactions_contract():
    raw = _make_sample_raw_transactions()
    cleaned, log_df = clean_transactions(raw)
    assert len(cleaned) == 6
    assert "Unknown" not in cleaned["country"].values
    assert not log_df.empty


def test_cutoff_no_temporal_leakage_and_id_exclusion():
    raw = _make_sample_raw_transactions()
    cleaned, _ = clean_transactions(raw)
    cleaned["region"] = cleaned["country"]
    cleaned["shipping_zone"] = "Short_Haul"
    cleaned["is_weekend"] = 0
    cleaned["is_peak_retail_month"] = 0

    dataset = build_cutoff_customer_dataset(cleaned, cutoff_date="2011-09-01")
    # Khách 10001 có đơn ngày 2011-09-10 -> target = 1, nhưng frequency trước cutoff phải là 2 (không tính đơn tháng 9)
    row_10001 = dataset[dataset["customer_id"] == "10001"].iloc[0]
    assert row_10001["frequency"] == 2
    assert row_10001["target_repeat_90d"] == 1
    # Quốc gia trước cutoff của 10001 là United Kingdom, không bị rò rỉ giá trị France sau cutoff
    assert row_10001["region"] == "United Kingdom"

    # Khách 10002 không mua sau cutoff -> target = 0
    row_10002 = dataset[dataset["customer_id"] == "10002"].iloc[0]
    assert row_10002["target_repeat_90d"] == 0

    # Kiểm tra preprocessor loại bỏ hoàn toàn customer_id và target_repeat_90d
    prep = build_sklearn_preprocessor(
        log_num_cols=FEATURE_WHITELIST["log_num_cols"],
        linear_num_cols=FEATURE_WHITELIST["linear_num_cols"],
        cat_cols=FEATURE_WHITELIST["cat_cols"],
    )
    X_trans = prep.fit_transform(dataset)
    assert np.isfinite(X_trans).all(), "Ma trận đặc trưng sau biến đổi chứa NaN hoặc Infinity."
    feature_names = prep.get_feature_names_out()
    assert not any("customer_id" in f or "target_repeat_90d" in f for f in feature_names)


def test_preprocessor_rejects_forbidden_or_negative_log_features():
    with pytest.raises(ValueError):
        build_sklearn_preprocessor(
            log_num_cols=["customer_id"],
            linear_num_cols=["cancel_rate"],
            cat_cols=["region"],
        )