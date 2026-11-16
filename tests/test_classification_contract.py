import numpy as np
import pandas as pd
import pytest

from src.pipeline import clean_transactions, integrate_external_sources, build_sklearn_preprocessor
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

    dataset = build_cutoff_customer_dataset(cleaned, cutoff_date="2011-09-01", observed_until="2011-12-09")
    # Khách 10001 có đơn ngày 2011-09-10 -> target = 1, nhưng frequency trước cutoff phải là 2 (không tính đơn tháng 9)
    row_10001 = dataset[dataset["customer_id"] == "10001"].iloc[0]
    assert row_10001["frequency"] == 2
    assert row_10001["target_repeat_90d"] == 1
    # Quốc gia trước cutoff của 10001 là United Kingdom, không bị rò rỉ giá trị France sau cutoff
    assert row_10001["region"] == "Domestic_UK"

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

CUTOFF = "2011-09-01"
COVERAGE = "2011-12-09"


def _row(invoice, timestamp, customer=1, qty=1, country="United Kingdom", price=2.0):
    return [invoice, "85123A", "ITEM", qty, timestamp, price, customer, country]


def _processed(rows):
    from src.pipeline import RAW_COLUMNS
    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], format="mixed")
    cleaned, _ = clean_transactions(raw)
    return integrate_external_sources(cleaned)


def _dataset(rows, **kwargs):
    args = {"cutoff_date": CUTOFF, "observed_until": COVERAGE, **kwargs}
    return build_cutoff_customer_dataset(_processed(rows), **args).set_index("customer_id")


def test_future_country_changes_no_historical_feature():
    past = [_row("A", "2011-08-01", country="Unspecified")]
    a = _dataset(past + [_row("B", "2011-09-02", country="France")])
    b = _dataset(past + [_row("B", "2011-09-02", country="Australia")])
    pd.testing.assert_frame_equal(a, b)
    assert a.loc["1", "region"] == "Unknown"
    assert a.loc["1", "shipping_zone"] == "Unknown"


def test_known_country_in_history_only():
    a = _dataset([_row("A", "2011-08-01", country="Unspecified"),
                  _row("B", "2011-08-02", country="France"),
                  _row("C", "2011-09-02", country="Australia")])
    assert a.loc["1", "region"] == "Western_Europe"
    assert a.loc["1", "shipping_zone"] == "Short_Haul"


def test_cross_cutoff_invoice_no_leakage_or_false_repeat():
    past = [_row("A", "2011-08-31 23:59:00")]
    a = _dataset(past)
    b = _dataset(past + [_row("A", "2011-09-01 00:01:00", qty=999)])
    pd.testing.assert_frame_equal(a, b)
    assert b.loc["1", "total_items"] == 1
    assert b.loc["1", "gross_valid_revenue"] == 2
    assert b.loc["1", "target_repeat_90d"] == 0
    assert b.loc["1", "peak_month_ratio"] == 0
    assert b.loc["1", "recency_days"] == pytest.approx(1 / 1440)


def test_invoice_id_is_scoped_to_customer_and_flags_use_invoice_start():
    a = _dataset([_row("A", "2011-07-29 23:59:00", customer=1),
                  _row("A", "2011-07-30 00:01:00", customer=1, qty=2),
                  _row("B", "2011-08-01", customer=2),
                  _row("A", "2011-09-01", customer=2)])
    assert a.loc["1", "weekend_ratio"] == 0
    assert a.loc["1", "frequency"] == 1
    assert a.loc["2", "frequency"] == 1
    assert a.loc["2", "target_repeat_90d"] == 1


@pytest.mark.parametrize("date,label", [("2011-09-01", 1), ("2011-11-29 23:59:59", 1),
                                         ("2011-11-30", 0), ("2011-12-02", 0)])
def test_horizon_half_open(date, label):
    a = _dataset([_row("A", "2011-08-01"), _row("B", date)])
    assert a.loc["1", "target_repeat_90d"] == label


def test_horizon_rejects_incomplete_coverage_and_wrong_length():
    rows = [_row("A", "2011-08-01")]
    with pytest.raises(ValueError, match="Thiếu cửa sổ"):
        _dataset(rows, observed_until="2011-11-29")
    with pytest.raises(ValueError, match="90 ngày"):
        _dataset(rows, prediction_horizon_days=99)
    assert _dataset(rows, observed_until="2011-11-30").loc["1", "target_repeat_90d"] == 0


@pytest.mark.parametrize("value", [-1.0, -2.0, np.inf, -np.inf])
def test_preprocessor_invalid_log_values(value):
    prep = build_sklearn_preprocessor(log_num_cols=["x"], cat_cols=[])
    with pytest.raises(ValueError):
        prep.fit_transform(pd.DataFrame({"x": [1.0, value]}))


def test_preprocessor_excludes_extra_columns_and_forbids_all_targets():
    frame = pd.DataFrame({"x": [0., 1.], "z": [-2., 1.], "customer_id": [1, 2],
                          "target_spend_90d": [100., 200.], "invoice_date": pd.to_datetime(["2011-08-01"]*2)})
    prep = build_sklearn_preprocessor(log_num_cols=["x"], linear_num_cols=["z"], cat_cols=[])
    assert np.isfinite(prep.fit_transform(frame)).all()
    assert list(prep.get_feature_names_out()) == ["log_num__x", "linear_num__z"]
    with pytest.raises(ValueError):
        build_sklearn_preprocessor(log_num_cols=["target_spend_90d"], cat_cols=[])


@pytest.mark.parametrize("column", ["Price", "Quantity"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_cleaning_directly_rejects_invalid_numeric(column, value):
    raw = _make_sample_raw_transactions()
    raw[column] = raw[column].astype(float)
    raw.loc[0, column] = value
    with pytest.raises(ValueError):
        clean_transactions(raw)


def test_runner_rebuilds_from_raw_even_if_model_output_exists(tmp_path, monkeypatch):
    """Test orchestration without requiring a parquet engine; real IO tested separately."""
    from src.classification_features import run_feature_engineering
    import src.classification_features as features
    processed = tmp_path / "processed"
    processed.mkdir()
    model_path = processed / "customer_modeling_dataset.parquet"
    model_path.write_text("old cache")
    calls = []
    fresh = _processed([_row("A", "2011-08-01"), _row("B", "2011-09-02")])

    def mock_pipeline(raw_dir, processed_dir, reports_dir):
        calls.append((raw_dir, processed_dir, reports_dir))
        return pd.Timestamp(COVERAGE)

    def mock_read(path):
        assert calls and path == processed / "retail_processed.parquet"
        return fresh.copy()

    written = []
    def mock_write(frame, path, index=False):
        written.append((frame.copy(), path))

    monkeypatch.setattr(features, "run_pipeline", mock_pipeline)
    monkeypatch.setattr(pd, "read_parquet", mock_read)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_write)
    for _ in range(2):
        result = run_feature_engineering(tmp_path / "raw", processed, tmp_path / "reports")
        assert result.loc[0, "target_repeat_90d"] == 1
    assert len(calls) == len(written) == 2
    assert all(path == model_path for _, path in written)


def test_real_parquet_roundtrip_from_raw(tmp_path):
    pytest.importorskip("pyarrow")
    from src.classification_features import run_feature_engineering
    from src.pipeline import RAW_COLUMNS
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [_row("A", "2011-08-01"), _row("B", "2011-09-02"),
            _row("C", "2011-12-09", customer=2)]
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(raw / "online_retail_II.csv", index=False)
    result = run_feature_engineering(raw, tmp_path / "processed", tmp_path / "reports")
    saved = pd.read_parquet(tmp_path / "processed/customer_modeling_dataset.parquet")
    pd.testing.assert_frame_equal(result, saved)
    assert result.loc[0, "target_repeat_90d"] == 1
