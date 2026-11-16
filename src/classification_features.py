"""Đặc trưng cấp khách hàng cho bài toán phân loại: khách có mua lại trong cửa sổ dự báo sau cutoff hay không."""
from pathlib import Path
import pandas as pd
import numpy as np

from src.pipeline import get_country_metadata, run_pipeline, unknown_country_mask

PREDICTION_HORIZON_DAYS = 90
INVOICE_KEYS = ["customer_id", "invoice"]


FEATURE_WHITELIST = {
    "log_num_cols": [
        "recency_days",
        "observation_span_days",
        "frequency",
        "total_items",
        "unique_products",
        "gross_valid_revenue",
        "avg_order_value",
        "avg_basket_size",
        "purchase_rate_per_30d",
        "interpurchase_mean_days",
        "cancelled_invoices",
    ],
    "linear_num_cols": [
        "weekend_ratio",
        "peak_month_ratio",
        "cancel_rate",
        "items_per_sku",
    ],
    "cat_cols": [
        "region",
        "shipping_zone",
    ],
}


def unify_invoice_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Gom timestamp theo cặp khách/hóa đơn, chỉ trong cửa sổ đã cắt."""
    out = df.copy()
    out["invoice_ts"] = out.groupby(INVOICE_KEYS)["invoice_date"].transform("min")
    return out


def _historical_geography(hist_df: pd.DataFrame) -> pd.DataFrame:
    """Chọn quốc gia quan sát phổ biến nhất trước cutoff; ánh xạ cả hai cột từ cùng quốc gia."""
    observed = hist_df["country_original"].astype("string").str.strip()
    unknown = unknown_country_mask(observed)
    known = hist_df.loc[~unknown, ["customer_id"]].copy()
    known["country"] = observed.loc[~unknown]
    # mode() chọn theo thứ tự xác định nếu nhiều quốc gia đồng hạng.
    modes = known.groupby("customer_id")["country"].agg(lambda s: s.mode().iloc[0])
    geo = hist_df[["customer_id"]].drop_duplicates().copy()
    geo["country"] = geo["customer_id"].map(modes).astype("string")
    geo = geo.merge(get_country_metadata()[["country", "region", "shipping_zone"]],
                    on="country", how="left", validate="many_to_one")
    geo["region"] = geo["region"].fillna("Other_International")
    geo["shipping_zone"] = geo["shipping_zone"].fillna("Other")
    geo.loc[geo["country"].isna(), ["region", "shipping_zone"]] = "Unknown"
    return geo[["customer_id", "region", "shipping_zone"]]


def build_cutoff_customer_dataset(
    df: pd.DataFrame,
    cutoff_date: str = "2011-09-01",
    prediction_horizon_days: int = PREDICTION_HORIZON_DAYS,
    *,
    observed_until: str | pd.Timestamp,
) -> pd.DataFrame:
    """X từ dòng quan sát trước cutoff; y từ đơn mua mới trong [cutoff, cutoff+90d).

    observed_until là ranh giới quan sát đáng tin cậy của nguồn gốc, không phải
    ngày cuối sau làm sạch. Không tạo nhãn nếu nguồn chưa phủ đủ cửa sổ.
    Hóa đơn nằm hai phía cutoff: chỉ dùng dòng lịch sử đã quan sát; các dòng
    tiếp nối của cùng (customer_id, invoice) không được tính là đơn mua mới.
    """
    if prediction_horizon_days != PREDICTION_HORIZON_DAYS:
        raise ValueError("target_repeat_90d yêu cầu horizon đúng 90 ngày.")
    cutoff_ts = pd.Timestamp(cutoff_date)
    coverage_end = pd.Timestamp(observed_until)
    if pd.isna(cutoff_ts) or pd.isna(coverage_end):
        raise ValueError("cutoff và observed_until phải là timestamp hợp lệ.")
    horizon_end_ts = cutoff_ts + pd.Timedelta(days=PREDICTION_HORIZON_DAYS)
    if coverage_end < horizon_end_ts:
        raise ValueError(f"Thiếu cửa sổ quan sát nhãn: cần đến {horizon_end_ts}, chỉ có đến {coverage_end}.")

    required = set(INVOICE_KEYS + ["invoice_date", "is_cancelled", "country_original",
                                  "quantity", "stockcode", "price", "total_amount"])
    if missing := required.difference(df.columns):
        raise ValueError(f"Thiếu cột: {sorted(missing)}; chạy lại pipeline từ dữ liệu gốc.")
    df = df.reset_index(drop=True).copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="raise")
    if df[INVOICE_KEYS + ["invoice_date", "is_cancelled"]].isna().any().any():
        raise ValueError("Thiếu khóa, timestamp hoặc trạng thái giao dịch.")
    if not df["is_cancelled"].isin([True, False]).all():
        raise ValueError("is_cancelled phải là cờ boolean.")
    if (df["invoice_date"] > coverage_end).any():
        raise ValueError("Dữ liệu có timestamp sau observed_until; kiểm tra lại nguồn.")
    for col in INVOICE_KEYS + ["stockcode"]:
        if df[col].isna().any() or df[col].astype("string").str.strip().eq("").any():
            raise ValueError(f"Khóa {col} thiếu hoặc rỗng.")
    for col in ["quantity", "price", "total_amount"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    values = df[["quantity", "price", "total_amount"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Quantity/price/total_amount phải hữu hạn.")
    df["is_cancelled"] = df["is_cancelled"].astype(bool)
    if not df["is_cancelled"].eq(df["invoice"].astype("string").str.startswith("C")).all():
        raise ValueError("is_cancelled không khớp mã hóa đơn.")
    q, price, amount = values.T
    if (price <= 0).any() or (q != np.floor(q)).any() or np.any(np.where(df["is_cancelled"], q >= 0, q <= 0)):
        raise ValueError("Giá hoặc dấu số lượng không hợp lệ.")
    if not np.allclose(amount, q * price, rtol=1e-9, atol=1e-8):
        raise ValueError("total_amount không khớp quantity * price.")
    df["line_amount"] = df["total_amount"]

    # Cắt bằng timestamp gốc TRƯỚC khi gom hóa đơn. Không đưa dòng tương lai về quá khứ.
    hist_df = unify_invoice_timestamps(df.loc[df["invoice_date"] < cutoff_ts])
    future_df = df.loc[(df["invoice_date"] >= cutoff_ts)
                       & (df["invoice_date"] < horizon_end_ts)
                       & (~df["is_cancelled"])].copy()
    past_keys = pd.MultiIndex.from_frame(hist_df[INVOICE_KEYS].drop_duplicates())
    future_keys = pd.MultiIndex.from_frame(future_df[INVOICE_KEYS])
    future_df = future_df.loc[~future_keys.isin(past_keys)]

    # Chỉ giữ khách hàng đã từng có ít nhất 1 đơn mua hợp lệ trước cutoff_date
    hist_valid = hist_df[~hist_df["is_cancelled"]].copy()
    if hist_valid.empty:
        raise ValueError("Không có giao dịch hợp lệ nào trước mốc cutoff_date.")

    # Tổng hợp cấp độ hóa đơn trước cutoff để tính chính xác hành vi đơn hàng
    inv_valid = hist_valid.groupby(["customer_id", "invoice"], as_index=False).agg(
        invoice_ts=("invoice_ts", "min"),
        invoice_revenue=("line_amount", "sum"),
        invoice_items=("quantity", "sum"),
        invoice_skus=("stockcode", "nunique"),
    )
    inv_valid["is_weekend"] = (inv_valid["invoice_ts"].dt.dayofweek >= 5).astype(int)
    inv_valid["is_peak_month"] = inv_valid["invoice_ts"].dt.month.isin([9, 10, 11, 12]).astype(int)
    inv_valid = inv_valid.sort_values(["customer_id", "invoice_ts"]).reset_index(drop=True)
    inv_valid["prev_invoice_ts"] = inv_valid.groupby("customer_id")["invoice_ts"].shift(1)
    inv_valid["days_since_prev"] = (
        (inv_valid["invoice_ts"] - inv_valid["prev_invoice_ts"]).dt.total_seconds() / 86400.0
    )

    cust_features = inv_valid.groupby("customer_id", as_index=False).agg(
        first_purchase_ts=("invoice_ts", "min"),
        last_purchase_ts=("invoice_ts", "max"),
        frequency=("invoice", "nunique"),
        gross_valid_revenue=("invoice_revenue", "sum"),
        avg_order_value=("invoice_revenue", "mean"),
        total_items=("invoice_items", "sum"),
        avg_basket_size=("invoice_items", "mean"),
        weekend_ratio=("is_weekend", "mean"),
        peak_month_ratio=("is_peak_month", "mean"),
        interpurchase_mean_days=("days_since_prev", "mean"),
    )

    unique_skus = hist_valid.groupby("customer_id", as_index=False).agg(
        unique_products=("stockcode", "nunique")
    )
    cust_features = cust_features.merge(unique_skus, on="customer_id", how="left")

    # Tính Recency và thời lượng quan sát theo đúng mốc cutoff_date
    cust_features["recency_days"] = (
        (cutoff_ts - cust_features["last_purchase_ts"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)
    cust_features["observation_span_days"] = (
        (cutoff_ts - cust_features["first_purchase_ts"]).dt.total_seconds() / 86400.0
    ).clip(lower=1.0)

    # Đặc trưng tỷ suất và độ đa dạng giỏ hàng
    cust_features["purchase_rate_per_30d"] = (
        cust_features["frequency"] / cust_features["observation_span_days"]
    ) * 30.0
    cust_features["items_per_sku"] = (
        cust_features["total_items"] / cust_features["unique_products"].clip(lower=1)
    )
    # Với khách chỉ mua 1 đơn trước cutoff, khoảng cách giữa các lần mua điền bằng thời lượng quan sát
    cust_features["interpurchase_mean_days"] = cust_features["interpurchase_mean_days"].fillna(
        cust_features["observation_span_days"]
    )

    # Chỉ dùng country_original trong lịch sử, không dùng region đã nội suy toàn kỳ.
    hist_geo = _historical_geography(hist_df)
    cust_features = cust_features.merge(hist_geo, on="customer_id", how="left", validate="one_to_one")

    # Thống kê đơn hủy trước cutoff_date
    hist_cancel = hist_df.groupby("customer_id", as_index=False).agg(
        total_hist_invoices=("invoice", "nunique"),
        cancelled_invoices=("invoice", lambda s: s[hist_df.loc[s.index, "is_cancelled"]].nunique()),
    )
    hist_cancel["cancel_rate"] = (
        hist_cancel["cancelled_invoices"] / hist_cancel["total_hist_invoices"].clip(lower=1)
    )
    cust_features = cust_features.merge(
        hist_cancel[["customer_id", "cancelled_invoices", "cancel_rate"]],
        on="customer_id",
        how="left",
    )

    # Tạo biến mục tiêu từ cửa sổ tương lai sau cutoff_date
    repeat_buyers = set(future_df["customer_id"].unique())
    cust_features["target_repeat_90d"] = cust_features["customer_id"].isin(repeat_buyers).astype(int)

    # Loại bỏ cột timestamp trung gian để giữ bảng đặc trưng sạch
    cust_features = cust_features.drop(columns=["first_purchase_ts", "last_purchase_ts"])
    feature_cols = sum(FEATURE_WHITELIST.values(), [])
    if cust_features["customer_id"].duplicated().any() or cust_features[feature_cols].isna().any().any():
        raise ValueError("Modeling dataset vi phạm tính duy nhất hoặc thiếu feature.")
    return cust_features


def run_feature_engineering(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    reports_dir: str | Path = "reports",
    cutoff_date: str = "2011-09-01",
) -> pd.DataFrame:
    """Luôn chạy lại nguồn -> pipeline -> modeling dataset; không dùng cache cũ."""
    processed = Path(processed_dir)
    observed_until = run_pipeline(raw_dir, processed, reports_dir)
    df = pd.read_parquet(processed / "retail_processed.parquet")
    dataset = build_cutoff_customer_dataset(df, cutoff_date=cutoff_date,
                                             observed_until=observed_until)
    output = processed / "customer_modeling_dataset.parquet"
    dataset.to_parquet(output, index=False)
    print(f"Cutoff: {cutoff_date} | Horizon: 90 ngày | Nguồn quan sát đến: {observed_until}")
    print(f"Khách hàng: {len(dataset):,} | Tỷ lệ mua lại: {dataset['target_repeat_90d'].mean():.2%}")
    return dataset


if __name__ == "__main__":
    run_feature_engineering()
