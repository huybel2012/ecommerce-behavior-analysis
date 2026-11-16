"""Đặc trưng cấp khách hàng cho bài toán phân loại: khách có mua lại trong cửa sổ dự báo sau cutoff hay không."""
import os
import pandas as pd
import numpy as np


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
    """Thống nhất mỗi hóa đơn sử dụng mốc thời gian sớm nhất (min invoice_date)."""
    out = df.copy()
    out["invoice_ts"] = out.groupby("invoice")["invoice_date"].transform("min")
    return out


def build_cutoff_customer_dataset(
    df: pd.DataFrame,
    cutoff_date: str = "2011-09-01",
    prediction_horizon_days: int = 99,
) -> pd.DataFrame:
    """
    Xây dựng tập dữ liệu huấn luyện phân loại không rò rỉ dữ liệu (Leakage-free):
    - Đặc trưng X: chỉ tổng hợp từ giao dịch trước cutoff_date.
    - Nhãn y (target_repeat_90d): khách hàng có quay lại mua trong cửa sổ dự báo sau cutoff_date hay không.
    """
    df = df.copy()
    if "line_amount" not in df.columns:
        # pipeline.py tạo total_amount = quantity * price
        df["line_amount"] = df["total_amount"]
    df_ts = unify_invoice_timestamps(df)
    cutoff_ts = pd.Timestamp(cutoff_date)
    horizon_end_ts = cutoff_ts + pd.Timedelta(days=prediction_horizon_days)

    hist_df = df_ts[df_ts["invoice_ts"] < cutoff_ts].copy()
    future_df = df_ts[
        (df_ts["invoice_ts"] >= cutoff_ts)
        & (df_ts["invoice_ts"] <= horizon_end_ts)
        & (~df_ts["is_cancelled"])
    ].copy()

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
        is_weekend=("is_weekend", "max"),
        is_peak_month=("is_peak_retail_month", "max"),
    )
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

    # Thông tin địa lý chỉ lấy từ dữ liệu lịch sử trước cutoff_date (không nội suy từ tương lai)
    hist_geo = hist_df.groupby("customer_id", as_index=False).agg(
        region=("region", lambda s: s.mode().iloc[0]),
        shipping_zone=("shipping_zone", lambda s: s.mode().iloc[0]),
    )
    cust_features = cust_features.merge(hist_geo, on="customer_id", how="left")

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
    return cust_features


def run_feature_engineering(
    processed_path: str = "data/processed/retail_processed.parquet",
    output_path: str = "data/processed/customer_modeling_dataset.parquet",
    cutoff_date: str = "2011-09-01",
):
    df = pd.read_parquet(processed_path)
    dataset = build_cutoff_customer_dataset(df, cutoff_date=cutoff_date)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    pos_rate = dataset["target_repeat_90d"].mean() * 100
    print(f"Đã tạo tập dữ liệu mô hình hóa theo mốc cutoff {cutoff_date}:")
    print(f"  - Số lượng khách hàng: {len(dataset):,}")
    print(f"  - Số lượng cột: {dataset.shape[1]}")
    print(f"  - Tỷ lệ khách quay lại mua hàng (target_repeat_90d = 1): {pos_rate:.2f}%")


if __name__ == "__main__":
    run_feature_engineering()