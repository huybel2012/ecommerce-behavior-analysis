"""Các đặc trưng mô tả thống nhất cho EDA và W9 (không dùng toàn kỳ làm feature dự báo)."""
import numpy as np
import pandas as pd


def build_orders(df_valid):
    orders = df_valid.groupby(["customer_id", "invoice"], as_index=False).agg(
        invoice_date=("invoice_date", "min"), order_value=("total_amount", "sum"),
        date_count=("invoice_date", "nunique")
    )
    orders["is_weekend"] = orders["invoice_date"].dt.day_of_week >= 5
    orders["is_peak_retail_month"] = orders["invoice_date"].dt.month.isin([9, 10, 11, 12])
    return orders


def aggregate_orders(order_df):
    return order_df.groupby("customer_id", as_index=False).agg(
        first_purchase=("invoice_date", "min"), last_purchase=("invoice_date", "max"),
        frequency=("invoice", "nunique"), monetary=("order_value", "sum"),
        avg_order_value=("order_value", "mean"), weekend_ratio=("is_weekend", "mean"),
        peak_month_ratio=("is_peak_retail_month", "mean")
    )


def classify_market(df_valid):
    """Nhãn thị trường từ quốc gia quan sát trong mẫu hiện tại, không từ nội suy toàn kỳ."""
    country = df_valid["country_original"].astype("string").str.strip()
    unknown = country.isna() | country.str.lower().isin(["", "unspecified", "unknown", "nan", "none", "european community"])
    country_check = pd.DataFrame({"customer_id": df_valid["customer_id"],
                                  "known_country": country.mask(unknown), "unknown": unknown})
    geo = country_check.groupby("customer_id", as_index=False).agg(
        known_count=("known_country", "nunique"), country=("known_country", "first"),
        has_unknown=("unknown", "any")
    )
    geo["is_uk"] = np.select(
        [geo["known_count"].gt(1).to_numpy(dtype=bool),
         (geo["known_count"].eq(0) | geo["has_unknown"]).to_numpy(dtype=bool),
         geo["country"].eq("United Kingdom").fillna(False).to_numpy(dtype=bool)],
        ["Mixed", "Unknown", "UK"], default="International"
    )
    geo["country"] = geo["country"].fillna("Unknown")
    geo.loc[geo["is_uk"].eq("Mixed"), "country"] = "Mixed"
    geo.loc[geo["is_uk"].eq("Unknown"), "country"] = "Unknown"
    return geo[["customer_id", "country", "is_uk"]]


def build_customer_profiles(df):
    """Một dòng/khách có ít nhất một đơn mua hợp lệ, với các chỉ số toàn kỳ."""
    df_valid = df.loc[~df["is_cancelled"].astype(bool)].copy()
    orders = build_orders(df_valid)
    snapshot_date = df["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    cust_valid = aggregate_orders(orders)
    cust_valid["recency_days"] = (snapshot_date - cust_valid["last_purchase"]).dt.days
    cust_valid["tenure_days"] = (cust_valid["last_purchase"] - cust_valid["first_purchase"]).dt.days + 1
    cust_valid["observation_days"] = (snapshot_date - cust_valid["first_purchase"].dt.normalize()).dt.days
    cust_valid = cust_valid.merge(df_valid.groupby("customer_id", as_index=False).agg(
        total_items=("quantity", "sum"), unique_products=("stockcode", "nunique"),
        region=("region", lambda s: s.mode().iloc[0]),
        shipping_zone=("shipping_zone", lambda s: s.mode().iloc[0])
    ), on="customer_id", validate="one_to_one")

    cancel_stats = df.groupby("customer_id")["invoice"].nunique().rename("total_invoices").to_frame()
    cancel_stats["cancelled_invoices"] = df.loc[df["is_cancelled"].astype(bool)].groupby("customer_id")["invoice"].nunique()
    cancel_stats["cancelled_invoices"] = cancel_stats["cancelled_invoices"].fillna(0).astype(int)
    cancel_stats["cancel_rate"] = cancel_stats["cancelled_invoices"] / cancel_stats["total_invoices"]
    customer_df = cust_valid.merge(cancel_stats.reset_index(), on="customer_id", validate="one_to_one")
    customer_df = customer_df.merge(classify_market(df_valid), on="customer_id", validate="one_to_one")
    customer_df["is_repeat_buyer"] = (customer_df["frequency"] > 1).astype(int)
    assert customer_df["customer_id"].is_unique and customer_df.notna().all().all()
    assert customer_df[["weekend_ratio", "peak_month_ratio", "cancel_rate"]].apply(lambda s: s.between(0, 1).all()).all()
    return customer_df
