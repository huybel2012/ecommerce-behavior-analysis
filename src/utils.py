"""Kiểm toán tổng hợp, không xuất giá trị định danh hay dòng mẫu."""
import pandas as pd
from src.pipeline import unknown_country_mask, missing_description_mask


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Báo cáo W3: chỉ số tổng hợp; không hiển thị dữ liệu từng người."""
    return pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })


def check_disguised_missing(df: pd.DataFrame, tokens=None) -> pd.DataFrame:
    """Token thiếu, không đếm lại NA thật; dùng đúng quy tắc theo cột."""
    generic = {"", "nan", "none", "null", "n/a", "?", "??", "unknown", "unspecified"}
    override = None if tokens is None else {str(t).strip().lower() for t in tokens}
    records = []
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        values = df[col].dropna()
        normalized = values.astype("string").str.strip().str.lower()
        name = str(col).lower()
        if override is not None:
            mask = normalized.isin(override)
        elif name in {"country", "country_original"}:
            mask = unknown_country_mask(values)
        elif name == "description":
            mask = missing_description_mask(values)
        else:
            mask = normalized.isin(generic)
        if mask.any():
            records.append({"column": col, "n_disguised": int(mask.sum())})
    return pd.DataFrame(records, columns=["column", "n_disguised"])
