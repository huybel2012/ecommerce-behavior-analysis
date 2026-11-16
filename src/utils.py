import pandas as pd
import numpy as np

def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Báo cáo chất lượng dữ liệu một lần nhìn cho mọi DataFrame (Week 3)."""
    return pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
        "sample": df.iloc[0] if len(df) > 0 else None,
    })

def check_disguised_missing(df: pd.DataFrame, tokens=None) -> pd.DataFrame:
    """Phát hiện các giá trị thiếu được ngụy trang dưới dạng chuỗi thông thường (Week 3)."""
    if tokens is None:
        tokens = ["?", "??", "nan", "NaN", "None", "null", "N/A", "n/a", "", " ", "Unspecified", "unknown", "check", "damaged"]
    records = []
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        s = df[col].dropna().astype(str).str.strip()
        mask = s.isin(tokens) | (s == "")
        if mask.sum() > 0:
            records.append({
                "column": col,
                "n_disguised": int(mask.sum()),
                "examples": s[mask].unique()[:5].tolist()
            })
    return pd.DataFrame(records)