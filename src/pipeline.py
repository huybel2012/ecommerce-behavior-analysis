import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer


def get_country_metadata() -> pd.DataFrame:
    """Bảng quy ước phân nhóm thủ công của dự án; xem docs/data_sources.md."""
    data = [
        # Domestic
        ("United Kingdom", "Domestic_UK", "UK", "GBP", "Domestic"),
        # Western Europe
        ("France", "Western_Europe", "EU", "EUR", "Short_Haul"),
        ("Germany", "Western_Europe", "EU", "EUR", "Short_Haul"),
        ("EIRE", "Western_Europe", "EU", "EUR", "Short_Haul"),
        ("Netherlands", "Western_Europe", "EU", "EUR", "Short_Haul"),
        ("Belgium", "Western_Europe", "EU", "EUR", "Short_Haul"),
        ("Switzerland", "Western_Europe", "Non_EU_Europe", "CHF", "Short_Haul"),
        ("Austria", "Western_Europe", "EU", "EUR", "Short_Haul"),
        ("Luxembourg", "Western_Europe", "EU", "EUR", "Short_Haul"),
        # Northern Europe
        ("Norway", "Northern_Europe", "Non_EU_Europe", "NOK", "Short_Haul"),
        ("Sweden", "Northern_Europe", "EU", "SEK", "Short_Haul"),
        ("Denmark", "Northern_Europe", "EU", "DKK", "Short_Haul"),
        ("Finland", "Northern_Europe", "EU", "EUR", "Short_Haul"),
        ("Iceland", "Northern_Europe", "Non_EU_Europe", "ISK", "Short_Haul"),
        # Southern Europe
        ("Spain", "Southern_Europe", "EU", "EUR", "Short_Haul"),
        ("Portugal", "Southern_Europe", "EU", "EUR", "Short_Haul"),
        ("Italy", "Southern_Europe", "EU", "EUR", "Short_Haul"),
        ("Greece", "Southern_Europe", "EU", "EUR", "Short_Haul"),
        ("Cyprus", "Southern_Europe", "EU", "EUR", "Short_Haul"),
        ("Malta", "Southern_Europe", "EU", "EUR", "Short_Haul"),
        # Eastern Europe
        ("Poland", "Eastern_Europe", "EU", "PLN", "Short_Haul"),
        ("Czech Republic", "Eastern_Europe", "EU", "CZK", "Short_Haul"),
        ("Lithuania", "Eastern_Europe", "EU", "EUR", "Short_Haul"),
        # Asia-Pacific & Middle East
        ("Japan", "Asia_Pacific", "APAC", "JPY", "Long_Haul"),
        ("Australia", "Asia_Pacific", "APAC", "AUD", "Long_Haul"),
        ("Singapore", "Asia_Pacific", "APAC", "SGD", "Long_Haul"),
        ("Hong Kong", "Asia_Pacific", "APAC", "HKD", "Long_Haul"),
        ("Korea", "Asia_Pacific", "APAC", "KRW", "Long_Haul"),
        ("Thailand", "Asia_Pacific", "APAC", "THB", "Long_Haul"),
        ("United Arab Emirates", "Middle_East", "MENA", "AED", "Long_Haul"),
        ("Israel", "Middle_East", "MENA", "ILS", "Long_Haul"),
        ("Bahrain", "Middle_East", "MENA", "BHD", "Long_Haul"),
        ("Saudi Arabia", "Middle_East", "MENA", "SAR", "Long_Haul"),
        ("Lebanon", "Middle_East", "MENA", "LBP", "Long_Haul"),
        # Americas & Africa
        ("USA", "North_America", "Americas", "USD", "Long_Haul"),
        ("Canada", "North_America", "Americas", "CAD", "Long_Haul"),
        ("Brazil", "South_America", "Americas", "BRL", "Long_Haul"),
        ("RSA", "Africa", "Africa", "ZAR", "Long_Haul"),
        ("Nigeria", "Africa", "Africa", "NGN", "Long_Haul"),
    ]
    return pd.DataFrame(
        data,
        columns=["country", "region", "trade_bloc", "currency", "shipping_zone"],
    )


def get_peak_season_calendar() -> pd.DataFrame:
    """Quy ước mùa vụ và tháng cao điểm do nhóm chọn, không phải dữ liệu thực nghiệm độc lập."""
    return pd.DataFrame({
        "month": list(range(1, 13)),
        "season": [
            "Winter", "Winter", "Spring", "Spring", "Spring", "Summer",
            "Summer", "Summer", "Autumn", "Autumn", "Autumn", "Winter"
        ],
        "is_peak_retail_month": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    })


RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


NON_PRODUCT_CODES = [
    "POST", "D", "M", "BANK CHARGES", "AMAZONFEE", "DOT",
    "CRUK", "S", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002"
]


UNKNOWN_COUNTRIES = {"", "unspecified", "unknown", "nan", "none", "n/a", "null", "european community"}

def unknown_country_mask(values):
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isna() | normalized.isin(UNKNOWN_COUNTRIES)


def validate_raw_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Kiểm tra schema và kiểu dữ liệu trước khi làm sạch; cho phép thiếu CID/mô tả."""
    if df.empty or df.columns.has_duplicates or set(df.columns) != set(RAW_COLUMNS):
        raise ValueError(f"{source}: cần bảng không rỗng với đúng 8 cột {RAW_COLUMNS}; nhận {list(df.columns)}")
    df = df[RAW_COLUMNS].copy()
    for col in ["Quantity", "Price", "Customer ID"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        values = df[col].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: {col} chứa giá trị không hữu hạn.")
        if col in ["Quantity", "Customer ID"] and not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{source}: {col} phải là số nguyên.")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="raise")
    required = ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate", "Country"]
    if df[required].isna().any().any():
        raise ValueError(f"{source}: thiếu giá trị ở cột giao dịch bắt buộc.")
    for col in ["Invoice", "StockCode", "Country"]:
        if df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{source}: {col} có chuỗi rỗng.")
    return df


def load_raw_data(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Ưu tiên CSV UTF-8; chỉ chấp nhận tên file Online Retail II đã biết."""
    from pathlib import Path
    directory = Path(raw_dir)
    csv_path = directory / "online_retail_II.csv"
    excel_path = directory / "online_retail_II.xlsx"
    if csv_path.is_file():
        df = validate_raw_data(pd.read_csv(csv_path, encoding="utf-8-sig"), str(csv_path))
    elif excel_path.is_file():
        sheets = ["Year 2009-2010", "Year 2010-2011"]
        parts = pd.read_excel(excel_path, sheet_name=sheets)
        df = pd.concat([validate_raw_data(parts[name], f"{excel_path.name}: {name}")
                        for name in sheets], ignore_index=True)
    else:
        raise FileNotFoundError(f"Cần {csv_path} hoặc {excel_path}; không tự chọn file khác.")
    print(f"Nguồn: {csv_path.name if csv_path.is_file() else excel_path.name} | {len(df):,} dòng")
    return df


def clean_transactions(df_raw: pd.DataFrame, *, drop_exact_duplicates: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Làm sạch dữ liệu giao dịch và ghi nhật ký xử lý (Cleaning Log)."""
    df = validate_raw_data(df_raw, "clean_transactions")
    logs = []

    def add_log(step, col, finding, action, rows, why):
        logs.append({
            "Step": step,
            "Column": col,
            "Finding": finding,
            "Action": action,
            "Rows": int(rows),
            "Why": why,
        })

    # 1. Chuẩn hóa tên cột và kiểu dữ liệu
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["invoice_date"] = pd.to_datetime(df["invoicedate"])
    df = df.drop(columns=["invoicedate"])
    for col in ["invoice", "stockcode", "country"]:
        df[col] = df[col].astype(str).str.strip()
    df["description"] = df["description"].astype("string").str.strip()
    add_log("1.1", "all_columns", "Tên cột chưa đồng nhất, InvoiceDate dạng chuỗi",
            "Chuẩn hóa snake_case, ép kiểu datetime và loại bỏ khoảng trắng thừa",
            len(df), "Đảm bảo tính nhất quán khi trích xuất thời gian và ghép bảng")

    # 2. Xóa trùng lặp hoàn toàn
    n_dups = df.duplicated().sum()
    if drop_exact_duplicates:
        df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    add_log("1.2", "all", f"{n_dups:,} dòng trùng lặp hoàn toàn",
            "Xóa các dòng trùng lặp, giữ bản ghi đầu" if drop_exact_duplicates else "Giữ dòng trùng (kịch bản độ nhạy)",
            n_dups if drop_exact_duplicates else 0, "Giả định các dòng giống hệt là trùng bản ghi; không xác minh được nguồn gốc từng dòng")

    # 3. Xử lý thiếu customer_id
    n_miss_cid = df["customer_id"].isna().sum()
    df = df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = df["customer_id"].astype(int).astype(str)
    add_log("1.3", "customer_id", f"{n_miss_cid:,} dòng thiếu mã khách hàng",
            "Loại bỏ các giao dịch không định danh được khách hàng",
            n_miss_cid, "Đơn vị phân tích là hành vi từng khách hàng")

    # 4. Chuẩn hóa mô tả sản phẩm bị thiếu hoặc lỗi ngụy trang
    bad_desc = {"nan", "none", "null", "n/a", "", "?", "??", "check", "damaged"}
    mask_desc = df["description"].isna() | df["description"].str.lower().isin(bad_desc)
    df.loc[mask_desc, "description"] = "UNKNOWN_PRODUCT"
    add_log("1.4", "description", f"{mask_desc.sum():,} dòng thiếu hoặc chứa ký tự lỗi",
            "Gán nhãn 'UNKNOWN_PRODUCT' thay vì xóa dòng",
            mask_desc.sum(), "Giữ lại lịch sử chi tiêu và tần suất đơn hàng")

    # Giữ quốc gia quan sát gốc để các phân tích theo cửa sổ thời gian không dùng nhãn nội suy toàn kỳ.
    df["country_original"] = df["country"]

    # 5. Suy diễn quốc gia 'Unspecified' theo lịch sử từng khách hàng
    unknown_before = unknown_country_mask(df["country"])
    # Chỉ nội suy các quốc gia chưa xác định; nguồn ứng viên phải là quốc gia đã biết.
    needs_country = df.loc[unknown_before, "customer_id"].unique()
    candidate_rows = df.loc[~unknown_before & df["customer_id"].isin(needs_country)]
    candidates = candidate_rows.groupby("customer_id")["country"].agg(lambda s: s.mode().iloc[0])
    replacements = df.loc[unknown_before, "customer_id"].map(candidates)
    restored = replacements.dropna()
    df.loc[restored.index, "country"] = restored
    n_restored = len(restored)
    n_unresolved = int(unknown_country_mask(df["country"]).sum())
    add_log("1.5", "country", f"{int(unknown_before.sum()):,} dòng chưa xác định; khôi phục {n_restored:,}",
            "Điền theo quốc gia đã biết phổ biến nhất của cùng customer_id (chỉ mô tả)",
            n_restored, "Rows = số dòng thực sự được khôi phục")
    add_log("1.5a", "country", f"{n_unresolved:,} dòng vẫn chưa xác định",
            "Giữ nguyên quốc gia gốc chưa xác định; gán Unknown khi merge",
            n_unresolved, "Rows = số dòng chưa khôi phục tại bước này, trước lọc phi sản phẩm")

    # 6. Loại bỏ mã phi sản phẩm
    mask_non_prod = df["stockcode"].str.upper().isin(NON_PRODUCT_CODES)
    df = df[~mask_non_prod].copy()
    add_log("1.6", "stockcode", f"{mask_non_prod.sum():,} dòng phí vận chuyển/điều chỉnh kế toán",
            "Loại bỏ khỏi bảng giao dịch sản phẩm",
            mask_non_prod.sum(), "Không phản ánh hành vi chọn mua sản phẩm")

    # 7. Gắn cờ đơn hủy/trả hàng & lọc giá trị bất khả thi
    df["is_cancelled"] = df["invoice"].str.startswith("C")
    add_log("1.7", "invoice", f"{df['is_cancelled'].sum():,} dòng đơn hủy/hoàn trả",
            "Giữ nguyên và gắn cờ is_cancelled=True",
            df["is_cancelled"].sum(), "Tín hiệu quan trọng phản ánh hành vi hoàn đơn")

    mask_invalid = (
        (df["price"] <= 0)
        | ((~df["is_cancelled"]) & (df["quantity"] <= 0))
        | (df["is_cancelled"] & (df["quantity"] >= 0))
    )
    df = df[~mask_invalid].copy()
    add_log("1.8", "price & quantity", f"{mask_invalid.sum():,} dòng giá <= 0 hoặc mâu thuẫn dấu số lượng",
            "Loại bỏ bản ghi bất khả thi",
            mask_invalid.sum(), "Tránh làm sai lệch doanh thu và giá trị giỏ hàng")

    df["total_amount"] = df["quantity"] * df["price"]
    if not np.isfinite(df["total_amount"].to_numpy(dtype=float)).all():
        raise ValueError("total_amount chứa giá trị không hữu hạn.")
    return df, pd.DataFrame(logs)


def integrate_external_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Ghép nối dữ liệu giao dịch với bảng thông tin thị trường và lịch mùa vụ."""
    # Merge 1: Thông tin khu vực & tiền tệ theo quốc gia
    country_meta = get_country_metadata()
    len_before_1 = len(df)
    df_merged = df.merge(country_meta, on="country", how="left", validate="many_to_one")
    len_after_1 = len(df_merged)

    unmatched_mask = df_merged["region"].isna()
    match_rate_1 = (1 - unmatched_mask.mean()) * 100
    print(f"[Merge 1 - Country Metadata] Trước: {len_before_1:,} | Sau: {len_after_1:,} | Tỷ lệ khớp: {match_rate_1:.2f}%")
    if unmatched_mask.sum() > 0:
        unmatched_countries = df_merged.loc[unmatched_mask, "country"].unique().tolist()
        print(f"  -> Các giá trị chưa khớp ({unmatched_mask.sum():,} dòng): {unmatched_countries}")
        # Xử lý các dòng chưa khớp bằng nhóm 'Other_International'
        df_merged["region"] = df_merged["region"].fillna("Other_International")
        df_merged["trade_bloc"] = df_merged["trade_bloc"].fillna("Other")
        df_merged["currency"] = df_merged["currency"].fillna("OTHER")
        df_merged["shipping_zone"] = df_merged["shipping_zone"].fillna("Other")

    unknown = unknown_country_mask(df_merged["country"])
    df_merged.loc[unknown, ["region", "trade_bloc", "shipping_zone"]] = "Unknown"
    df_merged.loc[unknown, "currency"] = "UNKNOWN"

    # Merge 2: Thông tin mùa vụ mua sắm theo tháng
    df_merged["month"] = df_merged["invoice_date"].dt.month
    df_merged["day_of_week"] = df_merged["invoice_date"].dt.day_of_week
    df_merged["hour"] = df_merged["invoice_date"].dt.hour
    df_merged["is_weekend"] = (df_merged["day_of_week"] >= 5).astype(int)

    season_meta = get_peak_season_calendar()
    len_before_2 = len(df_merged)
    df_merged = df_merged.merge(season_meta, on="month", how="left", validate="many_to_one")
    len_after_2 = len(df_merged)
    match_rate_2 = (1 - df_merged["season"].isna().mean()) * 100
    print(f"[Merge 2 - Season Calendar]  Trước: {len_before_2:,} | Sau: {len_after_2:,} | Tỷ lệ khớp: {match_rate_2:.2f}%")

    return df_merged


FORBIDDEN_MODEL_COLS = {"customer_id", "invoice", "invoice_ts", "invoice_date", "cutoff_date", "country_original"}


def _log1p_nonnegative(x):
    """log1p có kiểm tra: báo lỗi nếu gặp giá trị âm thay vì sinh NaN âm thầm."""
    arr = np.asarray(x, dtype=float)
    if np.nanmin(arr, initial=0.0) < 0:
        raise ValueError("Biến đưa vào log1p chứa giá trị âm; hãy chuyển sang nhóm linear_num_cols.")
    return np.log1p(arr)


def build_sklearn_preprocessor(
    log_num_cols: list[str],
    cat_cols: list[str],
    linear_num_cols: list[str] | None = None,
) -> ColumnTransformer:
    """
    Bộ biến đổi Scikit-Learn chống rò rỉ dữ liệu (fit trên train, transform trên test).

    - log_num_cols: biến số lệch phải, không âm -> median impute -> log1p -> RobustScaler.
    - linear_num_cols: biến tỷ lệ/bị chặn -> median impute -> StandardScaler.
    - cat_cols: biến phân loại -> impute 'Missing' -> One-Hot.
    - Mọi cột còn lại (customer_id, target_repeat_90d, ...) bị loại bỏ (remainder='drop').
    """
    linear_num_cols = list(linear_num_cols or [])
    log_num_cols, cat_cols = list(log_num_cols), list(cat_cols)

    all_cols = log_num_cols + linear_num_cols + cat_cols
    forbidden = sorted(c for c in all_cols if c in FORBIDDEN_MODEL_COLS or c.startswith("target_"))
    if forbidden:
        raise ValueError(f"Cột bị cấm dùng làm đặc trưng (định danh hoặc nhãn): {forbidden}")
    duplicated = sorted({c for c in all_cols if all_cols.count(c) > 1})
    if duplicated:
        raise ValueError(f"Cột xuất hiện ở nhiều nhóm đặc trưng: {duplicated}")

    log_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(_log1p_nonnegative, feature_names_out="one-to-one")),
        ("scaler", RobustScaler()),
    ])
    linear_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [("log_num", log_transformer, log_num_cols)]
    if linear_num_cols:
        transformers.append(("linear_num", linear_transformer, linear_num_cols))
    transformers.append(("cat", categorical_transformer, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def run_pipeline(raw_dir: str = "data/raw", processed_dir: str = "data/processed", reports_dir: str = "reports"):
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("1. Đang đọc dữ liệu gốc...")
    df_raw = load_raw_data(raw_dir)
    # Mốc cuối quan sát lấy từ nguồn gốc trước khi lọc giao dịch;
    # dùng timestamp cuối như ranh giới bảo thủ, không suy ra từ bảng đã lọc.
    observed_until = df_raw["InvoiceDate"].max()

    print("2. Đang làm sạch dữ liệu & tạo Cleaning Log...")
    df_clean, log_df = clean_transactions(df_raw)
    log_df.to_csv(os.path.join(reports_dir, "cleaning_log.csv"), index=False)

    print("3. Đang tích hợp nguồn dữ liệu phụ (Merge)...")
    df_final = integrate_external_sources(df_clean)

    out_path = os.path.join(processed_dir, "retail_processed.parquet")
    df_final.to_parquet(out_path, index=False)
    print(f"4. Hoàn tất! Đã lưu {len(df_final):,} dòng vào '{out_path}' và cập nhật '{reports_dir}/cleaning_log.csv'.")
    return observed_until


if __name__ == "__main__":
    run_pipeline()