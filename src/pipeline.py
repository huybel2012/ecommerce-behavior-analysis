import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer


def get_country_metadata() -> pd.DataFrame:
    """Bảng tham chiếu thông tin thị trường theo quốc gia (nguồn dữ liệu tích hợp)."""
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
    """Bảng tham chiếu mùa vụ bán lẻ & cao điểm lễ hội theo tháng."""
    return pd.DataFrame({
        "month": list(range(1, 13)),
        "season": [
            "Winter", "Winter", "Spring", "Spring", "Spring", "Summer",
            "Summer", "Summer", "Autumn", "Autumn", "Autumn", "Winter"
        ],
        "is_peak_retail_month": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    })


def load_raw_data(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Đọc dữ liệu gốc từ thư mục data/raw mà không làm thay đổi file gốc."""
    files = os.listdir(raw_dir)
    csv_files = [f for f in files if f.endswith(".csv")]
    xlsx_files = [f for f in files if f.endswith(".xlsx")]

    if csv_files:
        return pd.read_csv(os.path.join(raw_dir, csv_files[0]), encoding="ISO-8859-1")
    if xlsx_files:
        path = os.path.join(raw_dir, xlsx_files[0])
        return pd.concat(
            [
                pd.read_excel(path, sheet_name="Year 2009-2010"),
                pd.read_excel(path, sheet_name="Year 2010-2011"),
            ],
            ignore_index=True,
        )
    raise FileNotFoundError(f"Không tìm thấy file .csv hoặc .xlsx trong {raw_dir}")


def clean_transactions(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Làm sạch dữ liệu giao dịch và ghi nhật ký xử lý (Cleaning Log)."""
    df = df_raw.copy()
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
    for col in ["invoice", "stockcode", "description", "country"]:
        df[col] = df[col].astype(str).str.strip()
    add_log("1.1", "all_columns", "Tên cột chưa đồng nhất, InvoiceDate dạng chuỗi",
            "Chuẩn hóa snake_case, ép kiểu datetime và loại bỏ khoảng trắng thừa",
            len(df), "Đảm bảo tính nhất quán khi trích xuất thời gian và ghép bảng")

    # 2. Xóa trùng lặp hoàn toàn
    n_dups = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    add_log("1.2", "all", f"{n_dups:,} dòng trùng lặp hoàn toàn",
            "Xóa các dòng trùng lặp, giữ lại bản ghi đầu tiên",
            n_dups, "Lỗi xuất dữ liệu lặp từ hệ thống giao dịch")

    # 3. Xử lý thiếu customer_id
    n_miss_cid = df["customer_id"].isna().sum()
    df = df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = df["customer_id"].astype(int).astype(str)
    add_log("1.3", "customer_id", f"{n_miss_cid:,} dòng thiếu mã khách hàng",
            "Loại bỏ các giao dịch không định danh được khách hàng",
            n_miss_cid, "Đơn vị phân tích là hành vi từng khách hàng")

    # 4. Chuẩn hóa mô tả sản phẩm bị thiếu hoặc lỗi ngụy trang
    bad_desc = ["nan", "None", "", "?", "??", "check", "damaged"]
    mask_desc = df["description"].isin(bad_desc)
    df.loc[mask_desc, "description"] = "UNKNOWN_PRODUCT"
    add_log("1.4", "description", f"{mask_desc.sum():,} dòng thiếu hoặc chứa ký tự lỗi",
            "Gán nhãn 'UNKNOWN_PRODUCT' thay vì xóa dòng",
            mask_desc.sum(), "Giữ lại lịch sử chi tiêu và tần suất đơn hàng")

    # 5. Suy diễn quốc gia 'Unspecified' theo lịch sử từng khách hàng
    n_unspec = (df["country"] == "Unspecified").sum()
    if n_unspec > 0:
        cid_country = (
            df[df["country"] != "Unspecified"]
            .groupby("customer_id")["country"]
            .agg(lambda s: s.mode().iloc[0])
        )
        df["country"] = df.apply(
            lambda r: cid_country.get(r["customer_id"], "Unspecified")
            if r["country"] == "Unspecified" else r["country"],
            axis=1,
        )
    add_log("1.5", "country", f"{n_unspec:,} dòng có giá trị 'Unspecified'",
            "Điền theo quốc gia phổ biến nhất của cùng customer_id",
            n_unspec, "Khôi phục thông tin địa lý từ các đơn hàng khác của cùng khách")

    # 6. Loại bỏ mã phi sản phẩm
    non_prod = [
        "POST", "D", "M", "BANK CHARGES", "AMAZONFEE", "DOT",
        "CRUK", "S", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002"
    ]
    mask_non_prod = df["stockcode"].str.upper().isin(non_prod)
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
    return df, pd.DataFrame(logs)


def integrate_external_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Ghép nối dữ liệu giao dịch với bảng thông tin thị trường và lịch mùa vụ."""
    # Merge 1: Thông tin khu vực & tiền tệ theo quốc gia
    country_meta = get_country_metadata()
    len_before_1 = len(df)
    df_merged = df.merge(country_meta, on="country", how="left")
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

    # Merge 2: Thông tin mùa vụ mua sắm theo tháng
    df_merged["month"] = df_merged["invoice_date"].dt.month
    df_merged["day_of_week"] = df_merged["invoice_date"].dt.dayofweek
    df_merged["hour"] = df_merged["invoice_date"].dt.hour
    df_merged["is_weekend"] = (df_merged["day_of_week"] >= 5).astype(int)

    season_meta = get_peak_season_calendar()
    len_before_2 = len(df_merged)
    df_merged = df_merged.merge(season_meta, on="month", how="left")
    len_after_2 = len(df_merged)
    match_rate_2 = (1 - df_merged["season"].isna().mean()) * 100
    print(f"[Merge 2 - Season Calendar]  Trước: {len_before_2:,} | Sau: {len_after_2:,} | Tỷ lệ khớp: {match_rate_2:.2f}%")

    return df_merged


def build_sklearn_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Tạo bộ biến đổi Scikit-Learn chống rò rỉ dữ liệu (fit trên train, transform trên test)."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", RobustScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="passthrough",
    )


def run_pipeline(raw_dir: str = "data/raw", processed_dir: str = "data/processed", reports_dir: str = "reports"):
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("1. Đang đọc dữ liệu gốc...")
    df_raw = load_raw_data(raw_dir)

    print("2. Đang làm sạch dữ liệu & tạo Cleaning Log...")
    df_clean, log_df = clean_transactions(df_raw)
    log_df.to_csv(os.path.join(reports_dir, "cleaning_log.csv"), index=False)

    print("3. Đang tích hợp nguồn dữ liệu phụ (Merge)...")
    df_final = integrate_external_sources(df_clean)

    out_path = os.path.join(processed_dir, "retail_processed.parquet")
    df_final.to_parquet(out_path, index=False)
    print(f"4. Hoàn tất! Đã lưu {len(df_final):,} dòng vào '{out_path}' và cập nhật '{reports_dir}/cleaning_log.csv'.")


if __name__ == "__main__":
    run_pipeline()