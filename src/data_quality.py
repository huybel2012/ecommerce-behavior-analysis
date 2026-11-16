"""W3/W4: chỉ báo kiểm toán, sai lệch chọn mẫu và kiểm tra hóa đơn; không sửa/xóa dữ liệu."""
import numpy as np
import pandas as pd

from src.pipeline import NON_PRODUCT_CODES
from src.utils import check_disguised_missing


def _score(rate):
    # Ngưỡng rubric nội bộ, không phải tiêu chuẩn bên ngoài.
    return int(np.searchsorted([.80, .90, .95, .99], rate, side="right") + 1)


def quality_dimensions(raw):
    """Điểm 1–5, proxy đo được và giới hạn. Accuracy/Timeliness là đánh giá bằng chứng."""
    disguised = check_disguised_missing(raw)
    n_disguised = int(disguised['n_disguised'].sum()) if not disguised.empty else 0
    complete = 1 - (raw.isna().sum().sum() + n_disguised) / raw.size
    exact_unique = 1 - raw.duplicated().mean()
    strings = raw.select_dtypes(include=['object', 'string'])
    text_count = clean_text_count = 0
    for col in strings:
        text = strings[col].dropna().astype(str)
        text_count += len(text)
        clean_text_count += int(text.eq(text.str.strip()).sum())
    consistency = clean_text_count / text_count if text_count else 1.0
    cancelled = raw['Invoice'].astype(str).str.strip().str.startswith('C')
    validity = ((raw['Price'] > 0) & ((cancelled & (raw['Quantity'] < 0)) |
                                       (~cancelled & (raw['Quantity'] > 0)))).mean()
    last = raw['InvoiceDate'].max()
    first = raw['InvoiceDate'].min()
    return pd.DataFrame([
        ('Completeness', _score(complete), f'{complete:.2%} ô không thiếu thật/giả', 'Tính theo ô; Customer ID cần xem riêng.'),
        ('Accuracy', 2, 'Không có dữ liệu đối chứng giao dịch thực', '2/5 là mức bằng chứng hạn chế, không phải tỷ lệ giá trị đúng.'),
        ('Consistency', _score(consistency), f'{consistency:.2%} ô chuỗi không thừa khoảng trắng', 'Proxy hẹp; kiểm tra timestamp hóa đơn riêng ở W4.'),
        ('Validity', _score(validity), f'{validity:.2%} dòng thỏa giá > 0 và dấu quantity', 'Gồm cả điều chỉnh/phi sản phẩm nên không đồng nghĩa dòng lỗi.'),
        ('Uniqueness', _score(exact_unique), f'{exact_unique:.2%} dòng không trùng hoàn toàn sau dòng đầu', 'Không xác minh được bản ghi trùng là giao dịch lặp thật hay lỗi.'),
        ('Timeliness', 2, f'{first} → {last}', '2/5 cho tính hiện hành (dữ liệu lịch sử 2009–2011); phù hợp nghiên cứu hồi cứu, không dùng trực tiếp cho 2026.'),
    ], columns=['Dimension', 'Score_1_5', 'Evidence', 'Limitation'])


def missing_id_comparison(raw):
    """So sánh ở cấp dòng, trước bỏ CID; không suy ra số khách thiếu ID."""
    # Chuẩn hóa chuỗi và xử lý duplicate giống pipeline trước khi loại CID.
    d = raw.copy()
    for col in ['Invoice', 'StockCode', 'Description', 'Country']:
        d[col] = d[col].astype(str).str.strip()
    d = d.drop_duplicates().copy()
    d['id_status'] = np.where(d['Customer ID'].isna(), 'Missing_ID', 'Known_ID')
    cancel = d['Invoice'].str.startswith('C')
    eligible = ~d['StockCode'].str.upper().isin(NON_PRODUCT_CODES) & (d['Price'] > 0)
    buy = d.loc[eligible & ~cancel & (d['Quantity'] > 0)].copy()
    buy['gross_value'] = buy['Quantity'] * buy['Price']
    buy['month'] = buy['InvoiceDate'].dt.to_period('M').astype(str)
    statuses = ['Known_ID', 'Missing_ID']
    summary = d.groupby('id_status').size().reindex(statuses, fill_value=0).rename('all_lines').to_frame()
    summary = summary.join(buy.groupby('id_status').agg(
        purchase_lines=('gross_value','size'), gross_value_gbp=('gross_value','sum'),
        median_line_gbp=('gross_value','median'), median_quantity=('Quantity','median'),
        median_unit_price_gbp=('Price','median')))
    summary[['purchase_lines','gross_value_gbp']] = summary[['purchase_lines','gross_value_gbp']].fillna(0)
    for c in ['all_lines','purchase_lines','gross_value_gbp']:
        summary[c+'_share'] = summary[c] / summary[c].sum() if summary[c].sum() else np.nan
    def composition(col):
        table = pd.crosstab(buy[col], buy['id_status']).reindex(columns=statuses, fill_value=0)
        table['total'] = table[statuses].sum(axis=1)
        table['missing_share'] = table['Missing_ID'] / table['total']
        return table.sort_index()
    return summary, composition('month'), composition('Country')


def invoice_timestamp_audit(df, cutoffs=('2011-03-01','2011-06-01','2011-09-01')):
    """Kiểm tra đúng khóa khách/hóa đơn, chỉ giao dịch mua hợp lệ; không xóa."""
    valid = df.loc[~df['is_cancelled']]
    keys = ['customer_id','invoice']
    bounds = valid.groupby(keys).agg(first=('invoice_date','min'), last=('invoice_date','max'),
                                      timestamp_count=('invoice_date','nunique'), lines=('invoice_date','size'))
    multi = bounds.loc[bounds['timestamp_count'] > 1].copy()
    multi['span_minutes'] = (multi['last'] - multi['first']).dt.total_seconds()/60
    multi['cross_calendar_day'] = multi['first'].dt.normalize() != multi['last'].dt.normalize()
    spans = []
    for cutoff in map(pd.Timestamp, cutoffs):
        for boundary, kind in [(cutoff,'cutoff'), (cutoff+pd.Timedelta(days=90),'label_end')]:
            crossing = (bounds['first'] < boundary) & (bounds['last'] >= boundary)
            spans.append({'cutoff':cutoff, 'boundary':boundary,'kind':kind,'crossing_invoices':int(crossing.sum())})
    counts = pd.Series({'purchase_invoices':len(bounds),'multi_timestamp_invoices':len(multi),
                        'cross_calendar_day':int(multi['cross_calendar_day'].sum()),
                        'invoice_codes_shared_by_customers':int((valid.groupby('invoice')['customer_id'].nunique()>1).sum())})
    return counts, multi, pd.DataFrame(spans)


def describe_numeric(df, cols):
    """Thống kê location, spread, shape; CV chỉ cho biến trung bình dương."""
    x = df[cols].select_dtypes(include='number')
    out = x.agg(['count','mean','median','std','var','min','max','skew','kurt']).T
    for q in [.01,.25,.75,.95,.99]:
        out[f'p{int(q*100):02d}'] = x.quantile(q)
    out['IQR'] = out['p75'] - out['p25']
    out['range'] = out['max'] - out['min']
    out['CV'] = out['std'] / out['mean'].where(out['mean'] > 0)
    return out
