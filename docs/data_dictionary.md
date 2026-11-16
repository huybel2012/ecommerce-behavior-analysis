# Data Dictionary (W2)

## 1. Dữ liệu gốc Online Retail II

Một dòng là một mặt hàng trên hóa đơn, không phải một khách hoặc đơn duy nhất. Tên cột bên dưới đúng file CSV/XLSX của nhóm.

| Cột | Kiểu logic | Đơn vị / miền | Ý nghĩa | Thiếu thật (CSV) |
|---|---|---|---|---:|
| Invoice | mã chuỗi | mã hóa đơn | Mã chứng từ; tiền tố C chỉ hủy/hoàn trả | 0 |
| StockCode | mã chuỗi | mã sản phẩm | Có thể gồm ký tự; có mã phí/điều chỉnh | 0 |
| Description | văn bản | tên/mô tả | Mô tả hàng; gán UNKNOWN_PRODUCT nếu thiếu/token lỗi sau làm sạch | 4.382 |
| Quantity | số nguyên có dấu | số sản phẩm | Số lượng dòng hàng; âm với hoàn trả | 0 |
| InvoiceDate | datetime | ngày giờ theo nguồn, không có timezone | Thời điểm chứng từ | 0 |
| Price | số thực | GBP/sản phẩm | Đơn giá; có thể 0 hoặc âm ở raw, loại khi không dương | 0 |
| Customer ID | mã số nguyên (có thể đọc float do NA) | mã khách | Định danh khách; không thay thế giá trị thiếu bằng mã giả | 243.007 |
| Country | danh mục | quốc gia | Quốc gia cư trú theo nguồn; Unspecified là không rõ | 0 |

Nguồn và giấy phép: [data_sources.md](data_sources.md). Missing thật tính bằng `isna()` trên CSV; không tính các token thiếu ngụy trang. Không dùng Customer ID, Invoice, StockCode như đại lượng số liên tục.

## 2. Giao dịch làm sạch và tích hợp

`retail_cleaned_w4.parquet`: 11 cột đầu tiên dưới đây. `retail_processed.parquet`: 21 cột. Kiểu logic không phụ thuộc backend; mã giữ dạng chuỗi; boolean hoặc cờ 0/1. Dữ liệu sạch được kiểm tra `total_amount == quantity * price` trong notebook 01/W9.

| Cột | Kiểu | Đơn vị | Định nghĩa / xử lý |
|---|---|---|---|
| invoice | chuỗi | ID | Invoice sau strip |
| stockcode | chuỗi | ID | StockCode sau strip; loại mã phi sản phẩm theo danh sách pipeline |
| description | chuỗi | văn bản | Mô tả sau strip, token thiếu gán UNKNOWN_PRODUCT |
| quantity | nguyên có dấu | sản phẩm | Dương nếu mua; âm nếu hủy |
| price | thực dương | GBP/sản phẩm | Đơn giá gốc, không đổi tiền tệ |
| customer_id | chuỗi | ID | Customer ID không thiếu, chuẩn hóa từ số nguyên |
| country | chuỗi | danh mục | Quốc gia có thể nội suy theo mode toàn kỳ của khách khi Unspecified (chỉ dùng mô tả) |
| invoice_date | datetime | ngày giờ | InvoiceDate chuẩn hóa |
| country_original | chuỗi | danh mục | Quốc gia quan sát sau strip, trước nội suy; giữ Unspecified; dùng phân nhóm W9 |
| is_cancelled | boolean | True/False | Invoice bắt đầu C |
| total_amount | thực có dấu | GBP | quantity × price |
| region | chuỗi | danh mục | Nhóm khu vực từ bảng tham chiếu; fallback Other_International |
| trade_bloc | chuỗi | danh mục | Nhãn tĩnh từ bảng dự án; fallback Other; không đảm bảo đúng lịch sử |
| currency | chuỗi | danh mục | Nhãn tham chiếu tĩnh; fallback OTHER; giá giao dịch luôn GBP |
| shipping_zone | chuỗi | danh mục | Proxy Domestic/Short_Haul/Long_Haul/Other, không phải phí thực |
| month | nguyên | 1–12 | Tháng invoice_date |
| day_of_week | nguyên | 0–6 | Thứ hai = 0, chủ nhật = 6 |
| hour | nguyên | 0–23 | Giờ invoice_date |
| is_weekend | nguyên | 0/1 | day_of_week >= 5 |
| season | chuỗi | danh mục | Mùa khí tượng Bắc bán cầu theo tháng |
| is_peak_retail_month | nguyên | 0/1 | Tháng 9–12 = 1 |

## 3. Hồ sơ khách hàng mô tả toàn kỳ

`customer_profiles.parquet` tạo bởi notebook 03 (đơn vị một khách có >= 1 hóa đơn mua hợp lệ). Cùng hàm `build_customer_profiles()` trong notebook 02. Một đơn hợp lệ = cặp customer_id/invoice với các dòng không hủy; thời điểm đơn là timestamp sớm nhất. Các đặc trưng dưới đây được tính trên toàn bộ kỳ quan sát; **không dùng trực tiếp làm dữ liệu huấn luyện dự báo tương lai**.

| Cột | Kiểu | Đơn vị | Định nghĩa |
|---|---|---|---|
| customer_id | chuỗi | ID | Khóa duy nhất |
| first_purchase | datetime | ngày giờ | Đơn mua đầu ghi nhận (không nhất thiết là lần mua đầu thật) |
| last_purchase | datetime | ngày giờ | Đơn mua cuối ghi nhận |
| frequency | nguyên | đơn | Số invoice mua duy nhất |
| monetary | thực dương | GBP | Tổng giá trị các dòng mua, **gross purchases**, không trừ hoàn trả |
| avg_order_value | thực dương | GBP/đơn | monetary / frequency |
| weekend_ratio | thực | 0–1 | Tỷ lệ số đơn mua vào cuối tuần |
| peak_month_ratio | thực | 0–1 | Tỷ lệ số đơn mua trong tháng 9–12 |
| recency_days | nguyên | ngày | Số ngày nguyên từ last_purchase đến snapshot (nửa đêm sau ngày cuối dữ liệu) |
| tenure_days | nguyên | ngày | Số ngày nguyên giữa đơn đầu/cuối + 1 |
| observation_days | nguyên | ngày lịch | Snapshot trừ ngày đầu mua đã normalize; thời lượng cơ hội mua quan sát được |
| total_items | nguyên | sản phẩm | Tổng quantity trên dòng mua |
| unique_products | nguyên | sản phẩm | Số stockcode mua khác nhau |
| region | chuỗi | danh mục | Mode nhãn region theo dòng mua; nếu đồng hạng chọn mode đầu |
| shipping_zone | chuỗi | danh mục | Mode proxy theo dòng mua |
| total_invoices | nguyên | hóa đơn | Số invoice duy nhất, gồm mua và hủy |
| cancelled_invoices | nguyên | hóa đơn | Số invoice hủy duy nhất |
| cancel_rate | thực | 0–1 | cancelled_invoices / total_invoices, không phải tỷ lệ sản phẩm hoàn về |
| country | chuỗi | danh mục | Quốc gia duy nhất quan sát từ country_original; Unknown hoặc Mixed nếu không xác định |
| is_uk | chuỗi | UK/International/Unknown/Mixed | Mixed nếu >=2 quốc gia biết; Unknown nếu có bất kỳ quốc gia không xác định (trừ khi đã Mixed); UK nếu duy nhất United Kingdom; còn lại International |
| is_repeat_buyer | nguyên | 0/1 | frequency > 1 trong cửa sổ, không phải nhãn giữ chân tương lai |

Unknown gồm rỗng, unspecified, unknown, nan, none, european community; Channel Islands không thuộc UK. W9 loại Unknown/Mixed khỏi H1/H3, giữ H2. Nhóm `peak_group` (tạm trong notebook): Peak nếu peak_month_ratio >= 0.5, Other nếu thấp hơn; H2 so sánh trung bình log1p(monetary), không đo tác động của mùa vụ. Phân tích 365 ngày lấy giao dịch trong [ngày mua đầu ghi nhận, ngày đó + 365 ngày), chỉ cohort có đủ thời gian quan sát; nhãn địa lý xác định từ `country_original` của cửa sổ đó. So sánh riêng trên cohort chung để giảm nhiễu do thay đổi tập khách.

## 4. Cleaning log

`reports/cleaning_log.csv` (pipeline) và `reports/cleaning_log_w4.csv` (notebook 01, thêm IQR). Mỗi hàng là một bước, `Step` (mã), `Column` (trường), `Finding` (phát hiện), `Action` (hành động), `Rows` (số bản ghi liên quan, không nhất thiết bị xóa), `Why` (lý do). Không cộng tất cả Rows để suy ra số dòng loại vì có bước chuẩn hóa, gắn cờ, chẩn đoán và có thể trùng tập bản ghi.

## W11: Modeling dataset theo cutoff

`customer_modeling_dataset.parquet`: một dòng cho mỗi khách có ít nhất một đơn mua hợp lệ trước cutoff. Mọi feature chỉ dùng lịch sử trước cutoff, không dùng hồ sơ toàn kỳ của EDA.

| Cột | Định nghĩa |
|---|---|
| `customer_id` | Khóa khách hàng, không đưa vào X. |
| `recency_days` | Số ngày thực từ timestamp đơn mua cuối đã quan sát đến cutoff. |
| `observation_span_days` | Số ngày từ đơn đầu đã quan sát đến cutoff, tối thiểu 1 để tính tỷ suất. |
| `frequency` | Số hóa đơn mua khác nhau trong lịch sử. |
| `total_items`, `unique_products` | Tổng số lượng và số mã hàng khác nhau của các dòng mua lịch sử. |
| `gross_valid_revenue`, `avg_order_value` | Tổng giá trị mua và trung bình giá trị đơn mua, GBP, chưa trừ hoàn trả. |
| `avg_basket_size`, `items_per_sku` | Số sản phẩm trung bình mỗi đơn; tổng số lượng / số mã khác nhau. |
| `purchase_rate_per_30d` | Frequency / observation_span_days × 30. |
| `interpurchase_mean_days` | Khoảng cách trung bình giữa các đơn; nếu chỉ một đơn thì dùng observation_span_days. |
| `weekend_ratio`, `peak_month_ratio` | Tỷ lệ hóa đơn mua bắt đầu vào cuối tuần hoặc tháng 9–12. |
| `cancelled_invoices`, `cancel_rate` | Số đơn hủy lịch sử và tỷ lệ trên toàn bộ hóa đơn lịch sử. |
| `region`, `shipping_zone` | Hai nhãn lấy từ cùng quốc gia gốc phổ biến nhất đã quan sát trước cutoff; Unknown nếu không biết. |
| `target_repeat_90d` | 1 nếu có hóa đơn mua mới trong [cutoff, cutoff + 90 ngày), ngược lại 0 khi nguồn phủ đủ cửa sổ. Dòng tiếp nối hóa đơn cũ không tính là đơn mới. Không đưa vào X. |

Cutoff mặc định 01/09/2011; thời điểm quan sát lấy từ dữ liệu gốc trước lọc. Không tái sử dụng kết quả nội suy quốc gia toàn kỳ của EDA cho mô hình.
