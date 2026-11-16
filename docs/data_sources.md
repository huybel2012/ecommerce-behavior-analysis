# Nguồn dữ liệu và metadata (W2)

## Nguồn giao dịch chính

- Trích dẫn: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D.
- Trang mô tả và license: https://archive.ics.uci.edu/dataset/502/online+retail+ii ; CC BY 4.0 (yêu cầu ghi công, nêu thay đổi khi tái sử dụng): https://creativecommons.org/licenses/by/4.0/.
- File nhóm dùng: `online_retail_II.csv`, từ https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci (bản chuyển định dạng từ nguồn UCI, không phải bộ dữ liệu độc lập thứ hai). Giấy phép ghi ở đây theo UCI; cần đối chiếu điều khoản riêng của bản phân phối Kaggle nếu chia sẻ lại file. Repository chỉ chứa code và tài liệu, không phân phối raw data.
- File thay thế: `online_retail_II.xlsx` của UCI, gồm hai sheet `Year 2009-2010` và `Year 2010-2011`; ghép dọc hai sheet, không cộng thêm CSV nếu đồng thời tồn tại.
- Mô tả: giao dịch của một nhà bán lẻ không có cửa hàng vật lý đăng ký tại Anh; chủ yếu hàng quà tặng, nhiều người mua buôn. Phạm vi 01/12/2009–09/12/2011. Đơn vị quan sát một dòng hàng trong hóa đơn; không đồng nhất với đơn hàng/khách hàng. Giá đơn vị theo pound sterling (GBP). Không có dữ liệu clickstream, chi phí vận chuyển, khuyến mại hay nhãn can thiệp A/B.
- Kiểm tra trên CSV gốc nhóm cung cấp: 1.067.371 hàng × 8 cột, thiếu 243.007 Customer ID, thiếu 4.382 Description; các cột khác không thiếu thật. `Country` có giá trị `Unspecified`. UTF-8 (đọc bằng `utf-8-sig` hỗ trợ BOM); không dùng ISO-8859-1 vì làm sai ký tự GBP.
- SHA-256 file CSV nhóm cung cấp: `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9`. Tên và schema được kiểm tra ở `src/pipeline.py`; ưu tiên CSV đúng tên. Chỉ thiếu CID/Description được phép qua kiểm tra đầu vào; kiểm tra số hữu hạn, Quantity/CID nguyên, datetime hợp lệ; các giá trị giá/dấu số lượng bất khả thi được xử lý ở bước làm sạch.

## Bảng tham chiếu do dự án tạo

| Bảng | Nguồn / cơ sở | Khóa | Đặc trưng | Giới hạn |
|---|---|---|---|---|
| `get_country_metadata()` | Danh sách phân nhóm và nhãn tĩnh được định nghĩa thủ công trong mã nguồn dự án, **chưa có bộ dữ liệu địa lý độc lập hay bằng chứng đối chiếu toàn bộ nhãn** | `country` | `region`, `trade_bloc`, `currency`, `shipping_zone` | Vùng và shipping là quy ước phân tích, không phải địa chỉ hoặc khoảng cách thực; nhãn tiền tệ và khối thương mại không được phiên bản hóa theo lịch sử 2009–2011 (ví dụ Lithuania dùng EUR về sau). Không đổi giá tiền theo `currency`. Không suy ra chi phí/điều kiện thương mại thực tế. |
| `get_peak_season_calendar()` | Quy ước do dự án chọn: mùa khí tượng Bắc bán cầu và tháng 9–12 là nhóm peak; không phải dữ liệu quan sát về khuyến mại/lễ hội của nhà bán lẻ | `month` (1–12) | `season`, `is_peak_retail_month` | Có thể không phù hợp mọi thị trường; kết quả phụ thuộc định nghĩa peak và cần kiểm tra độ nhạy nếu thay mốc. |

Ghép trái nhiều-một với kiểm chứng tính duy nhất khóa (`validate="many_to_one"`); không nhân số bản ghi. Nhãn chưa khớp country dùng fallback `Other_International`, `Other`, `OTHER`, `Other`; không đồng nghĩa thị trường đã được xác định. Phân nhóm UK/International trong W9 dùng `country_original`, với nhãn Unknown/Mixed riêng, không dựa vào fallback. Các bảng được tạo từ quy tắc trong code, không tuyên bố là dữ liệu bên ngoài được thu thập độc lập.

## Biến đổi và lưu trữ

Giữ file gốc bất biến; sửa encoding, kiểm tra schema, chuẩn hóa tên và kiểu, xóa bản ghi giống hệt theo giả định, bỏ khách vô danh, gán mô tả thiếu, lưu `country_original` sau strip trước nội suy country, loại mã phi sản phẩm và bản ghi có giá/dấu số lượng không hợp lệ. Giữ giao dịch hủy có dấu âm để phân tích hoàn trả. Gắn đặc trưng ngày/giờ, metadata. Hồ sơ khách chỉ từ khách có đơn mua; định nghĩa ở [Data Dictionary](data_dictionary.md). Giới hạn chính: phạm vi thời gian/cơ sở bán lẻ, loại khách vô danh, sao chép hàng có thể là giao dịch hợp lệ, không biết lần mua đầu thật và không có thí nghiệm.
