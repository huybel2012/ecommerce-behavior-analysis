# Datasheet — Online Retail II (W12)

## Nguồn, mục đích và thành phần
- Chen, D. (2012), *Online Retail II*, UCI, DOI [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D). Một nhà bán lẻ online đăng ký tại Anh, chủ yếu bán quà tặng, cả khách mua buôn; giao dịch 01/12/2009–09/12/2011. UCI ghi người tạo Daqing Chen; phương thức đồng ý của từng khách và quy trình thu thập nội bộ không được mô tả đầy đủ.
- 1.067.371 dòng hàng, 8 cột (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country). Đơn vị dòng hàng, giá GBP. Không phải mẫu ngẫu nhiên dân số, không có clickstream, demographics, nhãn chiến dịch hoặc giá vốn.
- File dự án là bản CSV từ [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci), không phải nguồn độc lập. SHA-256 và schema tại [data_sources.md](data_sources.md). Hai bảng địa lý/lịch mùa do nhóm định nghĩa, không phải dữ liệu thu thập độc lập.

## Giấy phép, điều khoản, consent (kiểm tra 16/11/2026)
| Hạng mục | Bằng chứng / trạng thái |
|---|---|
| UCI | [Trang dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) công bố CC BY 4.0; ghi công, ghi rõ biến đổi khi tái sử dụng. [License](https://creativecommons.org/licenses/by/4.0/). |
| Công bố nguồn | [UCI donation policy](https://archive.ics.uci.edu/contribute/donation) yêu cầu quyền công bố và loại PII; chính sách chung không chứng minh sự đồng ý của từng khách trong dataset. Không tự khẳng định consent/GDPR compliance. |
| Kaggle | [Terms](https://www.kaggle.com/terms), bản hiệu lực 22/06/2025, quy định tuân thủ luật, hạn chế crawl/scrape và sử dụng dịch vụ. Trang mirror truy cập được nhưng không đọc được trường license riêng trong phiên kiểm tra; trước chia sẻ lại phải xác minh trực tiếp. Không phân phối file CSV gốc. |
| robots.txt | [UCI](https://archive.ics.uci.edu/robots.txt) và [Kaggle](https://www.kaggle.com/robots.txt): đã thử mở ngày 16/11/2026 nhưng công cụ không truy xuất nội dung; chưa xác nhận quy tắc. Nhóm cần lưu kết quả kiểm tra thủ công trước nghiệm thu rubric. |
| Cách lấy dữ liệu | Pipeline đọc file đã tải về từ trang phân phối; không có web scraper. Không có bằng chứng về thao tác tải ban đầu của thành viên, cần nhóm xác nhận. robots.txt không phải giấy phép. |

## Chất lượng, chọn mẫu và độ đại diện
- 243.007 dòng thiếu Customer ID trên dữ liệu thô (22,77%); 4.382 thiếu Description. Loại duplicate theo giả định, chuẩn hóa token thiếu, loại mã phi sản phẩm và giá/dấu số lượng sai; vẫn giữ dòng hủy hợp lệ. Giữ `country_original` để tránh leakage. Sau làm sạch: 794.437 dòng ở lần nghiệm thu W10.
- P2: dòng mua hợp lệ thiếu ID chiếm 22,60% số dòng và 13,11% gross mua hợp lệ sau chuẩn hóa/loại trùng; đây là sai lệch cấp dòng, không ước tính được số người vô danh. Hong Kong và Bermuda không có dòng mua hợp lệ có ID trong bảng kiểm toán W3. UK chiếm ưu thế; quốc tế, nhóm chưa rõ quốc gia, khách vô danh và khách mới bị thiếu đại diện.
- Phân tích toàn kỳ có 5.853 khách mua; test là 5.225 khách đã mua trước cutoff 01/09/2011. Không đồng nhất hai population. W12 đo thành phần snapshot riêng, nhóm dựa trên vùng modal trước cutoff, không dùng nhãn địa lý toàn kỳ.

## Quyền riêng tư, lưu giữ và phạm vi
Customer ID/Invoice là mã liên kết, không phải bằng chứng ẩn danh; timestamp, quốc gia, số lượng, giá và tổ hợp hành vi có thể tái định danh. Không thu thập thêm PII, không nối nguồn ngoài để nhận diện cá nhân. Giữ mã nội bộ để tạo profile/target, không đưa vào feature hoặc output công khai. `.gitignore` loại raw và CSV/Parquet; chỉ công bố tổng hợp, không công bố bảng cấp khách hoặc nhóm metric quá nhỏ. Lịch sử Git cũ có thể còn mẫu ID/hóa đơn; không rewrite lịch sử, không khẳng định đã xóa các bản sao.

**Kiểm tra k:** notebook 06 tính k tối thiểu theo distinct customer trên test snapshot với bộ quasi-ID (region, recency, frequency, gross lịch sử); sau đó tổng quát hóa thành UK/International/Unknown, recency ≤30/90/180/>180 ngày, frequency ≤1/3/10/>10, gross ≤100/500/2000/>2000 GBP, loại lớp tương đương có n<10. Chỉ công bố số tổng hợp, không xuất microdata tổng quát hóa. k là phép đo trên bộ quasi-ID giả định, không chứng minh ẩn danh dữ liệu gốc hay chống mọi tấn công (ví dụ homogeneous class). Privacy audit không thay đổi mẫu đánh giá mô hình. Kết quả được ghi tự động khi chạy notebook trên dữ liệu thật.

<!-- W12_RESULTS_START -->
W12: mô phỏng generalize/suppress trên test; không công bố bảng khách và không khẳng định dữ liệu gốc ẩn danh.

| Nhóm | customers | suppressed | k_min | customers_below_k | k_target |
| --- | --- | --- | --- | --- | --- |
| exact | 5225 | 0 | 1 | 5225 | 10 |
| generalized | 5225 | 0 | 1 | 172 | 10 |
| generalized_suppressed | 5053 | 172 | 10 | 0 | 10 |
<!-- W12_RESULTS_END -->

**Chính sách đề xuất cho nhóm:** giữ raw/processed trong máy có kiểm soát truy cập, chỉ thành viên cần xử lý; không tải lên dịch vụ khác, không chia sẻ bảng khách. Sau chấm và hết thời hạn phúc khảo, xóa bản dữ liệu cục bộ/backup không cần thiết; giữ code và báo cáo tổng hợp. Nhóm cần xác nhận thực thi chính sách, không tuyên bố đã xóa hay đã kiểm tra mọi máy.

**Phù hợp:** nghiên cứu hồi cứu về hành vi mua lại, gross GBP, học phương pháp thống kê/ML. **Không phù hợp:** dự báo doanh thu hiện tại, định giá cá nhân, quyết định tự động ảnh hưởng quyền lợi, dự báo cold-start, suy diễn nhân quả hoặc đại diện mọi quốc gia. Không dùng mô hình để loại khách quốc tế; cần dữ liệu mới, đánh giá lại và xem xét tác động trước sử dụng thực tế.
