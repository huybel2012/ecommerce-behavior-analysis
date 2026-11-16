# Model Card — Regression (W10/W12)

**Mục tiêu:** gross GBP của đơn mua mới trong 90 ngày ở khách đã từng mua trước cutoff; zero nếu không mua. Không trừ trả hàng, không phải lợi nhuận hay CLV. [Datasheet](datasheet.md).

**Mô hình/đánh giá:** DummyRegressor median được chọn bằng MAE CV train (hòa RMSE, tên); OLS, Ridge, Lasso là đối chứng. 7 feature số lịch sử, không ID/geography, fit preprocessing trong CV; clip dự báo âm về 0. Train/validation/test 01/03, 01/06, 01/09/2011 (4.513/4.909/5.225 khách); nhãn 90 ngày không chồng lấn. W12 tái fit Dummy median trên train, không chọn lại theo test. Test hồi cứu, không cold-start.

**Kết quả W10 đã nghiệm thu:** median dummy dự báo £0 mọi khách, test MAE £515,34, RMSE £3.144,94, R² −0,0276. OLS R² 0,1127; Ridge MAE £484,85 trên test nhưng không được chọn lại. Không đổi winner dựa trên test. Tỷ lệ zero 57,21% test. OLS diagnostics cho thấy residual không chuẩn, phương sai không đều; HC3 không giải quyết mọi vi phạm. Không suy diễn nhân quả.

**Fairness:** MAE (metric chọn model) và mean error = actual − predicted theo thị trường; positive là dự báo thiếu. RMSE để thấy sai số lớn, R² N/A khi target hằng; so sánh cả mean actual/zero share để hiểu chênh lệch nền. Nhóm theo modal địa lý lịch sử; công bố UK và Non-UK / Unknown (gộp International và Unknown). Nếu một nhóm n<30, ẩn cả hai nhóm. Gap nhóm gộp − UK không chứng minh phân biệt đối xử. Không công bố International riêng trong báo cáo mới; lịch sử Git vẫn có kết quả cũ. Bảng đầy đủ notebook 06.

<!-- W12_RESULTS_START -->
Temporal test đã xác thực; gap MAE Non-UK / Unknown − UK +549.10 GBP.

| Nhóm | n | mae_gbp | rmse_gbp | r2 | mean_error_gbp |
| --- | --- | --- | --- | --- | --- |
| Overall | 5225 | 515.3416 | 3144.9414 | -0.0276 | 515.3416 |
| UK | 4773 | 467.8408 | 2787.4855 | -0.0290 | 467.8408 |
| Non-UK / Unknown | 452 | 1016.9378 | 5681.8507 | -0.0331 | 1016.9378 |
<!-- W12_RESULTS_END -->

**Không khuyến nghị triển khai:** winner không phân biệt được mức chi tiêu, dự báo tổng £0 cũng không phù hợp để lập kế hoạch doanh thu/phân khúc. Chỉ baseline nghiên cứu, không quyết định quyền lợi từng khách. Cần cải tiến và kiểm tra dữ liệu ngoài thời gian đã xem, coverage nhóm, drift và chi phí sai số. Không công bố dự báo cấp khách.
