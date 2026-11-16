# Model Card — Classification (W11/W12)

**Mục tiêu:** dự báo khách từng mua có đơn mua mới trong 90 ngày. Phạm vi: một nhà bán lẻ UK, Online Retail II 2009–2011; không phải cold-start. Target 0/1, không tính đơn hủy hoặc tiếp nối đơn trước cutoff. [Datasheet](datasheet.md).

**Mô hình/đánh giá:** Random Forest (300 cây, depth 8, min leaf 10, balanced weights, seed 42). 17 feature từ lịch sử, gồm region và shipping zone; không dùng ID/nhãn. CV ROC-AUC trên train 01/03/2011 (n=4.513), threshold tối đa F1 trên validation 01/06 (n=4.909), test 01/09 (n=5.225), các nhãn 90 ngày không chồng lấn. Khách có thể lặp; test đã xem ở P0, hồi cứu không phải holdout mới. W12 tái fit đúng model train và tái tạo threshold trên validation, không lựa chọn lại hoặc tinh chỉnh theo test.

**Kết quả W11 đã nghiệm thu:** accuracy 0,7301; ROC-AUC 0,7936; AP 0,7639; F1 0,6579; recall 0,6064; precision 0,7190; threshold hiển thị 0,4805 (dùng giá trị đầy đủ trong code). Baseline accuracy 0,5721.

**Fairness:** ưu tiên chênh lệch recall/TPR (equal opportunity: tránh bỏ sót khách có khả năng quay lại), kiểm tra thêm FPR (chi phí liên hệ nhầm), precision, accuracy, ROC-AUC. Nhóm UK/International/Unknown theo lịch sử trước cutoff, không phải quốc tịch hay protected attributes. Chênh lệch Intl − UK mô tả, không kết luận nhân quả hoặc chứng nhận fairness. n<30 hoặc denominator<10 thì không báo metric tương ứng; không gộp Unknown vào quốc tế. Bảng đầy đủ notebook 06.

<!-- W12_RESULTS_START -->
W12 temporal test (đúng CSV tham chiếu), threshold validation 0.480468678468; gap recall Intl − UK +0.0822.

| Nhóm | n | accuracy | recall | fpr | precision | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 5225 | 0.7301 | 0.6064 | 0.1773 | 0.7190 | 0.7936 |
| UK | 4773 | 0.7301 | 0.5987 | 0.1729 | 0.7186 | 0.7927 |
| International | 445 | 0.7348 | 0.6810 | 0.2170 | 0.7371 | 0.8047 |
| Unknown | 7 | N/A | N/A | N/A | N/A | N/A |
<!-- W12_RESULTS_END -->

**Giới hạn và chống lạm dụng:** thiếu ID, lệch UK, drift, thiếu dữ liệu nhân khẩu học, bỏ sót khách; correlated features hạn chế diễn giải permutation importance. Chỉ nghiên cứu/khảo sát tổng hợp, không tự động phân biệt đối xử hay gán quyền lợi. Cần kiểm chứng trên dữ liệu mới, đánh giá nhóm và human review trước thực tế. Không công bố dự báo cấp khách; [quyền riêng tư](datasheet.md).
