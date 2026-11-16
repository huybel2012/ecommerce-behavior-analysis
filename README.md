# Phân Tích Hành Vi Mua Sắm Thương Mại Điện Tử (E-Commerce Shopping Behavior Analysis)

Dự án môn **Khoa học Dữ liệu (INFO3020)** phân tích hành vi mua sắm trực tuyến trên bộ dữ liệu *Online Retail II UCI*, bao gồm toàn bộ quy trình: kiểm toán chất lượng dữ liệu, làm sạch, tích hợp dữ liệu, phân tích khám phá (EDA), kiểm định giả thuyết thống kê và xây dựng mô hình phân loại dự đoán khách hàng quay lại mua hàng.

---

## 1. Cấu trúc thư mục

```text
ecommerce-behavior-analysis/
├── data/
│   ├── raw/                 # Dữ liệu thô tải từ Kaggle (KHÔNG commit lên Git)
│   └── processed/           # Dữ liệu sạch sau khi chạy Pipeline
├── notebooks/               # Các Jupyter Notebook theo lộ trình môn học
│   ├── 01_w3_w4_audit_and_cleaning.ipynb
│   ├── 02_w5_pipeline_and_merge.ipynb
│   ├── 03_w6_w7_eda_and_viz.ipynb
│   ├── 04_w9_statistical_inference.ipynb
│   └── 05_w11_classification_modeling.ipynb
├── reports/
│   ├── figures/             # Biểu đồ xuất ra (PNG, 300 dpi)
│   ├── cleaning_log.csv     # Nhật ký xử lý dữ liệu
│   └── model_card.md        # Thẻ mô hình & đánh giá đạo đức dữ liệu
├── src/
│   ├── __init__.py
│   └── utils.py             # Hàm kiểm toán chất lượng dữ liệu dùng chung
├── .gitignore
├── README.md
└── requirements.txt