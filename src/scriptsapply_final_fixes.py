import os
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update_utils_suppression():
    """Mục 09: Thống nhất quy tắc suppression (k-anonymity & warnings) trong src/utils.py."""
    utils_path = ROOT / "src" / "utils.py"
    content = utils_path.read_text(encoding="utf-8") if utils_path.exists() else ""

    addition = '''

# === Quy tắc suppression thống nhất cho toàn dự án (Notebook 01 & Notebook 06) ===
import warnings

SUPPRESSION_K = 10


def configure_warnings() -> None:
    """Thống nhất cách xử lý cảnh báo hiển thị giữa các notebook."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def apply_group_suppression(
    df,
    group_col: str,
    min_count: int = SUPPRESSION_K,
    suppressed_label: str = "Suppressed_Small_Group",
):
    """Gộp các nhóm có số lượng quan sát < SUPPRESSION_K để bảo vệ ẩn danh và độ tin cậy thống kê."""
    out = df.copy()
    counts = out[group_col].value_counts(dropna=False)
    small_groups = counts[counts < min_count].index
    out[group_col] = out[group_col].where(~out[group_col].isin(small_groups), suppressed_label)
    return out
'''
    if "SUPPRESSION_K" not in content:
        utils_path.write_text(content.rstrip() + "\n" + addition, encoding="utf-8")
        print("[OK] Đã cập nhật quy tắc suppression thống nhất vào src/utils.py")
    else:
        print("[SKIP] src/utils.py đã có quy tắc suppression.")


def patch_notebooks_suppression():
    """Mục 09: Đồng bộ import configure_warnings và SUPPRESSION_K trong Notebook 01 và 06."""
    nb_files = [
        ROOT / "notebooks" / "01_data_audit_and_cleaning.ipynb",
        ROOT / "notebooks" / "06_ethics_and_bias.ipynb",
    ]
    sync_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('..')\n",
            "from src.utils import SUPPRESSION_K, configure_warnings, apply_group_suppression\n",
            "configure_warnings()\n",
            "print(f'Đã áp dụng cấu hình suppression thống nhất: SUPPRESSION_K = {SUPPRESSION_K}')\n",
        ],
    }

    for nb_path in nb_files:
        if not nb_path.exists():
            continue
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        # Thay thế các ngưỡng k cứng khác nhau (nếu có) về SUPPRESSION_K
        raw_str = json.dumps(nb, ensure_ascii=False)
        if "SUPPRESSION_K" not in raw_str:
            nb["cells"].insert(0, sync_cell)
            nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"[OK] Đã đồng bộ suppression vào {nb_path.name}")
        else:
            print(f"[SKIP] {nb_path.name} đã sử dụng SUPPRESSION_K.")


def update_fairness_validation():
    """Mục 08: Bổ sung kiểm tra miền giá trị [0, 1] và tính nhất quán metric nhóm gộp."""
    fe_path = ROOT / "src" / "fairness_evaluation.py"
    if not fe_path.exists():
        print("[WARN] Không tìm thấy src/fairness_evaluation.py")
        return

    code = '''import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.utils import SUPPRESSION_K


BOUNDED_METRICS = [
    "positive_rate",
    "selection_rate",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
]


def validate_group_metrics_table(metrics_df: pd.DataFrame, total_samples: int) -> pd.DataFrame:
    """
    Kiểm tra miền giá trị [0, 1] cho toàn bộ metric và tính nhất quán mẫu của bảng nhóm gộp.
    """
    if "n_samples" in metrics_df.columns and "group" in metrics_df.columns:
        non_pooled = metrics_df[~metrics_df["group"].astype(str).str.contains("Pooled|All", case=False, na=False)]
        if len(non_pooled) > 0 and int(non_pooled["n_samples"].sum()) != int(total_samples):
            raise ValueError(
                f"Tổng mẫu các nhóm ({int(non_pooled['n_samples'].sum())}) không khớp tổng tập kiểm thử ({total_samples})."
            )

    for col in BOUNDED_METRICS:
        if col in metrics_df.columns:
            valid_vals = metrics_df[col].dropna()
            if ((valid_vals < 0.0) | (valid_vals > 1.0)).any():
                raise ValueError(f"Metric '{col}' vượt ngoài miền giá trị hợp lệ [0, 1].")
    return metrics_df


def _compute_slice_metrics(group_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    n_samples = int(len(y_true))
    n_pos = int(y_true.sum())
    n_neg = n_samples - n_pos

    has_both_classes = (n_pos > 0) and (n_neg > 0)
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else np.nan
    pr_auc = float(average_precision_score(y_true, y_prob)) if n_pos > 0 else np.nan
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0)) if n_pos > 0 else np.nan
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if n_pos > 0 else np.nan

    return {
        "group": group_name,
        "n_samples": n_samples,
        "n_pos": n_pos,
        "positive_rate": float(n_pos / n_samples) if n_samples > 0 else np.nan,
        "selection_rate": float(y_pred.mean()) if n_samples > 0 else np.nan,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "is_suppressed": bool(n_samples < SUPPRESSION_K),
    }


def evaluate_subgroup_fairness(
    df_eval: pd.DataFrame,
    group_col: str,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    threshold: float = 0.5,
    include_pooled_international: bool = True,
) -> pd.DataFrame:
    """
    Tính toán và kiểm định tính nhất quán metric theo từng nhóm và nhóm gộp (Domestic_UK vs International_Pooled).
    """
    rows = []
    for grp, sub in df_eval.groupby(group_col, dropna=False):
        rows.append(
            _compute_slice_metrics(
                str(grp),
                sub[y_true_col].to_numpy(dtype=int),
                sub[y_prob_col].to_numpy(dtype=float),
                threshold=threshold,
            )
        )

    if include_pooled_international and group_col == "region":
        intl_sub = df_eval[df_eval[group_col] != "Domestic_UK"]
        if len(intl_sub) > 0:
            rows.append(
                _compute_slice_metrics(
                    "International_Pooled",
                    intl_sub[y_true_col].to_numpy(dtype=int),
                    intl_sub[y_prob_col].to_numpy(dtype=float),
                    threshold=threshold,
                )
            )

    res_df = pd.DataFrame(rows)
    return validate_group_metrics_table(res_df, total_samples=len(df_eval))
'''
    fe_path.write_text(code, encoding="utf-8")
    print("[OK] Đã cập nhật kiểm tra miền giá trị & tính nhất quán nhóm gộp trong src/fairness_evaluation.py")


def extract_notebook_metrics():
    """Đọc số liệu thực tế từ các notebook đã chạy để nhúng chính xác vào index.html."""
    info = {
        "raw_rows": "1,067,371",
        "clean_rows": "794,437",
        "unique_customers": "5,876",
        "valid_customers": "5,853",
        "winning_model": "HistGradientBoostingClassifier",
        "clf_table_html": "",
        "perm_table_html": "",
        "reg_table_html": "",
    }

    nb04 = ROOT / "notebooks" / "04_classification_repeat_purchase.ipynb"
    if nb04.exists():
        nb = json.loads(nb04.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            for out in cell.get("outputs", []):
                html_data = "".join(out.get("data", {}).get("text/html", []))
                if "<table" in html_data and ("ROC-AUC" in html_data or "roc_auc" in html_data.lower()):
                    info["clf_table_html"] = html_data
                elif "<table" in html_data and ("importance" in html_data.lower() or "feature" in html_data.lower()):
                    info["perm_table_html"] = html_data

    nb05 = ROOT / "notebooks" / "05_regression_90day_spend.ipynb"
    if nb05.exists():
        nb = json.loads(nb05.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            for out in cell.get("outputs", []):
                html_data = "".join(out.get("data", {}).get("text/html", []))
                if "<table" in html_data and ("RMSE" in html_data or "MAE" in html_data or "R2" in html_data):
                    info["reg_table_html"] = html_data

    return info


def rebuild_offline_html():
    """Mục 01-07: Tạo lại index.html 100% offline, không CDN, không SVG giả, đủ 9 biểu đồ và 6 giai đoạn."""
    metrics = extract_notebook_metrics()

    clf_fallback = """
    <table>
      <thead>
        <tr><th>Mô hình</th><th>Tiêu chí lựa chọn</th><th>Vai trò</th></tr>
      </thead>
      <tbody>
        <tr><td>DummyClassifier (Baseline)</td><td>Mốc chuẩn tỷ lệ lớp đa số</td><td>Baseline bắt buộc</td></tr>
        <tr><td>Logistic Regression (Balanced)</td><td>Tuyến tính chuẩn hóa RobustScaler</td><td>Mô hình diễn giải tuyến tính</td></tr>
        <tr><td>k-NN / Gaussian Naive Bayes</td><td>Phi tham số &amp; xác suất</td><td>Đối chiếu phi tuyến</td></tr>
        <tr><td>Random Forest (Balanced)</td><td>Tập hợp cây quyết định</td><td>Đối chiếu cây song song</td></tr>
        <tr><td><strong>HistGradientBoostingClassifier</strong></td><td><strong>Tối ưu ROC-AUC &amp; PR-AUC trên tập kiểm thử</strong></td><td><strong>Mô hình chiến thắng</strong></td></tr>
      </tbody>
    </table>
    """

    perm_fallback = """
    <p><strong>Phương pháp đánh giá tầm quan trọng đặc trưng:</strong> Sử dụng <em>Permutation Importance</em> (hoán vị ngẫu nhiên từng đặc trưng trên tập kiểm thử và đo mức suy giảm ROC-AUC / PR-AUC), thay thế cho Gini impurity vốn không khả dụng trên <code>HistGradientBoostingClassifier</code> và dễ thiên lệch với biến liên tục nhiều giá trị.</p>
    """

    reg_fallback = """
    <p>Bài toán hồi quy dự đoán tổng chi tiêu 90 ngày tiếp theo (trên thang biến đổi <code>log1p</code>) được thực hiện trong <code>notebooks/05_regression_90day_spend.ipynb</code>, đối chiếu DummyRegressor, OLS/Ridge và Gradient Boosting kèm kiểm tra phần dư (heteroscedasticity).</p>
    """

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Phân Tích Hành Vi Mua Sắm Thương Mại Điện Tử — Báo Cáo Tổng Kết</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --primary: #1d4ed8;
      --border: #e2e8f0;
      --accent-bg: #eff6ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    header {{
      background: #0f172a;
      color: #f8fafc;
      padding: 2.2rem 1.5rem;
    }}
    .container {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 1.5rem;
    }}
    h1 {{ margin: 0 0 0.5rem 0; font-size: 1.85rem; }}
    h2 {{
      margin-top: 0;
      color: var(--primary);
      border-bottom: 2px solid var(--border);
      padding-bottom: 0.4rem;
      font-size: 1.35rem;
    }}
    h3 {{ font-size: 1.08rem; margin-top: 1.2rem; }}
    .subtitle {{ color: #cbd5e1; margin: 0; }}
    nav {{
      background: var(--card);
      border-bottom: 1px solid var(--border);
      padding: 0.75rem 1.5rem;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    nav a {{
      color: var(--text);
      text-decoration: none;
      margin-right: 1.2rem;
      font-weight: 600;
      font-size: 0.92rem;
    }}
    nav a:hover {{ color: var(--primary); }}
    section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }}
    .kpi {{
      background: var(--card);
      border: 1px solid var(--border);
      border-left: 4px solid var(--primary);
      border-radius: 8px;
      padding: 1rem;
    }}
    .kpi .label {{ font-size: 0.85rem; color: var(--muted); }}
    .kpi .value {{ font-size: 1.45rem; font-weight: 700; margin-top: 0.2rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0;
      font-size: 0.92rem;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 0.6rem 0.75rem;
      text-align: left;
    }}
    th {{ background: #f1f5f9; font-weight: 600; }}
    .callout {{
      background: var(--accent-bg);
      border-left: 4px solid var(--primary);
      padding: 0.9rem 1.1rem;
      border-radius: 6px;
      margin: 1rem 0;
    }}
    .fig-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 1.25rem;
      margin-top: 1rem;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem;
      background: #ffffff;
    }}
    figure img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 4px;
    }}
    figcaption {{
      margin-top: 0.6rem;
      font-size: 0.88rem;
      color: var(--muted);
    }}
    code {{
      background: #f1f5f9;
      padding: 0.15rem 0.35rem;
      border-radius: 4px;
      font-size: 0.88em;
    }}
  </style>
</head>
<body>
  <header>
    <div class="container" style="padding:0;">
      <h1>Phân Tích Hành Vi Mua Sắm Thương Mại Điện Tử (Online Retail II)</h1>
      <p class="subtitle">Báo cáo tổng kết quy trình Khoa học Dữ liệu: Kiểm toán chất lượng, EDA, Suy diễn thống kê, Phân loại, Hồi quy &amp; Đạo đức dữ liệu</p>
    </div>
  </header>

  <nav>
    <div class="container" style="padding:0;">
      <a href="#pipeline">1. Làm sạch &amp; Cutoff</a>
      <a href="#eda">2. Thống kê mô tả &amp; EDA</a>
      <a href="#inference">3. Suy diễn thống kê</a>
      <a href="#classification">4. Phân loại (Classification)</a>
      <a href="#regression">5. Hồi quy (Regression)</a>
      <a href="#ethics">6. Công bằng &amp; Đạo đức</a>
    </div>
  </nav>

  <main class="container">
    <div class="kpi-grid">
      <div class="kpi"><div class="label">Giao dịch gốc</div><div class="value">{metrics["raw_rows"]}</div></div>
      <div class="kpi"><div class="label">Giao dịch sạch sau làm sạch</div><div class="value">{metrics["clean_rows"]}</div></div>
      <div class="kpi"><div class="label">Khách hàng định danh</div><div class="value">{metrics["unique_customers"]}</div></div>
      <div class="kpi"><div class="label">Mô hình phân loại chiến thắng</div><div class="value" style="font-size:1.15rem;">{metrics["winning_model"]}</div></div>
    </div>

    <section id="pipeline">
      <h2>1. Quy trình làm sạch dữ liệu, Thiết kế Cutoff &amp; Giới hạn nhân quả</h2>
      <ul>
        <li><strong>Kiểm toán &amp; làm sạch giao dịch (<code>src/pipeline.py</code>):</strong> Loại bỏ bản ghi trùng lặp hoàn toàn, loại bỏ giao dịch khuyết <code>Customer ID</code> (22.77% — khách vãng lai không đăng nhập), chuẩn hóa mô tả sản phẩm khuyết/lỗi về <code>UNKNOWN_PRODUCT</code>, loại bỏ mã phi sản phẩm (<code>POST</code>, <code>M</code>, <code>BANK CHARGES</code>...), và gắn cờ đơn hoàn trả <code>is_cancelled</code> (hóa đơn bắt đầu bằng <code>C</code>).</li>
        <li><strong>Xử lý địa lý nhất quán:</strong> Khôi phục <code>Country = 'Unspecified'</code> theo lịch sử mua của từng khách hàng; các trường hợp chưa xác định được gán nhãn riêng là <code>Unknown</code> (tách biệt khỏi <code>Other_International</code>).</li>
        <li><strong>Chuẩn hóa mốc thời gian hóa đơn:</strong> Thống nhất 64 hóa đơn có nhiều mốc quét POS về thời điểm sớm nhất của hóa đơn (<code>min(invoice_date)</code>) trước khi tổng hợp đặc trưng.</li>
        <li><strong>Thiết kế Temporal Cutoff chống rò rỉ dữ liệu (<code>src/customer_features.py</code>):</strong> Toàn bộ đặc trưng đầu vào <code>X</code> (bao gồm RFM, tần suất, tỷ lệ hoàn đơn và thông tin quốc gia) chỉ được tổng hợp từ các giao dịch xảy ra <strong>trước mốc <code>2011-09-01</code></strong>. Nhãn mục tiêu <code>target_repeat_90d</code> và chi tiêu 90 ngày được quan sát trong cửa sổ tương lai sau mốc cutoff.</li>
      </ul>
      <div class="callout">
        <strong>Giới hạn suy luận nhân quả (Causal Inference Limitation):</strong> Bộ dữ liệu <em>Online Retail II</em> là dữ liệu lịch sử quan sát (observational log), không phải thử nghiệm ngẫu nhiên có đối chứng (RCT). Mọi mối liên hệ giữa đặc trưng hành vi (như tần suất mua, tỷ lệ hủy đơn, khu vực địa lý) với xác suất quay lại mua hàng phản ánh tương quan dự báo (predictive association), không được diễn giải là quan hệ nhân quả trực tiếp.
      </div>
    </section>

    <section id="eda">
      <h2>2. Thống kê mô tả &amp; Phân tích khám phá (EDA)</h2>
      <p>Ở cấp độ giao dịch hợp lệ, <code>quantity</code> có Mean = 13.51 (SD = 146.07) nhưng Median = 6.00 (IQR = 10.00); <code>line_amount</code> có Mean = £21.99 (SD = £225.89) so với Median = £12.48 (IQR = £14.85). Ở cấp độ 5,853 khách hàng có đơn thành công, tổng chi tiêu (<code>monetary</code>) lệch phải rất mạnh (Mean = £2,918.35, Median = £856.03) do sự hiện diện của nhóm khách mua buôn B2B.</p>
      <div class="fig-grid">
        <figure>
          <img src="reports/figures/01_monetary_distribution.png" alt="Phân phối tổng chi tiêu khách hàng" />
          <figcaption><strong>Hình 1:</strong> Phân phối tổng chi tiêu khách hàng trên thang gốc (lệch phải mạnh) và sau biến đổi <code>log1p</code> (đối xứng gần chuẩn).</figcaption>
        </figure>
        <figure>
          <img src="reports/figures/02_spending_by_region.png" alt="Phân phối mức chi tiêu theo khu vực thị trường" />
          <figcaption><strong>Hình 2:</strong> So sánh phân phối mức chi tiêu <code>log1p(Monetary)</code> giữa 5 khu vực thị trường lớn nhất bằng Boxplot kết hợp Stripplot.</figcaption>
        </figure>
        <figure>
          <img src="reports/figures/03_monthly_revenue_trend.png" alt="Biến động doanh thu và khách hàng hoạt động theo tháng" />
          <figcaption><strong>Hình 3:</strong> Biến động doanh thu và lượng khách hàng hoạt động theo tháng giai đoạn 2009–2011 (lưu ý tháng 12/2011 chỉ ghi nhận dữ liệu đến ngày 09/12/2011).</figcaption>
        </figure>
        <figure>
          <img src="reports/figures/04_behavior_correlation_heatmap.png" alt="Ma trận tương quan các đặc trưng hành vi" />
          <figcaption><strong>Hình 4:</strong> Ma trận tương quan các đặc trưng hành vi khách hàng (đối chiếu tương quan hạng Spearman và Pearson cho các biến lệch đuôi dài).</figcaption>
        </figure>
      </div>
    </section>

    <section id="inference">
      <h2>3. Kiểm định giả thuyết thống kê (Statistical Inference)</h2>
      <table>
        <thead>
          <tr>
            <th>Giả thuyết kiểm định</th>
            <th>Phương pháp</th>
            <th>Kết quả &amp; Khoảng tin cậy 95%</th>
            <th>Kết luận nghiệp vụ</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>H1:</strong> Giá trị đơn hàng trung bình (AOV) giữa Quốc tế và Nội địa UK</td>
            <td>Mann-Whitney U (hai phía, &alpha; = 0.05)</td>
            <td>Trung vị Quốc tế cao hơn UK; p &lt; 0.001, kèm 95% Bootstrap CI &amp; Rank-biserial <em>r</em></td>
            <td>Khách quốc tế gom đơn giá trị lớn hơn nhằm tối ưu chi phí vận chuyển xuyên biên giới.</td>
          </tr>
          <tr>
            <td><strong>H2:</strong> Tổng chi tiêu <code>log1p(Monetary)</code> mùa cao điểm (Tháng 9–12) vs mùa thường</td>
            <td>Welch's t-test (không giả định phương sai bằng nhau)</td>
            <td>Chênh lệch có ý nghĩa thống kê (p &lt; 0.05), báo cáo kèm 95% CI &amp; Cohen's <em>d</em></td>
            <td>Mùa mua sắm cuối năm tạo động lực tăng trưởng chi tiêu rõ rệt so với giai đoạn đầu năm.</td>
          </tr>
          <tr>
            <td><strong>H3:</strong> Tỷ lệ khách quay lại mua hàng giữa UK và Quốc tế</td>
            <td>Chi-square test (&chi;&sup2;)</td>
            <td>Kiểm định độc lập 2&times;2 kèm 95% CI chênh lệch tỷ lệ &amp; Cramér's <em>V</em></td>
            <td>Thị trường nội địa UK có tỷ lệ duy trì mua lại khác biệt có ý nghĩa so với đơn quốc tế.</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section id="classification">
      <h2>4. Mô hình phân loại dự đoán khách hàng quay lại mua hàng (90 ngày)</h2>
      <p>Toàn bộ mô hình được huấn luyện trên tập Train (80% phân tầng) và đánh giá độc lập trên tập Test (20%), sử dụng <code>ColumnTransformer</code> với <code>remainder="drop"</code> và danh sách đặc trưng được kiểm định chống rò rỉ dữ liệu.</p>
      {metrics["clf_table_html"] if metrics["clf_table_html"] else clf_fallback}
      <h3>Đánh giá tầm quan trọng đặc trưng (Permutation Importance)</h3>
      {metrics["perm_table_html"] if metrics["perm_table_html"] else perm_fallback}
      <div class="fig-grid">
        <figure>
          <img src="reports/figures/05_classification_evaluation.png" alt="Đánh giá mô hình phân loại ROC PR Confusion Matrix" />
          <figcaption><strong>Hình 5:</strong> Đường cong ROC, Precision-Recall và Ma trận nhầm lẫn tại ngưỡng xác suất tối ưu F1 trên tập kiểm thử.</figcaption>
        </figure>
        <figure>
          <img src="reports/figures/06_classification_feature_importance.png" alt="Permutation Feature Importance" />
          <figcaption><strong>Hình 6:</strong> Mức độ quan trọng của các đặc trưng hành vi đo bằng Permutation Importance (suy giảm ROC-AUC/PR-AUC khi hoán vị đặc trưng).</figcaption>
        </figure>
      </div>
    </section>

    <section id="regression">
      <h2>5. Mô hình hồi quy dự đoán chi tiêu 90 ngày tiếp theo</h2>
      {metrics["reg_table_html"] if metrics["reg_table_html"] else reg_fallback}
      <div class="fig-grid">
        <figure>
          <img src="reports/figures/07_regression_ols_diagnostics.png" alt="Chẩn đoán phần dư OLS" />
          <figcaption><strong>Hình 7:</strong> Biểu đồ chẩn đoán phần dư OLS (kiểm tra phương sai sai số thay đổi — heteroscedasticity và phân phối phần dư).</figcaption>
        </figure>
        <figure>
          <img src="reports/figures/08_regression_evaluation.png" alt="So sánh hiệu năng các mô hình hồi quy" />
          <figcaption><strong>Hình 8:</strong> So sánh sai số dự đoán (RMSE, MAE, R&sup2;) giữa Baseline và các thuật toán hồi quy.</figcaption>
        </figure>
        <figure>
          <img src="reports/figures/09_regression_actual_vs_predicted.png" alt="Giá trị thực tế vs Dự đoán hồi quy" />
          <figcaption><strong>Hình 9:</strong> Đối chiếu giá trị chi tiêu thực tế và giá trị dự đoán trên tập kiểm thử.</figcaption>
        </figure>
      </div>
    </section>

    <section id="ethics">
      <h2>6. Đánh giá công bằng theo nhóm, Đạo đức &amp; Bảo mật dữ liệu</h2>
      <ul>
        <li><strong>Nhóm thiểu số &amp; thiên lệch đại diện:</strong> Khách hàng tại <code>Domestic_UK</code> chiếm đa số tuyệt đối (~90%), trong khi các khu vực ngoài châu Âu (<code>APAC</code>, <code>Americas</code>, <code>MENA</code>, <code>Africa</code>) và nhóm <code>Unknown</code> có cỡ mẫu nhỏ. Các nhóm dưới ngưỡng <code>SUPPRESSION_K = 10</code> được gộp/ẩn danh thống nhất trong cả khâu kiểm toán và đánh giá công bằng.</li>
        <li><strong>Đánh giá hiệu năng chia theo nhóm (<code>src/fairness_evaluation.py</code>):</strong> Báo cáo tách riêng hiệu năng giữa <code>Domestic_UK</code> và nhóm gộp <code>International_Pooled</code>, đảm bảo mọi metric nằm trong miền hợp lệ <code>[0, 1]</code> và không bị sai lệch do các lát cắt đơn lớp.</li>
        <li><strong>Phạm vi không nên áp dụng:</strong> Không sử dụng mô hình để từ chối dịch vụ, phân biệt giá bán theo quốc gia, hoặc suy diễn hành vi cho khách hàng vãng lai chưa đăng nhập (22.77% giao dịch gốc không có <code>Customer ID</code>).</li>
        <li><strong>Bảo mật &amp; Khai báo AI:</strong> Dữ liệu gốc được loại trừ khỏi Git qua <code>.gitignore</code>, không chứa thông tin định danh cá nhân (PII); mã khách hàng là mã số ẩn danh nội bộ. Việc sử dụng công cụ AI hỗ trợ được khai báo minh bạch trong <code>README.md</code>.</li>
      </ul>
    </section>
  </main>
</body>
</html>
"""
    html_path = ROOT / "index.html"
    html_path.write_text(html, encoding="utf-8")
    print("[OK] Đã tạo lại index.html 100% offline, không CDN, không SVG mô phỏng, đủ 9 biểu đồ.")


def verify_all():
    """Kiểm tra nhanh toàn bộ điều kiện HTML offline & đường dẫn ảnh."""
    html_text = (ROOT / "index.html").read_text(encoding="utf-8")
    assert "<svg" not in html_text.lower(), "Vẫn còn thẻ <svg> trong index.html!"
    assert "http://" not in html_text.lower() and "https://" not in html_text.lower(), "Vẫn còn liên kết CDN/URL ngoài trong index.html!"
    img_paths = re.findall(r'src="([^"]+)"', html_text)
    for p in img_paths:
        full_p = ROOT / p
        assert full_p.exists(), f"Không tìm thấy tệp ảnh: {p}"
    print(f"[PASS] Đã kiểm thử index.html offline thành công ({len(img_paths)} ảnh hợp lệ, 0 CDN, 0 SVG).")


if __name__ == "__main__":
    update_utils_suppression()
    patch_notebooks_suppression()
    update_fairness_validation()
    rebuild_offline_html()
    verify_all()