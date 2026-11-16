"""Kiểm thử nhỏ không cần dataset thật: python -m unittest discover -s tests -v."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from src.pipeline import RAW_COLUMNS, load_raw_data, validate_raw_data
from src.customer_features import build_orders, aggregate_orders, classify_market


def sample_raw():
    return pd.DataFrame([["123456", "12345", "Gift Voucher £80.00", 1,
                          "2009-12-01 12:00:00", 80.0, 12345, "United Kingdom"]], columns=RAW_COLUMNS)


class TestInput(unittest.TestCase):
    def test_utf8_and_exact_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "unrelated.csv").write_text("bad\n1\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                load_raw_data(path)
            sample_raw().to_csv(path / "online_retail_II.csv", index=False, encoding="utf-8-sig")
            self.assertEqual(load_raw_data(path).iloc[0]["Description"], "Gift Voucher £80.00")

    def test_schema_and_required_values(self):
        with self.assertRaises(ValueError):
            validate_raw_data(sample_raw().drop(columns="Price"), "test")
        data = sample_raw()
        data.loc[0, "Price"] = float("inf")
        with self.assertRaises(ValueError):
            validate_raw_data(data, "test")
        data = sample_raw()
        data.loc[0, "Customer ID"] = float("nan")
        self.assertTrue(validate_raw_data(data, "test")["Customer ID"].isna().all())

    def test_excel_two_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with pd.ExcelWriter(path / "online_retail_II.xlsx", engine="openpyxl") as writer:
                for name in ["Year 2009-2010", "Year 2010-2011"]:
                    sample_raw().to_excel(writer, sheet_name=name, index=False)
            self.assertEqual(len(load_raw_data(path)), 2)


class TestCustomerFeatures(unittest.TestCase):
    def test_ratio_by_invoice_not_line(self):
        data = pd.DataFrame({"customer_id": ["1"] * 4, "invoice": ["A", "A", "A", "B"],
                             "invoice_date": pd.to_datetime(["2010-10-02"] * 3 + ["2010-04-05"]),
                             "total_amount": [1, 2, 3, 4]})
        row = aggregate_orders(build_orders(data)).iloc[0]
        self.assertEqual(row.frequency, 2)
        self.assertEqual(row.peak_month_ratio, 0.5)
        self.assertEqual(row.weekend_ratio, 0.5)
        self.assertEqual(row.avg_order_value, 5)

    def test_market_from_observed_window(self):
        data = pd.DataFrame({"customer_id": ["1", "1", "2", "3", "4", "4"],
                             "country_original": ["United Kingdom", "France", "Unspecified", "France", "United Kingdom", "Unspecified"],
                             "country": ["United Kingdom", "France", "United Kingdom", "France", "United Kingdom", "United Kingdom"]})
        labels = classify_market(data).set_index("customer_id")["is_uk"].to_dict()
        self.assertEqual(labels, {"1": "Mixed", "2": "Unknown", "3": "International", "4": "Unknown"})
        self.assertEqual(classify_market(data.iloc[:1]).iloc[0]["is_uk"], "UK")


if __name__ == "__main__":
    unittest.main()
