import unittest
from unittest.mock import patch

from scripts import verify_data
from src.utils.report_universe import classify_overall_status


class VerifyDataContractTests(unittest.TestCase):
    def test_classify_overall_status_ignores_list_payload(self):
        status = classify_overall_status(
            [
                {"status": "PASS"},
                [{"status": "FAIL"}],
                {"status": "WARN_PARTIAL"},
                {"note": "no status"},
            ]
        )
        self.assertEqual(status, "WARN")

    @patch("scripts.verify_data._client")
    @patch("scripts.verify_data.load_report_required_stock_universe")
    @patch("scripts.verify_data.load_report_required_etf_universe")
    @patch("scripts.verify_data.load_report_required_macro_series")
    @patch("scripts.verify_data.fetch_view_rows")
    @patch("scripts.verify_data.fetch_latest_macro_rows")
    @patch("scripts.verify_data.fetch_watchlist_rows")
    @patch("scripts.verify_data.fetch_raw_oldest_dates")
    def test_build_verification_report_separates_diagnostics(
        self,
        mock_raw_dates,
        mock_watchlist_rows,
        mock_macro_rows,
        mock_fetch_view_rows,
        mock_macro_cfg,
        mock_etf_cfg,
        mock_stock_cfg,
        mock_client,
    ):
        mock_client.return_value = object()
        mock_stock_cfg.return_value = [{"symbol": "005930", "is_active": True}]
        mock_etf_cfg.return_value = [{"symbol": "396500", "is_active": True, "exclude_from_signal": False}]
        mock_macro_cfg.return_value = [{"symbol": "S&P500", "series_id": "S&P500", "is_active": True}]
        mock_fetch_view_rows.side_effect = [
            [{"symbol": "396500", "stale_days": 0, "exclude_from_signal": False}],
            [{"symbol": "005930", "base_date": "2026-05-08"}],
            [{"latest_stock_price_date": "2026-05-08"}],
            [{"base_date": "2026-05-08"}],
        ]
        mock_macro_rows.return_value = [{"series_id": "S&P500", "base_date": "2026-05-08"}]
        mock_watchlist_rows.return_value = [{"symbol": "005930", "base_date": "2026-05-08"}]
        mock_raw_dates.return_value = {}

        report = verify_data.build_verification_report("2026-05-08")
        self.assertIn("diagnostics", report)
        self.assertIn("freshness_view_rows", report["diagnostics"])
        self.assertEqual(report["overall_status"], "PASS")


if __name__ == "__main__":
    unittest.main()
