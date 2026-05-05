import unittest

from src.utils.report_universe import (
    evaluate_etf_coverage,
    evaluate_macro_freshness,
    evaluate_raw_retention,
    evaluate_watchlist_coverage,
)


class ReportFreshnessLogicTests(unittest.TestCase):
    def test_etf_coverage_detects_stale_primary(self):
        result = evaluate_etf_coverage(
            [{"symbol": "396500", "role": "primary", "is_active": True, "exclude_from_signal": False}],
            [{"symbol": "396500", "stale_days": 4, "exclude_from_signal": False}],
            stale_warn_days=3,
        )
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["stale_primary"], ["396500"])

    def test_macro_freshness_detects_missing_series(self):
        result = evaluate_macro_freshness(
            [{"symbol": "US10Y", "series_id": "US10Y", "is_active": True}],
            [],
            "2026-05-05",
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["missing"], ["US10Y"])

    def test_watchlist_coverage_detects_missing_supply(self):
        result = evaluate_watchlist_coverage(
            ["005930", "000660"],
            [{"symbol": "005930"}, {"symbol": "000660"}],
            [{"symbol": "005930"}],
        )
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["missing_supplies"], ["000660"])

    def test_raw_retention_detects_overage(self):
        result = evaluate_raw_retention(
            "2026-05-05",
            {"raw_stock_prices_daily": "2026-02-01", "raw_market_rankings": "2026-04-20"},
        )
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["issues"][0]["table"], "raw_stock_prices_daily")


if __name__ == "__main__":
    unittest.main()
