import unittest

from src.services.supabase_stockdata_reader import SupabaseStockDataReader


class SupabaseStockDataReaderTests(unittest.TestCase):
    def setUp(self):
        self.reader = SupabaseStockDataReader.__new__(SupabaseStockDataReader)
        self.reader._last_watchlist_diagnostics = {}

    def test_change_rate_recalculated_from_change_amount_and_previous_value(self):
        warnings = []
        rate = self.reader._normalize_percent_change_rate(
            field="sp500",
            explicit_rate=-40.62,
            change_value=-41.0,
            previous_value=7241.0,
            warnings=warnings,
        )
        self.assertAlmostEqual(rate, -41.0 / 7241.0, places=6)
        self.assertEqual(warnings, [])

    def test_decimal_change_rate_is_preserved(self):
        warnings = []
        rate = self.reader._normalize_percent_change_rate(
            field="sp500",
            explicit_rate=-0.0057,
            change_value=None,
            previous_value=None,
            warnings=warnings,
        )
        self.assertAlmostEqual(rate, -0.0057, places=6)

    def test_percent_unit_change_rate_is_normalized(self):
        warnings = []
        rate = self.reader._normalize_percent_change_rate(
            field="sp500",
            explicit_rate=-0.57,
            change_value=None,
            previous_value=None,
            warnings=warnings,
        )
        self.assertAlmostEqual(rate, -0.0057, places=6)

    def test_abnormal_change_rate_becomes_warning(self):
        warnings = []
        rate = self.reader._normalize_percent_change_rate(
            field="sp500",
            explicit_rate=-40.62,
            change_value=None,
            previous_value=None,
            warnings=warnings,
        )
        self.assertIsNone(rate)
        self.assertTrue(any("anomaly" in warning for warning in warnings))

    def test_normalize_readiness_derives_blocked_sections(self):
        normalized = self.reader.normalize_report_readiness(
            {
                "kr_full_market_price_ready": False,
                "kis_universe_ready": True,
                "kis_volume_ranking_ready": True,
                "kr_trading_value_ranking_ready": True,
                "kr_market_cap_ranking_ready": True,
                "report_allowed_sections": ["macro", "us_market"],
                "report_blocked_sections": [],
            }
        )
        self.assertEqual(normalized["display_mode"], "KIS_UNIVERSE_ONLY")
        self.assertIn("kr_full_market_trading_value_top", normalized["report_blocked_sections"])
        self.assertIn("kr_full_market_market_cap_top", normalized["report_blocked_sections"])
        self.assertIn("watchlist_signal", normalized["report_allowed_sections"])

    def test_watchlist_snapshot_filters_to_active_symbols(self):
        class BaseReaderStub:
            def fetch_static_stock_universe(self):
                return [
                    {"symbol": "005930", "name": "삼성전자", "market": "KOSPI"},
                    {"symbol": "000660", "name": "SK하이닉스", "market": "KOSPI"},
                ]

        self.reader.base_reader = BaseReaderStub()
        self.reader._fetch_view_rows = lambda *args, **kwargs: [
            {"symbol": "005930", "name": "삼성전자", "market": "KOSPI", "base_date": "2026-05-08", "close_price": 1, "data_status": "FRESH"},
            {"symbol": "000660", "name": "SK하이닉스", "market": "KOSPI", "base_date": "2026-05-08", "close_price": 1, "data_status": "FRESH"},
            {"symbol": "123456", "name": "과거종목", "market": "KOSPI", "base_date": "2026-05-08", "close_price": 1, "data_status": "FRESH"},
        ]
        self.reader._fetch_watchlist_quality_map = lambda symbols: {}
        rows = self.reader.get_watchlist_snapshot("2026-05-08")
        self.assertEqual([row["symbol"] for row in rows], ["000660", "005930"])
        self.assertEqual(self.reader._last_watchlist_diagnostics["raw_row_count"], 3)
        self.assertEqual(self.reader._last_watchlist_diagnostics["active_row_count"], 2)

    def test_resolve_source_mixed_from_quality_flag(self):
        self.assertTrue(
            self.reader._resolve_source_mixed(
                explicit_value=None,
                data_quality_flag="SOURCE_MIXED",
                consistency_status=None,
                quality_hint=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
