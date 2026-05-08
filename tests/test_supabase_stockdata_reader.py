import unittest

from src.services.supabase_stockdata_reader import SupabaseStockDataReader


class SupabaseStockDataReaderTests(unittest.TestCase):
    def setUp(self):
        self.reader = SupabaseStockDataReader.__new__(SupabaseStockDataReader)

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


if __name__ == "__main__":
    unittest.main()
