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


if __name__ == "__main__":
    unittest.main()
