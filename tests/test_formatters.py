import unittest

from src.utils.formatters import (
    NA_TEXT,
    format_bp,
    format_index,
    format_market_cap,
    format_pct,
    format_price,
    format_spread_bp,
    format_trading_value,
    format_usdkrw,
    format_volume,
    safe_change_rate,
)


class FormatterTests(unittest.TestCase):
    def test_format_usdkrw(self):
        self.assertEqual(format_usdkrw(1450.25), "1달러 = 1,450.25원")

    def test_format_index(self):
        self.assertEqual(format_index(5123.456), "5,123.46")

    def test_format_pct(self):
        self.assertEqual(format_pct(0.012), "+1.20%")
        self.assertEqual(format_pct(1.2), "+1.20%")

    def test_safe_change_rate(self):
        self.assertAlmostEqual(safe_change_rate(0.012), 0.012)
        self.assertAlmostEqual(safe_change_rate(1.2), 0.012)

    def test_format_price_and_volume(self):
        self.assertEqual(format_price(12345), "12,345원")
        self.assertEqual(format_volume(12345678), "12,345,678주")

    def test_format_trading_value(self):
        self.assertEqual(format_trading_value(123_456_789_000), "1,235억원")

    def test_format_market_cap(self):
        self.assertEqual(format_market_cap(432_100_000_000_000), "432.1조원")
        self.assertEqual(format_market_cap(None), NA_TEXT)

    def test_format_spread_bp(self):
        self.assertEqual(format_spread_bp(50.2), "+50.2bp")
        self.assertEqual(format_bp(-20.5), "-20.5bp")
        self.assertEqual(format_spread_bp(None), NA_TEXT)


if __name__ == "__main__":
    unittest.main()
