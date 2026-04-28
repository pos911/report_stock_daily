import unittest

from src.utils.formatters import (
    NA_TEXT,
    format_index,
    format_market_cap,
    format_percent,
    format_price,
    format_trading_value,
    format_usdkrw,
    format_volume,
)


class FormatterTests(unittest.TestCase):
    def test_format_usdkrw(self):
        self.assertEqual(format_usdkrw(1450.25), "1달러 = 1,450.25원")

    def test_format_index(self):
        self.assertEqual(format_index(5123.456), "5,123.46")

    def test_format_percent(self):
        self.assertEqual(format_percent(1.2345), "+1.23%")
        self.assertEqual(format_percent(-0.42), "-0.42%")

    def test_format_price_and_volume(self):
        self.assertEqual(format_price(12345), "12,345원")
        self.assertEqual(format_volume(12345678), "12,345,678주")

    def test_format_trading_value(self):
        self.assertEqual(format_trading_value(123_456_789_000), "1,235억원")

    def test_format_market_cap(self):
        self.assertEqual(format_market_cap(432_100_000_000_000), "432.1조원")
        self.assertEqual(format_market_cap(None), NA_TEXT)


if __name__ == "__main__":
    unittest.main()
