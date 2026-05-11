import unittest

from src.utils.formatters import (
    NA_TEXT,
    detect_market_value_anomaly,
    detect_stock_price_anomaly,
    format_bp,
    format_index,
    format_market_cap,
    format_pct,
    format_price,
    format_spread_bp,
    format_trading_value,
    format_usdkrw,
    format_volume,
    format_yield_spread,
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
        self.assertEqual(format_spread_bp(0.468), "+46.8bp")
        self.assertEqual(format_bp(-20.5), "-20.5bp")
        self.assertEqual(format_spread_bp(None), NA_TEXT)
        self.assertEqual(format_yield_spread(0.468, -4.4), "+46.8bp / 전일대비 -4.4bp")

    def test_market_anomaly_warning(self):
        self.assertEqual(
            detect_market_value_anomaly("KOSPI", 7822.24, change_rate=0.001, as_of_date="2026-05-11", target_date="2026-05-11"),
            "지수 원천 확인 필요",
        )
        self.assertIsNone(detect_market_value_anomaly("KOSPI", 2700))
        self.assertIsNotNone(detect_market_value_anomaly("KOSPI", 0))

    def test_stock_price_anomaly_warning(self):
        self.assertIsNone(detect_stock_price_anomaly("000660", "SK하이닉스", 1_686_000))
        self.assertIsNone(detect_stock_price_anomaly("005930", "삼성전자", 68000))
        self.assertIsNotNone(
            detect_stock_price_anomaly(
                "005930",
                "삼성전자",
                268500,
                data_quality_flag="SOURCE_MIXED",
            )
        )
        self.assertIsNotNone(
            detect_stock_price_anomaly(
                "005930",
                "삼성전자",
                268500,
                previous_price=120000,
            )
        )


if __name__ == "__main__":
    unittest.main()
