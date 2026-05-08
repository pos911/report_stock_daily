import unittest

from src.signals.sector_impact import build_sector_morning_impacts


class SectorImpactTests(unittest.TestCase):
    def test_fresh_semiconductor_etf_with_sox_strength_is_positive(self):
        impacts = build_sector_morning_impacts(
            {"regime_label": "Risk-on", "positive_drivers": ["SOX +2.00", "Nasdaq +1.00"], "negative_drivers": []},
            [{"symbol": "396500", "name": "TIGER 반도체TOP10", "sector_group": "반도체", "theme_group": "반도체", "role": "primary", "exclude_from_signal": False, "data_status": "FRESH", "change_rate_1d": 0.02, "return_20d": 0.10, "trading_value_ratio_20d": 2.5, "warnings": []}],
            [],
            [],
        )
        self.assertEqual(impacts[0]["label"], "우호")
        self.assertIn("%", impacts[0]["etf_reason"])

    def test_stale_shipbuilding_etf_limits_etf_reason(self):
        impacts = build_sector_morning_impacts(
            {"regime_label": "Neutral", "positive_drivers": [], "negative_drivers": []},
            [{"symbol": "494670", "name": "TIGER 조선TOP10", "sector_group": "조선", "theme_group": "조선", "role": "primary", "exclude_from_signal": False, "data_status": "STALE", "stale_days": 4, "warnings": []}],
            [],
            [],
        )
        self.assertIn("stale", " ".join(impacts[0]["warnings"]).lower())
        self.assertIn("오래", impacts[0]["etf_reason"])

    def test_leverage_etf_is_not_primary_signal(self):
        impacts = build_sector_morning_impacts(
            {"regime_label": "Mild risk-on", "positive_drivers": [], "negative_drivers": []},
            [{"symbol": "462330", "name": "KODEX 2차전지산업레버리지", "sector_group": "2차전지", "theme_group": "2차전지", "role": "primary", "exclude_from_signal": True, "data_status": "FRESH", "warnings": []}],
            [],
            [],
        )
        self.assertIn("Speculative ETF excluded", impacts[0]["warnings"])
        self.assertIn("과열 참고 신호", impacts[0]["etf_reason"])


if __name__ == "__main__":
    unittest.main()
