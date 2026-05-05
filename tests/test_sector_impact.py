import unittest

from src.signals.sector_impact import build_sector_morning_impacts


class SectorImpactTests(unittest.TestCase):
    def test_fresh_semiconductor_etf_with_sox_strength_is_positive(self):
        impacts = build_sector_morning_impacts(
            {"regime_label": "Risk-on", "positive_drivers": ["SOX +2.00", "Nasdaq +1.00"], "negative_drivers": []},
            [{"symbol": "396500", "name": "TIGER Semiconductor TOP10", "sector_group": "Semiconductor", "theme_group": "Semiconductor", "role": "primary", "exclude_from_signal": False, "data_status": "FRESH", "change_rate_1d": 0.02, "return_20d": 0.10, "trading_value_ratio_20d": 2.5, "warnings": []}],
            [],
            [],
        )
        self.assertEqual(impacts[0]["label"], "우호")

    def test_stale_shipbuilding_etf_excludes_etf_weight(self):
        impacts = build_sector_morning_impacts(
            {"regime_label": "Neutral", "positive_drivers": [], "negative_drivers": []},
            [{"symbol": "494670", "name": "TIGER Shipbuilding TOP10", "sector_group": "Shipbuilding", "theme_group": "Shipbuilding", "role": "primary", "exclude_from_signal": False, "data_status": "STALE", "stale_days": 4, "warnings": []}],
            [],
            [],
        )
        self.assertIn("stale", " ".join(impacts[0]["warnings"]).lower())
        self.assertEqual(impacts[0]["data_status"], "STALE")

    def test_leverage_etf_is_excluded_from_primary_score(self):
        impacts = build_sector_morning_impacts(
            {"regime_label": "Mild risk-on", "positive_drivers": [], "negative_drivers": []},
            [{"symbol": "462330", "name": "KODEX Secondary Battery Industry Leverage", "sector_group": "Battery", "theme_group": "Battery", "role": "primary", "exclude_from_signal": True, "data_status": "FRESH", "warnings": []}],
            [],
            [],
        )
        self.assertIn("Excluded", " ".join(impacts[0]["warnings"]))
        self.assertNotEqual(impacts[0]["label"], "우호")

    def test_fresh_primary_beats_fresh_leverage_reference(self):
        impacts = build_sector_morning_impacts(
            {"regime_label": "Neutral", "positive_drivers": [], "negative_drivers": []},
            [
                {"symbol": "305720", "name": "KODEX 2차전지산업", "sector_group": "Battery", "theme_group": "Battery", "role": "primary", "exclude_from_signal": False, "data_status": "FRESH", "change_rate_1d": 0.01, "return_20d": 0.05, "warnings": []},
                {"symbol": "462330", "name": "KODEX 2차전지산업레버리지", "sector_group": "Battery", "theme_group": "Battery", "role": "satellite", "exclude_from_signal": True, "data_status": "FRESH", "change_rate_1d": 0.08, "return_20d": 0.20, "warnings": []},
            ],
            [],
            [],
        )
        self.assertEqual(impacts[0]["etf_symbol"], "305720")
        self.assertIn("Speculative ETF excluded", impacts[0]["warnings"])

    def test_supply_text_is_omitted_when_null(self):
        impacts = build_sector_morning_impacts(
            {"regime_label": "Neutral", "positive_drivers": [], "negative_drivers": []},
            [{"symbol": "091170", "name": "KODEX Banks", "sector_group": "Financials", "theme_group": "Banks", "role": "primary", "exclude_from_signal": False, "data_status": "FRESH", "warnings": []}],
            [],
            [],
        )
        self.assertEqual(impacts[0]["investor_reason"], "Investor flow unavailable")


if __name__ == "__main__":
    unittest.main()
