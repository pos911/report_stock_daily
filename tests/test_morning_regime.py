import unittest

from src.signals.morning_regime import build_global_morning_regime


class MorningRegimeTests(unittest.TestCase):
    def test_sox_and_nasdaq_strength_create_positive_driver(self):
        regime = build_global_morning_regime(
            {
                "sp500_change_rate": 0.006,
                "nasdaq_change_rate": 0.01,
                "sox_change_rate": 0.02,
                "vix": 14,
                "usdkrw_change_rate": -0.004,
                "dxy_change_rate": -0.004,
                "us10y_change_bp": -6,
                "us3y_change_bp": -6,
                "brent_change_rate": -0.01,
                "advancing_ratio": 0.60,
                "kospi_foreign_net_buy": 1,
                "kospi_institutional_net_buy": 1,
            },
            {"xnys_is_open": True, "carry_forward_fields": [], "missing_required_data": ""},
        )
        self.assertEqual(regime["regime_label"], "Risk-on")
        self.assertTrue(any("SOX" in driver for driver in regime["positive_drivers"]))

    def test_usdkrw_and_us10y_rise_create_negative_driver(self):
        regime = build_global_morning_regime(
            {
                "sp500_change_rate": 0,
                "nasdaq_change_rate": 0,
                "sox_change_rate": 0,
                "vix": 21,
                "usdkrw_change_rate": 0.004,
                "dxy_change_rate": 0.004,
                "us10y_change_bp": 6,
                "us3y_change_bp": 6,
                "brent_change_rate": 0.03,
                "advancing_ratio": 0.40,
                "kospi_foreign_net_buy": -1,
                "kospi_institutional_net_buy": -1,
            },
            {"xnys_is_open": True, "carry_forward_fields": [], "missing_required_data": ""},
        )
        self.assertIn(regime["regime_label"], {"Cautious", "Risk-off"})
        self.assertTrue(any("US10Y" in driver for driver in regime["negative_drivers"]))

    def test_null_metrics_become_warnings(self):
        regime = build_global_morning_regime({}, {"xnys_is_open": True, "carry_forward_fields": [], "missing_required_data": ""})
        self.assertTrue(regime["warnings"])

    def test_xnys_closed_halves_us_weight(self):
        open_regime = build_global_morning_regime(
            {"sp500_change_rate": 0.01, "nasdaq_change_rate": 0.01, "sox_change_rate": 0.02, "vix": 14, "usdkrw_change_rate": 0, "dxy_change_rate": 0, "us10y_change_bp": 0, "us3y_change_bp": 0, "brent_change_rate": 0, "advancing_ratio": 0.5, "kospi_foreign_net_buy": 0, "kospi_institutional_net_buy": 0},
            {"xnys_is_open": True, "carry_forward_fields": [], "missing_required_data": ""},
        )
        carry_regime = build_global_morning_regime(
            {"sp500_change_rate": 0.01, "nasdaq_change_rate": 0.01, "sox_change_rate": 0.02, "vix": 14, "usdkrw_change_rate": 0, "dxy_change_rate": 0, "us10y_change_bp": 0, "us3y_change_bp": 0, "brent_change_rate": 0, "advancing_ratio": 0.5, "kospi_foreign_net_buy": 0, "kospi_institutional_net_buy": 0},
            {"xnys_is_open": False, "carry_forward_fields": ["sp500"], "missing_required_data": ""},
        )
        self.assertLess(carry_regime["score"], open_regime["score"])


if __name__ == "__main__":
    unittest.main()
