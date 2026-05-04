import unittest

from src.analysis.gemini_analyzer import GeminiAnalyzer
from src.jobs.generate_report import _interpret_us_10y_3y_spread, _score_watchlist_snapshot
from scripts.backtest_signal_score import BacktestConfig, build_dataset_plan


class YieldCurveAndSignalTests(unittest.TestCase):
    def test_interpret_us_10y_3y_spread_positive(self):
        result = _interpret_us_10y_3y_spread(4.412, 3.91)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["spread_bp"], 50.2, places=1)
        self.assertEqual(result["regime"], "mildly_positive")
        self.assertIn("정상", result["plain_korean_summary"])

    def test_interpret_us_10y_3y_spread_missing(self):
        self.assertIsNone(_interpret_us_10y_3y_spread(4.1, None))

    def test_compact_macro_data_includes_us3y_and_spread(self):
        compact = GeminiAnalyzer._compact_global_macro_data(
            {
                "base_date": "2026-05-04",
                "us10y": 4.412,
                "us3y": 3.91,
                "us10y_us3y_spread": 0.502,
                "us10y_us3y_spread_bp": 50.2,
                "yield_curve_regime": "mildly_positive",
            }
        )
        self.assertEqual(compact["us3y"], 3.91)
        self.assertEqual(compact["us10y_us3y_spread_bp"], 50.2)
        self.assertEqual(compact["yield_curve_regime"], "mildly_positive")

    def test_signal_score_uses_new_labels(self):
        snapshot = {
            "symbol": "005930",
            "market": "KOSPI",
            "price": {"close_price": 80000, "trading_value": 500_000_000_000},
            "supply": {"foreign_net_buy": 100000, "institutional_net_buy": 100000},
            "features": {
                "return_5d": 0.03,
                "moving_avg_5": 79000,
                "moving_avg_20": 77000,
                "volatility_20d": 0.03,
                "foreign_flow_zscore": 1.5,
            },
            "fundamentals_diag": {"display": {"per": "12.3배", "pbr": "1.2배"}},
            "short_diag": {"needs_review": False},
            "short_selling": {"short_ratio": 1.2},
        }
        ranking_lookup = {"005930": {"volume_rank": 2, "trading_value_rank": 1}}
        macro = {"usdkrw": 1430, "us10y": 4.2}

        score = _score_watchlist_snapshot(snapshot, ranking_lookup, macro)
        self.assertIn(score["label"], {"비중확대 후보", "보유/관찰", "관망", "리스크 축소 후보", "판단 유보"})
        self.assertNotIn(score["label"], {"BUY", "SELL", "HOLD"})

    def test_backtest_skeleton_plan_is_buildable(self):
        plan = build_dataset_plan(BacktestConfig(date_from="2026-01-01", date_to="2026-05-01", market="KOSPI", dry_run=True))
        self.assertEqual(plan["signal_model_version"], "v0.1_unbacktested")
        self.assertIn("normalized_market_rankings_daily", plan["tables"]["rankings"])
        self.assertEqual(plan["forward_windows"], [1, 5, 20])


if __name__ == "__main__":
    unittest.main()
