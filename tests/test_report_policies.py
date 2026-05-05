import unittest

from src.utils.report_universe import PIPELINE_FREQUENCIES, REPORT_FEATURE_CONTRACT, RETENTION_POLICIES


class ReportPolicyTests(unittest.TestCase):
    def test_feature_contract_contains_required_features(self):
        self.assertEqual(
            REPORT_FEATURE_CONTRACT,
            [
                "return_5d",
                "return_20d",
                "return_60d",
                "trading_value_ratio_20d",
                "volatility_20d",
                "near_52w_high_pct",
                "foreign_flow_direction",
                "short_ratio",
                "value_quality_score",
            ],
        )

    def test_retention_policy_matches_spec(self):
        self.assertEqual(RETENTION_POLICIES["raw_stock_prices_daily"], 60)
        self.assertEqual(RETENTION_POLICIES["raw_market_rankings"], 60)
        self.assertEqual(RETENTION_POLICIES["raw_macro"], 90)
        self.assertEqual(RETENTION_POLICIES["pipeline_run_logs"], 180)

    def test_weekly_and_monthly_frequency_split(self):
        self.assertIn("stocks_master_full_refresh", PIPELINE_FREQUENCIES["weekly"])
        self.assertIn("market_trading_calendar_sync", PIPELINE_FREQUENCIES["monthly"])
        self.assertNotIn("market_trading_calendar_sync", PIPELINE_FREQUENCIES["weekly"])


if __name__ == "__main__":
    unittest.main()
