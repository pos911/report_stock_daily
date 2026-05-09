import unittest

from src.signals.watchlist_morning import build_watchlist_morning_scores


class WatchlistMorningTests(unittest.TestCase):
    def test_component_scores_affect_signal_label(self):
        rows = build_watchlist_morning_scores(
            [
                {
                    "symbol": "005930",
                    "name": "삼성전자",
                    "sector_group": "반도체",
                    "close_price": 80000,
                    "return_5d": 0.03,
                    "return_20d": 0.08,
                    "return_60d": 0.15,
                    "trading_value_ratio_20d": 2.1,
                    "foreign_net_buy": 10,
                    "institutional_net_buy": 10,
                    "roe": 15,
                    "debt_ratio": 20,
                    "short_ratio": 1,
                    "data_status": "FRESH",
                }
            ],
            {"regime_label": "Risk-on"},
            [{"sector_group": "반도체", "label": "우호", "score": 80}],
        )
        self.assertIn(rows[0]["signal_label"], {"강한 모멘텀 후보", "보유·관찰"})
        self.assertNotEqual(rows[0]["quant_reasons"][0], rows[0]["positive_factors"][0])

    def test_source_mixed_downgrades_label(self):
        rows = build_watchlist_morning_scores(
            [
                {
                    "symbol": "005930",
                    "name": "삼성전자",
                    "sector_group": "반도체",
                    "close_price": 80000,
                    "return_5d": 0.10,
                    "return_20d": 0.25,
                    "trading_value_ratio_20d": 2.1,
                    "foreign_net_buy": 10,
                    "institutional_net_buy": 10,
                    "data_status": "FRESH",
                    "source_mixed": True,
                }
            ],
            {"regime_label": "Risk-on"},
            [{"sector_group": "반도체", "label": "우호", "score": 80}],
        )
        self.assertNotEqual(rows[0]["signal_label"], "강한 모멘텀 후보")
        self.assertIn("혼합", " ".join(rows[0]["negative_factors"]))

    def test_return_20d_none_is_not_used_in_reasons(self):
        rows = build_watchlist_morning_scores(
            [
                {
                    "symbol": "278470",
                    "name": "에이피알",
                    "sector_group": "화장품/소비재",
                    "close_price": 100000,
                    "return_5d": 0.02,
                    "return_20d": None,
                    "trading_value_ratio_20d": None,
                    "data_status": "FRESH",
                }
            ],
            {"regime_label": "Neutral"},
            [],
        )
        reasons = " ".join(rows[0]["quant_reasons"])
        self.assertNotIn("20일 수익률", reasons)

    def test_buy_hold_sell_not_used(self):
        rows = build_watchlist_morning_scores(
            [{"symbol": "005930", "name": "삼성전자", "sector_group": "반도체", "close_price": 80000, "data_status": "FRESH"}],
            {"regime_label": "Neutral"},
            [],
        )
        serialized = str(rows[0]).upper()
        self.assertNotIn("BUY", serialized)
        self.assertNotIn("SELL", serialized)
        self.assertNotIn("HOLD", serialized)

    def test_missing_data_keeps_watchlist_row(self):
        rows = build_watchlist_morning_scores(
            [{"symbol": "017670", "name": "SK텔레콤", "sector_group": "통신", "close_price": None, "data_status": "DATA_MISSING"}],
            {"regime_label": "Neutral"},
            [],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["signal_label"], "판단 유보")

    def test_core_priority_is_preferred(self):
        rows = build_watchlist_morning_scores(
            [
                {"symbol": "071050", "name": "한국금융지주", "sector_group": "금융/증권", "close_price": 1, "return_5d": 0.01, "trading_value_ratio_20d": 1.2, "foreign_net_buy": 1, "institutional_net_buy": 1, "roe": 10, "debt_ratio": 40, "data_status": "FRESH"},
                {"symbol": "999999", "name": "기타종목", "sector_group": "금융/증권", "close_price": 1, "return_5d": 0.01, "trading_value_ratio_20d": 1.2, "foreign_net_buy": 1, "institutional_net_buy": 1, "roe": 10, "debt_ratio": 40, "data_status": "FRESH"},
            ],
            {"regime_label": "Neutral"},
            [{"sector_group": "금융/증권", "label": "중립", "score": 55}],
        )
        self.assertEqual(rows[0]["symbol"], "071050")


if __name__ == "__main__":
    unittest.main()
