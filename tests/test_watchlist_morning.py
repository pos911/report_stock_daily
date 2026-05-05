import unittest

from src.signals.watchlist_morning import build_watchlist_morning_scores


class WatchlistMorningTests(unittest.TestCase):
    def test_component_scores_affect_label(self):
        rows = build_watchlist_morning_scores(
            [
                {
                    "symbol": "005930",
                    "name": "Samsung Electronics",
                    "sector_group": "Semiconductor",
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
            {"market_tone": "우호"},
            [{"sector_group": "Semiconductor", "label": "우호", "intraday_checkpoints": ["SOX 연동 확인"]}],
        )
        self.assertIn(rows[0]["label"], {"우호", "중립~우호"})

    def test_buy_hold_sell_not_used(self):
        rows = build_watchlist_morning_scores(
            [{"symbol": "005930", "name": "Samsung Electronics", "sector_group": "Semiconductor", "close_price": 80000, "data_status": "FRESH"}],
            {"market_tone": "중립"},
            [],
        )
        serialized = str(rows[0]).upper()
        self.assertNotIn("BUY", serialized)
        self.assertNotIn("SELL", serialized)
        self.assertNotIn("HOLD", serialized)

    def test_missing_data_keeps_watchlist_row(self):
        rows = build_watchlist_morning_scores(
            [{"symbol": "017670", "name": "SK Telecom", "sector_group": "Telecom", "close_price": None, "data_status": "DATA_MISSING"}],
            {"market_tone": "중립"},
            [],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "데이터 부족")


if __name__ == "__main__":
    unittest.main()
