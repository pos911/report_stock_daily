import unittest

from src.data.supabase_reader import SupabaseReader
from src.jobs.generate_report import _format_watchlist_section


class WatchlistContractTests(unittest.TestCase):
    def test_watchlist_snapshot_view_is_preferred(self):
        reader = SupabaseReader.__new__(SupabaseReader)
        reader.fetch_report_watchlist_snapshot_view = lambda: [
            {
                "symbol": "005930",
                "name": "Samsung Electronics",
                "market": "KOSPI",
                "base_date": "2026-05-05",
                "close_price": 80000,
                "trading_value": 1000000,
                "foreign_net_buy": 10,
                "institutional_net_buy": 20,
                "individual_net_buy": -30,
                "foreign_holding_ratio": 55.0,
                "return_5d": 0.03,
                "return_20d": 0.05,
                "return_60d": 0.10,
                "trading_value_ratio_20d": 1.2,
                "short_ratio": 1.0,
                "short_value": 10000,
                "per": 12.0,
                "pbr": 1.5,
                "roe": 10.0,
                "debt_ratio": 20.0,
                "data_status": "FRESH",
            }
        ]
        reader.get_latest_valid_price_date = lambda report_date=None, lookback_days=7: {"base_date": "2026-05-04"}
        reader.fetch_static_universe_stock_snapshot = lambda price_base_date=None: []

        bundle = reader.get_watchlist_snapshots(report_date="2026-05-05")
        self.assertEqual(bundle["price_meta"]["source"], "report_watchlist_snapshot_view")
        self.assertEqual(bundle["snapshots"][0]["symbol"], "005930")

    def test_watchlist_empty_warning_is_rendered(self):
        lines = _format_watchlist_section([])
        joined = "\n".join(lines)
        self.assertIn("watchlist empty", joined.lower())


if __name__ == "__main__":
    unittest.main()
