import unittest

from src.data.supabase_reader import SupabaseReader


class ReaderRankingLogicTests(unittest.TestCase):
    def test_get_latest_valid_price_date_prefers_more_valid_rows(self):
        reader = SupabaseReader.__new__(SupabaseReader)
        reader.fetch_price_date_candidates = lambda report_date=None, lookback_days=7: ["2026-05-02", "2026-05-01"]
        reader.fetch_price_rows_by_date = lambda base_date: (
            [{"symbol": "005930", "close_price": 1000, "volume": 10, "trading_value": 10000}]
            if base_date == "2026-05-01"
            else [{"symbol": "005930", "close_price": None, "volume": None, "trading_value": None}]
        )

        latest = reader.get_latest_valid_price_date("2026-05-02")
        self.assertEqual(latest["base_date"], "2026-05-01")
        self.assertEqual(latest["valid_rows"], 1)

    def test_price_fallback_rankings_split_kospi_kosdaq_etf_etn(self):
        reader = SupabaseReader.__new__(SupabaseReader)
        reader.fetch_stocks_master_map = lambda: {
            "005930": {"symbol": "005930", "name": "삼성전자", "market": "KOSPI"},
            "035720": {"symbol": "035720", "name": "카카오", "market": "KOSDAQ"},
            "305720": {"symbol": "305720", "name": "KODEX 2차전지산업레버리지", "market": "ETF"},
            "530036": {"symbol": "530036", "name": "삼성 인버스 2X WTI원유 선물 ETN", "market": "ETN"},
        }
        reader.fetch_price_rows_by_date = lambda base_date: [
            {"symbol": "005930", "base_date": base_date, "close_price": 1000, "volume": 50, "trading_value": 50000, "market_cap": 1000000},
            {"symbol": "035720", "base_date": base_date, "close_price": 2000, "volume": 40, "trading_value": 40000, "market_cap": 900000},
            {"symbol": "305720", "base_date": base_date, "close_price": 3000, "volume": 30, "trading_value": 30000, "market_cap": 800000},
            {"symbol": "530036", "base_date": base_date, "close_price": 4000, "volume": 20, "trading_value": 20000, "market_cap": 700000},
        ]

        sections = reader._build_price_fallback_rankings("2026-05-02", limit=5)
        self.assertEqual(sections["volume"]["KOSPI"][0]["symbol"], "005930")
        self.assertEqual(sections["volume"]["KOSDAQ"][0]["symbol"], "035720")
        self.assertEqual(sections["volume"]["ETF"][0]["symbol"], "305720")
        self.assertEqual(sections["volume"]["ETN"][0]["symbol"], "530036")

    def test_get_latest_market_rankings_filters_legacy_kis_trading_value(self):
        reader = SupabaseReader.__new__(SupabaseReader)
        reader._candidate_ranking_dates = lambda report_date=None, lookback_days=7: ["2026-05-02"]
        reader.get_latest_valid_price_date = lambda report_date=None, lookback_days=7: {"base_date": "2026-05-02", "valid_rows": 3, "total_rows": 3}
        reader._build_price_map_for_date = lambda base_date: (
            {
                "005930": {"symbol": "005930", "base_date": base_date, "close_price": 1000, "volume": 100, "trading_value": 100000, "market_cap": 500000},
                "000660": {"symbol": "000660", "base_date": base_date, "close_price": 2000, "volume": 200, "trading_value": 200000, "market_cap": 600000},
            },
            {"base_date": base_date, "row_count": 2, "valid_rows": 2},
        )
        reader.fetch_stocks_master_map = lambda: {
            "005930": {"symbol": "005930", "name": "삼성전자", "market": "KOSPI"},
            "000660": {"symbol": "000660", "name": "SK하이닉스", "market": "KOSPI"},
        }
        reader._fetch_market_ranking_rows_for_dates = lambda candidate_dates: [
            {"symbol": "005930", "base_date": "2026-05-02", "market": "KOSPI", "rank_type": "volume", "rank": 1, "source": "KIS"},
            {"symbol": "000660", "base_date": "2026-05-02", "market": "KOSPI", "rank_type": "trading_value", "rank": 1, "source": "KIS"},
        ]
        reader._build_price_fallback_rankings = lambda price_base_date, limit=10: {
            "volume": {"KOSPI": [], "KOSDAQ": [], "ETF": [], "ETN": []},
            "trading_value": {
                "KOSPI": [{"symbol": "000660", "display_symbol": "000660", "name": "SK하이닉스", "market": "KOSPI", "asset_type": "COMMON_STOCK", "rank_type": "trading_value", "rank": 1, "source": "VALID_PRICE_FALLBACK", "close_price": 2000, "volume": 200, "trading_value": 200000, "market_cap": 600000, "ranking_base_date": price_base_date, "price_base_date": price_base_date}],
                "KOSDAQ": [],
                "ETF": [],
                "ETN": [],
            },
            "market_cap": {"KOSPI": [], "KOSDAQ": [], "ETF": [], "ETN": []},
        }

        bundle = reader.get_latest_market_rankings(report_date="2026-05-02", limit=5)
        self.assertEqual(bundle["sections"]["volume"]["KOSPI"][0]["symbol"], "005930")
        self.assertEqual(bundle["sections"]["trading_value"]["KOSPI"][0]["source"], "VALID_PRICE_FALLBACK")
        self.assertTrue(bundle["fallback_used"])
        self.assertEqual(len(bundle["diagnostics"]["legacy_source_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
