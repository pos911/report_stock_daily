import unittest

from src.jobs.generate_report import _build_simple_non_morning_report
from src.reports.morning_report import generate_morning_brief
from src.services.supabase_stockdata_reader import SupabaseStockDataReader


class ReportReadinessSectionTests(unittest.TestCase):
    def setUp(self):
        self.bundle = {
            "freshness": {
                "target_date": "2026-05-09",
                "xkrx_is_open": True,
                "xnys_is_open": True,
                "latest_macro_date": "2026-05-09",
                "latest_stock_price_date": "2026-05-09",
            },
            "macro": {"kospi": 2700, "kosdaq": 850, "usdkrw": 1400},
            "sector_etfs": [],
            "watchlist": [
                {"name": "삼성전자", "symbol": "005930", "close_price": 75000, "change_rate_1d": 0.012, "signal_label": "보유·관찰"}
            ],
            "rankings": [
                {"source": "KIS", "rank_type": "volume", "symbol": "005930", "name": "삼성전자", "rank": 1, "volume": 1000000, "market": "KOSPI"}
            ],
            "readiness": {
                "kr_full_market_price_ready": False,
                "kis_universe_ready": True,
                "kis_volume_ranking_ready": True,
                "kr_trading_value_ranking_ready": False,
                "kr_market_cap_ranking_ready": False,
                "etf_etn_ready": True,
                "display_mode": "KIS_UNIVERSE_ONLY",
                "allowed_korean_sections": ["kis_volume_top", "watchlist_signal", "etf_etn"],
                "blocked_korean_sections": ["kr_full_market_trading_value_top", "kr_full_market_market_cap_top"],
                "report_allowed_sections": ["macro", "us_market", "kis_volume_top", "watchlist_signal", "etf_etn"],
                "report_blocked_sections": ["kr_full_market_trading_value_top", "kr_full_market_market_cap_top"],
                "data_limitation_note": "국내 전종목 가격 커버리지가 부족해 거래대금·시총 기준 전체시장 Top은 생략합니다. 거래량 상위는 KIS ranking 기준, 종목 점검은 관심종목·KIS 후보군 기준으로 제공합니다.",
            },
        }

    def test_normalize_report_readiness_builds_display_mode(self):
        reader = SupabaseStockDataReader.__new__(SupabaseStockDataReader)
        normalized = reader.normalize_report_readiness(self.bundle["readiness"])
        self.assertEqual(normalized["display_mode"], "KIS_UNIVERSE_ONLY")
        self.assertIn("kis_volume_top", normalized["allowed_korean_sections"])
        self.assertIn("kr_full_market_market_cap_top", normalized["blocked_korean_sections"])

    def test_morning_uses_kis_ranking_wording_when_available(self):
        text = generate_morning_brief(self.bundle, "2026-05-09")["report_text"]
        self.assertIn("KIS ranking 기준", text)

    def test_regular_hides_full_market_sections_when_not_ready(self):
        text = _build_simple_non_morning_report("regular", "2026-05-09", self.bundle)
        self.assertNotIn("\n4. 전체시장 거래대금 Top", text)
        self.assertNotIn("\n5. 전체시장 시총 Top", text)
        self.assertIn("KIS 거래량 순위 기준", text)
        self.assertIn("관심종목 장중 반응", text)

    def test_closing_hides_full_market_sections_when_not_ready(self):
        text = _build_simple_non_morning_report("closing", "2026-05-09", self.bundle)
        self.assertNotIn("\n4. 전체시장 거래대금 Top", text)
        self.assertNotIn("\n5. 전체시장 시총 Top", text)

    def test_report_has_no_buy_hold_sell(self):
        text = _build_simple_non_morning_report("regular", "2026-05-09", self.bundle).upper()
        self.assertNotIn("BUY", text)
        self.assertNotIn("SELL", text)
        self.assertNotIn("HOLD", text)

    def test_kis_universe_is_not_described_as_full_market(self):
        text = _build_simple_non_morning_report("regular", "2026-05-09", self.bundle)
        self.assertNotIn("국내 전체시장 거래대금 상위", text)
        self.assertNotIn("오늘 한국시장 전체 Top", text)


if __name__ == "__main__":
    unittest.main()
