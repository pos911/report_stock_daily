import unittest

from src.jobs.generate_report import _build_simple_non_morning_report


class RegularClosingReportSectionTests(unittest.TestCase):
    def setUp(self):
        self.bundle = {
            "freshness": {"xkrx_is_open": True, "xnys_is_open": True},
            "macro": {"kospi": 2700, "kosdaq": 850, "usdkrw": 1452, "brent_change_rate": 0.0},
            "watchlist": [
                {
                    "name": "삼성전자",
                    "symbol": "005930",
                    "close_price": 75000,
                    "change_rate_1d": 1.2,
                    "signal_label": "관찰",
                    "signal_score": 66,
                    "source_mixed": True,
                    "trading_value_ratio_20d": None,
                    "data_status": "FRESH",
                }
            ],
            "rankings": [{"source": "KIS", "rank_type": "volume", "symbol": "005930", "name": "삼성전자", "rank": 1, "volume": 1000000, "market": "KOSPI"}],
            "readiness": {
                "display_mode": "KIS_UNIVERSE_ONLY",
                "report_allowed_sections": ["macro", "us_market", "kis_volume_top", "watchlist_signal"],
                "blocked_korean_sections": ["kr_full_market_trading_value_top", "kr_full_market_market_cap_top"],
                "data_limitation_note": "국내 전종목 가격 커버리지가 부족해 거래대금·시총 기준 전체시장 Top은 생략합니다.",
            },
            "sector_etfs": [],
        }

    def test_regular_sections_are_sequential(self):
        text = _build_simple_non_morning_report("regular", "2026-05-09", self.bundle)
        for section in ["1. 시장 상태", "2. 장중 핵심 요약", "3. 오전 View 점검", "4. 국내 데이터 범위", "5. KIS 거래량 순위 기준", "6. 관심종목 장중 반응", "7. 오후 체크포인트"]:
            self.assertIn(section, text)

    def test_closing_sections_are_sequential(self):
        text = _build_simple_non_morning_report("closing", "2026-05-09", self.bundle)
        for section in ["1. 마감 데이터 상태", "2. 마감 요약", "3. 오늘의 핵심 키워드", "4. 국내 데이터 범위", "5. KIS 거래량 순위 기준 마감 점검", "6. 관심종목 마감 진단", "7. 내일의 전략"]:
            self.assertIn(section, text)

    def test_regular_summary_uses_usdkrw_threshold_wording(self):
        text = _build_simple_non_morning_report("regular", "2026-05-09", self.bundle)
        self.assertIn("환율 1,450원대", text)


if __name__ == "__main__":
    unittest.main()
