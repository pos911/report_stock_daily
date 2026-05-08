import unittest

from src.jobs.generate_report import _build_simple_non_morning_report


class RegularClosingReportSectionTests(unittest.TestCase):
    def setUp(self):
        self.bundle = {
            "freshness": {"xkrx_is_open": True, "xnys_is_open": True},
            "macro": {"kospi": 2700, "kosdaq": 850, "usdkrw": 1400},
            "watchlist": [{"name": "삼성전자", "symbol": "005930", "close_price": 75000, "change_rate_1d": 1.2, "signal_label": "보유·관찰", "signal_score": 66}],
            "rankings": [{"source": "KIS", "rank_type": "volume", "symbol": "005930", "name": "삼성전자", "rank": 1, "volume": 1000000, "market": "KOSPI"}],
            "readiness": {
                "display_mode": "KIS_UNIVERSE_ONLY",
                "report_allowed_sections": ["macro", "us_market", "kis_volume_top", "watchlist_signal"],
                "blocked_korean_sections": ["kr_full_market_trading_value_top", "kr_full_market_market_cap_top"],
                "data_limitation_note": "국내 전종목 가격 커버리지가 부족해 거래대금·시총 기준 전체시장 Top은 생략합니다.",
            },
        }

    def test_regular_sections_are_sequential(self):
        text = _build_simple_non_morning_report("regular", "2026-05-09", self.bundle)
        for section in ["1. 시장 상태", "2. 장중 핵심 요약", "3. 국내 데이터 범위", "4. KIS 거래량 순위 기준", "5. 관심종목·랭킹 후보 Signal", "6. 오후 체크포인트"]:
            self.assertIn(section, text)

    def test_closing_sections_are_sequential(self):
        text = _build_simple_non_morning_report("closing", "2026-05-09", self.bundle)
        for section in ["1. 마감 데이터 상태", "2. 마감 요약", "3. 국내 데이터 범위", "4. KIS 거래량 순위 기준 마감 점검", "5. 관심종목·후보군 마감 점검", "6. 다음 거래일 체크포인트"]:
            self.assertIn(section, text)


if __name__ == "__main__":
    unittest.main()
