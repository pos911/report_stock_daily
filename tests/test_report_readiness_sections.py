import unittest
from unittest.mock import MagicMock
from src.reports.morning_report import generate_morning_brief
from src.jobs.generate_report import _build_simple_non_morning_report

class TestReportReadinessSections(unittest.TestCase):
    def setUp(self):
        self.bundle = {
            "freshness": {
                "target_date": "2024-05-06",
                "xkrx_is_open": True,
                "xnys_is_open": True,
                "latest_macro_date": "2024-05-03",
                "latest_stock_price_date": "2024-05-03"
            },
            "macro": {
                "kospi": 2700,
                "kosdaq": 850,
                "usdkrw": 1350,
                "us10y_us3y_spread_bp": -15
            },
            "sector_etfs": [],
            "watchlist": [
                {"name": "Samsung Electronics", "symbol": "005930", "close_price": 75000, "change_rate_1d": 0.012, "label": "보유/관찰"}
            ],
            "rankings": [
                {"source": "KIS", "rank_type": "volume", "symbol": "005930", "name": "Samsung Electronics", "rank_value": 1000000}
            ],
            "readiness": {
                "kr_full_market_price_ready": True,
                "report_allowed_sections": ["morning_macro", "morning_sector", "morning_watchlist", "kis_volume_top", "watchlist_signal"],
                "report_blocked_sections": []
            }
        }

    def test_full_report_when_all_ready(self):
        result = generate_morning_brief(self.bundle, "2024-05-06")
        report_text = result["report_text"]
        
        self.assertIn("3. 야간 글로벌 시장", report_text)
        self.assertIn("4. 한국장 예상 영향", report_text)
        self.assertIn("6. 관심종목", report_text)
        self.assertNotIn("데이터 커버리지 안내", report_text)

    def test_block_macro_section(self):
        self.bundle["readiness"]["report_blocked_sections"] = ["morning_macro"]
        result = generate_morning_brief(self.bundle, "2024-05-06")
        report_text = result["report_text"]
        
        self.assertNotIn("3. 야간 글로벌 시장", report_text)
        self.assertIn("4. 한국장 예상 영향", report_text)

    def test_coverage_limitation_note(self):
        self.bundle["readiness"]["kr_full_market_price_ready"] = False
        result = generate_morning_brief(self.bundle, "2024-05-06")
        report_text = result["report_text"]
        
        self.assertIn("데이터 커버리지 안내", report_text)
        self.assertIn("KIS 거래량 상위", report_text)

    def test_simple_non_morning_report_readiness(self):
        # Regular report with full readiness
        report_text = _build_simple_non_morning_report("regular", "2024-05-06", self.bundle)
        
        self.assertIn("[Regular Brief | 2024-05-06]", report_text)
        self.assertIn("국내 전종목 가격 커버리지: 충족", report_text)
        self.assertIn("4. KIS 거래량 순위 기준", report_text)
        self.assertIn("5. 관심종목·랭킹 후보 기반 Signal", report_text)
        self.assertIn("Samsung Electronics(005930)", report_text)
        self.assertIn("75,000원", report_text)
        self.assertIn("보유/관찰", report_text)

    def test_simple_non_morning_report_blocked(self):
        # Regular report with restricted readiness
        self.bundle["readiness"]["kr_full_market_price_ready"] = False
        self.bundle["readiness"]["report_allowed_sections"] = ["watchlist_signal"]
        self.bundle["readiness"]["report_blocked_sections"] = ["kis_volume_top", "trading_value_ranking"]
        
        report_text = _build_simple_non_morning_report("regular", "2024-05-06", self.bundle)
        
        self.assertIn("국내 전종목 가격 커버리지: 미충족", report_text)
        self.assertIn("3. 국내 데이터 범위", report_text)
        self.assertNotIn("4. KIS 거래량 순위 기준", report_text)
        self.assertIn("5. 관심종목·랭킹 후보 기반 Signal", report_text)
        self.assertIn("※ 생략된 섹션: kis_volume_top, trading_value_ranking", report_text)

if __name__ == "__main__":
    unittest.main()
