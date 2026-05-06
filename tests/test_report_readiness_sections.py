import unittest
from unittest.mock import MagicMock
from src.reports.morning_report import generate_morning_brief

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
            "macro": {"sp500": 5000},
            "sector_etfs": [],
            "watchlist": [],
            "rankings": [],
            "readiness": {
                "kr_full_market_price_ready": True,
                "report_allowed_sections": ["morning_macro", "morning_sector", "morning_watchlist"],
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

if __name__ == "__main__":
    unittest.main()
