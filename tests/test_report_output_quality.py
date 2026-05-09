import unittest

from src.jobs.generate_report import _build_simple_non_morning_report
from src.reports.morning_report import generate_morning_brief


class ReportOutputQualityTests(unittest.TestCase):
    def setUp(self):
        self.bundle = {
            "freshness": {
                "target_date": "2026-05-08",
                "latest_macro_date": "2026-05-08",
                "latest_stock_price_date": "2026-05-08",
                "latest_ranking_date": "2026-05-08",
                "latest_supply_date": "2026-05-08",
                "latest_breadth_date": "2026-05-08",
                "sector_etf_coverage_status": "PASS",
                "watchlist_coverage_status": "PASS",
                "stale_warnings": "",
                "missing_required_data": "",
                "xkrx_is_open": True,
                "xnys_is_open": True,
                "carry_forward_fields": [],
            },
            "macro": {
                "sp500": 5000,
                "sp500_change_value": 10,
                "sp500_change_rate": 0.002,
                "nasdaq": 18000,
                "nasdaq_change_value": 20,
                "nasdaq_change_rate": 0.002,
                "sox": 5000,
                "sox_change_value": 10,
                "sox_change_rate": 0.002,
                "vix": 15,
                "vix_change_value": -0.1,
                "vix_change_rate": -0.01,
                "usdkrw": 1452,
                "usdkrw_change_value": 0.0,
                "usdkrw_change_rate": 0.0,
                "dxy": 100,
                "dxy_change_value": 0.0,
                "dxy_change_rate": 0.0,
                "us10y": 4.0,
                "us10y_change_bp": 0.0,
                "us3y": 3.8,
                "us3y_change_bp": 0.0,
                "us10y_us3y_spread": 0.468,
                "us10y_us3y_spread_change_bp": -4.4,
                "kr10y": 3.1,
                "kr10y_change_bp": 0.0,
                "brent": 80,
                "brent_change_value": 0.0,
                "brent_change_rate": 0.0,
                "wti": 78,
                "wti_change_value": 0.0,
                "wti_change_rate": 0.0,
                "advancing_ratio": 0.5,
                "kospi": 7498,
                "kospi_foreign_net_buy": 0,
                "kospi_institutional_net_buy": 0,
                "kosdaq": 880,
            },
            "sector_etfs": [
                {
                    "symbol": "396500",
                    "name": "TIGER 반도체TOP10",
                    "sector_group": "반도체",
                    "theme_group": "반도체",
                    "role": "primary",
                    "exclude_from_signal": False,
                    "data_status": "STALE_BUT_USABLE",
                    "stale_days": 1,
                    "change_rate_1d": 0.03,
                    "return_20d": None,
                    "trading_value_ratio_20d": None,
                    "foreign_net_buy": 1,
                    "institutional_net_buy": -1,
                    "warnings": [],
                }
            ],
            "watchlist": [
                {
                    "name": "삼성전자",
                    "symbol": "005930",
                    "close_price": 268500,
                    "change_rate_1d": 1.2,
                    "signal_label": "관찰",
                    "signal_score": 81,
                    "sector_group": "반도체",
                    "return_5d": 0.02,
                    "return_20d": None,
                    "return_60d": 0.04,
                    "trading_value_ratio_20d": None,
                    "foreign_net_buy": 1,
                    "institutional_net_buy": 1,
                    "roe": 10,
                    "debt_ratio": 20,
                    "short_ratio": 1,
                    "data_status": "FRESH",
                    "source_mixed": True,
                    "stale_days": 0,
                }
            ],
            "watchlist_diagnostics": {"raw_row_count": 21, "active_row_count": 7},
            "rankings": [{"source": "KIS", "rank_type": "volume", "symbol": "005930", "name": "삼성전자", "rank": 1, "volume": 1000000, "market": "KOSPI"}],
            "readiness": {"display_mode": "KIS_UNIVERSE_ONLY", "allowed_korean_sections": ["kis_volume_top", "watchlist_signal"], "blocked_korean_sections": ["kr_full_market_trading_value_top", "kr_full_market_market_cap_top"], "report_allowed_sections": ["macro", "us_market", "kis_volume_top", "watchlist_signal"], "report_blocked_sections": ["kr_full_market_trading_value_top", "kr_full_market_market_cap_top"], "data_limitation_note": "국내 전종목 가격 커버리지가 부족해 거래대금·시총 기준 전체시장 Top은 생략합니다. 국내 종목은 KIS ranking·관심종목 후보군 중심으로 제한 제공합니다."},
        }

    def test_banned_words_are_absent(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        regular = _build_simple_non_morning_report("regular", "2026-05-08", self.bundle)
        closing = _build_simple_non_morning_report("closing", "2026-05-08", self.bundle)
        for text in [morning, regular, closing]:
            upper = text.upper()
            self.assertNotIn("BUY", upper)
            self.assertNotIn("SELL", upper)
            self.assertNotIn("HOLD", upper)
            self.assertNotIn("비중확대 후보", text)
            self.assertNotIn("리스크 축소 후보", text)
            self.assertNotIn("전체시장 거래대금 상위", text)

    def test_allowed_labels_are_present(self):
        regular = _build_simple_non_morning_report("regular", "2026-05-08", self.bundle)
        self.assertIn("관찰", regular)

    def test_morning_avoids_awkward_phrase(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        self.assertNotIn("기울어 있습니다를 바탕으로", morning)
        self.assertNotIn("우호적입니다를 바탕으로", morning)
        self.assertNotIn("부담입니다를 바탕으로", morning)
        self.assertNotIn("기대은", morning)

    def test_scale_warning_only_once(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        self.assertEqual(morning.count("일부 지수·종목 가격은 원천 스케일 확인이 필요합니다."), 1)

    def test_closing_has_recap_line(self):
        closing = _build_simple_non_morning_report("closing", "2026-05-08", self.bundle)
        self.assertIn("KIS 거래량 순위", closing)
        self.assertIn("마감 복기", closing)

    def test_stale_but_usable_etf_is_secondary_signal(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        self.assertIn("보조 신호", morning)
        self.assertNotIn("20일 기준 추세", morning)

    def test_spread_format_is_bp(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        self.assertIn("10Y-3Y spread: +46.8bp / 전일대비 -4.4bp", morning)


if __name__ == "__main__":
    unittest.main()
