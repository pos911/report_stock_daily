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
                "sox_change_rate": 0.012,
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
                "kosdaq": 1207.72,
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
                    "foreign_net_buy": -1,
                    "institutional_net_buy": 1,
                    "roe": 10,
                    "debt_ratio": 20,
                    "short_ratio": 1,
                    "data_status": "FRESH",
                    "source_mixed": False,
                    "stale_days": 0,
                },
                {
                    "name": "SK하이닉스",
                    "symbol": "000660",
                    "close_price": 1686000,
                    "change_rate_1d": 0.8,
                    "signal_label": "보유·관찰",
                    "signal_score": 66,
                    "sector_group": "반도체",
                    "return_5d": 0.12,
                    "return_20d": 0.22,
                    "return_60d": 0.30,
                    "trading_value_ratio_20d": 2.5,
                    "foreign_net_buy": 1,
                    "institutional_net_buy": 1,
                    "roe": 15,
                    "debt_ratio": 20,
                    "short_ratio": 1,
                    "data_status": "FRESH",
                    "source_mixed": False,
                    "stale_days": 0,
                },
                {
                    "name": "에이피알",
                    "symbol": "278470",
                    "close_price": 100000,
                    "change_rate_1d": -0.5,
                    "signal_label": "관망",
                    "signal_score": 50,
                    "sector_group": "화장품/소비재",
                    "return_5d": -0.04,
                    "return_20d": 0.10,
                    "return_60d": 0.20,
                    "trading_value_ratio_20d": 1.4,
                    "foreign_net_buy": -1,
                    "institutional_net_buy": 0,
                    "roe": 12,
                    "debt_ratio": 80,
                    "short_ratio": 4.5,
                    "data_status": "FRESH",
                    "source_mixed": False,
                    "stale_days": 0,
                },
                {
                    "name": "미래에셋증권",
                    "symbol": "006800",
                    "close_price": 12000,
                    "change_rate_1d": 0.4,
                    "signal_label": "보유·관찰",
                    "signal_score": 55,
                    "sector_group": "금융/증권",
                    "return_5d": 0.03,
                    "return_20d": 0.07,
                    "return_60d": 0.09,
                    "trading_value_ratio_20d": 1.3,
                    "foreign_net_buy": 1,
                    "institutional_net_buy": 1,
                    "roe": 10,
                    "debt_ratio": 100,
                    "short_ratio": 1.2,
                    "data_status": "FRESH",
                    "source_mixed": False,
                    "stale_days": 0,
                },
                {
                    "name": "POSCO홀딩스",
                    "symbol": "005490",
                    "close_price": 400000,
                    "change_rate_1d": 0.2,
                    "signal_label": "관망",
                    "signal_score": 52,
                    "sector_group": "철강/소재",
                    "return_5d": 0.01,
                    "return_20d": 0.03,
                    "return_60d": 0.11,
                    "trading_value_ratio_20d": 1.1,
                    "foreign_net_buy": 0,
                    "institutional_net_buy": 1,
                    "roe": 9,
                    "debt_ratio": 60,
                    "short_ratio": 1.0,
                    "data_status": "FRESH",
                    "source_mixed": False,
                    "stale_days": 0,
                },
            ],
            "watchlist_diagnostics": {"raw_row_count": 21, "active_row_count": 7},
            "rankings": [
                {"source": "KIS", "rank_type": "volume", "symbol": "005930", "name": "삼성전자", "rank": 1, "volume": 1000000, "market": "KOSPI"},
                {"source": "KIS", "rank_type": "volume", "symbol": "000660", "name": "SK하이닉스", "rank": 2, "volume": 900000, "market": "KOSPI"},
            ],
            "readiness": {
                "display_mode": "KIS_UNIVERSE_ONLY",
                "allowed_korean_sections": ["kis_volume_top", "watchlist_signal"],
                "blocked_korean_sections": ["kr_full_market_trading_value_top", "kr_full_market_market_cap_top"],
                "report_allowed_sections": ["macro", "us_market", "kis_volume_top", "watchlist_signal"],
                "report_blocked_sections": ["kr_full_market_trading_value_top", "kr_full_market_market_cap_top"],
                "data_limitation_note": "국내 전종목 가격 커버리지가 부족해 거래대금·시총 기준 전체시장 Top은 생략합니다. 국내 종목은 KIS ranking·관심종목 후보군 중심으로 제한 제공합니다.",
            },
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
        self.assertNotIn("부담는", morning)

    def test_scale_warning_not_shown_for_high_level_only(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        self.assertNotIn("일부 지수·종목 가격은 원천 스케일 확인이 필요합니다.", morning)

    def test_closing_has_recap_line(self):
        closing = _build_simple_non_morning_report("closing", "2026-05-08", self.bundle)
        self.assertIn("KIS 거래량 순위", closing)
        self.assertIn("상대 강도", closing)

    def test_stale_but_usable_etf_is_secondary_signal(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        self.assertIn("보조 신호", morning)
        self.assertNotIn("20일 기준 상승 추세가 유지되고 있습니다", morning)

    def test_spread_format_is_bp(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        self.assertIn("10Y-3Y spread: +46.8bp / 전일대비 -4.4bp", morning)

    def test_watchlist_checkpoints_are_not_identical(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        checkpoint_lines = [line for line in morning.splitlines() if "체크포인트:" in line]
        self.assertGreaterEqual(len(checkpoint_lines), 5)
        self.assertGreaterEqual(len(set(checkpoint_lines)), 3)
        repeated = max(checkpoint_lines.count(line) for line in checkpoint_lines)
        self.assertLess(repeated, 3)

    def test_symbol_specific_checkpoint_keywords_exist(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        self.assertRegex(morning, r"(?s)삼성전자\(005930\).*?(SOX|외국인|KIS 거래량)", "삼성전자 체크포인트가 충분히 구체적이어야 합니다.")
        self.assertRegex(morning, r"(?s)SK하이닉스\(000660\).*?(SOX|과열|반도체 ETF)", "SK하이닉스 체크포인트가 충분히 구체적이어야 합니다.")
        self.assertRegex(morning, r"(?s)에이피알\(278470\).*?(공매도|화장품|반등 거래대금)", "에이피알 체크포인트가 충분히 구체적이어야 합니다.")
        self.assertRegex(morning, r"(?s)미래에셋증권\(006800\).*?(증권업종|금리|거래대금)", "미래에셋증권 체크포인트가 충분히 구체적이어야 합니다.")
        self.assertRegex(morning, r"(?s)POSCO홀딩스\(005490\).*?(2차전지|철강|원자재)", "POSCO홀딩스 체크포인트가 충분히 구체적이어야 합니다.")

    def test_morning_has_scenario_section(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        self.assertIn("오늘의 시나리오", morning)
        self.assertIn("공격적 관점", morning)
        self.assertIn("보수적 관점", morning)

    def test_regular_has_morning_view_review(self):
        regular = _build_simple_non_morning_report("regular", "2026-05-08", self.bundle)
        self.assertIn("오전 View 점검", regular)

    def test_closing_has_tomorrow_strategy(self):
        closing = _build_simple_non_morning_report("closing", "2026-05-08", self.bundle)
        self.assertIn("내일의 전략", closing)


    def test_banned_grammar_fragments_are_absent(self):
        morning = generate_morning_brief(self.bundle, "2026-05-08")["report_text"]
        regular = _build_simple_non_morning_report("regular", "2026-05-08", self.bundle)
        closing = _build_simple_non_morning_report("closing", "2026-05-08", self.bundle)
        for text in [morning, regular, closing]:
            self.assertNotIn("2차전지과", text)
            self.assertNotIn("은(는)", text)
            self.assertNotIn("대한전선가", text)
            self.assertNotIn("부담는", text)
            self.assertNotIn("기대은", text)
            self.assertNotIn("있습니다를 바탕으로", text)

    def test_regular_and_closing_explain_conservative_reassessment(self):
        regular = _build_simple_non_morning_report("regular", "2026-05-08", self.bundle)
        closing = _build_simple_non_morning_report("closing", "2026-05-08", self.bundle)
        expected = "장중/마감 Signal은 현재 price/signal 기준의 보수적 재평가입니다."
        self.assertIn(expected, regular)
        self.assertIn(expected, closing)


if __name__ == "__main__":
    unittest.main()
