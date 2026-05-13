import datetime
import os
import unittest
from unittest.mock import MagicMock, patch

from src.analysis.gemini_interpretation import _build_anchor_terms, _sanitize_payload
from src.jobs.generate_report import _build_simple_non_morning_report, _resolve_use_gemini, _sanitize_final_report_text, run_report
from src.reports.morning_report import generate_morning_brief


def _bundle():
    return {
        "freshness": {
            "target_date": "2026-05-11",
            "xkrx_is_open": True,
            "xnys_is_open": True,
            "watchlist_coverage_status": "PASS",
            "stale_warnings": "",
            "carry_forward_fields": [],
        },
        "macro": {
            "sp500": 7200,
            "sp500_change_value": 20,
            "sp500_change_rate": 0.003,
            "nasdaq": 25000,
            "nasdaq_change_value": 120,
            "nasdaq_change_rate": 0.005,
            "sox": 11000,
            "sox_change_value": 200,
            "sox_change_rate": 0.018,
            "vix": 17.0,
            "vix_change_value": -0.1,
            "vix_change_rate": -0.005,
            "usdkrw": 1448.0,
            "dxy": 100.0,
            "us10y": 4.2,
            "us10y_change_bp": 1.0,
            "us3y": 3.7,
            "us3y_change_bp": -2.0,
            "us10y_us3y_spread": 0.468,
            "us10y_us3y_spread_change_bp": -4.4,
            "brent": 80.0,
            "brent_change_value": 1.0,
            "brent_change_rate": 0.01,
            "wti": 77.0,
            "wti_change_value": 0.5,
            "wti_change_rate": 0.006,
            "kospi": 2600,
            "kosdaq": 850,
            "advancing_ratio": 0.55,
        },
        "sector_etfs": [
            {"symbol": "396500", "name": "TIGER 반도체TOP10", "sector_group": "반도체", "change_rate_1d": 0.02, "data_status": "FRESH"},
        ],
        "watchlist": [
            {"symbol": "000660", "name": "SK하이닉스", "close_price": 1686000, "change_rate_1d": 0.019, "signal_label": "강한 모멘텀 후보", "signal_score": 78, "sector_group": "반도체", "source_mixed": False, "data_status": "FRESH"},
            {"symbol": "005930", "name": "삼성전자", "close_price": 268500, "change_rate_1d": 0.011, "signal_label": "보유·관찰", "signal_score": 64, "sector_group": "반도체", "source_mixed": False, "data_status": "FRESH"},
        ],
        "rankings": [
            {"symbol": "005930", "name": "삼성전자", "market": "KOSPI", "rank_type": "volume", "rank": 1, "volume": 1000000, "source": "KIS"},
        ],
        "readiness": {
            "display_mode": "KIS_UNIVERSE_ONLY",
            "report_allowed_sections": ["macro", "us_market", "kis_volume_top", "watchlist_signal", "etf_etn"],
            "report_blocked_sections": ["kr_full_market_trading_value_top", "kr_full_market_market_cap_top"],
            "allowed_korean_sections": ["kis_volume_top", "watchlist_signal", "etf_etn"],
            "blocked_korean_sections": ["kr_full_market_trading_value_top", "kr_full_market_market_cap_top"],
            "data_limitation_note": "국내 리포트는 KIS 유니버스 기반으로 운영합니다. 전체시장 거래대금·시총 Top은 사용하지 않고, KIS 거래량 후보와 관심종목 중심으로 해석합니다.",
            "kr_full_market_price_ready": False,
            "kis_volume_ranking_ready": True,
            "kis_universe_ready": True,
        },
        "contract_failed_views": [],
        "watchlist_diagnostics": {"raw_row_count": 7, "active_row_count": 7},
    }


class GeminiInterpretationLayerTests(unittest.TestCase):
    def test_resolve_use_gemini_from_env(self):
        with patch.dict(os.environ, {"REPORT_USE_GEMINI": "true"}, clear=False):
            self.assertTrue(_resolve_use_gemini(None))

    def test_morning_gemini_insight_is_injected(self):
        result = generate_morning_brief(
            _bundle(),
            "2026-05-11",
            gemini_insight={
                "scenario_summary": "시초가 강세보다 거래대금 유지 여부를 먼저 봅니다.",
                "aggressive_view": "반도체 거래대금이 유지되면 선별 대응이 가능합니다.",
                "conservative_view": "환율이 다시 상승하면 추격은 늦춰야 합니다.",
                "must_watch": ["삼성전자 거래대금 유지", "SK하이닉스 과열 흡수 여부"],
            },
        )["report_text"]
        self.assertIn("Gemini 보강 해석", result)
        self.assertIn("시초가 강세보다 거래대금 유지 여부를 먼저 봅니다.", result)

    def test_morning_gemini_filters_rank_specific_kis_must_watch(self):
        result = generate_morning_brief(
            _bundle(),
            "2026-05-11",
            gemini_insight={
                "scenario_summary": "환율과 반도체 흐름을 함께 확인합니다.",
                "must_watch": [
                    "흥아해운, 이노인스트루먼트의 KIS 거래량 1위 유지 여부 확인",
                    "USD/KRW 방향 확인",
                ],
            },
        )["report_text"]
        self.assertNotIn("KIS 거래량 1위 유지 여부", result)
        self.assertIn("USD/KRW 방향 확인", result)

    def test_regular_gemini_insight_is_injected(self):
        text = _build_simple_non_morning_report(
            "regular",
            "2026-05-11",
            _bundle(),
            gemini_insight={
                "view_vs_actual_status": "유지",
                "view_vs_actual_reason": "오전 반도체 우위가 KIS 거래량과 관심종목 반응에서 유지됩니다.",
                "kis_volume_interpretation": ["삼성전자는 KIS 거래량 1위라 주도 지속 여부의 핵심 확인 대상입니다."],
                "watchlist_comments": {"000660": "SK하이닉스는 거래대금 유지 여부가 핵심입니다."},
                "next_checkpoints": ["KIS 거래량 상위가 오후에도 유지되는지 확인"],
            },
        )
        self.assertIn("Gemini 해석: 오전 View 대비 현재 판단은 유지입니다.", text)
        self.assertIn("해석: 삼성전자는 KIS 거래량 1위라 주도 지속 여부의 핵심 확인 대상입니다.", text)
        self.assertIn("SK하이닉스는 거래대금 유지 여부가 핵심입니다.", text)

    def test_closing_gemini_insight_is_injected(self):
        text = _build_simple_non_morning_report(
            "closing",
            "2026-05-11",
            _bundle(),
            gemini_insight={
                "key_drivers": ["반도체 거래량 집중"],
                "market_review_status": "추세 지속",
                "market_review_reason": "반도체 중심으로 강도가 유지됐습니다.",
                "watchlist_review": {"000660": "SK하이닉스는 강세가 유지됐지만 과열 확인이 필요합니다."},
                "tomorrow_strategy": {
                    "aggressive_condition": "반도체 거래량 상위가 유지될 때",
                    "conservative_condition": "환율이 재상승할 때",
                    "must_check": ["USD/KRW", "KIS 거래량 상위 지속 여부"],
                },
            },
        )
        self.assertIn("Gemini 키워드: 반도체 거래량 집중", text)
        self.assertIn("마감 해석: 추세 지속 / 반도체 중심으로 강도가 유지됐습니다.", text)
        self.assertIn("반도체 거래량 상위가 유지될 때", text)

    def test_sanitize_removes_forbidden_terms_and_unknown_numbers(self):
        payload = {
            "scenario_summary": "BUY 관점입니다. 999.99% 급등을 기대합니다. 반도체 거래대금 유지 여부를 확인합니다.",
            "must_watch": ["외국인 선물 순매수 확인", "USD/KRW 방향 확인"],
        }
        cleaned = _sanitize_payload(payload, {"80.0", "1448.0", "468", "0.468"}, _build_anchor_terms(_bundle()))
        self.assertNotIn("BUY", str(cleaned))
        self.assertNotIn("외국인 선물 순매수", str(cleaned))
        self.assertNotIn("999.99%", str(cleaned))
        self.assertIn("USD/KRW 방향 확인", str(cleaned))

    def test_final_report_sanitize_removes_forbidden_gemini_lines(self):
        report_text = "[Regular Brief | 2026-05-11]\n- Gemini 해석: 관련 뉴스를 참고하십시오.\n- 정상 문장: KIS 거래량 유지 여부를 확인합니다.\n"
        sanitized, removed_terms = _sanitize_final_report_text(report_text)
        self.assertNotIn("관련 뉴스", sanitized)
        self.assertIn("KIS 거래량 유지 여부", sanitized)
        self.assertIn("관련 뉴스", removed_terms)

    @patch("src.jobs.generate_report._save_report")
    @patch("src.jobs.generate_report.generate_morning_brief")
    @patch("src.jobs.generate_report._generate_gemini_insight")
    @patch("src.jobs.generate_report.SupabaseStockDataReader")
    @patch("src.jobs.generate_report.SupabaseReader")
    def test_no_gemini_does_not_call_interpreter(self, mock_base_reader_cls, mock_reader_cls, mock_generate, mock_morning, mock_save):
        base_reader = MagicMock()
        base_reader.telegram_bot_token = None
        base_reader.telegram_chat_id = None
        base_reader.fetch_market_calendar_status.return_value = {"report_market_mode": "FULL_REPORT", "xkrx_is_open": True, "xnys_is_open": True}
        mock_base_reader_cls.return_value = base_reader
        mock_reader = MagicMock()
        mock_reader.get_report_contract_bundle.return_value = _bundle()
        mock_reader_cls.return_value = mock_reader
        mock_morning.return_value = {"report_text": "ok", "snapshot": {}}
        run_report("morning", datetime.datetime(2026, 5, 11, 8, 0), report_date="2026-05-11", send_enabled=False, use_gemini=False)
        mock_generate.assert_not_called()

    @patch("src.jobs.generate_report._save_report")
    @patch("src.jobs.generate_report.generate_morning_brief")
    @patch("src.jobs.generate_report._generate_gemini_insight")
    @patch("src.jobs.generate_report.SupabaseStockDataReader")
    @patch("src.jobs.generate_report.SupabaseReader")
    @patch("src.jobs.generate_report.config.get", return_value="key")
    def test_use_gemini_calls_interpreter(self, _mock_config_get, mock_base_reader_cls, mock_reader_cls, mock_generate, mock_morning, mock_save):
        base_reader = MagicMock()
        base_reader.telegram_bot_token = None
        base_reader.telegram_chat_id = None
        base_reader.fetch_market_calendar_status.return_value = {"report_market_mode": "FULL_REPORT", "xkrx_is_open": True, "xnys_is_open": True}
        mock_base_reader_cls.return_value = base_reader
        mock_reader = MagicMock()
        mock_reader.get_report_contract_bundle.return_value = _bundle()
        mock_reader_cls.return_value = mock_reader
        mock_generate.return_value = {"scenario_summary": "ok"}
        mock_morning.return_value = {"report_text": "ok", "snapshot": {}}
        run_report("morning", datetime.datetime(2026, 5, 11, 8, 0), report_date="2026-05-11", send_enabled=False, use_gemini=True)
        mock_generate.assert_called_once()

    @patch("src.jobs.generate_report._save_report")
    @patch("src.jobs.generate_report.generate_morning_brief")
    @patch("src.jobs.generate_report._generate_gemini_insight")
    @patch("src.jobs.generate_report.SupabaseStockDataReader")
    @patch("src.jobs.generate_report.SupabaseReader")
    @patch("src.jobs.generate_report.config.get", return_value=None)
    def test_missing_api_key_falls_back(self, _mock_config_get, mock_base_reader_cls, mock_reader_cls, mock_generate, mock_morning, mock_save):
        base_reader = MagicMock()
        base_reader.telegram_bot_token = None
        base_reader.telegram_chat_id = None
        base_reader.fetch_market_calendar_status.return_value = {"report_market_mode": "FULL_REPORT", "xkrx_is_open": True, "xnys_is_open": True}
        mock_base_reader_cls.return_value = base_reader
        mock_reader = MagicMock()
        mock_reader.get_report_contract_bundle.return_value = _bundle()
        mock_reader_cls.return_value = mock_reader
        mock_morning.return_value = {"report_text": "ok", "snapshot": {}}
        with patch.dict(os.environ, {}, clear=True):
            run_report("morning", datetime.datetime(2026, 5, 11, 8, 0), report_date="2026-05-11", send_enabled=False, use_gemini=True)
        mock_generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
