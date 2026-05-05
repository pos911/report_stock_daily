import json
import tempfile
import unittest
from pathlib import Path

from src.reports.morning_report import generate_morning_brief, save_morning_snapshot


class MorningReportTests(unittest.TestCase):
    def _bundle(self, contract_fallback_used: bool = False):
        return {
            "contract_fallback_used": contract_fallback_used,
            "contract_failed_views": ["report_watchlist_snapshot_view"] if contract_fallback_used else [],
            "freshness": {
                "target_date": "2026-05-05",
                "latest_macro_date": "2026-05-04",
                "latest_stock_price_date": "2026-05-04",
                "latest_ranking_date": "2026-05-04",
                "latest_supply_date": "2026-05-04",
                "latest_breadth_date": "2026-05-04",
                "sector_etf_coverage_status": "WARN",
                "watchlist_coverage_status": "WARN",
                "stale_warnings": "sector_etf: 494670",
                "missing_required_data": "",
                "xkrx_is_open": False,
                "xnys_is_open": True,
                "carry_forward_fields": [],
            },
            "macro": {
                "sp500": 5000,
                "sp500_change_value": 50,
                "sp500_change_rate": 0.01,
                "nasdaq": 18000,
                "nasdaq_change_value": 200,
                "nasdaq_change_rate": 0.011,
                "sox": 5200,
                "sox_change_value": 100,
                "sox_change_rate": 0.02,
                "vix": 14,
                "vix_change_value": -1,
                "vix_change_rate": -0.08,
                "usdkrw": 1360,
                "usdkrw_change_value": -5,
                "usdkrw_change_rate": -0.004,
                "dxy": 104,
                "dxy_change_value": -0.4,
                "dxy_change_rate": -0.004,
                "us10y": 4.2,
                "us10y_change_bp": -6,
                "us3y": 3.9,
                "us3y_change_bp": -5,
                "us10y_us3y_spread": 0.3,
                "us10y_us3y_spread_change_bp": 5,
                "kr10y": 3.1,
                "kr10y_change_bp": 1,
                "brent": 82,
                "brent_change_value": 0.5,
                "brent_change_rate": 0.006,
                "wti": 78,
                "wti_change_value": 0.4,
                "wti_change_rate": 0.005,
                "advancing_ratio": 0.58,
                "kospi_foreign_net_buy": 1,
                "kospi_institutional_net_buy": 1,
            },
            "sector_etfs": [
                {
                    "symbol": "396500",
                    "name": "TIGER Semiconductor TOP10",
                    "sector_group": "Semiconductor",
                    "theme_group": "Semiconductor",
                    "role": "primary",
                    "exclude_from_signal": False,
                    "data_status": "FRESH",
                    "change_rate_1d": 0.02,
                    "return_20d": 0.12,
                    "trading_value_ratio_20d": 2.5,
                    "warnings": [],
                    "foreign_net_buy": 10,
                    "institutional_net_buy": 8,
                },
                {
                    "symbol": "305720",
                    "name": "KODEX Secondary Battery Industry",
                    "sector_group": "Battery",
                    "theme_group": "Battery",
                    "role": "primary",
                    "exclude_from_signal": False,
                    "data_status": "STALE",
                    "stale_days": 4,
                    "change_rate_1d": 0.01,
                    "return_20d": 0.08,
                    "trading_value_ratio_20d": 1.5,
                    "warnings": [],
                },
            ],
            "watchlist": [
                {
                    "symbol": "000660",
                    "name": "SK hynix",
                    "market": "KOSPI",
                    "sector_group": "Semiconductor",
                    "close_price": 210000,
                    "return_5d": 0.04,
                    "return_20d": 0.10,
                    "return_60d": 0.18,
                    "trading_value_ratio_20d": 2.4,
                    "foreign_net_buy": 12,
                    "institutional_net_buy": 6,
                    "roe": 18,
                    "debt_ratio": 25,
                    "short_ratio": 1,
                    "data_status": "FRESH",
                },
                {
                    "symbol": "005930",
                    "name": "Samsung Electronics",
                    "market": "KOSPI",
                    "sector_group": "Semiconductor",
                    "close_price": 80000,
                    "return_5d": 0.03,
                    "return_20d": 0.08,
                    "return_60d": 0.12,
                    "trading_value_ratio_20d": 2.0,
                    "foreign_net_buy": 10,
                    "institutional_net_buy": 10,
                    "roe": 15,
                    "debt_ratio": 20,
                    "short_ratio": 1,
                    "data_status": "FRESH",
                },
                {
                    "symbol": "247540",
                    "name": "Ecopro BM",
                    "market": "KOSDAQ",
                    "sector_group": "Battery",
                    "close_price": 210000,
                    "return_5d": 0.02,
                    "return_20d": 0.06,
                    "return_60d": 0.11,
                    "trading_value_ratio_20d": 1.4,
                    "foreign_net_buy": -2,
                    "institutional_net_buy": 5,
                    "roe": 11,
                    "debt_ratio": 90,
                    "short_ratio": 2,
                    "data_status": "FRESH",
                },
                {
                    "symbol": "012330",
                    "name": "Hyundai Mobis",
                    "market": "KOSPI",
                    "sector_group": "Automobile",
                    "close_price": 260000,
                    "return_5d": 0.02,
                    "return_20d": 0.04,
                    "return_60d": 0.07,
                    "trading_value_ratio_20d": 1.3,
                    "foreign_net_buy": 5,
                    "institutional_net_buy": -2,
                    "roe": 13,
                    "debt_ratio": 40,
                    "short_ratio": 1,
                    "data_status": "FRESH",
                },
                {
                    "symbol": "017670",
                    "name": "SK Telecom",
                    "market": "KOSPI",
                    "sector_group": "Telecom",
                    "close_price": None,
                    "data_status": "DATA_MISSING",
                },
            ],
            "rankings": [{"symbol": "005930", "name": "Samsung Electronics", "rank_type": "trading_value", "rank": 1}],
        }

    def test_contract_view_success_removes_fallback_warning(self):
        text = generate_morning_brief(self._bundle(contract_fallback_used=False), "2026-05-05")["report_text"]
        self.assertNotIn("fallback 데이터를 사용했습니다", text)

    def test_contract_view_404_shows_fallback_warning(self):
        text = generate_morning_brief(self._bundle(contract_fallback_used=True), "2026-05-05")["report_text"]
        self.assertIn("리포트 contract view 일부 미조회로 fallback 데이터를 사용했습니다.", text)
        self.assertNotIn("contract fallback used:", text)

    def test_xkrx_holiday_is_rendered_for_2026_05_05(self):
        text = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"]
        self.assertIn("- 한국장: 휴장", text)
        self.assertIn("2026-05-05", text)

    def test_sector_names_are_rendered_in_korean(self):
        text = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"]
        self.assertIn("반도체", text)
        self.assertIn("2차전지", text)
        for banned in ["Semiconductor", "Battery", "Defense", "Shipbuilding", "Financials", "Healthcare"]:
            self.assertNotIn(banned, text)

    def test_one_line_judgment_has_at_least_three_ground_types(self):
        lines = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"].splitlines()
        idx = lines.index("2. 오늘의 한 줄 판단")
        sentence = lines[idx + 1]
        self.assertIn("글로벌 지표", sentence)
        self.assertIn("ETF", sentence)
        self.assertTrue("수급" in sentence or "리스크" in sentence)

    def test_global_market_section_is_not_empty(self):
        text = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"]
        for label in ["S&P500", "Nasdaq", "SOX", "VIX", "USD/KRW", "DXY", "US10Y", "US3Y", "10Y-3Y spread", "Brent", "WTI"]:
            self.assertIn(label, text)

    def test_watchlist_section_renders_at_least_three_names(self):
        text = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"]
        self.assertGreaterEqual(text.count("- 장전 판단:"), 3)
        self.assertIn("관심종목 5개 중 주요 5개 표시. 나머지 0개는 snapshot에 저장합니다.", text)

    def test_watchlist_display_and_snapshot_split_for_17_names(self):
        bundle = self._bundle()
        bundle["watchlist"] = [
            {
                "symbol": f"{i:06d}",
                "name": f"종목{i}",
                "market": "KOSPI",
                "sector_group": "반도체",
                "close_price": 10000 + i,
                "return_5d": 0.01,
                "return_20d": 0.02,
                "return_60d": 0.03,
                "trading_value_ratio_20d": 1.2,
                "foreign_net_buy": 1,
                "institutional_net_buy": 1,
                "roe": 10,
                "debt_ratio": 50,
                "short_ratio": 1,
                "data_status": "FRESH",
            }
            for i in range(1, 18)
        ]
        result = generate_morning_brief(bundle, "2026-05-05")
        self.assertIn("관심종목 17개 중 주요 6개 표시. 나머지 11개는 snapshot에 저장합니다.", result["report_text"])
        self.assertEqual(len(result["snapshot"]["watchlist_morning_scores"]), 17)

    def test_watchlist_names_prefer_korean_labels(self):
        text = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"]
        self.assertIn("SK하이닉스(000660)", text)
        self.assertIn("삼성전자(005930)", text)
        self.assertIn("현대모비스(012330)", text)
        self.assertNotIn("Samsung Electronics(005930)", text)
        self.assertNotIn("SK hynix(000660)", text)

    def test_holiday_report_removes_live_intraday_checkpoints(self):
        text = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"]
        self.assertIn("한국장 휴장으로 장중 체크포인트는 없습니다.", text)
        for banned in [
            "09:30 외국인 KOSPI200 선물 방향",
            "10:30 주도 섹터 거래대금 유지 여부",
            "12:30 아침 주도 테마 유지 여부",
            "장 초반 급등 후 이익실현 여부 확인",
        ]:
            self.assertNotIn(banned, text)

    def test_holiday_one_line_judgment_is_cautious(self):
        lines = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"].splitlines()
        sentence = lines[lines.index("2. 오늘의 한 줄 판단") + 1]
        self.assertIn("한국장은 휴장입니다.", sentence)
        self.assertIn("다음 거래일", sentence)

    def test_sentence_join_quality(self):
        text = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"]
        self.assertNotIn("입니다.가", text)
        self.assertNotIn("습니다.로", text)
        self.assertNotIn("; ", text)

    def test_watchlist_empty_warning_is_rendered(self):
        bundle = self._bundle()
        bundle["watchlist"] = []
        result = generate_morning_brief(bundle, "2026-05-05")
        self.assertIn("watchlist empty", result["report_text"].lower())

    def test_snapshot_core_fields_are_not_empty(self):
        snapshot = generate_morning_brief(self._bundle(), "2026-05-05")["snapshot"]
        for key in [
            "regime_label",
            "regime_score",
            "positive_drivers",
            "negative_drivers",
            "top_sectors",
            "sector_etf_signals",
            "watchlist_morning_scores",
            "risk_flags",
            "intraday_checkpoints",
            "data_freshness_manifest",
            "watchlist_coverage_status",
        ]:
            self.assertTrue(snapshot.get(key) not in (None, "", []), key)

    def test_snapshot_file_is_created(self):
        result = generate_morning_brief(self._bundle(), "2026-05-05")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_morning_snapshot(Path(tmpdir), "2026-05-05", result["snapshot"])
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["report_date"], "2026-05-05")


if __name__ == "__main__":
    unittest.main()
