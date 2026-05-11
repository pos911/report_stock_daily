import json
import tempfile
import unittest
from pathlib import Path

from src.reports.morning_report import generate_morning_brief, save_morning_snapshot


class MorningReportTests(unittest.TestCase):
    def _bundle(self):
        return {
            "freshness": {
                "target_date": "2026-05-05",
                "latest_macro_date": "2026-05-05",
                "latest_stock_price_date": "2026-05-04",
                "latest_ranking_date": "2026-05-05",
                "latest_supply_date": "2026-05-04",
                "latest_breadth_date": "2026-05-04",
                "sector_etf_coverage_status": "WARN",
                "watchlist_coverage_status": "PASS",
                "stale_warnings": "sector_etf: 091170, 091180",
                "missing_required_data": "",
                "xkrx_is_open": False,
                "xnys_is_open": True,
                "carry_forward_fields": [],
            },
            "macro": {
                "sp500": 7259.22,
                "sp500_change_value": 58.47,
                "sp500_change_rate": 0.0081,
                "nasdaq": 25326.12,
                "nasdaq_change_value": 258.32,
                "nasdaq_change_rate": 0.0103,
                "sox": 10980.58,
                "sox_change_value": 445.92,
                "sox_change_rate": 0.0423,
                "vix": 17.38,
                "vix_change_value": -0.3,
                "vix_change_rate": -0.017,
                "usdkrw": 1484.8,
                "usdkrw_change_value": 0.0,
                "usdkrw_change_rate": 0.0,
                "dxy": 98.48,
                "dxy_change_value": -0.02,
                "dxy_change_rate": -0.0002,
                "us10y": 4.42,
                "us10y_change_bp": -3.0,
                "us3y": 3.98,
                "us3y_change_bp": 7.0,
                "us10y_us3y_spread": 0.468,
                "us10y_us3y_spread_change_bp": -4.4,
                "kr10y": 3.93,
                "kr10y_change_bp": 0.0,
                "brent": 110.43,
                "brent_change_value": -2.19,
                "brent_change_rate": -0.0194,
                "wti": 102.62,
                "wti_change_value": -1.34,
                "wti_change_rate": -0.0129,
                "advancing_ratio": 0.58,
                "kospi_foreign_net_buy": 1,
                "kospi_institutional_net_buy": 1,
                "kospi": 7498,
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
                    "data_status": "FRESH",
                    "change_rate_1d": 0.0514,
                    "return_20d": 0.3687,
                    "trading_value_ratio_20d": 1.27,
                    "foreign_net_buy": 10,
                    "institutional_net_buy": -3,
                    "warnings": ["OVERHEATED_20D"],
                },
                {
                    "symbol": "305720",
                    "name": "KODEX 2차전지산업",
                    "sector_group": "2차전지",
                    "theme_group": "2차전지",
                    "role": "primary",
                    "exclude_from_signal": False,
                    "data_status": "FRESH",
                    "change_rate_1d": 0.0378,
                    "return_20d": 0.2756,
                    "trading_value_ratio_20d": 1.10,
                    "foreign_net_buy": 8,
                    "institutional_net_buy": -2,
                    "warnings": [],
                },
            ],
            "watchlist": [
                {"symbol": "000660", "name": "SK하이닉스", "market": "KOSPI", "sector_group": "반도체", "close_price": 1686000, "return_5d": 0.1812, "return_20d": None, "return_60d": 0.40, "trading_value_ratio_20d": None, "foreign_net_buy": 12, "institutional_net_buy": 2, "roe": 18, "debt_ratio": 25, "short_ratio": 1, "data_status": "FRESH", "source_mixed": False, "stale_days": 0},
                {"symbol": "005930", "name": "삼성전자", "market": "KOSPI", "sector_group": "반도체", "close_price": 268500, "return_5d": 0.0592, "return_20d": None, "return_60d": 0.24, "trading_value_ratio_20d": None, "foreign_net_buy": 10, "institutional_net_buy": 4, "roe": 15, "debt_ratio": 20, "short_ratio": 1, "data_status": "FRESH", "source_mixed": False, "stale_days": 0},
                {"symbol": "071050", "name": "한국금융지주", "market": "KOSPI", "sector_group": "금융/증권", "close_price": 100000, "return_5d": 0.0116, "return_20d": 0.2358, "return_60d": 0.3113, "trading_value_ratio_20d": 1.10, "foreign_net_buy": 3, "institutional_net_buy": 1, "roe": 14, "debt_ratio": 160, "short_ratio": 5.2, "data_status": "STALE_BUT_USABLE", "source_mixed": False, "stale_days": 1},
            ],
            "watchlist_diagnostics": {"raw_row_count": 21, "active_row_count": 7},
            "rankings": [{"symbol": "005930", "name": "삼성전자", "rank_type": "volume", "rank": 1, "source": "KIS", "market": "KOSPI"}],
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
                "data_limitation_note": "국내 리포트는 KIS 유니버스 기반으로 운영합니다. 전체시장 거래대금·시총 Top은 사용하지 않고, KIS 거래량 후보와 관심종목 중심으로 해석합니다.",
            },
            "contract_failed_views": [],
        }

    def test_holiday_report_uses_next_trading_day_language(self):
        text = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"]
        self.assertIn("관심종목 다음 거래일 점검", text)
        self.assertIn("다음 거래일 확인 포인트", text)
        self.assertNotIn("장중 체크포인트", text)
        self.assertNotIn("09:30", text)

    def test_morning_report_uses_korean_names(self):
        text = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"]
        self.assertIn("반도체", text)
        self.assertIn("2차전지", text)
        self.assertIn("SK하이닉스(000660)", text)
        self.assertIn("삼성전자(005930)", text)
        self.assertNotIn("Semiconductor", text)
        self.assertNotIn("Battery", text)

    def test_one_line_judgment_is_compact(self):
        lines = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"].splitlines()
        judgment = lines[lines.index("2. 오늘의 한 줄 판단") + 1]
        self.assertLessEqual(judgment.count("."), 3)
        self.assertIn("다음 거래일", judgment)

    def test_scale_warning_is_not_for_high_level_only(self):
        text = generate_morning_brief(self._bundle(), "2026-05-05")["report_text"]
        self.assertNotIn("일부 지수·종목 가격은 원천 스케일 확인이 필요합니다.", text)

    def test_snapshot_contains_readiness_fields(self):
        snapshot = generate_morning_brief(self._bundle(), "2026-05-05")["snapshot"]
        for key in [
            "stockdata_readiness",
            "report_allowed_sections",
            "report_blocked_sections",
            "kr_full_market_price_ready",
            "kis_volume_ranking_ready",
            "kis_universe_ready",
            "display_mode",
            "data_limitation_note",
            "scale_warnings",
            "watchlist_diagnostics",
        ]:
            self.assertIn(key, snapshot)

    def test_snapshot_file_is_created(self):
        result = generate_morning_brief(self._bundle(), "2026-05-05")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_morning_snapshot(Path(tmpdir), "2026-05-05", result["snapshot"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["report_date"], "2026-05-05")
            self.assertEqual(payload["display_mode"], "KIS_UNIVERSE_ONLY")
            self.assertEqual(payload["watchlist_diagnostics"]["raw_row_count"], 21)


if __name__ == "__main__":
    unittest.main()
