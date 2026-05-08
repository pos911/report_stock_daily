import unittest

from src.reports.morning_report import generate_morning_brief


class ReportOutputQualityTests(unittest.TestCase):
    def test_morning_report_avoids_repetitive_system_words(self):
        bundle = {
            "freshness": {
                "target_date": "2026-05-09",
                "latest_macro_date": "2026-05-09",
                "latest_stock_price_date": "2026-05-09",
                "latest_ranking_date": "2026-05-09",
                "latest_supply_date": "2026-05-09",
                "latest_breadth_date": "2026-05-09",
                "sector_etf_coverage_status": "PASS",
                "watchlist_coverage_status": "PASS",
                "stale_warnings": "",
                "missing_required_data": "",
                "xkrx_is_open": True,
                "xnys_is_open": True,
                "carry_forward_fields": [],
            },
            "macro": {"sp500": 5000, "sp500_change_value": 10, "sp500_change_rate": 0.002, "nasdaq": 18000, "nasdaq_change_value": 20, "nasdaq_change_rate": 0.002, "sox": 5000, "sox_change_value": 10, "sox_change_rate": 0.002, "vix": 15, "vix_change_value": -0.1, "vix_change_rate": -0.01, "usdkrw": 1400, "usdkrw_change_value": 0.0, "usdkrw_change_rate": 0.0, "dxy": 100, "dxy_change_value": 0.0, "dxy_change_rate": 0.0, "us10y": 4.0, "us10y_change_bp": 0.0, "us3y": 3.8, "us3y_change_bp": 0.0, "us10y_us3y_spread": 0.2, "us10y_us3y_spread_change_bp": 0.0, "kr10y": 3.1, "kr10y_change_bp": 0.0, "brent": 80, "brent_change_value": 0.0, "brent_change_rate": 0.0, "wti": 78, "wti_change_value": 0.0, "wti_change_rate": 0.0, "advancing_ratio": 0.5, "kospi_foreign_net_buy": 0, "kospi_institutional_net_buy": 0},
            "sector_etfs": [],
            "watchlist": [],
            "rankings": [],
            "readiness": {"display_mode": "MACRO_ONLY", "allowed_korean_sections": [], "blocked_korean_sections": [], "report_allowed_sections": ["macro", "us_market"], "report_blocked_sections": [], "data_limitation_note": ""},
        }
        text = generate_morning_brief(bundle, "2026-05-09")["report_text"]
        for banned in ["WARNING", "N/A", "Not available", "fallback", "NO_DATA"]:
            self.assertNotIn(banned, text)


if __name__ == "__main__":
    unittest.main()
