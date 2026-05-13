import unittest
import datetime as dt

from src.services.supabase_stockdata_reader import SupabaseStockDataReader


class SupabaseStockDataReaderTests(unittest.TestCase):
    def setUp(self):
        self.reader = SupabaseStockDataReader.__new__(SupabaseStockDataReader)
        self.reader._last_watchlist_diagnostics = {}
        def _calendar_status(_self, report_date=None):
            base = dt.date.fromisoformat(str(report_date or "2026-05-13"))
            previous = (base - dt.timedelta(days=1)).isoformat()
            return {
                "xkrx_previous_trading_day": previous,
                "xnys_previous_trading_day": previous,
            }
        self.reader.base_reader = type(
            "BaseReaderStub",
            (),
            {
                "fetch_market_calendar_status": _calendar_status,
                "fetch_static_stock_universe": lambda _self: [],
            },
        )()

    def test_change_rate_recalculated_from_change_amount_and_previous_value(self):
        warnings = []
        rate = self.reader._normalize_percent_change_rate(
            field="sp500",
            explicit_rate=-40.62,
            change_value=-41.0,
            previous_value=7241.0,
            warnings=warnings,
        )
        self.assertAlmostEqual(rate, -41.0 / 7241.0, places=6)
        self.assertEqual(warnings, [])

    def test_decimal_change_rate_is_preserved(self):
        warnings = []
        rate = self.reader._normalize_percent_change_rate(
            field="sp500",
            explicit_rate=-0.0057,
            change_value=None,
            previous_value=None,
            warnings=warnings,
        )
        self.assertAlmostEqual(rate, -0.0057, places=6)

    def test_percent_unit_change_rate_is_normalized(self):
        warnings = []
        rate = self.reader._normalize_percent_change_rate(
            field="sp500",
            explicit_rate=-0.57,
            change_value=None,
            previous_value=None,
            warnings=warnings,
        )
        self.assertAlmostEqual(rate, -0.0057, places=6)

    def test_abnormal_change_rate_becomes_warning(self):
        warnings = []
        rate = self.reader._normalize_percent_change_rate(
            field="sp500",
            explicit_rate=-40.62,
            change_value=None,
            previous_value=None,
            warnings=warnings,
        )
        self.assertIsNone(rate)
        self.assertTrue(any("anomaly" in warning for warning in warnings))

    def test_normalize_readiness_derives_blocked_sections(self):
        normalized = self.reader.normalize_report_readiness(
            {
                "kr_full_market_price_ready": False,
                "kis_universe_ready": True,
                "kis_volume_ranking_ready": True,
                "kr_trading_value_ranking_ready": True,
                "kr_market_cap_ranking_ready": True,
                "report_allowed_sections": ["macro", "us_market"],
                "report_blocked_sections": [],
            }
        )
        self.assertEqual(normalized["display_mode"], "KIS_UNIVERSE_ONLY")
        self.assertIn("kr_full_market_trading_value_top", normalized["report_blocked_sections"])
        self.assertIn("kr_full_market_market_cap_top", normalized["report_blocked_sections"])
        self.assertIn("watchlist_signal", normalized["report_allowed_sections"])

    def test_watchlist_snapshot_filters_to_active_symbols(self):
        class BaseReaderStub:
            def fetch_static_stock_universe(self):
                return [
                    {"symbol": "005930", "name": "삼성전자", "market": "KOSPI"},
                    {"symbol": "000660", "name": "SK하이닉스", "market": "KOSPI"},
                ]

        self.reader.base_reader = BaseReaderStub()
        self.reader._fetch_view_rows = lambda *args, **kwargs: [
            {"symbol": "005930", "name": "삼성전자", "market": "KOSPI", "base_date": "2026-05-08", "close_price": 1, "data_status": "FRESH"},
            {"symbol": "000660", "name": "SK하이닉스", "market": "KOSPI", "base_date": "2026-05-08", "close_price": 1, "data_status": "FRESH"},
            {"symbol": "123456", "name": "과거종목", "market": "KOSPI", "base_date": "2026-05-08", "close_price": 1, "data_status": "FRESH"},
        ]
        self.reader._fetch_watchlist_quality_map = lambda symbols: {}
        rows = self.reader.get_watchlist_snapshot("2026-05-08")
        self.assertEqual([row["symbol"] for row in rows], ["000660", "005930"])
        self.assertEqual(self.reader._last_watchlist_diagnostics["raw_row_count"], 3)
        self.assertEqual(self.reader._last_watchlist_diagnostics["active_row_count"], 2)

    def test_watchlist_snapshot_uses_latest_row_on_or_before_target_date(self):
        class BaseReaderStub:
            def fetch_static_stock_universe(self):
                return [
                    {"symbol": "005930", "name": "삼성전자", "market": "KOSPI"},
                ]

        self.reader.base_reader = BaseReaderStub()
        self.reader._fetch_view_rows = lambda *args, **kwargs: [
            {"symbol": "005930", "name": "삼성전자", "market": "KOSPI", "base_date": "2026-05-09", "close_price": 9, "data_status": "FRESH"},
            {"symbol": "005930", "name": "삼성전자", "market": "KOSPI", "base_date": "2026-05-08", "close_price": 8, "data_status": "FRESH"},
            {"symbol": "005930", "name": "삼성전자", "market": "KOSPI", "base_date": "2026-05-07", "close_price": 7, "data_status": "FRESH"},
        ]
        self.reader._fetch_watchlist_quality_map = lambda symbols: {}
        rows = self.reader.get_watchlist_snapshot("2026-05-08")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["base_date"], "2026-05-08")
        self.assertEqual(rows[0]["close_price"], 8)

    def test_morning_macro_snapshot_excludes_future_rows(self):
        self.reader._fetch_view_rows = lambda *args, **kwargs: [
            {"base_date": "2026-05-09", "sp500": 9},
            {"base_date": "2026-05-08", "sp500": 8},
            {"base_date": "2026-05-07", "sp500": 7},
        ]
        self.reader._fetch_intraday_macro_rows_for_date = lambda base_date: []
        self.reader._fetch_latest_row_on_or_before = lambda *args, **kwargs: {"base_date": "2026-05-07", "sp500": 7}
        self.reader._fetch_previous_macro_row = lambda current_base_date: {}
        self.reader._inject_deltas = lambda current, previous, warnings=None: None
        self.reader._attach_breadth = lambda macro, breadth, warnings: macro.update({"breadth": breadth, "advancing_ratio": None})
        self.reader._select_previous_macro_row = SupabaseStockDataReader._select_previous_macro_row.__get__(self.reader, SupabaseStockDataReader)
        macro = self.reader.get_morning_macro_snapshot("2026-05-08")
        self.assertEqual(macro["effective_date"], "2026-05-07")
        self.assertEqual(macro["base_date"], "2026-05-07")
        self.assertEqual(macro["sp500"], 7)

    def test_morning_bundle_uses_previous_trading_day_cutoff(self):
        self.reader.get_report_data_freshness = lambda target_date=None: {}
        self.reader.get_macro_snapshot = lambda report_type, target_date=None: {"base_date": "2026-05-12", "contract_fallback_used": False}
        self.reader.get_sector_etf_signals = lambda target_date=None: [{"latest_price_date": target_date, "contract_fallback_used": False}]
        self.reader.get_watchlist_snapshot = lambda target_date=None: [{"base_date": target_date, "contract_fallback_used": False}]
        self.reader.get_market_rankings = lambda target_date=None: [{"base_date": target_date, "contract_fallback_used": False}]
        self.reader.normalize_report_readiness = lambda readiness=None: {}
        self.reader.base_reader.fetch_stockdata_report_readiness = lambda target_date=None: {}
        bundle = self.reader.get_report_contract_bundle("morning", "2026-05-13")
        self.assertEqual(bundle["session_cutoff_date"], "2026-05-12")
        self.assertEqual(bundle["watchlist"][0]["base_date"], "2026-05-12")
        self.assertEqual(bundle["rankings"][0]["base_date"], "2026-05-12")

    def test_regular_macro_prefers_same_day_intraday_row(self):
        self.reader._fetch_intraday_macro_rows_for_date = lambda base_date: [
            {"base_date": "2026-05-13", "series_id": "KOSPI", "captured_at": "2026-05-13T10:31:00+09:00", "quality_flag": "OK", "value": 7822.24, "source": "KIS", "source_symbol": "0001"},
            {"base_date": "2026-05-13", "series_id": "KOSPI", "captured_at": "2026-05-13T10:29:00+09:00", "quality_flag": "INVALID", "value": 1},
        ]
        self.reader._fetch_view_rows = lambda *args, **kwargs: []
        self.reader._fetch_latest_row_on_or_before = lambda *args, **kwargs: {"base_date": "2026-05-13"}
        self.reader._fetch_previous_intraday_macro_row = lambda current_base_date: {}
        self.reader._fetch_previous_macro_row = lambda current_base_date: {}
        self.reader._inject_deltas = lambda current, previous, warnings=None: None
        self.reader._attach_breadth = lambda macro, breadth, warnings: macro.update({"breadth": breadth, "advancing_ratio": None})
        macro = self.reader.get_macro_snapshot("regular", "2026-05-13")
        self.assertEqual(macro["base_date"], "2026-05-13")
        self.assertEqual(macro["macro_source_mode"], "intraday_blended")
        self.assertEqual(macro["source"], "KIS")
        self.assertEqual(macro["source_symbol"], "0001")

    def test_closing_macro_falls_back_to_daily_when_intraday_missing(self):
        self.reader._fetch_intraday_macro_rows_for_date = lambda base_date: []
        self.reader._fetch_view_rows = lambda *args, **kwargs: [{"base_date": "2026-05-13", "kospi": 7800}]
        self.reader._fetch_latest_row_on_or_before = lambda *args, **kwargs: {"base_date": "2026-05-13", "kospi": 7800}
        self.reader._fetch_previous_macro_row = lambda current_base_date: {}
        self.reader._inject_deltas = lambda current, previous, warnings=None: None
        self.reader._attach_breadth = lambda macro, breadth, warnings: macro.update({"breadth": breadth, "advancing_ratio": None})
        self.reader._select_previous_macro_row = SupabaseStockDataReader._select_previous_macro_row.__get__(self.reader, SupabaseStockDataReader)
        macro = self.reader.get_macro_snapshot("closing", "2026-05-13")
        self.assertEqual(macro["base_date"], "2026-05-13")
        self.assertEqual(macro["macro_source_mode"], "daily")

    def test_sector_etf_signals_exclude_future_rows(self):
        self.reader._fetch_view_rows = lambda *args, **kwargs: [
            {"symbol": "396500", "latest_price_date": "2026-05-09", "data_status": "FRESH"},
            {"symbol": "396500", "latest_price_date": "2026-05-08", "data_status": "FRESH"},
            {"symbol": "305720", "latest_price_date": "2026-05-08", "data_status": "FRESH"},
        ]
        self.reader._normalize_sector_etf_row = lambda row, contract_fallback_used=False, target_date=None: dict(row)
        rows = self.reader.get_sector_etf_signals("2026-05-08")
        self.assertEqual({row["symbol"]: row["latest_price_date"] for row in rows}, {"396500": "2026-05-08", "305720": "2026-05-08"})

    def test_market_rankings_use_latest_base_date_on_or_before_target(self):
        self.reader._fetch_view_rows = lambda *args, **kwargs: [
            {"symbol": "A", "base_date": "2026-05-09", "rank_type": "volume"},
            {"symbol": "B", "base_date": "2026-05-08", "rank_type": "volume", "rank": 2},
            {"symbol": "D", "base_date": "2026-05-08", "rank_type": "volume", "rank": 1},
            {"symbol": "C", "base_date": "2026-05-07", "rank_type": "volume", "rank": 3},
        ]
        self.reader._normalize_ranking_row = lambda row, contract_fallback_used=False, target_date=None: dict(row)
        rows = self.reader.get_market_rankings("2026-05-08")
        self.assertEqual([row["symbol"] for row in rows], ["D", "B"])

    def test_resolve_source_mixed_from_quality_flag(self):
        self.assertTrue(
            self.reader._resolve_source_mixed(
                explicit_value=None,
                data_quality_flag="SOURCE_MIXED",
                consistency_status=None,
                quality_hint=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
