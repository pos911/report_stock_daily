import datetime
import unittest

from src.data.supabase_reader import SupabaseReader
from src.jobs.generate_report import (
    _build_market_closed_skip_text,
    _should_include_kr_sections,
    _should_include_us_sections,
    _should_skip_all_markets,
)


class MarketCalendarReportModeTests(unittest.TestCase):
    def test_determine_full_report(self):
        self.assertEqual(SupabaseReader._determine_report_market_mode(True, True, True), "FULL_REPORT")

    def test_determine_korea_only(self):
        self.assertEqual(SupabaseReader._determine_report_market_mode(True, False, True), "KOREA_ONLY")

    def test_determine_us_only(self):
        self.assertEqual(SupabaseReader._determine_report_market_mode(False, True, True), "US_ONLY")

    def test_determine_skip_all_markets_closed(self):
        self.assertEqual(SupabaseReader._determine_report_market_mode(False, False, True), "SKIP_ALL_MARKETS_CLOSED")

    def test_determine_calendar_unknown(self):
        self.assertEqual(SupabaseReader._determine_report_market_mode(True, True, False), "CALENDAR_UNKNOWN")

    def test_skip_all_markets_disables_sections(self):
        calendar_status = {"report_market_mode": "SKIP_ALL_MARKETS_CLOSED"}
        self.assertTrue(_should_skip_all_markets(calendar_status))
        self.assertFalse(_should_include_kr_sections(calendar_status))
        self.assertFalse(_should_include_us_sections(calendar_status))

    def test_us_only_excludes_korea_sections(self):
        calendar_status = {"report_market_mode": "US_ONLY"}
        self.assertFalse(_should_include_kr_sections(calendar_status))
        self.assertTrue(_should_include_us_sections(calendar_status))

    def test_korea_only_excludes_new_us_interpretation_path(self):
        calendar_status = {"report_market_mode": "KOREA_ONLY"}
        self.assertTrue(_should_include_kr_sections(calendar_status))
        self.assertFalse(_should_include_us_sections(calendar_status))

    def test_skip_text_contains_status(self):
        text = _build_market_closed_skip_text(
            "morning",
            datetime.datetime(2026, 5, 5, 8, 0),
            {
                "report_date": "2026-05-05",
                "xkrx_reason": "Children's Day",
                "xnys_reason": "Weekend",
                "xkrx_previous_trading_day": "2026-05-04",
                "xnys_previous_trading_day": "2026-05-02",
                "report_market_mode": "SKIP_ALL_MARKETS_CLOSED",
            },
        )
        self.assertIn("SKIPPED_REPORT_MARKET_CLOSED", text)
        self.assertIn("XKRX", text)
        self.assertIn("XNYS", text)


if __name__ == "__main__":
    unittest.main()
