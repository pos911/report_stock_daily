import unittest
from pathlib import Path

from src.utils.report_universe import (
    active_symbols,
    load_legacy_target_stocks,
    load_report_required_etf_universe,
    load_report_required_macro_series,
    load_report_required_stock_universe,
    prioritize_detail_targets,
    validate_legacy_watchlist_migration,
)


class ReportUniverseTests(unittest.TestCase):
    def test_report_required_stock_universe_loads(self):
        rows = load_report_required_stock_universe()
        self.assertGreaterEqual(len(rows), 5)
        self.assertTrue(any(row["symbol"] == "005930" for row in rows))

    def test_report_required_etf_universe_loads_and_marks_exclusions(self):
        rows = load_report_required_etf_universe()
        leverage = next(row for row in rows if row["symbol"] == "462330")
        self.assertTrue(leverage["exclude_from_signal"])
        self.assertIn("레버리지", leverage["exclude_reason"])

    def test_report_required_etf_universe_is_koreanized(self):
        rows = load_report_required_etf_universe()
        joined = " ".join(f"{row['name']} {row['sector_group']} {row.get('theme_group') or ''}" for row in rows)
        for banned in ["Semiconductor", "Battery", "Defense", "Shipbuilding", "Financials", "Healthcare", "AI Power"]:
            self.assertNotIn(banned, joined)
        self.assertTrue(any(row["sector_group"] == "반도체" for row in rows))

    def test_report_required_stock_universe_is_koreanized(self):
        rows = load_report_required_stock_universe()
        joined = " ".join(f"{row['name']} {row['sector_group']}" for row in rows)
        for banned in ["Samsung Electronics", "SK hynix", "Hyundai Mobis", "Semiconductor", "Automobile", "Financials"]:
            self.assertNotIn(banned, joined)
        self.assertTrue(any(row["name"] == "삼성전자" for row in rows))

    def test_report_required_macro_series_loads(self):
        rows = load_report_required_macro_series()
        symbols = {row["symbol"] for row in rows}
        self.assertIn("US10Y", symbols)
        self.assertIn("US3Y", symbols)
        self.assertIn("KOSPI", symbols)

    def test_active_symbols_filters_inactive(self):
        result = active_symbols(
            [
                {"symbol": "005930", "is_active": True},
                {"symbol": "000660", "is_active": False},
            ]
        )
        self.assertEqual(result, ["005930"])

    def test_prioritize_detail_targets_caps_to_limit(self):
        rows = prioritize_detail_targets(
            static_rows=[{"symbol": "005930", "enabled": True}, {"symbol": "000660", "enabled": True}],
            report_stock_rows=[{"symbol": "058470", "is_active": True}],
            report_etf_rows=[{"symbol": "396500", "is_active": True, "market": "ETF"}],
            ranking_rows=[
                {"symbol": "035420", "rank_type": "trading_value", "rank": 1, "trading_value": 1000},
                {"symbol": "051910", "rank_type": "trading_value", "rank": 2, "trading_value": 900},
            ],
            detail_limit=3,
            max_limit=5,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["symbol"], "005930")

    def test_legacy_target_stocks_are_migrated(self):
        legacy_rows = load_legacy_target_stocks()
        report_rows = load_report_required_stock_universe()
        result = validate_legacy_watchlist_migration(legacy_rows, static_rows=[], report_rows=report_rows)
        self.assertEqual(result["legacy_count"], len(legacy_rows))
        self.assertEqual(result["missing_symbols"], [])

    def test_non_ranking_watchlist_symbol_is_kept_in_detail_universe(self):
        rows = prioritize_detail_targets(
            static_rows=[],
            report_stock_rows=[{"symbol": "017670", "is_active": True}],
            report_etf_rows=[],
            ranking_rows=[{"symbol": "005930", "rank_type": "trading_value", "rank": 1, "trading_value": 1000}],
            detail_limit=10,
            max_limit=10,
        )
        symbols = {row["symbol"] for row in rows}
        self.assertIn("017670", symbols)
        self.assertIn("005930", symbols)

    def test_deploy_seed_sql_has_no_english_sector_group(self):
        sql = (Path(__file__).resolve().parent.parent / "sql" / "deploy_report_universe_tables.sql").read_text(encoding="utf-8")
        for banned in ["'Semiconductor'", "'Battery'", "'Defense'", "'Shipbuilding'", "'Financials'", "'Healthcare'", "'AI Power'"]:
            self.assertNotIn(banned, sql)


if __name__ == "__main__":
    unittest.main()
