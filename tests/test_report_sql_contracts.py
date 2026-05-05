from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent


class ReportSqlContractTests(unittest.TestCase):
    def test_sector_etf_signal_view_contains_required_columns(self):
        sql = (ROOT / "sql" / "views" / "report_sector_etf_signal_view.sql").read_text(encoding="utf-8")
        for token in [
            "stale_days",
            "data_status",
            "return_5d",
            "return_20d",
            "return_60d",
            "trading_value_ratio_20d",
            "foreign_holding_ratio",
        ]:
            self.assertIn(token, sql)

    def test_data_freshness_view_contains_coverage_fields(self):
        sql = (ROOT / "sql" / "views" / "report_data_freshness_view.sql").read_text(encoding="utf-8")
        for token in [
            "latest_stock_price_date",
            "latest_ranking_date",
            "sector_etf_coverage_status",
            "watchlist_coverage_status",
            "missing_required_data",
        ]:
            self.assertIn(token, sql)

    def test_watchlist_snapshot_view_unions_static_and_report_required_sources(self):
        sql = (ROOT / "sql" / "views" / "report_watchlist_snapshot_view.sql").read_text(encoding="utf-8")
        self.assertIn("static_stock_universe", sql)
        self.assertIn("report_required_stock_universe", sql)
        self.assertIn("select distinct on (symbol)", sql.lower())


if __name__ == "__main__":
    unittest.main()
