"""
Verify report-consumption data readiness.

This script focuses on report-required coverage rather than broad ingestion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from supabase import create_client

from src.utils import config
from src.utils.report_universe import (
    active_symbols,
    classify_overall_status,
    evaluate_etf_coverage,
    evaluate_macro_freshness,
    evaluate_raw_retention,
    evaluate_watchlist_coverage,
    load_report_required_etf_universe,
    load_report_required_macro_series,
    load_report_required_stock_universe,
)


def _client():
    return create_client(
        config.get("url", section="supabase"),
        config.get("service_role_key", section="supabase"),
    )


def fetch_view_rows(client, view_name: str) -> list[dict]:
    try:
        response = client.table(view_name).select("*").limit(5000).execute()
        return response.data or []
    except Exception:
        return []


def fetch_latest_macro_rows(client, series_ids: list[str]) -> list[dict]:
    rows = []
    for series_id in series_ids:
        try:
            response = (
                client.table("normalized_macro_series")
                .select("series_id, base_date, value")
                .eq("series_id", series_id)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
        except Exception:
            response = (
                client.table("normalized_global_macro_daily")
                .select("base_date")
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            if response.data:
                rows.append({"series_id": series_id, "base_date": response.data[0]["base_date"]})
            continue
        if response.data:
            rows.append(response.data[0])
    return rows


def fetch_watchlist_rows(client, symbols: list[str], table_name: str) -> list[dict]:
    if not symbols:
        return []
    response = (
        client.table(table_name)
        .select("symbol, base_date")
        .in_("symbol", symbols)
        .order("base_date", desc=True)
        .limit(5000)
        .execute()
    )
    rows = response.data or []
    latest_by_symbol = {}
    for row in rows:
        symbol = row.get("symbol")
        if symbol and symbol not in latest_by_symbol:
            latest_by_symbol[symbol] = row
    return list(latest_by_symbol.values())


def fetch_raw_oldest_dates(client, table_names: list[str]) -> dict[str, str | None]:
    oldest = {}
    for table_name in table_names:
        try:
            date_column = "created_at" if table_name == "pipeline_run_logs" else "base_date"
            response = (
                client.table(table_name)
                .select(date_column)
                .order(date_column)
                .limit(1)
                .execute()
            )
            oldest[table_name] = response.data[0][date_column][:10] if response.data else None
        except Exception:
            oldest[table_name] = None
    return oldest


def build_verification_report(check_date: str) -> dict:
    client = _client()
    required_stocks = load_report_required_stock_universe(project_root)
    required_etfs = load_report_required_etf_universe(project_root)
    required_macro = load_report_required_macro_series(project_root)

    etf_view_rows = fetch_view_rows(client, "report_sector_etf_signal_view")
    watchlist_view_rows = fetch_view_rows(client, "report_watchlist_snapshot_view")
    freshness_rows = fetch_view_rows(client, "report_data_freshness_view")
    ranking_rows = fetch_view_rows(client, "report_market_ranking_view")

    etf_result = evaluate_etf_coverage(required_etfs, etf_view_rows, stale_warn_days=3)
    macro_result = evaluate_macro_freshness(
        required_macro,
        fetch_latest_macro_rows(client, [row.get("series_id") or row["symbol"] for row in required_macro]),
        check_date,
    )
    watchlist_symbols = active_symbols(required_stocks)
    watchlist_result = evaluate_watchlist_coverage(
        watchlist_symbols,
        watchlist_view_rows or fetch_watchlist_rows(client, watchlist_symbols, "normalized_stock_prices_daily"),
        fetch_watchlist_rows(client, watchlist_symbols, "normalized_stock_supply_daily"),
    )
    ranking_latest = max((row.get("base_date") for row in ranking_rows if row.get("base_date")), default=None)
    ranking_result = {
        "status": "FAIL" if not ranking_latest else "WARN" if ranking_latest < check_date else "PASS",
        "latest_base_date": ranking_latest,
    }
    raw_retention_result = evaluate_raw_retention(
        check_date,
        fetch_raw_oldest_dates(
            client,
            ["raw_stock_prices_daily", "raw_market_rankings", "raw_macro", "pipeline_run_logs"],
        ),
    )

    exclusion_violations = [
        row["symbol"]
        for row in etf_view_rows
        if row.get("exclude_from_signal") is False and str(row.get("symbol")) in {item["symbol"] for item in required_etfs if item.get("exclude_from_signal")}
    ]
    exclusion_result = {
        "status": "FAIL" if exclusion_violations else "PASS",
        "violations": exclusion_violations,
    }

    results = {
        "etf_coverage": etf_result,
        "macro_freshness": macro_result,
        "watchlist_coverage": watchlist_result,
        "ranking_freshness": ranking_result,
        "raw_retention": raw_retention_result,
        "signal_exclusion": exclusion_result,
        "freshness_view_rows": freshness_rows,
    }
    results["overall_status"] = classify_overall_status(results.values())
    return results


def _parse_args():
    parser = argparse.ArgumentParser(description="Verify report-oriented StockData readiness")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    return parser.parse_args()


def main():
    args = _parse_args()
    report = build_verification_report(args.date)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
