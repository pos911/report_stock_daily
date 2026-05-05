"""
Cleanup raw StockData tables with retention rules.

Usage:
    python scripts/cleanup_raw_tables.py
    python scripts/cleanup_raw_tables.py --apply
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
from src.utils.report_universe import RETENTION_POLICIES


DATE_COLUMNS = {
    "raw_stock_prices_daily": "base_date",
    "raw_market_rankings": "base_date",
    "raw_macro": "base_date",
    "pipeline_run_logs": "created_at",
}


def _client():
    return create_client(
        config.get("url", section="supabase"),
        config.get("service_role_key", section="supabase"),
    )


def build_cleanup_plan(current_date: str, retention_days: dict[str, int] | None = None) -> list[dict]:
    policies = retention_days or RETENTION_POLICIES
    today = dt.date.fromisoformat(current_date)
    plan = []
    for table, days in policies.items():
        cutoff = today - dt.timedelta(days=days)
        plan.append(
            {
                "table": table,
                "retention_days": days,
                "date_column": DATE_COLUMNS[table],
                "cutoff_date": cutoff.isoformat(),
            }
        )
    return plan


def count_candidate_rows(client, table: str, date_column: str, cutoff_date: str) -> int:
    response = (
        client.table(table)
        .select(date_column)
        .lt(date_column, cutoff_date)
        .limit(100000)
        .execute()
    )
    return len(response.data or [])


def delete_candidate_rows(client, table: str, date_column: str, cutoff_date: str):
    return client.table(table).delete().lt(date_column, cutoff_date).execute()


def write_pipeline_log(client, mode: str, summary: dict):
    payload = {
        "pipeline_name": "cleanup_raw_tables",
        "status": mode.upper(),
        "details": json.dumps(summary, ensure_ascii=False),
    }
    try:
        client.table("pipeline_run_logs").insert(payload).execute()
    except Exception:
        pass


def execute_cleanup(current_date: str, apply: bool = False) -> dict:
    client = _client()
    summary = {"current_date": current_date, "apply": apply, "tables": []}
    for item in build_cleanup_plan(current_date):
        count = count_candidate_rows(client, item["table"], item["date_column"], item["cutoff_date"])
        table_summary = {**item, "candidate_rows": count, "deleted_rows": 0}
        if apply and count:
            delete_candidate_rows(client, item["table"], item["date_column"], item["cutoff_date"])
            table_summary["deleted_rows"] = count
        summary["tables"].append(table_summary)
    write_pipeline_log(client, "apply" if apply else "dry_run", summary)
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description="Cleanup raw tables with retention rules")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--apply", action="store_true", help="Actually delete old rows.")
    return parser.parse_args()


def main():
    args = _parse_args()
    summary = execute_cleanup(args.date, apply=args.apply)
    for item in summary["tables"]:
        print(
            f"{item['table']}: cutoff={item['cutoff_date']} "
            f"candidates={item['candidate_rows']} deleted={item['deleted_rows']}"
        )


if __name__ == "__main__":
    main()
