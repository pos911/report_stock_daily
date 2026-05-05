from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import config


VIEW_FILES = [
    PROJECT_ROOT / "sql" / "views" / "report_morning_macro_view.sql",
    PROJECT_ROOT / "sql" / "views" / "report_sector_etf_signal_view.sql",
    PROJECT_ROOT / "sql" / "views" / "report_watchlist_snapshot_view.sql",
    PROJECT_ROOT / "sql" / "views" / "report_market_ranking_view.sql",
    PROJECT_ROOT / "sql" / "views" / "report_data_freshness_view.sql",
]


def _load_sql() -> str:
    return "\n\n".join(path.read_text(encoding="utf-8").strip() for path in VIEW_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy report contract views to Supabase Postgres.")
    parser.add_argument("--db-url", dest="db_url", help="Override database URL.")
    parser.add_argument("--dry-run", action="store_true", help="Print the SQL bundle without executing it.")
    args = parser.parse_args()

    db_url = args.db_url or config.get("DATABASE_URL") or config.get("connection_string", section="supabase")
    sql_bundle = _load_sql()

    if args.dry_run:
        print(sql_bundle)
        return

    if not db_url:
        raise SystemExit("Database URL is required. Set DATABASE_URL or config.supabase.connection_string.")

    with psycopg.connect(db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_bundle)
    print("report contract views deployed")


if __name__ == "__main__":
    main()
