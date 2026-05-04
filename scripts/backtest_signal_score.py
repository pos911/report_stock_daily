from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.supabase_reader import SupabaseReader


SIGNAL_MODEL_VERSION = "v0.1_unbacktested"


@dataclass
class BacktestConfig:
    date_from: str | None = None
    date_to: str | None = None
    market: str | None = None
    dry_run: bool = False


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest skeleton for Signal Score v0.1")
    parser.add_argument("--date-from", dest="date_from")
    parser.add_argument("--date-to", dest="date_to")
    parser.add_argument("--market", choices=["KOSPI", "KOSDAQ"], help="ETF/ETN are excluded in v0.1 skeleton.")
    parser.add_argument("--dry-run", action="store_true", help="Print dataset plan only.")
    return parser


def build_dataset_plan(config: BacktestConfig) -> dict:
    return {
        "signal_model_version": SIGNAL_MODEL_VERSION,
        "date_from": config.date_from,
        "date_to": config.date_to,
        "market": config.market,
        "tables": {
            "rankings": "normalized_market_rankings_daily",
            "prices": "normalized_stock_prices_daily",
            "supply": "normalized_stock_supply_daily",
            "short_selling": "normalized_stock_short_selling",
            "fundamentals_ratios": "normalized_stock_fundamentals_ratios",
        },
        "forward_windows": [1, 5, 20],
        "score_buckets": [
            "score >= 4",
            "score 1~3",
            "score -1~0",
            "score <= -2",
        ],
        "metrics": [
            "forward_return_1d",
            "forward_return_5d",
            "forward_return_20d",
            "hit_rate",
            "average_return",
            "median_return",
            "max_drawdown_proxy",
        ],
        "notes": [
            "This script is a skeleton and does not claim validated performance.",
            "Watchlist labels in the report remain heuristic until this backtest is completed.",
            "ETF/ETN are excluded from the initial score validation pass.",
        ],
    }


def main() -> None:
    args = build_argument_parser().parse_args()
    config = BacktestConfig(
        date_from=args.date_from,
        date_to=args.date_to,
        market=args.market,
        dry_run=args.dry_run,
    )
    plan = build_dataset_plan(config)

    if config.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    reader = SupabaseReader()
    readiness = reader.fetch_report_readiness()
    payload = {
        "signal_model_version": SIGNAL_MODEL_VERSION,
        "dataset_plan": plan,
        "report_readiness_reference": readiness,
        "status": "skeleton_only",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
