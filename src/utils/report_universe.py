from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.utils.market_assets import canonicalize_symbol

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
LEGACY_TARGET_STOCKS_PATH = CONFIG_DIR / "target_stocks.json"

DEFAULT_DETAIL_LIMIT = 300
MAX_DETAIL_LIMIT = 500

RETENTION_POLICIES = {
    "raw_stock_prices_daily": 60,
    "raw_market_rankings": 60,
    "raw_macro": 90,
    "pipeline_run_logs": 180,
}

PIPELINE_FREQUENCIES = {
    "daily": [
        "normalized_global_macro_daily",
        "normalized_market_rankings_daily",
        "normalized_stock_prices_daily",
        "market_breadth_daily",
        "normalized_stock_supply_daily",
        "etf_etn_ranking",
        "report_required_etf_coverage",
    ],
    "daily_close": [
        "normalized_stock_short_selling",
        "normalized_stock_snapshots_daily",
        "sector_etf_coverage_check",
        "data_quality_verification",
    ],
    "weekly": [
        "stocks_master_full_refresh",
        "etf_etn_master_full_refresh",
        "normalized_stock_fundamentals_ratios",
        "static_universe_validation",
        "raw_retention_cleanup",
    ],
    "monthly": [
        "market_trading_calendar_sync",
        "macro_series_master_validation",
        "long_horizon_data_quality_summary",
    ],
}

REPORT_FEATURE_CONTRACT = [
    "return_5d",
    "return_20d",
    "return_60d",
    "trading_value_ratio_20d",
    "volatility_20d",
    "near_52w_high_pct",
    "foreign_flow_direction",
    "short_ratio",
    "value_quality_score",
]

SIGNAL_EXCLUSION_KEYWORDS = (
    "lever",
    "leverage",
    "2x",
    "3x",
    "inverse",
    "inverter",
    "bear",
    "bull",
    "covered call",
    "coverdcall",
    "synthetic",
    "futures",
    "etn",
    "선물",
    "레버리지",
    "인버스",
    "커버드콜",
    "합성",
)


@dataclass(frozen=True)
class UniverseEntry:
    symbol: str
    name: str
    market: str | None = None
    sector_group: str | None = None
    theme_group: str | None = None
    role: str | None = None
    is_active: bool = True
    notes: str | None = None


def _config_path(filename: str, root: Path | None = None) -> Path:
    base = root or PROJECT_ROOT
    return base / "config" / filename


def load_yaml_file(path: Path) -> list[dict]:
    if yaml is not None:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or []
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a top-level YAML list")
        return payload
    return _load_simple_yaml_list(path)


def _load_simple_yaml_list(path: Path) -> list[dict]:
    items: list[dict] = []
    current: dict | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if current:
                items.append(current)
            current = {}
            key, value = _split_yaml_key_value(stripped[2:])
            current[key] = _parse_simple_yaml_value(value)
            continue
        if current is None:
            raise ValueError(f"{path} must contain a YAML list")
        key, value = _split_yaml_key_value(stripped)
        current[key] = _parse_simple_yaml_value(value)
    if current:
        items.append(current)
    return items


def _split_yaml_key_value(text: str) -> tuple[str, str]:
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _parse_simple_yaml_value(value: str):
    if value in {"", "null", "Null", "NULL"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def _normalize_entries(rows: Iterable[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        symbol = canonicalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        item = dict(row)
        item["symbol"] = symbol
        item["is_active"] = bool(item.get("is_active", True))
        normalized.append(item)
    return normalized


def load_report_required_stock_universe(root: Path | None = None) -> list[dict]:
    return _normalize_entries(load_yaml_file(_config_path("report_required_stock_universe.yml", root)))


def load_report_required_etf_universe(root: Path | None = None) -> list[dict]:
    rows = _normalize_entries(load_yaml_file(_config_path("report_required_etf_universe.yml", root)))
    normalized: list[dict] = []
    for row in rows:
        item = dict(row)
        item["exclude_from_signal"] = should_exclude_from_signal(item)
        if item["exclude_from_signal"] and not item.get("exclude_reason"):
            item["exclude_reason"] = infer_signal_exclusion_reason(item)
        normalized.append(item)
    return normalized


def load_report_required_macro_series(root: Path | None = None) -> list[dict]:
    return _normalize_entries(load_yaml_file(_config_path("report_required_macro_series.yml", root)))


def load_legacy_target_stocks(root: Path | None = None) -> list[dict]:
    import json

    path = (root or PROJECT_ROOT) / "config" / "target_stocks.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) or []
    return _normalize_entries(payload)


def infer_signal_exclusion_reason(row: dict) -> str | None:
    name = f"{row.get('name', '')} {row.get('notes', '')}".lower()
    if "etn" in name:
        return "ETN is not allowed as a primary sector signal instrument."
    for keyword in SIGNAL_EXCLUSION_KEYWORDS:
        if keyword in name:
            return f"Excluded from signal because it is a {keyword} product."
    return row.get("exclude_reason")


def should_exclude_from_signal(row: dict) -> bool:
    if row.get("exclude_from_signal") is True:
        return True
    if row.get("exclude_from_signal") is False:
        return False
    return infer_signal_exclusion_reason(row) is not None


def active_symbols(rows: Iterable[dict]) -> list[str]:
    return [canonicalize_symbol(row.get("symbol")) for row in rows if row.get("is_active", True)]


def merge_report_stock_with_static_universe(static_rows: Iterable[dict], report_rows: Iterable[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for row in static_rows:
        symbol = canonicalize_symbol(row.get("symbol"))
        if not symbol or not row.get("enabled", row.get("is_active", True)):
            continue
        merged[symbol] = {
            "symbol": symbol,
            "name": row.get("name") or symbol,
            "market": row.get("market"),
            "enabled": True,
            "source": "static_stock_universe",
            "role": row.get("role") or "watchlist",
        }
    for row in report_rows:
        symbol = canonicalize_symbol(row.get("symbol"))
        if not symbol or not row.get("is_active", True):
            continue
        base = merged.get(symbol, {})
        merged[symbol] = {
            **base,
            **row,
            "symbol": symbol,
            "source": "report_required_stock_universe",
            "enabled": True,
        }
    return list(merged.values())


def validate_legacy_watchlist_migration(
    legacy_rows: Iterable[dict],
    static_rows: Iterable[dict],
    report_rows: Iterable[dict],
) -> dict:
    legacy_list = list(legacy_rows)
    static_symbols = {
        canonicalize_symbol(row.get("symbol"))
        for row in static_rows
        if row.get("enabled", row.get("is_active", True))
    }
    report_symbols = {
        canonicalize_symbol(row.get("symbol"))
        for row in report_rows
        if row.get("is_active", True)
    }
    migrated = []
    missing = []
    for row in legacy_list:
        symbol = canonicalize_symbol(row.get("symbol"))
        if symbol in static_symbols or symbol in report_symbols:
            migrated.append(symbol)
        else:
            missing.append(symbol)
    return {
        "legacy_count": len(legacy_list),
        "migrated_symbols": migrated,
        "missing_symbols": missing,
    }


def prioritize_detail_targets(
    static_rows: Iterable[dict],
    report_stock_rows: Iterable[dict],
    report_etf_rows: Iterable[dict],
    ranking_rows: Iterable[dict],
    ranking_extension_rows: Iterable[dict] | None = None,
    detail_limit: int = DEFAULT_DETAIL_LIMIT,
    max_limit: int = MAX_DETAIL_LIMIT,
) -> list[dict]:
    limit = min(max(detail_limit, 1), max_limit)
    registry: dict[str, dict] = {}

    def register(row: dict, priority: int, source: str, asset_bucket: str):
        symbol = canonicalize_symbol(row.get("symbol"))
        if not symbol:
            return
        trading_value = _numeric(row.get("trading_value"))
        current = registry.get(symbol)
        candidate = {
            **(current or {}),
            **row,
            "symbol": symbol,
            "priority": max(priority, (current or {}).get("priority", -1)),
            "priority_source": source if current is None or priority >= current.get("priority", -1) else current.get("priority_source"),
            "asset_bucket": asset_bucket,
            "trading_value": trading_value,
        }
        registry[symbol] = candidate

    for row in static_rows:
        if row.get("enabled", row.get("is_active", True)):
            register(row, 100, "static_stock_universe", "stock")
    for row in report_stock_rows:
        if row.get("is_active", True):
            register(row, 95, "report_required_stock_universe", "stock")
    for row in report_etf_rows:
        if row.get("is_active", True):
            register({**row, "market": row.get("market") or "ETF"}, 90, "report_required_etf_universe", "etf")
    for row in ranking_rows:
        rank_type = str(row.get("rank_type") or "").lower()
        priority = {"trading_value": 85, "volume": 80, "market_cap": 75}.get(rank_type, 70)
        register(row, priority, f"ranking:{rank_type}", "ranking")
    for row in ranking_extension_rows or []:
        register(row, 65, "ranking_extension", "ranking_extension")

    ranked = sorted(
        registry.values(),
        key=lambda row: (
            row.get("priority", 0),
            _numeric(row.get("trading_value")),
            _invert_rank(row.get("rank")),
            row.get("symbol"),
        ),
        reverse=True,
    )
    return ranked[:limit]


def evaluate_etf_coverage(required_rows: Iterable[dict], latest_rows: Iterable[dict], stale_warn_days: int = 3) -> dict:
    latest_map = {canonicalize_symbol(row.get("symbol")): row for row in latest_rows if row.get("symbol")}
    missing = []
    stale = []
    excluded_violations = []

    for row in required_rows:
        if not row.get("is_active", True):
            continue
        symbol = canonicalize_symbol(row.get("symbol"))
        latest = latest_map.get(symbol)
        if not latest:
            missing.append(symbol)
            continue
        stale_days = int(latest.get("stale_days") or 0)
        if stale_days >= stale_warn_days and str(row.get("role") or "").lower() == "primary":
            stale.append(symbol)
        if not row.get("exclude_from_signal") and latest.get("exclude_from_signal") is True:
            excluded_violations.append(symbol)

    if missing:
        status = "FAIL"
    elif stale:
        status = "WARN"
    else:
        status = "PASS"
    return {
        "status": status,
        "missing": missing,
        "stale_primary": stale,
        "exclusion_violations": excluded_violations,
    }


def evaluate_macro_freshness(required_rows: Iterable[dict], latest_rows: Iterable[dict], expected_date: str) -> dict:
    latest_map = {str(row.get("series_id") or row.get("symbol")): row for row in latest_rows}
    missing = []
    stale = []
    for row in required_rows:
        if not row.get("is_active", True):
            continue
        series_id = str(row.get("series_id") or row.get("symbol"))
        latest = latest_map.get(series_id)
        if not latest:
            missing.append(series_id)
            continue
        if str(latest.get("base_date")) < expected_date:
            stale.append(series_id)
    status = "FAIL" if missing else "WARN" if stale else "PASS"
    return {"status": status, "missing": missing, "stale": stale}


def evaluate_watchlist_coverage(required_symbols: Iterable[str], price_rows: Iterable[dict], supply_rows: Iterable[dict]) -> dict:
    required = {canonicalize_symbol(symbol) for symbol in required_symbols if symbol}
    prices = {canonicalize_symbol(row.get("symbol")) for row in price_rows if row.get("symbol")}
    supplies = {canonicalize_symbol(row.get("symbol")) for row in supply_rows if row.get("symbol")}
    missing_prices = sorted(required - prices)
    missing_supplies = sorted(required - supplies)
    status = "FAIL" if missing_prices else "WARN" if missing_supplies else "PASS"
    return {
        "status": status,
        "missing_prices": missing_prices,
        "missing_supplies": missing_supplies,
    }


def evaluate_raw_retention(current_date: str, raw_stats: dict[str, str | None]) -> dict:
    issues = []
    today = dt.date.fromisoformat(current_date)
    for table, latest_oldest_date in raw_stats.items():
        retention_days = RETENTION_POLICIES.get(table)
        if retention_days is None or not latest_oldest_date:
            continue
        oldest = dt.date.fromisoformat(str(latest_oldest_date))
        age = (today - oldest).days
        if age > retention_days:
            issues.append({"table": table, "age_days": age, "retention_days": retention_days})
    return {
        "status": "WARN" if issues else "PASS",
        "issues": issues,
    }


def classify_overall_status(results: Iterable[dict]) -> str:
    statuses = [str(result.get("status")) for result in results]
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def _numeric(value) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _invert_rank(rank) -> float:
    try:
        return -float(rank)
    except (TypeError, ValueError):
        return 0.0
