from __future__ import annotations

import datetime as dt
from typing import Any

from src.data.supabase_reader import SupabaseReader
from src.utils.market_assets import canonicalize_symbol, normalize_market_label
from src.utils.report_universe import load_report_required_etf_universe, load_report_required_stock_universe


NUMERIC_FIELDS_FOR_DELTA = (
    "sp500",
    "nasdaq",
    "sox",
    "vix",
    "usdkrw",
    "dxy",
    "us10y",
    "us3y",
    "kr10y",
    "brent",
    "wti",
    "gold",
    "copper",
    "hy_spread",
    "kospi",
    "kosdaq",
)

BP_FIELDS = {"us10y", "us3y", "kr10y", "hy_spread"}
PERCENT_FIELDS = {"sp500", "nasdaq", "sox", "vix", "usdkrw", "dxy", "brent", "wti", "gold", "copper", "kospi", "kosdaq"}
MAJOR_INDEX_FIELDS = {"sp500", "nasdaq", "sox", "vix"}


class SupabaseStockDataReader:
    def __init__(self, base_reader: SupabaseReader | None = None):
        self.base_reader = base_reader or SupabaseReader()
        self.client = self.base_reader.client

    def get_report_contract_bundle(self, report_type: str = "morning", target_date: str | None = None) -> dict:
        if report_type != "morning":
            raise ValueError("Only morning contract bundle is supported here.")
        freshness = self.get_report_data_freshness(target_date)
        macro = self.get_morning_macro_snapshot(target_date)
        sector_etfs = self.get_sector_etf_signals(target_date)
        watchlist = self.get_watchlist_snapshot(target_date)
        rankings = self.get_market_rankings(target_date)
        fallback_used = bool(
            freshness.get("contract_fallback_used")
            or macro.get("contract_fallback_used")
            or any(row.get("contract_fallback_used") for row in sector_etfs)
            or any(row.get("contract_fallback_used") for row in watchlist)
            or any(row.get("contract_fallback_used") for row in rankings)
        )
        failed_views = []
        if freshness.get("contract_fallback_used"):
            failed_views.append("report_data_freshness_view")
        if macro.get("contract_fallback_used"):
            failed_views.append("report_morning_macro_view")
        if any(row.get("contract_fallback_used") for row in sector_etfs):
            failed_views.append("report_sector_etf_signal_view")
        if any(row.get("contract_fallback_used") for row in watchlist):
            failed_views.append("report_watchlist_snapshot_view")
        if any(row.get("contract_fallback_used") for row in rankings):
            failed_views.append("report_market_ranking_view")
        readiness = self.base_reader.fetch_stockdata_report_readiness(target_date)
        return {
            "freshness": freshness,
            "macro": macro,
            "sector_etfs": sector_etfs,
            "watchlist": watchlist,
            "rankings": rankings,
            "readiness": readiness,
            "contract_fallback_used": fallback_used,
            "contract_failed_views": failed_views,
        }

    def get_report_data_freshness(self, target_date: str | None = None) -> dict:
        normalized_date = self._normalize_date(target_date)
        rows = self._fetch_view_rows("report_data_freshness_view", limit=5, order_column="latest_stock_price_date")
        row = dict(rows[0]) if rows else self._build_freshness_fallback()
        row["target_date"] = normalized_date
        row["contract_fallback_used"] = not bool(rows)
        row["contract_warnings"] = []
        if row["contract_fallback_used"]:
            row["contract_warnings"].append("contract fallback used: report_data_freshness_view unavailable")

        calendar_status = self.base_reader.fetch_market_calendar_status(normalized_date)
        row["xkrx_is_open"] = calendar_status.get("xkrx_is_open")
        row["xnys_is_open"] = calendar_status.get("xnys_is_open")
        row["xkrx_reason"] = calendar_status.get("xkrx_reason")
        row["xnys_reason"] = calendar_status.get("xnys_reason")
        row["report_market_mode"] = calendar_status.get("report_market_mode")

        carry_forward_fields = []
        if row.get("xkrx_is_open") and not row.get("xnys_is_open"):
            carry_forward_fields.extend(["sp500", "nasdaq", "sox", "vix", "dxy", "us10y", "us3y", "brent", "wti"])
        row["carry_forward_fields"] = carry_forward_fields
        row["watchlist_coverage_status"] = row.get("watchlist_coverage_status") or "UNKNOWN"
        row["sector_etf_coverage_status"] = row.get("sector_etf_coverage_status") or "UNKNOWN"
        row["stale_warnings"] = row.get("stale_warnings") or ""
        row["missing_required_data"] = row.get("missing_required_data") or ""
        return row

    def get_morning_macro_snapshot(self, target_date: str | None = None) -> dict:
        rows = self._fetch_view_rows("report_morning_macro_view", limit=2, order_column="base_date")
        macro = dict(rows[0]) if rows else self._fetch_latest_row("normalized_global_macro_daily")
        macro["contract_fallback_used"] = not bool(rows)
        warnings = []
        if macro["contract_fallback_used"]:
            warnings.append("contract fallback used: report_morning_macro_view unavailable")

        if not macro:
            macro = {"base_date": None}
        previous = self._fetch_previous_macro_row(macro.get("base_date"))
        self._inject_deltas(macro, previous, warnings)
        breadth = self._fetch_latest_row("market_breadth_daily")
        macro["breadth"] = breadth
        advances = self._to_float(breadth.get("advances"))
        declines = self._to_float(breadth.get("declines"))
        if advances is not None and declines is not None and (advances + declines) > 0:
            macro["advancing_ratio"] = advances / (advances + declines)
        else:
            macro["advancing_ratio"] = None
            warnings.append("market breadth unavailable")
        macro["warnings"] = warnings
        return macro

    def get_sector_etf_signals(self, target_date: str | None = None) -> list[dict]:
        rows = self._fetch_view_rows("report_sector_etf_signal_view", limit=500, order_column="latest_price_date")
        if rows:
            normalized = [
                self._normalize_sector_etf_row(row, contract_fallback_used=False, target_date=target_date)
                for row in rows
            ]
            return normalized
        return [self._normalize_sector_etf_row(row, contract_fallback_used=True) for row in self._build_sector_etf_fallback(target_date)]

    def get_watchlist_snapshot(self, target_date: str | None = None) -> list[dict]:
        view_rows = self._fetch_view_rows("report_watchlist_snapshot_view", limit=2000, order_column="base_date")
        contract_fallback_used = not bool(view_rows)
        normalized_rows = [
            self._normalize_watchlist_row(row, contract_fallback_used=contract_fallback_used, target_date=target_date)
            for row in view_rows
        ]
        if contract_fallback_used:
            normalized_rows = [self._normalize_watchlist_row(row, contract_fallback_used=True) for row in self._build_watchlist_fallback(target_date)]
        required = load_report_required_stock_universe()
        by_symbol = {row["symbol"]: row for row in normalized_rows}
        for required_row in required:
            symbol = required_row["symbol"]
            if symbol not in by_symbol:
                by_symbol[symbol] = {
                    "symbol": symbol,
                    "name": required_row.get("name") or symbol,
                    "market": required_row.get("market"),
                    "sector_group": required_row.get("sector_group"),
                    "close_price": None,
                    "change_rate_1d": None,
                    "return_5d": None,
                    "return_20d": None,
                    "return_60d": None,
                    "trading_value": None,
                    "trading_value_ratio_20d": None,
                    "foreign_net_buy": None,
                    "institutional_net_buy": None,
                    "individual_net_buy": None,
                    "foreign_holding_ratio": None,
                    "short_ratio": None,
                    "short_value": None,
                    "per": None,
                    "pbr": None,
                    "roe": None,
                    "debt_ratio": None,
                    "data_status": "DATA_MISSING",
                    "contract_fallback_used": contract_fallback_used,
                    "warnings": ["watchlist row missing from report_watchlist_snapshot_view"],
                }
        return [by_symbol[symbol] for symbol in sorted(by_symbol)]

    def get_market_rankings(self, target_date: str | None = None) -> list[dict]:
        rows = self._fetch_view_rows("report_market_ranking_view", limit=1000, order_column="base_date")
        if rows:
            return [
                self._normalize_ranking_row(row, contract_fallback_used=False, target_date=target_date)
                for row in rows
            ]
        latest_date = self._get_latest_base_date("normalized_market_rankings_daily")
        
        # [NEW] Apply source filtering rules for fallback
        # volume: source='KIS', trading_value/market_cap: source in ('KRX', 'VALID_PRICE_FALLBACK')
        all_fallback_rows = self._fetch_rows(
            "normalized_market_rankings_daily",
            columns="*",
            limit=2000,
            filters=[("eq", "base_date", latest_date)] if latest_date else [],
        )
        
        filtered_rows = []
        for row in all_fallback_rows:
            rt = row.get("rank_type")
            src = row.get("source")
            if rt == "volume":
                if src == "KIS":
                    filtered_rows.append(row)
            elif rt in ("trading_value", "market_cap"):
                if src in ("KRX", "VALID_PRICE_FALLBACK"):
                    filtered_rows.append(row)
            else:
                filtered_rows.append(row)
                
        return [self._normalize_ranking_row(row, contract_fallback_used=True, target_date=target_date) for row in filtered_rows]

    def fetch_report_contract_bundle(self) -> dict:
        return self.get_report_contract_bundle("morning")

    def _normalize_date(self, value: str | None) -> str:
        if value:
            return str(value)
        return dt.date.today().isoformat()

    def _fetch_view_rows(self, view_name: str, limit: int = 1000, order_column: str | None = None) -> list[dict]:
        try:
            query = self.client.table(view_name).select("*")
            if order_column:
                query = query.order(order_column, desc=True)
            response = query.limit(limit).execute()
            return response.data or []
        except Exception:
            return []

    def _fetch_rows(self, table_name: str, columns: str = "*", limit: int = 1000, filters: list[tuple[str, str, Any]] | None = None) -> list[dict]:
        try:
            query = self.client.table(table_name).select(columns)
            for op, column, value in filters or []:
                query = getattr(query, op)(column, value)
            response = query.limit(limit).execute()
            return response.data or []
        except Exception:
            return []

    def _fetch_latest_row(self, table_name: str) -> dict:
        try:
            response = (
                self.client.table(table_name)
                .select("*")
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            return (response.data or [{}])[0]
        except Exception:
            return {}

    def _fetch_previous_macro_row(self, current_base_date: str | None) -> dict:
        if not current_base_date:
            return {}
        try:
            response = (
                self.client.table("normalized_global_macro_daily")
                .select("*")
                .lt("base_date", current_base_date)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            return (response.data or [{}])[0]
        except Exception:
            return {}

    def _get_latest_base_date(self, table_name: str) -> str | None:
        row = self._fetch_latest_row(table_name)
        return row.get("base_date")

    def _inject_deltas(self, current: dict, previous: dict, warnings: list[str] | None = None):
        warnings = warnings if warnings is not None else []
        for field in NUMERIC_FIELDS_FOR_DELTA:
            current_value = self._to_float(current.get(field))
            previous_value = self._to_float(previous.get(field))
            if current_value is None or previous_value is None:
                current[f"{field}_change_value"] = None
                if field in PERCENT_FIELDS:
                    current[f"{field}_change_rate"] = self._normalize_percent_change_rate(
                        field=field,
                        explicit_rate=current.get(f"{field}_change_rate"),
                        change_value=None,
                        previous_value=previous_value,
                        warnings=warnings,
                    )
                if field in BP_FIELDS:
                    current[f"{field}_change_bp"] = None
                continue
            change_value = current_value - previous_value
            current[f"{field}_change_value"] = change_value
            if field in BP_FIELDS:
                current[f"{field}_change_bp"] = change_value * 100
            else:
                current[f"{field}_change_rate"] = self._normalize_percent_change_rate(
                    field=field,
                    explicit_rate=current.get(f"{field}_change_rate"),
                    change_value=change_value,
                    previous_value=previous_value,
                    warnings=warnings,
                )
        if current.get("us10y") is not None and current.get("us3y") is not None:
            current["us10y_us3y_spread"] = self._to_float(current.get("us10y")) - self._to_float(current.get("us3y"))
            prev_spread = None
            if previous.get("us10y") is not None and previous.get("us3y") is not None:
                prev_spread = self._to_float(previous.get("us10y")) - self._to_float(previous.get("us3y"))
            current["us10y_us3y_spread_change_bp"] = (
                (current["us10y_us3y_spread"] - prev_spread) * 100 if prev_spread is not None else None
            )

    def _normalize_percent_change_rate(
        self,
        field: str,
        explicit_rate,
        change_value: float | None,
        previous_value: float | None,
        warnings: list[str],
    ) -> float | None:
        recalculated = None
        if change_value is not None and previous_value not in (None, 0):
            recalculated = change_value / previous_value
        if recalculated is not None:
            if field in MAJOR_INDEX_FIELDS and abs(recalculated) > 0.10:
                warnings.append(f"{field} change rate recalculated above 10%")
            return recalculated

        numeric = self._to_float(explicit_rate)
        if numeric is None:
            return None

        normalized = numeric / 100 if abs(numeric) > 0.2 else numeric
        if field in MAJOR_INDEX_FIELDS and abs(normalized) > 0.10:
            warnings.append(f"{field} change rate anomaly")
            return None
        return normalized

    def _build_freshness_fallback(self) -> dict:
        return {
            "latest_stock_price_date": self._get_latest_base_date("normalized_stock_prices_daily"),
            "latest_ranking_date": self._get_latest_base_date("normalized_market_rankings_daily"),
            "latest_macro_date": self._get_latest_base_date("normalized_global_macro_daily"),
            "latest_supply_date": self._get_latest_base_date("normalized_stock_supply_daily"),
            "latest_short_selling_date": self._get_latest_base_date("normalized_stock_short_selling"),
            "latest_breadth_date": self._get_latest_base_date("market_breadth_daily"),
            "sector_etf_coverage_status": "UNKNOWN",
            "watchlist_coverage_status": "UNKNOWN",
            "stale_warnings": "",
            "missing_required_data": "",
        }

    def _normalize_sector_etf_row(self, row: dict, contract_fallback_used: bool, target_date: str | None = None) -> dict:
        normalized = dict(row)
        normalized["symbol"] = canonicalize_symbol(row.get("symbol"))
        normalized["market"] = "ETF"
        normalized["contract_fallback_used"] = contract_fallback_used
        normalized["warnings"] = list(row.get("warnings") or []) if isinstance(row.get("warnings"), list) else []
        stale_days = self._resolve_stale_days(row.get("latest_price_date"), row.get("target_date"), row.get("stale_days"), target_date)
        normalized["stale_days"] = stale_days
        status = str(row.get("data_status") or "NO_DATA").upper()
        if status == "FRESH":
            normalized["data_status"] = "FRESH"
        elif stale_days is not None and stale_days <= 3:
            normalized["data_status"] = "STALE_BUT_USABLE"
        elif stale_days is not None:
            normalized["data_status"] = "STALE"
        else:
            normalized["data_status"] = "NO_DATA"
        return_20d = self._to_float(row.get("return_20d"))
        if return_20d is not None and return_20d >= 0.30:
            normalized["warnings"].append("OVERHEATED_20D")
        return normalized

    def _normalize_watchlist_row(self, row: dict, contract_fallback_used: bool, target_date: str | None = None) -> dict:
        normalized = {
            **row,
            "symbol": canonicalize_symbol(row.get("symbol")),
            "market": normalize_market_label(row.get("market")),
            "contract_fallback_used": contract_fallback_used,
            "warnings": list(row.get("warnings") or []) if isinstance(row.get("warnings"), list) else [],
        }
        base_date = normalized.get("base_date")
        normalized["data_status"] = self._status_for_row(base_date, normalized.get("data_status"), target_date, missing_label="DATA_MISSING")
        return normalized

    def _normalize_ranking_row(self, row: dict, contract_fallback_used: bool, target_date: str | None = None) -> dict:
        normalized = {
            **row,
            "symbol": canonicalize_symbol(row.get("symbol")),
            "market": normalize_market_label(row.get("market")),
            "contract_fallback_used": contract_fallback_used,
        }
        base_date = normalized.get("base_date")
        normalized["data_status"] = self._status_for_row(base_date, normalized.get("data_status"), target_date, stale_but_usable_days=None)
        return normalized

    def _build_sector_etf_fallback(self, target_date: str | None = None) -> list[dict]:
        target = self._normalize_date(target_date)
        required_rows = load_report_required_etf_universe()
        symbols = [row["symbol"] for row in required_rows if row.get("is_active", True)]
        price_history = self._fetch_rows_for_symbols(
            "normalized_stock_prices_daily",
            "symbol, base_date, close_price, volume, trading_value",
            symbols,
            limit=20000,
        )
        price_metrics = self._build_price_metrics_map(price_history)
        supply_map = self._pick_latest_by_symbol(
            self._fetch_rows_for_symbols(
                "normalized_stock_supply_daily",
                "symbol, base_date, foreign_net_buy, institutional_net_buy, individual_net_buy, foreign_holding_ratio",
                symbols,
                limit=5000,
            )
        )

        fallback_rows = []
        for item in required_rows:
            symbol = item["symbol"]
            metric = price_metrics.get(symbol, {})
            supply = supply_map.get(symbol, {})
            stale_days = self._calc_stale_days(metric.get("base_date"), target)
            fallback_rows.append(
                {
                    "symbol": symbol,
                    "name": item.get("name") or symbol,
                    "sector_group": item.get("sector_group"),
                    "theme_group": item.get("theme_group"),
                    "role": item.get("role") or "fallback",
                    "exclude_from_signal": item.get("exclude_from_signal", False),
                    "latest_price_date": metric.get("base_date"),
                    "target_date": target,
                    "stale_days": stale_days,
                    "data_status": self._stale_status(metric.get("base_date"), target),
                    "close_price": metric.get("close_price"),
                    "volume": metric.get("volume"),
                    "trading_value": metric.get("trading_value"),
                    "change_rate_1d": metric.get("change_rate_1d"),
                    "return_5d": metric.get("return_5d"),
                    "return_20d": metric.get("return_20d"),
                    "return_60d": metric.get("return_60d"),
                    "trading_value_20d_avg": metric.get("trading_value_20d_avg"),
                    "trading_value_ratio_20d": metric.get("trading_value_ratio_20d"),
                    "high_52w": metric.get("high_52w"),
                    "near_52w_high_pct": metric.get("near_52w_high_pct"),
                    "foreign_net_buy": supply.get("foreign_net_buy"),
                    "institutional_net_buy": supply.get("institutional_net_buy"),
                    "individual_net_buy": supply.get("individual_net_buy"),
                    "foreign_holding_ratio": supply.get("foreign_holding_ratio"),
                    "warnings": ["contract fallback used: report_sector_etf_signal_view unavailable"],
                }
            )
        return fallback_rows

    def _build_watchlist_fallback(self, target_date: str | None = None) -> list[dict]:
        target = self._normalize_date(target_date)
        required_rows = load_report_required_stock_universe()
        symbols = [row["symbol"] for row in required_rows if row.get("is_active", True)]
        price_history = self._fetch_rows_for_symbols(
            "normalized_stock_prices_daily",
            "symbol, base_date, close_price, volume, trading_value",
            symbols,
            limit=20000,
        )
        price_metrics = self._build_price_metrics_map(price_history)
        supply_map = self._pick_latest_by_symbol(
            self._fetch_rows_for_symbols(
                "normalized_stock_supply_daily",
                "symbol, base_date, foreign_net_buy, institutional_net_buy, individual_net_buy, foreign_holding_ratio",
                symbols,
                limit=5000,
            )
        )
        short_map = self._pick_latest_by_symbol(
            self._fetch_rows_for_symbols(
                "normalized_stock_short_selling",
                "symbol, base_date, short_ratio, short_value",
                symbols,
                limit=5000,
            )
        )
        fundamentals_map = self._pick_latest_by_symbol(
            self._fetch_rows_for_symbols(
                "normalized_stock_fundamentals_ratios",
                "symbol, base_date, per, pbr, roe, debt_ratio",
                symbols,
                limit=5000,
            )
        )

        fallback_rows = []
        for item in required_rows:
            symbol = item["symbol"]
            price = price_metrics.get(symbol, {})
            supply = supply_map.get(symbol, {})
            short = short_map.get(symbol, {})
            fundamentals = fundamentals_map.get(symbol, {})
            fallback_rows.append(
                {
                    "symbol": symbol,
                    "name": item.get("name") or symbol,
                    "market": item.get("market"),
                    "sector_group": item.get("sector_group"),
                    "base_date": price.get("base_date"),
                    "close_price": price.get("close_price"),
                    "change_rate_1d": price.get("change_rate_1d"),
                    "return_5d": price.get("return_5d"),
                    "return_20d": price.get("return_20d"),
                    "return_60d": price.get("return_60d"),
                    "trading_value": price.get("trading_value"),
                    "trading_value_ratio_20d": price.get("trading_value_ratio_20d"),
                    "foreign_net_buy": supply.get("foreign_net_buy"),
                    "institutional_net_buy": supply.get("institutional_net_buy"),
                    "individual_net_buy": supply.get("individual_net_buy"),
                    "foreign_holding_ratio": supply.get("foreign_holding_ratio"),
                    "short_ratio": short.get("short_ratio"),
                    "short_value": short.get("short_value"),
                    "per": fundamentals.get("per"),
                    "pbr": fundamentals.get("pbr"),
                    "roe": fundamentals.get("roe"),
                    "debt_ratio": fundamentals.get("debt_ratio"),
                    "data_status": self._stale_status(price.get("base_date"), target, missing_label="DATA_MISSING"),
                    "warnings": ["contract fallback used: report_watchlist_snapshot_view unavailable"],
                }
            )
        return fallback_rows

    def _fetch_rows_for_symbols(self, table_name: str, columns: str, symbols: list[str], limit: int = 10000) -> list[dict]:
        if not symbols:
            return []
        try:
            response = (
                self.client.table(table_name)
                .select(columns)
                .in_("symbol", symbols)
                .order("base_date", desc=True)
                .limit(limit)
                .execute()
            )
            return response.data or []
        except Exception:
            return []

    def _pick_latest_by_symbol(self, rows: list[dict]) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for row in rows:
            symbol = canonicalize_symbol(row.get("symbol"))
            if not symbol or symbol in latest:
                continue
            latest[symbol] = row
        return latest

    def _build_price_metrics_map(self, rows: list[dict]) -> dict[str, dict]:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            symbol = canonicalize_symbol(row.get("symbol"))
            if not symbol:
                continue
            grouped.setdefault(symbol, []).append(row)

        metrics: dict[str, dict] = {}
        for symbol, symbol_rows in grouped.items():
            ordered = sorted(symbol_rows, key=lambda item: str(item.get("base_date") or ""), reverse=True)
            latest = ordered[0]
            close_price = self._to_float(latest.get("close_price"))
            prev_close = self._to_float(ordered[1].get("close_price")) if len(ordered) > 1 else None
            close_5d_ago = self._to_float(ordered[5].get("close_price")) if len(ordered) > 5 else None
            close_20d_ago = self._to_float(ordered[20].get("close_price")) if len(ordered) > 20 else None
            close_60d_ago = self._to_float(ordered[60].get("close_price")) if len(ordered) > 60 else None
            trading_value = self._to_float(latest.get("trading_value"))
            trading_values = [self._to_float(row.get("trading_value")) for row in ordered[:20]]
            trading_values = [value for value in trading_values if value is not None]
            trading_value_20d_avg = (sum(trading_values) / len(trading_values)) if trading_values else None
            high_52w = max((self._to_float(row.get("close_price")) or 0.0) for row in ordered[:252]) if ordered else None
            metrics[symbol] = {
                "base_date": latest.get("base_date"),
                "close_price": latest.get("close_price"),
                "volume": latest.get("volume"),
                "trading_value": latest.get("trading_value"),
                "change_rate_1d": ((close_price / prev_close) - 1) if close_price is not None and prev_close not in (None, 0) else None,
                "return_5d": ((close_price / close_5d_ago) - 1) if close_price is not None and close_5d_ago not in (None, 0) else None,
                "return_20d": ((close_price / close_20d_ago) - 1) if close_price is not None and close_20d_ago not in (None, 0) else None,
                "return_60d": ((close_price / close_60d_ago) - 1) if close_price is not None and close_60d_ago not in (None, 0) else None,
                "trading_value_20d_avg": trading_value_20d_avg,
                "trading_value_ratio_20d": (trading_value / trading_value_20d_avg) if trading_value is not None and trading_value_20d_avg not in (None, 0) else None,
                "high_52w": high_52w,
                "near_52w_high_pct": ((close_price / high_52w) * 100) if close_price is not None and high_52w not in (None, 0) else None,
            }
        return metrics

    def _stale_status(self, base_date: str | None, target_date: str, missing_label: str = "NO_DATA") -> str:
        stale_days = self._calc_stale_days(base_date, target_date)
        if stale_days is None:
            return missing_label
        if stale_days >= 4:
            return "STALE"
        if stale_days >= 1:
            return "STALE_BUT_USABLE"
        return "FRESH"

    def _calc_stale_days(self, base_date: str | None, target_date: str | None) -> int | None:
        if not base_date or not target_date:
            return None
        try:
            return (dt.date.fromisoformat(str(target_date)) - dt.date.fromisoformat(str(base_date))).days
        except ValueError:
            return None

    def _resolve_stale_days(self, latest_price_date, row_target_date, row_stale_days, override_target_date: str | None) -> int | None:
        effective_target_date = override_target_date or row_target_date
        recalculated = self._calc_stale_days(latest_price_date, effective_target_date)
        if recalculated is not None:
            return recalculated
        try:
            if row_stale_days is None:
                return None
            return int(row_stale_days)
        except (TypeError, ValueError):
            return None

    def _status_for_row(
        self,
        base_date,
        current_status,
        target_date: str | None,
        missing_label: str = "NO_DATA",
        stale_but_usable_days: int | None = 3,
    ) -> str:
        stale_days = self._calc_stale_days(base_date, self._normalize_date(target_date) if target_date else None)
        if stale_days is None:
            return str(current_status or missing_label).upper()
        if stale_days >= 4:
            return "STALE"
        if stale_but_usable_days is not None and stale_days >= 1:
            return "STALE_BUT_USABLE"
        return "FRESH"

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
