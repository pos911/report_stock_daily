import datetime
import re
from collections import defaultdict

from src.data.supabase_reader import ETF_PATTERN, SupabaseReader


ETF_NAME_FALLBACK = re.compile(
    r"(ETF|ACE|KODEX|TIGER|KBSTAR|SOL|HANARO|ARIRANG|KOSEF)",
    re.IGNORECASE,
)


class SupabaseStockDataReader:
    MACRO_COLUMNS = (
        "base_date, sp500, sp500_change_rate, nasdaq, nasdaq_change_rate, "
        "kospi, kospi_change_rate, kosdaq, kosdaq_change_rate, usdkrw, dxy, "
        "us10y, kr10y, sox, vix, wti, brent, gold, copper, bdry, hy_spread, "
        "kospi_individual_net_buy, kospi_foreign_net_buy, kospi_institutional_net_buy, "
        "kosdaq_individual_net_buy, kosdaq_foreign_net_buy, kosdaq_institutional_net_buy, "
        "available_at"
    )
    BREADTH_COLUMNS = (
        "base_date, advances, declines, unchanged, advancing_volume, declining_volume, available_at"
    )
    DERIVATIVE_COLUMNS = (
        "base_date, kospi200_futures, futures_basis, open_interest, night_futures_return, "
        "expiration_flag, available_at"
    )
    STATIC_UNIVERSE_COLUMNS = "symbol, name, market, enabled, source_file, updated_at"
    PRICE_COLUMNS = (
        "symbol, base_date, open_price, high_price, low_price, close_price, volume, "
        "trading_value, market_cap, outstanding_shares, available_at"
    )
    SUPPLY_COLUMNS = (
        "symbol, base_date, individual_net_buy, foreign_net_buy, institutional_net_buy, "
        "pension_net_buy, corporate_net_buy, foreign_holding_ratio, available_at"
    )
    FUNDAMENTAL_COLUMNS = (
        "symbol, base_date, per, pbr, roe, debt_ratio, source, available_at"
    )
    SHORT_COLUMNS = (
        "symbol, base_date, short_volume, short_value, short_ratio, source, available_at"
    )
    EVENT_COLUMNS = (
        "symbol, base_date, event_type, event_score, sentiment_score, available_at"
    )
    FEATURE_COLUMNS = "symbol, base_date, feature_name, feature_value, available_at"
    STOCK_FEATURE_NAMES = [
        "return_5d",
        "moving_avg_5",
        "moving_avg_20",
        "volatility_20d",
        "foreign_flow_zscore",
    ]

    def __init__(self, base_reader: SupabaseReader | None = None):
        self.base_reader = base_reader or SupabaseReader()
        self.client = self.base_reader.client

    def fetch_morning_bundle(self, top_n: int = 5) -> dict:
        macro = self._fetch_latest_row("normalized_global_macro_daily", self.MACRO_COLUMNS)
        breadth = self._fetch_latest_row("market_breadth_daily", self.BREADTH_COLUMNS)
        derivatives = self._fetch_latest_row("normalized_derivatives_daily", self.DERIVATIVE_COLUMNS)
        static_universe = self.fetch_enabled_static_universe()
        latest_price_base_date = self._fetch_latest_base_date("normalized_stock_prices_daily")
        top_volume = self.fetch_top_volume_by_market(latest_price_base_date, top_n=top_n)
        static_snapshots = self.fetch_static_universe_snapshots(static_universe)
        return {
            "macro": macro,
            "breadth": breadth,
            "derivatives": derivatives,
            "static_universe": static_universe,
            "latest_price_base_date": latest_price_base_date,
            "top_volume": top_volume,
            "static_snapshots": static_snapshots,
        }

    def fetch_enabled_static_universe(self) -> list[dict]:
        try:
            response = (
                self.client.table("static_stock_universe")
                .select(self.STATIC_UNIVERSE_COLUMNS)
                .eq("enabled", True)
                .order("symbol")
                .execute()
            )
            return response.data or []
        except Exception as exc:
            print(f"[WARNING] static_stock_universe 조회 실패: {exc}")
            return []

    def fetch_top_volume_by_market(self, latest_price_base_date: str | None, top_n: int = 5) -> dict:
        result = {
            "KOSPI": [],
            "KOSDAQ": [],
            "ETF": [],
            "base_date": latest_price_base_date,
            "unclassified_count": 0,
        }
        if not latest_price_base_date:
            return result

        try:
            price_rows = (
                self.client.table("normalized_stock_prices_daily")
                .select(self.PRICE_COLUMNS)
                .eq("base_date", latest_price_base_date)
                .order("volume", desc=True)
                .limit(5000)
                .execute()
                .data
                or []
            )
        except Exception as exc:
            print(f"[WARNING] 거래량 상위 가격 조회 실패: {exc}")
            return result

        active_master = self._fetch_active_master_map()
        for row in price_rows:
            symbol = row.get("symbol")
            volume = row.get("volume")
            if not symbol or volume is None:
                continue
            master = active_master.get(symbol)
            if not master:
                continue
            market_bucket = self._classify_market(master.get("market"), master.get("name", ""))
            if market_bucket not in ("KOSPI", "KOSDAQ", "ETF"):
                result["unclassified_count"] += 1
                continue
            if len(result[market_bucket]) >= top_n:
                continue
            result[market_bucket].append(
                {
                    **row,
                    "name": master.get("name") or symbol,
                    "market": master.get("market"),
                }
            )
            if all(len(result[k]) >= top_n for k in ("KOSPI", "KOSDAQ", "ETF")):
                break
        return result

    def fetch_static_universe_snapshots(self, static_universe: list[dict]) -> list[dict]:
        symbols = [item["symbol"] for item in static_universe if item.get("symbol")]
        if not symbols:
            return []

        universe_map = {item["symbol"]: item for item in static_universe}
        price_map = self._pick_latest_rows_by_symbol(
            self._fetch_rows_for_symbols("normalized_stock_prices_daily", self.PRICE_COLUMNS, symbols)
        )
        supply_rows = self._fetch_rows_for_symbols("normalized_stock_supply_daily", self.SUPPLY_COLUMNS, symbols)
        supply_map = self._pick_supply_rows(symbols, price_map, supply_rows)
        fundamental_map = self._pick_latest_rows_by_symbol(
            self._fetch_rows_for_symbols("normalized_stock_fundamentals_ratios", self.FUNDAMENTAL_COLUMNS, symbols)
        )
        short_map = self._pick_latest_rows_by_symbol(
            self._fetch_rows_for_symbols("normalized_stock_short_selling", self.SHORT_COLUMNS, symbols)
        )
        event_map = self._pick_latest_rows_by_symbol(
            self._fetch_rows_for_symbols("normalized_stock_events_daily", self.EVENT_COLUMNS, symbols)
        )
        feature_map = self._pivot_latest_features(symbols)

        snapshots = []
        for symbol in symbols:
            universe_row = universe_map.get(symbol, {})
            price_row = price_map.get(symbol, {})
            supply_row = supply_map.get(symbol, {})
            fundamental_row = fundamental_map.get(symbol, {})
            short_row = short_map.get(symbol, {})
            event_row = event_map.get(symbol, {})
            snapshots.append(
                {
                    "symbol": symbol,
                    "name": universe_row.get("name", symbol),
                    "market": universe_row.get("market"),
                    "source_file": universe_row.get("source_file"),
                    "updated_at": universe_row.get("updated_at"),
                    "price": price_row,
                    "supply": supply_row,
                    "fundamentals": fundamental_row,
                    "short_selling": short_row,
                    "event": event_row,
                    "features": feature_map.get(symbol, {}),
                }
            )
        return snapshots

    def _fetch_latest_base_date(self, table_name: str) -> str | None:
        try:
            response = (
                self.client.table(table_name)
                .select("base_date")
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            if response.data:
                return response.data[0].get("base_date")
        except Exception as exc:
            print(f"[WARNING] {table_name} 최신 base_date 조회 실패: {exc}")
        return None

    def _fetch_latest_row(self, table_name: str, columns: str) -> dict:
        try:
            response = (
                self.client.table(table_name)
                .select(columns)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            return (response.data or [{}])[0]
        except Exception as exc:
            print(f"[WARNING] {table_name} 최신 row 조회 실패: {exc}")
            return {}

    def _fetch_rows_for_symbols(self, table_name: str, columns: str, symbols: list[str]) -> list[dict]:
        if not symbols:
            return []
        try:
            response = (
                self.client.table(table_name)
                .select(columns)
                .in_("symbol", symbols)
                .order("base_date", desc=True)
                .limit(5000)
                .execute()
            )
            return response.data or []
        except Exception as exc:
            print(f"[WARNING] {table_name} 심볼 rows 조회 실패: {exc}")
            return []

    def _fetch_active_master_map(self) -> dict:
        try:
            response = (
                self.client.table("stocks_master")
                .select("symbol, name, market, is_active, updated_at")
                .eq("is_active", True)
                .execute()
            )
            return {row["symbol"]: row for row in (response.data or []) if row.get("symbol")}
        except Exception as exc:
            print(f"[WARNING] stocks_master active universe 조회 실패: {exc}")
            return {}

    @staticmethod
    def _classify_market(market_value: str | None, name: str) -> str | None:
        market_text = (market_value or "").upper()
        if "KOSPI" in market_text:
            return "KOSPI"
        if "KOSDAQ" in market_text:
            return "KOSDAQ"
        if "ETF" in market_text:
            return "ETF"
        if ETF_PATTERN.search(name or "") or ETF_NAME_FALLBACK.search(name or ""):
            return "ETF"
        return None

    @staticmethod
    def _pick_latest_rows_by_symbol(rows: list[dict]) -> dict:
        latest_rows = {}
        for row in rows or []:
            symbol = row.get("symbol")
            if not symbol or symbol in latest_rows:
                continue
            latest_rows[symbol] = row
        return latest_rows

    @staticmethod
    def _pick_supply_rows(symbols: list[str], price_map: dict, supply_rows: list[dict]) -> dict:
        rows_by_symbol = defaultdict(list)
        for row in supply_rows or []:
            symbol = row.get("symbol")
            if symbol:
                rows_by_symbol[symbol].append(row)

        selected = {}
        for symbol in symbols:
            rows = rows_by_symbol.get(symbol, [])
            if not rows:
                selected[symbol] = {}
                continue
            price_date = (price_map.get(symbol) or {}).get("base_date")
            exact_date_row = None
            for row in rows:
                if price_date and row.get("base_date") == price_date:
                    exact_date_row = row
                    break
            selected[symbol] = exact_date_row or rows[0]
        return selected

    def _pivot_latest_features(self, symbols: list[str]) -> dict:
        if not symbols:
            return {}
        try:
            response = (
                self.client.table("feature_store_daily")
                .select(self.FEATURE_COLUMNS)
                .in_("symbol", symbols)
                .in_("feature_name", self.STOCK_FEATURE_NAMES)
                .order("base_date", desc=True)
                .limit(10000)
                .execute()
            )
            rows = response.data or []
        except Exception as exc:
            print(f"[WARNING] feature_store_daily 조회 실패: {exc}")
            return {}

        pivoted = defaultdict(dict)
        seen = set()
        for row in rows:
            symbol = row.get("symbol")
            feature_name = row.get("feature_name")
            if not symbol or not feature_name:
                continue
            key = (symbol, feature_name)
            if key in seen:
                continue
            seen.add(key)
            pivoted[symbol][feature_name] = row.get("feature_value")
            pivoted[symbol]["base_date"] = row.get("base_date")
            pivoted[symbol]["available_at"] = row.get("available_at")
        return dict(pivoted)
