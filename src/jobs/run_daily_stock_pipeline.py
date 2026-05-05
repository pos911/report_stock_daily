"""
Daily stock detail pipeline for report-required coverage.

Policy changes:
- Do not depend on config/target_stocks.json.
- Build the default detail universe from static_stock_universe,
  report_required_stock_universe, report_required_etf_universe, and
  latest market rankings.
- Keep XKRX calendar guardrails and skip writes on market holidays.
- Reuse normalized_stock_prices_daily / supply / short tables.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

import requests
from supabase import Client, create_client

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils import config
from src.utils.report_universe import (
    DEFAULT_DETAIL_LIMIT,
    MAX_DETAIL_LIMIT,
    load_report_required_etf_universe,
    load_report_required_stock_universe,
    prioritize_detail_targets,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"
REQUEST_DELAY = 0.4
BATCH_UPSERT_SIZE = 500


class KISClient:
    def __init__(self):
        self.app_key = config.get("app_key", section="kis")
        self.app_secret = config.get("app_secret", section="kis")
        self._token: str | None = None
        self._token_expires: datetime.datetime | None = None

    def _get_token(self) -> str:
        now = datetime.datetime.now()
        if self._token and self._token_expires and now < self._token_expires:
            return self._token

        response = requests.post(
            f"{KIS_BASE_URL}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        self._token = payload["access_token"]
        expires_in = int(payload.get("expires_in", 86400))
        self._token_expires = now + datetime.timedelta(seconds=expires_in - 60)
        return self._token

    def _headers(self, tr_id: str) -> dict:
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self._get_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
        }

    def fetch_price_history(self, symbol: str, start_date: str, end_date: str, market: str = "J") -> list[dict]:
        response = requests.get(
            f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            headers=self._headers("FHKST03010100"),
            params={
                "FID_COND_MRKT_DIV_CODE": market,
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start_date,
                "FID_INPUT_DATE_2": end_date,
                "FID_PERIOD_DIV_CODE": "D",
                "FID_ORG_ADJ_PRC": "0",
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("output2", []) if payload.get("rt_cd") == "0" else []

    def fetch_supply_history(self, symbol: str, start_date: str, end_date: str) -> list[dict]:
        response = requests.get(
            f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-investor",
            headers=self._headers("FHKST01010900"),
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start_date,
                "FID_INPUT_DATE_2": end_date,
                "FID_PERIOD_DIV_CODE": "D",
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("output", []) if payload.get("rt_cd") == "0" else []

    def fetch_short_selling(self, symbol: str, start_date: str, end_date: str) -> list[dict]:
        response = requests.get(
            f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-short-sale",
            headers=self._headers("FHPST04560000"),
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start_date,
                "FID_INPUT_DATE_2": end_date,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("output", []) if payload.get("rt_cd") == "0" else []

    def fetch_stock_info(self, symbol: str) -> dict | None:
        response = requests.get(
            f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/search-stock-info",
            headers=self._headers("CTPF1002R"),
            params={"PRDT_TYPE_CD": "300", "PDNO": symbol},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("output") if payload.get("rt_cd") == "0" else None

    def fetch_fundamentals(self, symbol: str) -> dict | None:
        response = requests.get(
            f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price",
            headers=self._headers("FHKST01010100"),
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("output") if payload.get("rt_cd") == "0" else None


class SupabaseUpserter:
    def __init__(self):
        self.client: Client = create_client(
            config.get("url", section="supabase"),
            config.get("service_role_key", section="supabase"),
        )

    def upsert(self, table: str, rows: list[dict], conflict_cols: list[str]) -> int:
        if not rows:
            return 0
        inserted = 0
        for index in range(0, len(rows), BATCH_UPSERT_SIZE):
            batch = rows[index : index + BATCH_UPSERT_SIZE]
            self.client.table(table).upsert(batch, on_conflict=",".join(conflict_cols)).execute()
            inserted += len(batch)
        return inserted

    def get_latest_date(self, table: str, symbol: str) -> str | None:
        try:
            response = (
                self.client.table(table)
                .select("base_date")
                .eq("symbol", symbol)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            return response.data[0]["base_date"] if response.data else None
        except Exception:
            return None

    def fetch_enabled_static_universe(self) -> list[dict]:
        try:
            response = (
                self.client.table("static_stock_universe")
                .select("symbol, name, market, enabled")
                .eq("enabled", True)
                .execute()
            )
            return response.data or []
        except Exception as exc:
            logger.warning("static_stock_universe fetch failed: %s", exc)
            return []

    def fetch_latest_rankings(self, limit_per_bucket: int = 50) -> list[dict]:
        try:
            latest = (
                self.client.table("normalized_market_rankings_daily")
                .select("base_date")
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            latest_date = latest.data[0]["base_date"] if latest.data else None
            if not latest_date:
                return []
            response = (
                self.client.table("normalized_market_rankings_daily")
                .select("symbol, name, market, rank_type, rank, trading_value, volume, market_cap, base_date, source")
                .eq("base_date", latest_date)
                .in_("rank_type", ["volume", "trading_value", "market_cap"])
                .lte("rank", limit_per_bucket)
                .execute()
            )
            return response.data or []
        except Exception as exc:
            logger.warning("normalized_market_rankings_daily fetch failed: %s", exc)
            return []

    def fetch_symbol_metadata(self, symbols: list[str]) -> dict[str, dict]:
        if not symbols:
            return {}
        try:
            response = (
                self.client.table("stocks_master")
                .select("symbol, name, market, is_active")
                .in_("symbol", symbols)
                .execute()
            )
            return {row["symbol"]: row for row in (response.data or []) if row.get("symbol")}
        except Exception as exc:
            logger.warning("stocks_master metadata fetch failed: %s", exc)
            return {}

    def is_xkrx_open(self, target_date: str) -> bool:
        date_columns = ("calendar_date", "trade_date", "base_date", "date")
        bool_columns = ("is_open", "open", "is_trading_day", "trading_day", "is_business_day")
        for date_col in date_columns:
            try:
                response = (
                    self.client.table("market_trading_calendar")
                    .select("*")
                    .eq("exchange_code", "XKRX")
                    .eq(date_col, target_date)
                    .limit(1)
                    .execute()
                )
            except Exception:
                continue
            row = (response.data or [None])[0]
            if not row:
                continue
            for bool_col in bool_columns:
                if bool_col in row:
                    return bool(row.get(bool_col))
            status_text = str(row.get("status") or row.get("session_status") or "").lower()
            if "open" in status_text:
                return True
            if any(token in status_text for token in ("closed", "holiday", "skip")):
                return False
        return datetime.date.fromisoformat(target_date).weekday() < 5

    def upsert_master(self, symbol: str, name: str, market: str, listed_date: str | None):
        row = {"symbol": symbol, "name": name, "market": market, "is_active": True}
        if listed_date:
            row["listed_date"] = listed_date
        self.client.table("stocks_master").upsert(row, on_conflict="symbol").execute()


def _safe_float(value) -> float | None:
    try:
        text = str(value).replace(",", "").strip()
        return float(text) if text else None
    except Exception:
        return None


def _fmt_date(raw: str) -> str | None:
    raw = str(raw).strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return None


def _date_range_chunks(start: str, end: str, chunk_days: int = 90) -> list[tuple[str, str]]:
    start_date = datetime.date.fromisoformat(start)
    end_date = datetime.date.fromisoformat(end)
    chunks = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + datetime.timedelta(days=chunk_days - 1), end_date)
        chunks.append((current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        current = chunk_end + datetime.timedelta(days=1)
    return chunks


def run_symbol(
    symbol: str,
    name: str,
    market: str,
    start_date: str,
    end_date: str,
    kis: KISClient,
    supa: SupabaseUpserter,
    skip_master: bool = False,
):
    logger.info("[%s] collect %s (%s -> %s)", symbol, name, start_date, end_date)

    if not skip_master:
        info = kis.fetch_stock_info(symbol)
        listed_date = _fmt_date((info or {}).get("lstg_stcn_dt", ""))
        try:
            supa.upsert_master(symbol, name, market, listed_date)
        except Exception as exc:
            logger.warning("[%s] stocks_master upsert failed: %s", symbol, exc)
        time.sleep(REQUEST_DELAY)

    price_rows = []
    for start_chunk, end_chunk in _date_range_chunks(start_date, end_date):
        try:
            rows = kis.fetch_price_history(symbol, start_chunk, end_chunk)
        except Exception as exc:
            logger.warning("[%s] price fetch failed: %s", symbol, exc)
            rows = []
        for row in rows:
            base_date = _fmt_date(row.get("stck_bsop_date", ""))
            if not base_date:
                continue
            price_rows.append(
                {
                    "symbol": symbol,
                    "base_date": base_date,
                    "open": _safe_float(row.get("stck_oprc")),
                    "high": _safe_float(row.get("stck_hgpr")),
                    "low": _safe_float(row.get("stck_lwpr")),
                    "close": _safe_float(row.get("stck_clpr")),
                    "volume": _safe_float(row.get("acml_vol")),
                    "amount": _safe_float(row.get("acml_tr_pbmn")),
                }
            )
        time.sleep(REQUEST_DELAY)
    supa.upsert("normalized_stock_prices_daily", price_rows, ["symbol", "base_date"])

    supply_rows = []
    for start_chunk, end_chunk in _date_range_chunks(start_date, end_date):
        try:
            rows = kis.fetch_supply_history(symbol, start_chunk, end_chunk)
        except Exception as exc:
            logger.warning("[%s] supply fetch failed: %s", symbol, exc)
            rows = []
        for row in rows:
            base_date = _fmt_date(row.get("stck_bsop_date", ""))
            if not base_date:
                continue
            supply_rows.append(
                {
                    "symbol": symbol,
                    "base_date": base_date,
                    "individual_net_buy": _safe_float(row.get("prsn_ntby_qty")),
                    "foreign_net_buy": _safe_float(row.get("frgn_ntby_qty")),
                    "institutional_net_buy": _safe_float(row.get("orgn_ntby_qty")),
                    "pension_net_buy": _safe_float(row.get("pnsn_ntby_qty")),
                    "corporate_net_buy": _safe_float(row.get("corp_ntby_qty")),
                }
            )
        time.sleep(REQUEST_DELAY)
    supa.upsert("normalized_stock_supply_daily", supply_rows, ["symbol", "base_date"])

    short_rows = []
    for start_chunk, end_chunk in _date_range_chunks(start_date, end_date):
        try:
            rows = kis.fetch_short_selling(symbol, start_chunk, end_chunk)
        except Exception as exc:
            logger.warning("[%s] short selling fetch failed: %s", symbol, exc)
            rows = []
        for row in rows:
            base_date = _fmt_date(row.get("stck_bsop_date", ""))
            if not base_date:
                continue
            short_rows.append(
                {
                    "symbol": symbol,
                    "base_date": base_date,
                    "short_sell_volume": _safe_float(row.get("smtn_slby_qty")),
                    "short_sell_amount": _safe_float(row.get("smtn_slby_tr_pbmn")),
                    "short_sell_ratio": _safe_float(row.get("slby_tr_pbmn_smtn_rate")),
                }
            )
        time.sleep(REQUEST_DELAY)
    supa.upsert("normalized_stock_short_selling", short_rows, ["symbol", "base_date"])

    try:
        fundamentals = kis.fetch_fundamentals(symbol)
    except Exception as exc:
        logger.warning("[%s] fundamentals fetch failed: %s", symbol, exc)
        fundamentals = None
    if fundamentals:
        supa.upsert(
            "normalized_stock_fundamentals_ratios",
            [
                {
                    "symbol": symbol,
                    "base_date": end_date,
                    "per": _safe_float(fundamentals.get("per")),
                    "pbr": _safe_float(fundamentals.get("pbr")),
                    "market_cap": _safe_float(fundamentals.get("hts_avls")),
                }
            ],
            ["symbol", "base_date"],
        )
    time.sleep(REQUEST_DELAY)


def _parse_args():
    parser = argparse.ArgumentParser(description="Daily Stock Pipeline")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols. If omitted, build the report-required detail universe.",
    )
    parser.add_argument("--start-date", default=None, help="Collection start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Collection end date in YYYY-MM-DD.")
    parser.add_argument("--detail-limit", type=int, default=DEFAULT_DETAIL_LIMIT)
    parser.add_argument("--max-detail-limit", type=int, default=MAX_DETAIL_LIMIT)
    return parser.parse_args()


def _load_universe(symbols_arg: str | None, supa: SupabaseUpserter, detail_limit: int, max_detail_limit: int) -> list[dict]:
    if symbols_arg:
        codes = {symbol.strip() for symbol in symbols_arg.split(",") if symbol.strip()}
        metadata = supa.fetch_symbol_metadata(list(codes))
        return [metadata.get(code) or {"symbol": code, "name": code, "market": "KOSPI"} for code in codes]

    return prioritize_detail_targets(
        static_rows=supa.fetch_enabled_static_universe(),
        report_stock_rows=load_report_required_stock_universe(project_root),
        report_etf_rows=load_report_required_etf_universe(project_root),
        ranking_rows=supa.fetch_latest_rankings(limit_per_bucket=50),
        detail_limit=detail_limit,
        max_limit=max_detail_limit,
    )


def main():
    args = _parse_args()
    now_kst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    end_date = args.end_date or now_kst.strftime("%Y-%m-%d")

    supa = SupabaseUpserter()
    if not supa.is_xkrx_open(end_date):
        logger.info("XKRX is closed on %s. Skip ingestion and do not write carry-forward rows.", end_date)
        return

    universe = _load_universe(args.symbols, supa, args.detail_limit, args.max_detail_limit)
    logger.info("Selected %s symbols for detailed collection.", len(universe))
    logger.info("Universe preview: %s", [row["symbol"] for row in universe[:20]])

    kis = KISClient()
    for stock in universe:
        symbol = stock["symbol"]
        name = stock.get("name", symbol)
        market = stock.get("market", "KOSPI")

        if args.start_date:
            start_date = args.start_date
        else:
            latest = supa.get_latest_date("normalized_stock_prices_daily", symbol)
            start_date = (
                (datetime.date.fromisoformat(latest) + datetime.timedelta(days=1)).isoformat()
                if latest
                else "2020-01-02"
            )

        if start_date > end_date:
            logger.info("[%s] already fresh (start=%s > end=%s).", symbol, start_date, end_date)
            continue

        try:
            run_symbol(
                symbol=symbol,
                name=name,
                market=market,
                start_date=start_date,
                end_date=end_date,
                kis=kis,
                supa=supa,
            )
        except Exception as exc:
            logger.error("[%s] processing error: %s", symbol, exc, exc_info=True)

    logger.info("Daily stock pipeline completed.")


if __name__ == "__main__":
    main()
