import os
import re
import json
import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import requests
import time
from src.utils import config
from src.utils.market_assets import (
    canonicalize_symbol,
    deduplicate_by_canonical_symbol,
    has_minimum_top_data,
    infer_asset_type,
    is_common_stock_top_eligible,
    is_etf_etn_top_eligible,
    is_allowed_ranking_source,
    normalize_market_label,
    ranking_market_matches_master,
)

# NEWS constants moved into class or kept global
NEWS_FETCH_TIMEOUT = 60
NEWS_FETCH_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3
MAX_NEWS_ITEM_CHARS = 150
MAX_NEWS_CONTEXT_CHARS = 3000

# ETF 판별 정규식
ETF_PATTERN = re.compile(r"(KODEX|TIGER|KBSTAR|SOL|ACE|KOSEF|ARIRANG)", re.IGNORECASE)

class SupabaseReader:
    def __init__(self):
        """
        Initialize Supabase client using the unified config loader.
        Priority: 1. Environment Variables, 2. config/api_keys.json
        """
        self.url = config.get("url", section="supabase")
        self.key = config.get("service_role_key", section="supabase") or config.get("supabase_key")

        if not self.url or not self.key:
            raise ValueError("Supabase URL and Key must be provided via Env or JSON.")

        self.client: Client = create_client(self.url, self.key)
        self.page_size = 1000

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _fetch_stock_name_map(self):
        """Fetches all (symbol, name) mappings from stocks_master."""
        try:
            response = self.client.table("stocks_master").select("symbol, name").execute()
            if response.data:
                return {item["symbol"]: item["name"] for item in response.data}
        except Exception as e:
            print(f"[WARNING] _fetch_stock_name_map 실패: {e}")
        return {}

    def _execute_paged_query(self, table_name: str, columns: str, query_mutator=None, batch_size: int | None = None):
        batch = batch_size or self.page_size
        start = 0
        rows = []
        while True:
            query = self.client.table(table_name).select(columns)
            if query_mutator is not None:
                query = query_mutator(query)
            response = query.range(start, start + batch - 1).execute()
            data = response.data or []
            rows.extend(data)
            if len(data) < batch:
                break
            start += batch
        return rows

    def _fetch_and_ffill_timeseries(self, table_name, lookback_days=5, as_of_utc_iso: str = None):
        """
        Fetches the last N days of data from a table, applies forward fill,
        and returns the most recent record as a dictionary.
        """
        try:
            query = (
                self.client.table(table_name)
                .select("*")
                .order("base_date", desc=True)
                .limit(lookback_days)
            )
            if as_of_utc_iso:
                query = query.lte("available_at", as_of_utc_iso)
            response = query.execute()
            if not response.data:
                # available_at 필터 결과가 비었어도 legacy 데이터가 있을 수 있어 fallback 조회
                response = (
                    self.client.table(table_name)
                    .select("*")
                    .order("base_date", desc=True)
                    .limit(lookback_days)
                    .execute()
                )
                if not response.data:
                    return None

            df = pd.DataFrame(response.data).sort_values("base_date")
            df = df.ffill()
            return df.iloc[-1].to_dict()
        except Exception as e:
            # available_at 컬럼/권한 미지원 시 과거 방식으로 fallback
            try:
                response = (
                    self.client.table(table_name)
                    .select("*")
                    .order("base_date", desc=True)
                    .limit(lookback_days)
                    .execute()
                )
                if response.data:
                    df = pd.DataFrame(response.data).sort_values("base_date")
                    df = df.ffill()
                    return df.iloc[-1].to_dict()
            except Exception:
                pass
            print(f"[WARNING] _fetch_and_ffill_timeseries({table_name}) 실패: {e}")
            return None

    def _inject_stock_names(self, data, name_map):
        """
        Injects 'stock_name' into each item in data list using name_map.
        """
        if not data:
            return []
        for item in data:
            symbol = item.get("symbol")
            if symbol and symbol in name_map:
                item["stock_name"] = name_map[symbol]
        return data

    def _get_latest_base_date(self, table_name: str):
        """테이블별 최신 base_date를 조회한다."""
        try:
            resp = (
                self.client.table(table_name)
                .select("base_date")
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data:
                return resp.data[0]["base_date"]
        except Exception as e:
            print(f"[WARNING] _get_latest_base_date({table_name}) 실패: {e}")
        return None

    def _get_latest_base_date_available(self, table_name: str, as_of_utc_iso: str = None):
        """
        available_at 기준으로 사용 가능한 최신 base_date를 조회한다.
        - 테이블에 available_at이 없으면 일반 최신 base_date로 fallback.
        """
        if not as_of_utc_iso:
            return self._get_latest_base_date(table_name)
        try:
            resp = (
                self.client.table(table_name)
                .select("base_date, available_at")
                .lte("available_at", as_of_utc_iso)
                .order("base_date", desc=True)
                .order("available_at", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data:
                return resp.data[0]["base_date"]
            # available_at이 NULL인 레거시 데이터만 존재하는 구간 fallback
            return self._get_latest_base_date(table_name)
        except Exception:
            # available_at 없는 테이블/권한 이슈는 일반 latest로 fallback
            return self._get_latest_base_date(table_name)
        return self._get_latest_base_date(table_name)

    def _fetch_latest_row_by_date(self, table_name: str, base_date: str, select_cols: str = "*"):
        """지정한 base_date 행을 조회하고 forward fill 후 마지막 레코드를 반환한다."""
        if not base_date:
            return None
        try:
            resp = (
                self.client.table(table_name)
                .select(select_cols)
                .eq("base_date", base_date)
                .execute()
            )
            if not resp.data:
                return None
            df = pd.DataFrame(resp.data).sort_values("base_date")
            df = df.ffill()
            return df.iloc[-1].to_dict()
        except Exception as e:
            print(f"[WARNING] _fetch_latest_row_by_date({table_name}, {base_date}) 실패: {e}")
            return None

    def _fetch_macro_series_snapshot(self, base_date: str, as_of_utc_iso: str = None):
        """normalized_macro_series를 series_id:close_val 맵으로 변환한다.
        실제 스키마: base_date, series_id, close_val
        """
        if not base_date:
            return None
        try:
            # close_val 컬럼 우선, value 컬럼은 레거시 폴백
            for col in ("close_val", "value"):
                try:
                    query = (
                        self.client.table("normalized_macro_series")
                        .select(f"series_id, {col}, base_date")
                        .eq("base_date", base_date)
                    )
                    if as_of_utc_iso:
                        try:
                            query = query.lte("available_at", as_of_utc_iso)
                        except Exception:
                            pass
                    resp = query.execute()
                    if resp.data:
                        snapshot = {"base_date": base_date}
                        for row in resp.data:
                            series_id = row.get("series_id")
                            if series_id:
                                snapshot[series_id] = row.get(col)
                        return snapshot
                except Exception:
                    continue
            return None
        except Exception as e:
            print(f"[WARNING] _fetch_macro_series_snapshot({base_date}) 실패: {e}")
            return None

    def get_latest_date(self, as_of_utc_iso: str = None):
        """
        기준 날짜 조회 우선순위:
        1. feature_store_daily (feature_name='volume') - StockData 파이프라인 기준일
        2. normalized_stock_prices_daily - 직접 가격 테이블 폴백
        """
        try:
            query = (
                self.client
                .table("feature_store_daily")
                .select("base_date, available_at")
                .eq("feature_name", "volume")
            )
            if as_of_utc_iso:
                query = query.lte("available_at", as_of_utc_iso)

            resp = query.order("base_date", desc=True).order("available_at", desc=True).limit(1).execute()
            if resp.data:
                return resp.data[0]["base_date"]
            # available_at이 NULL인 historical row만 있는 경우 fallback
            resp = (
                self.client
                .table("feature_store_daily")
                .select("base_date, available_at")
                .eq("feature_name", "volume")
            )
            if as_of_utc_iso:
                query = query.lte("available_at", as_of_utc_iso)

            resp = query.order("base_date", desc=True).order("available_at", desc=True).limit(1).execute()
            if resp.data:
                return resp.data[0]["base_date"]
        except Exception as e:
            # available_at 컬럼이 없거나 인덱스/권한 이슈 시 fallback
            try:
                resp = (
                    self.client
                    .table("feature_store_daily")
                    .select("base_date")
                    .eq("feature_name", "volume")
                    .order("base_date", desc=True)
                    .limit(1)
                    .execute()
                )
                if resp.data:
                    return resp.data[0]["base_date"]
            except Exception:
                pass
            print(f"[WARNING] get_latest_date 실패: {e}")
        # feature_store_daily 실패 시 normalized_stock_prices_daily 폴백
        try:
            resp = (
                self.client
                .table("normalized_stock_prices_daily")
                .select("base_date")
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data:
                return resp.data[0]["base_date"]
        except Exception as e2:
            print(f"[WARNING] get_latest_date 폴백(prices) 실패: {e2}")
        return None

    # -------------------------------------------------------------------------
    # Public data fetchers
    # -------------------------------------------------------------------------

    def fetch_macro_and_market_data(self):
        """
        최신 매크로 및 시장 폭 데이터 수집.

        반환 딕셔너리 구조:
          - 'normalized_macro_series'  : normalized_macro_series 테이블 최신 행
          - 'market_breadth_daily'     : market_breadth_daily 테이블 최신 행
          - 'momentum'                 : feature_store_daily에서 symbol='GLOBAL'인
                                         _1d_chg/_5d_chg/market_breadth 피처 전체
        """
        results = {}

        as_of_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. normalized_macro_series
        try:
            latest_macro_date = self._get_latest_base_date_available("normalized_macro_series", as_of_utc)
            results["normalized_macro_series"] = self._fetch_macro_series_snapshot(latest_macro_date, as_of_utc)
        except Exception as e:
            print(f"[WARNING] normalized_macro_series 조회 실패: {e}")
            results["normalized_macro_series"] = None

        # 1-b. normalized_global_macro_daily (consumer spec 핵심)
        try:
            latest_global_macro_date = self._get_latest_base_date_available("normalized_global_macro_daily", as_of_utc)
            # KOSPI/KOSDAQ, 투자자별 수급, 해외 지수 컬럼이 계속 확장되므로 전체 row를 가져온다.
            results["normalized_global_macro_daily"] = self._fetch_latest_row_by_date(
                "normalized_global_macro_daily",
                latest_global_macro_date,
            )
        except Exception as e:
            print(f"[WARNING] normalized_global_macro_daily 조회 실패: {e}")
            results["normalized_global_macro_daily"] = None

        # 2. market_breadth_daily
        try:
            results["market_breadth_daily"] = self._fetch_and_ffill_timeseries(
                "market_breadth_daily",
                lookback_days=5,
                as_of_utc_iso=as_of_utc,
            )
        except Exception as e:
            print(f"[WARNING] market_breadth_daily 조회 실패: {e}")
            results["market_breadth_daily"] = None

        # 3. momentum 피처: feature_store_daily에서 symbol='GLOBAL'인 데이터 추출
        try:
            latest_date = self.get_latest_date(as_of_utc_iso=as_of_utc)
            if latest_date:
                momentum_resp = (
                    self.client
                    .table("feature_store_daily")
                    .select("symbol, feature_name, feature_value, base_date")
                    .eq("base_date", latest_date)
                    .eq("symbol", "GLOBAL")
                    .or_(
                        "feature_name.ilike.%_1d_chg,"
                        "feature_name.ilike.%_5d_chg,"
                        "feature_name.ilike.market_breadth%"
                    )
                    .execute()
                )
                results["momentum"] = momentum_resp.data if momentum_resp.data else []
            else:
                results["momentum"] = []
        except Exception as e:
            print(f"[WARNING] GLOBAL momentum 피처 조회 실패: {e}")
            results["momentum"] = []

        return results

    def fetch_data_quality_guardrails(self):
        """
        소비자 규격 기반 데이터 품질 가드레일 요약:
        1) zero_volume_pct 급증 추적
        2) 테이블별 최신일 lag_days
        3) pipeline_run_logs WARN/FAIL 탐지
        """
        kst_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        kst_today = kst_now.date()
        as_of_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        table_names = [
            "normalized_stock_prices_daily",
            "normalized_stock_supply_daily",
            "normalized_stock_fundamentals_ratios",
            "normalized_global_macro_daily",
            "market_breadth_daily",
            "normalized_derivatives_daily",
            "feature_store_daily",
        ]

        latest_by_table = {}
        lag_days_by_table = {}
        for table_name in table_names:
            latest_date = self._get_latest_base_date_available(table_name, as_of_utc)
            latest_by_table[table_name] = latest_date
            if latest_date:
                try:
                    lag_days_by_table[table_name] = max(
                        0,
                        (kst_today - datetime.date.fromisoformat(str(latest_date))).days,
                    )
                except Exception:
                    lag_days_by_table[table_name] = None
            else:
                lag_days_by_table[table_name] = None

        zero_volume = {
            "latest_base_date": None,
            "previous_base_date": None,
            "latest_zero_volume_pct": None,
            "previous_zero_volume_pct": None,
            "delta_pct": None,
        }
        try:
            latest_d = self._get_latest_base_date_available("normalized_stock_prices_daily", as_of_utc)
            prev_d = None
            if latest_d:
                prev_resp = (
                    self.client.table("normalized_stock_prices_daily")
                    .select("base_date")
                    .lt("base_date", latest_d)
                    .order("base_date", desc=True)
                    .limit(1)
                    .execute()
                )
                if prev_resp.data:
                    prev_d = prev_resp.data[0].get("base_date")

            if latest_d and prev_d:
                universe_resp = self.client.table("stocks_master").select("symbol, is_active").execute()
                active_symbols = {
                    item["symbol"]
                    for item in (universe_resp.data or [])
                    if item.get("is_active") is True
                }

                price_resp = (
                    self.client.table("normalized_stock_prices_daily")
                    .select("symbol, base_date, volume")
                    .in_("base_date", [latest_d, prev_d])
                    .execute()
                )
                latest_rows = []
                prev_rows = []
                for row in price_resp.data or []:
                    symbol = row.get("symbol")
                    if active_symbols and symbol not in active_symbols:
                        continue
                    if row.get("base_date") == latest_d:
                        latest_rows.append(row)
                    elif row.get("base_date") == prev_d:
                        prev_rows.append(row)

                def calc_zero_pct(rows):
                    if not rows:
                        return None
                    zero_count = 0
                    total_count = 0
                    for r in rows:
                        vol = r.get("volume")
                        try:
                            v = float(vol) if vol is not None else 0.0
                        except Exception:
                            v = 0.0
                        if v <= 0:
                            zero_count += 1
                        total_count += 1
                    return round((zero_count / total_count) * 100, 2) if total_count else None

                latest_pct = calc_zero_pct(latest_rows)
                prev_pct = calc_zero_pct(prev_rows)
                delta = round(latest_pct - prev_pct, 2) if latest_pct is not None and prev_pct is not None else None

                zero_volume = {
                    "latest_base_date": latest_d,
                    "previous_base_date": prev_d,
                    "latest_zero_volume_pct": latest_pct,
                    "previous_zero_volume_pct": prev_pct,
                    "delta_pct": delta,
                }
        except Exception as e:
            print(f"[WARNING] zero_volume_pct 계산 실패: {e}")

        log_alerts = []
        try:
            recent_from = (kst_today - datetime.timedelta(days=3)).isoformat()
            logs_resp = (
                self.client.table("pipeline_run_logs")
                .select("job_name, target_date, status, records_processed, error_message")
                .gte("target_date", recent_from)
                .order("target_date", desc=True)
                .limit(120)
                .execute()
            )
            grouped_alerts = {}
            for row in logs_resp.data or []:
                status = (row.get("status") or "").upper()
                if status in {"WARN", "FAIL", "FAILED", "ERROR"}:
                    key = (row.get("target_date"), row.get("job_name"), status)
                    if key not in grouped_alerts:
                        grouped_alerts[key] = {
                            **row,
                            "status": status,
                            "occurrences": 1,
                        }
                    else:
                        grouped_alerts[key]["occurrences"] += 1
                        if row.get("error_message") and not grouped_alerts[key].get("error_message"):
                            grouped_alerts[key]["error_message"] = row.get("error_message")
            log_alerts = list(grouped_alerts.values())
            log_alerts = log_alerts[:20]
        except Exception as e:
            print(f"[WARNING] pipeline_run_logs 조회 실패: {e}")

        return {
            "as_of_kst_datetime": kst_now.isoformat(),
            "latest_base_date_by_table": latest_by_table,
            "lag_days_by_table": lag_days_by_table,
            "zero_volume_guardrail": zero_volume,
            "pipeline_alert_logs": log_alerts,
        }

    def fetch_top_volume_stocks(self, limit=10):
        """
        feature_store_daily 기준 최신 날짜의 거래량 상위 종목 조회 및 데이터 농축.

        1. 거래량 상위 종목 기호(symbols) 추출
        2. 해당 기호들의 [foreign_flow_zscore, return_5d, moving_avg_20] (feature_store_daily)
        3. 해당 기호들의 [per, pbr, market_cap] (fundamentals) - Join/Enrichment
        """
        # 1. stocks_master → symbol : name/market 맵
        try:
            resp = self.client.table("stocks_master").select("symbol, name, market").execute()
            market_map = {item["symbol"]: item.get("market", "Unknown") for item in resp.data}
            name_map = {item["symbol"]: item.get("name", "Unknown") for item in resp.data}
            # ETF 판별 보정
            for sym, name in name_map.items():
                if ETF_PATTERN.search(name):
                    market_map[sym] = "ETF"
        except Exception as e:
            print(f"[ERROR] stocks_master 조회 실패: {e}")
            return {"KOSPI": [], "KOSDAQ": [], "ETF": []}

        # 2. 기준 날짜
        as_of_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        latest_date = self.get_latest_date(as_of_utc_iso=as_of_utc)
        if not latest_date:
            return {"KOSPI": [], "KOSDAQ": [], "ETF": []}

        # 3. volume 상위 추출 (feature_store_daily → normalized_stock_prices_daily 폴백)
        sorted_vols = []
        try:
            vol_resp = (
                self.client.table("feature_store_daily")
                .select("symbol, feature_value")
                .eq("base_date", latest_date)
                .eq("feature_name", "volume")
                .execute()
            )
            if vol_resp.data:
                sorted_vols = sorted(
                    vol_resp.data,
                    key=lambda x: float(x.get("feature_value") or 0),
                    reverse=True,
                )
        except Exception:
            pass

        # feature_store_daily 없거나 비어 있으면 normalized_stock_prices_daily 폴백
        if not sorted_vols:
            try:
                price_resp = (
                    self.client.table("normalized_stock_prices_daily")
                    .select("symbol, volume")
                    .eq("base_date", latest_date)
                    .execute()
                )
                sorted_vols = [
                    {"symbol": r["symbol"], "feature_value": r.get("volume") or 0}
                    for r in (price_resp.data or [])
                ]
                sorted_vols.sort(key=lambda x: float(x.get("feature_value") or 0), reverse=True)
            except Exception as e:
                print(f"[WARNING] volume 폴백(prices) 조회 실패: {e}")
                return {"KOSPI": [], "KOSDAQ": [], "ETF": []}

        if not sorted_vols:
            return {"KOSPI": [], "KOSDAQ": [], "ETF": []}


        # 분류 및 심볼 리스트 추출
        result = {"KOSPI": [], "KOSDAQ": [], "ETF": []}
        all_top_symbols = []
        for item in sorted_vols:
            sym = item["symbol"]
            market = market_map.get(sym)
            if market in result and len(result[market]) < limit:
                all_top_symbols.append(sym)
                result[market].append({
                    "symbol": sym,
                    "stock_name": name_map.get(sym, "Unknown"),
                    "market": market,
                    "volume_value": float(item["feature_value"]) if item.get("feature_value") is not None else 0.0,
                    "base_date": latest_date,
                })
            if all(len(v) >= limit for v in result.values()):
                break

        if not all_top_symbols:
            return result

        # 4. 데이터 농축 (Enrichment)
        # 4-1. feature_store_daily (zscore, return, ma)
        try:
            feat_resp = (
                self.client.table("feature_store_daily")
                .select("symbol, feature_name, feature_value")
                .eq("base_date", latest_date)
                .in_("symbol", all_top_symbols)
                .in_("feature_name", ["foreign_flow_zscore", "return_5d", "moving_avg_20"])
                .execute()
            )
            for m in feat_resp.data:
                for k in result:
                    for stock in result[k]:
                        if stock["symbol"] == m["symbol"]:
                            stock[m["feature_name"]] = float(m["feature_value"]) if m["feature_value"] is not None else None
        except Exception as e:
            print(f"[WARNING] Enrichment(features) 실패: {e}")

        # 4-2. fundamentals (per, pbr, market_cap) - Fallback 포함
        try:
            try:
                fund_resp = (
                    self.client.table("normalized_stock_fundamentals_ratios")
                    .select("symbol, per, pbr, market_cap")
                    .eq("base_date", latest_date)
                    .in_("symbol", all_top_symbols)
                    .execute()
                )
            except Exception:
                # market_cap 컬럼이 없을 경우 대비 fallback
                fund_resp = (
                    self.client.table("normalized_stock_fundamentals_ratios")
                    .select("symbol, per, pbr")
                    .eq("base_date", latest_date)
                    .in_("symbol", all_top_symbols)
                    .execute()
                )
            
            for f in fund_resp.data:
                for k in result:
                    for stock in result[k]:
                        if stock["symbol"] == f["symbol"]:
                            stock.update({k: v for k, v in f.items() if k != "symbol"})
        except Exception as e:
            print(f"[WARNING] Enrichment(fundamentals) 실패: {e}")

        return result

    def fetch_target_stocks_data(self, target_symbols):
        """
        타겟 종목 분석 데이터 수집.
        - 모든 종목 코드는 6자리 zero-padded 처리
        - market_cap 컬럼 제외 (스키마 미존재)
        - supply: long format(investor_type, net_buy_vol, net_buy_amt) → wide format 변환
        - short_selling: 실제 컬럼명 사용
        - 데이터 없는 종목은 'pending' 표시 (StockData 파이프라인 적재 대기)
        """
        if not target_symbols:
            return {}

        # 종목 코드 6자리 정규화
        normalized_symbols = [str(s).zfill(6) for s in target_symbols]

        name_map = self._fetch_stock_name_map()
        results = {}
        as_of_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        latest_date = self.get_latest_date(as_of_utc_iso=as_of_utc)
        if not latest_date:
            print("[WARNING] fetch_target_stocks_data: 기준 날짜를 찾을 수 없음.")
            return {}

        print(f"[INFO] fetch_target_stocks_data: latest_date={latest_date}, symbols={normalized_symbols}")

        # 1. Fundamentals — market_cap 제외, per/pbr/dividend_yield만 조회
        try:
            fund_data = (
                self.client.table("normalized_stock_fundamentals_ratios")
                .select("symbol, per, pbr, dividend_yield, base_date")
                .eq("base_date", latest_date)
                .in_("symbol", normalized_symbols)
                .execute()
            )
            fund_rows = self._inject_stock_names(fund_data.data or [], name_map)
        except Exception:
            # dividend_yield 없으면 per/pbr만
            try:
                fund_data = (
                    self.client.table("normalized_stock_fundamentals_ratios")
                    .select("symbol, per, pbr, base_date")
                    .eq("base_date", latest_date)
                    .in_("symbol", normalized_symbols)
                    .execute()
                )
                fund_rows = self._inject_stock_names(fund_data.data or [], name_map)
            except Exception as e:
                print(f"[WARNING] Fundamentals 조회 실패: {e}")
                fund_rows = []
        results["normalized_stock_fundamentals_ratios"] = fund_rows

        # 2. 수급 — long format(investor_type, net_buy_vol, net_buy_amt) → wide format 변환
        try:
            supply_data = (
                self.client.table("normalized_stock_supply_daily")
                .select("symbol, base_date, investor_type, net_buy_vol, net_buy_amt")
                .eq("base_date", latest_date)
                .in_("symbol", normalized_symbols)
                .execute()
            )
            # long → wide pivot
            supply_wide = self._pivot_supply_long_to_wide(supply_data.data or [], name_map)
        except Exception:
            # 레거시 wide format 폴백
            try:
                supply_data = (
                    self.client.table("normalized_stock_supply_daily")
                    .select("symbol, base_date, individual_net_buy, foreign_net_buy, institutional_net_buy, pension_net_buy")
                    .eq("base_date", latest_date)
                    .in_("symbol", normalized_symbols)
                    .execute()
                )
                supply_wide = self._inject_stock_names(supply_data.data or [], name_map)
            except Exception as e:
                print(f"[WARNING] normalized_stock_supply_daily 조회 실패: {e}")
                supply_wide = []
        results["normalized_stock_supply_daily"] = supply_wide

        # 3. 공매도 — 실제 컬럼명 사용
        try:
            short_data = (
                self.client.table("normalized_stock_short_selling")
                .select("symbol, base_date, short_vol, short_amt, short_balance_vol, short_balance_amt")
                .eq("base_date", latest_date)
                .in_("symbol", normalized_symbols)
                .execute()
            )
            short_rows = self._inject_stock_names(short_data.data or [], name_map)
        except Exception:
            # 레거시 컬럼명 폴백
            try:
                short_data = (
                    self.client.table("normalized_stock_short_selling")
                    .select("*")
                    .eq("base_date", latest_date)
                    .in_("symbol", normalized_symbols)
                    .execute()
                )
                short_rows = self._inject_stock_names(short_data.data or [], name_map)
            except Exception as e:
                print(f"[WARNING] normalized_stock_short_selling 조회 실패: {e}")
                short_rows = []
        results["normalized_stock_short_selling"] = short_rows

        # 4. feature_store_daily (있으면 사용, 없으면 조용히 건너뜀)
        try:
            feature_data = (
                self.client.table("feature_store_daily")
                .select("*")
                .eq("base_date", latest_date)
                .in_("symbol", normalized_symbols)
                .execute()
            )
            results["feature_store_daily"] = self._inject_stock_names(feature_data.data or [], name_map)
        except Exception as e:
            print(f"[INFO] feature_store_daily 없거나 접근 불가 (StockData 적재 대기): {e}")
            results["feature_store_daily"] = []

        # 5. 가격 데이터 (close_price, change_pct 등)
        try:
            # change_pct 컬럼이 없을 경우 대비 (StockData 적재 시점에 따라 다름)
            try:
                price_data = (
                    self.client.table("normalized_stock_prices_daily")
                    .select("symbol, base_date, open_price, high_price, low_price, close_price, volume, change_pct")
                    .eq("base_date", latest_date)
                    .in_("symbol", normalized_symbols)
                    .execute()
                )
            except Exception:
                price_data = (
                    self.client.table("normalized_stock_prices_daily")
                    .select("symbol, base_date, open_price, high_price, low_price, close_price, volume")
                    .eq("base_date", latest_date)
                    .in_("symbol", normalized_symbols)
                    .execute()
                )
            results["normalized_stock_prices_daily"] = self._inject_stock_names(price_data.data or [], name_map)
        except Exception as e:
            print(f"[WARNING] normalized_stock_prices_daily 조회 실패: {e}")
            results["normalized_stock_prices_daily"] = []

        # 데이터 없는 종목 로그
        found_symbols = {r["symbol"] for r in results.get("normalized_stock_prices_daily", []) if r.get("symbol")}
        missing = [s for s in normalized_symbols if s not in found_symbols]
        if missing:
            print(f"[INFO] 수집 대기 중 (StockData 미적재): {missing}")

        return results

    def _pivot_supply_long_to_wide(self, rows: list, name_map: dict) -> list:
        """
        normalized_stock_supply_daily long format → wide format 변환.
        investor_type 예시: '개인', '외국인', '기관합계', '연기금', '법인'
        """
        INVESTOR_KEY_MAP = {
            "개인":    "individual_net_buy",
            "외국인":  "foreign_net_buy",
            "기관합계": "institutional_net_buy",
            "기관":    "institutional_net_buy",
            "연기금":  "pension_net_buy",
            "법인":    "corporate_net_buy",
        }
        # symbol, base_date 기준으로 그룹핑
        grouped: dict = {}
        for row in rows:
            sym = row.get("symbol")
            dt  = row.get("base_date")
            if not sym or not dt:
                continue
            key = (sym, dt)
            if key not in grouped:
                grouped[key] = {"symbol": sym, "base_date": dt}
                if sym in name_map:
                    grouped[key]["stock_name"] = name_map[sym]
            inv_type = row.get("investor_type", "")
            col = INVESTOR_KEY_MAP.get(inv_type)
            if col:
                grouped[key][col] = row.get("net_buy_vol")
                grouped[key][col.replace("net_buy", "net_buy_amt")] = row.get("net_buy_amt")
            else:
                # 매핑 안 된 투자자 유형은 raw key로 보존
                grouped[key][f"{inv_type}_net_buy_vol"] = row.get("net_buy_vol")
        return list(grouped.values())

    # -------------------------------------------------------------------------
    # News methods (Unified into SupabaseReader)
    # -------------------------------------------------------------------------

    def fetch_news_document(self):
        """Fetches news document from Google Docs (moved from news_reader)."""
        url = config.get("news_url", section="google_docs")
        if not url:
            print("Warning: Google Docs news URL not found. Skipping news.")
            return ""

        last_error = None
        for attempt in range(1, NEWS_FETCH_RETRIES + 1):
            try:
                response = requests.get(url, timeout=NEWS_FETCH_TIMEOUT)
                response.raise_for_status()
                return response.text
            except Exception as exc:
                last_error = exc
                print(f"Warning: news fetch attempt {attempt}/{NEWS_FETCH_RETRIES} failed: {exc}")
                if attempt < NEWS_FETCH_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        print(f"Error fetching news document after retries: {last_error}")
        return ""

    def prepare_news_context(self, news_text: str) -> str:
        """Prepares news context for LLM (moved from news_reader)."""
        if not news_text:
            return ""

        text = news_text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
        
        kst = datetime.timezone(datetime.timedelta(hours=9))
        now = datetime.datetime.now(kst)
        
        items = []
        seen = set()
        current = []
        keep_current = True

        def flush():
            nonlocal current, keep_current
            if not current or not keep_current:
                current = []
                keep_current = True
                return
            item = " ".join(current)
            item = " ".join(item.split()).strip(" -•*|\t")
            if not item:
                return
            normalized_key = item.lower()
            if normalized_key in seen:
                return
            seen.add(normalized_key)
            if len(item) > MAX_NEWS_ITEM_CHARS:
                item = item[: MAX_NEWS_ITEM_CHARS - 1].rstrip() + "…"
            items.append(item)
            current = []
            keep_current = True

        for raw_line in text.split("\n"):
            line = " ".join(raw_line.strip().split())
            if not line:
                flush()
                continue
                
            # 불필요한 메타데이터 스킵
            if line.startswith(("갱신 시각:", "시간:", "유지 기준:")) or line.startswith('{"type":"text"') or line.startswith("📢"):
                continue

            # 새로운 뉴스 항목 시작
            if line.startswith("["):
                flush()
                # 12시간 이내 필터링 로직
                time_match = re.search(r'\[.*?(\d{1,2})[:/.-](\d{2}).*?\]', line)
                if time_match:
                    hour, minute = int(time_match.group(1)), int(time_match.group(2))
                    try:
                        target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        if target_time > now:
                            target_time -= datetime.timedelta(days=1)
                        if (now - target_time) > datetime.timedelta(hours=12):
                            keep_current = False
                    except ValueError:
                        pass
                
            if current and not raw_line.startswith((" ", "\t", "-", "•", "*")):
                flush()

            current.append(line)

        flush()

        if not items:
            return ""

        rendered = "\n".join(f"- {item}" for item in items)
        return rendered[:MAX_NEWS_CONTEXT_CHARS]

    # -------------------------------------------------------------------------
    # Report spec v3 helpers
    # -------------------------------------------------------------------------

    def fetch_latest_global_macro_snapshot(self):
        columns = (
            "base_date, usdkrw, dxy, us10y, us3y, kr10y, kospi, kospi_change_rate, "
            "kosdaq, kosdaq_change_rate, nasdaq, nasdaq_change_rate, sp500, "
            "sp500_change_rate, sox, vix, wti, brent, gold, copper, bdry, "
            "hy_spread, kospi_individual_net_buy, kospi_foreign_net_buy, "
            "kospi_institutional_net_buy, kosdaq_individual_net_buy, "
            "kosdaq_foreign_net_buy, kosdaq_institutional_net_buy, available_at"
        )
        try:
            response = (
                self.client.table("normalized_global_macro_daily")
                .select(columns)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            row = (response.data or [{}])[0]
            us10y = row.get("us10y")
            us3y = row.get("us3y")
            try:
                if us10y is not None and us3y is not None:
                    spread = float(us10y) - float(us3y)
                    row["us10y_us3y_spread"] = spread
                    row["us10y_us3y_spread_bp"] = spread * 100
                else:
                    row["us10y_us3y_spread"] = None
                    row["us10y_us3y_spread_bp"] = None
            except (TypeError, ValueError):
                row["us10y_us3y_spread"] = None
                row["us10y_us3y_spread_bp"] = None
            return row
        except Exception as exc:
            print(f"[WARNING] fetch_latest_global_macro_snapshot failed: {exc}")
            return {}

    def fetch_latest_market_breadth(self):
        columns = "base_date, advances, declines, unchanged, advancing_volume, declining_volume, available_at"
        try:
            response = (
                self.client.table("market_breadth_daily")
                .select(columns)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            return (response.data or [{}])[0]
        except Exception as exc:
            print(f"[WARNING] fetch_latest_market_breadth failed: {exc}")
            return {}

    def fetch_latest_derivatives_snapshot(self):
        columns = (
            "base_date, kospi200_futures, futures_basis, open_interest, "
            "night_futures_return, expiration_flag, available_at"
        )
        try:
            response = (
                self.client.table("normalized_derivatives_daily")
                .select(columns)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            return (response.data or [{}])[0]
        except Exception as exc:
            print(f"[WARNING] fetch_latest_derivatives_snapshot failed: {exc}")
            return {}

    def fetch_static_stock_universe(self):
        columns = "symbol, name, market, enabled, source_file, updated_at"
        try:
            response = (
                self.client.table("static_stock_universe")
                .select(columns)
                .eq("enabled", True)
                .order("market")
                .order("symbol")
                .execute()
            )
            rows = response.data or []
            normalized_rows = []
            for row in rows:
                symbol = canonicalize_symbol(row.get("symbol"))
                if not symbol:
                    continue
                normalized_rows.append({**row, "symbol": symbol})
            return normalized_rows
        except Exception as exc:
            print(f"[WARNING] fetch_static_stock_universe failed: {exc}")
            return []

    def _fetch_rows_for_symbols(self, table_name: str, columns: str, symbols: list[str], limit: int = 10000):
        if not symbols:
            return []
        expanded_symbols = self._expand_symbol_aliases(symbols)
        try:
            response = (
                self.client.table(table_name)
                .select(columns)
                .in_("symbol", expanded_symbols)
                .order("base_date", desc=True)
                .limit(limit)
                .execute()
            )
            return response.data or []
        except Exception as exc:
            print(f"[WARNING] _fetch_rows_for_symbols({table_name}) failed: {exc}")
            return []

    @staticmethod
    def _expand_symbol_aliases(symbols: list[str]) -> list[str]:
        aliases = []
        for symbol in symbols:
            canonical = canonicalize_symbol(symbol)
            if not canonical:
                continue
            aliases.append(canonical)
            aliases.append(canonical.lstrip("0") or "0")
            aliases.append(f"Q{canonical}")
        deduped = []
        seen = set()
        for item in aliases:
            if item and item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    @staticmethod
    def _pick_latest_rows_by_symbol(rows: list[dict]) -> dict:
        latest_rows = {}
        for row in rows or []:
            symbol = canonicalize_symbol(row.get("symbol"))
            if symbol and symbol not in latest_rows:
                latest_rows[symbol] = {**row, "symbol": symbol}
        return latest_rows

    @staticmethod
    def _pick_rows_matching_price_date(symbols: list[str], price_map: dict, rows: list[dict]) -> dict:
        grouped = {}
        for row in rows or []:
            symbol = canonicalize_symbol(row.get("symbol"))
            if not symbol:
                continue
            row = {**row, "symbol": symbol}
            grouped.setdefault(symbol, []).append(row)

        selected = {}
        for symbol in symbols:
            canonical = canonicalize_symbol(symbol)
            symbol_rows = grouped.get(canonical, [])
            if not symbol_rows:
                selected[canonical] = {}
                continue
            price_date = (price_map.get(canonical) or {}).get("base_date")
            exact = next((row for row in symbol_rows if row.get("base_date") == price_date), None)
            selected[canonical] = exact or symbol_rows[0]
        return selected

    def fetch_stock_feature_pivot(self, symbols, feature_names):
        if not symbols or not feature_names:
            return {}
        expanded_symbols = self._expand_symbol_aliases(symbols)
        try:
            response = (
                self.client.table("feature_store_daily")
                .select("symbol, base_date, feature_name, feature_value, available_at")
                .in_("symbol", expanded_symbols)
                .in_("feature_name", feature_names)
                .order("base_date", desc=True)
                .limit(20000)
                .execute()
            )
            rows = response.data or []
        except Exception as exc:
            print(f"[WARNING] fetch_stock_feature_pivot failed: {exc}")
            return {}

        pivoted = {}
        seen = set()
        for row in rows:
            symbol = canonicalize_symbol(row.get("symbol"))
            feature_name = row.get("feature_name")
            if not symbol or not feature_name:
                continue
            key = (symbol, feature_name)
            if key in seen:
                continue
            seen.add(key)
            pivoted.setdefault(symbol, {})
            pivoted[symbol][feature_name] = row.get("feature_value")
            pivoted[symbol]["base_date"] = row.get("base_date")
            pivoted[symbol]["available_at"] = row.get("available_at")
        return pivoted

    def fetch_latest_stock_events(self, symbols):
        columns = "symbol, base_date, event_type, event_score, sentiment_score, available_at"
        return self._pick_latest_rows_by_symbol(
            self._fetch_rows_for_symbols("normalized_stock_events_daily", columns, symbols)
        )

    def fetch_latest_short_selling(self, symbols):
        columns = "symbol, base_date, short_volume, short_value, short_ratio, source, available_at"
        return self._pick_latest_rows_by_symbol(
            self._fetch_rows_for_symbols("normalized_stock_short_selling", columns, symbols)
        )

    def fetch_fundamentals_ratio_history(self, symbols, lookback_rows: int = 2000):
        columns = "symbol, base_date, per, pbr, roe, debt_ratio, source, available_at"
        rows = self._fetch_rows_for_symbols(
            "normalized_stock_fundamentals_ratios",
            columns,
            symbols,
            limit=max(lookback_rows, len(symbols) * 10),
        )
        history = {}
        for row in rows:
            symbol = row.get("symbol")
            if not symbol:
                continue
            history.setdefault(symbol, []).append(row)
        return history

    def fetch_latest_fundamentals_raw(self, symbols):
        columns = (
            "symbol, base_date, revenue, operating_income, net_income, "
            "total_assets, total_liabilities, total_equity, available_at"
        )
        return self._pick_latest_rows_by_symbol(
            self._fetch_rows_for_symbols("normalized_stock_fundamentals", columns, symbols)
        )

    def fetch_latest_base_date_and_count(self, table_name: str, date_column: str = "base_date"):
        latest_date = None
        row_count = 0
        try:
            latest_resp = (
                self.client.table(table_name)
                .select(date_column)
                .order(date_column, desc=True)
                .limit(1)
                .execute()
            )
            if latest_resp.data:
                latest_date = latest_resp.data[0].get(date_column)
            if latest_date:
                count_resp = (
                    self.client.table(table_name)
                    .select(date_column, count="exact")
                    .eq(date_column, latest_date)
                    .limit(1)
                    .execute()
                )
                row_count = count_resp.count or 0
        except Exception as exc:
            print(f"[WARNING] fetch_latest_base_date_and_count({table_name}) failed: {exc}")
        return {"base_date": latest_date, "row_count": row_count}

    def get_latest_valid_price_date(self, report_date: str | None = None, lookback_days: int = 7):
        candidates = self.fetch_price_date_candidates(report_date, lookback_days=lookback_days)
        best = None
        for base_date in candidates:
            rows = self.fetch_price_rows_by_date(base_date)
            valid_rows = 0
            for row in rows:
                if has_minimum_top_data(row):
                    valid_rows += 1
            candidate = {"base_date": base_date, "valid_rows": valid_rows, "total_rows": len(rows)}
            if best is None or (candidate["valid_rows"], candidate["base_date"]) > (best["valid_rows"], best["base_date"]):
                best = candidate
        return best or {"base_date": None, "valid_rows": 0, "total_rows": 0}

    @staticmethod
    def _normalize_price_row_fields(row: dict) -> dict:
        normalized = dict(row or {})
        field_map = {
            "stck_oprc": "open_price",
            "stck_hgpr": "high_price",
            "stck_lwpr": "low_price",
            "stck_clpr": "close_price",
            "acml_vol": "volume",
            "acml_tr_pbmn": "trading_value",
            "close": "close_price",
            "amount": "trading_value",
            "open": "open_price",
            "high": "high_price",
            "low": "low_price",
        }
        for source_field, target_field in field_map.items():
            if target_field not in normalized or normalized.get(target_field) in (None, "", "-"):
                if source_field in normalized:
                    normalized[target_field] = normalized.get(source_field)
        return normalized

    def _build_price_map_for_date(self, base_date: str | None) -> tuple[dict, dict]:
        if not base_date:
            return {}, {"base_date": None, "row_count": 0, "valid_rows": 0}
        rows = [self._normalize_price_row_fields(row) for row in self.fetch_price_rows_by_date(base_date)]
        price_map = {}
        valid_rows = 0
        for row in rows:
            canonical = canonicalize_symbol(row.get("symbol"))
            if not canonical:
                continue
            row = {**row, "symbol": canonical}
            if canonical not in price_map:
                price_map[canonical] = row
            if has_minimum_top_data(row):
                valid_rows += 1
        return price_map, {"base_date": base_date, "row_count": len(rows), "valid_rows": valid_rows}

    def _fetch_market_ranking_rows_for_dates(self, candidate_dates: list[str]) -> list[dict]:
        if not candidate_dates:
            return []
        try:
            return self._execute_paged_query(
                "normalized_market_rankings_daily",
                "*",
                query_mutator=lambda query: query.in_("base_date", candidate_dates),
            )
        except Exception as exc:
            print(f"[WARNING] _fetch_market_ranking_rows_for_dates failed: {exc}")
            return []

    def _candidate_ranking_dates(self, report_date: str | None = None, lookback_days: int = 7) -> list[str]:
        candidates = []
        try:
            query = self.client.table("normalized_market_rankings_daily").select("base_date").order("base_date", desc=True).limit(max(lookback_days * 4, 20))
            if report_date:
                query = query.lte("base_date", report_date)
            response = query.execute()
            for row in response.data or []:
                base_date = str(row.get("base_date"))
                if base_date and base_date not in candidates:
                    candidates.append(base_date)
        except Exception as exc:
            print(f"[WARNING] _candidate_ranking_dates failed: {exc}")
        return candidates[: max(lookback_days, 5)]

    def _enrich_ranking_rows(self, rows: list[dict], price_map: dict, master_map: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        enriched = []
        market_mismatches = []
        q_prefix_rows = []
        legacy_source_rows = []

        for row in rows or []:
            raw_symbol = row.get("symbol")
            canonical = canonicalize_symbol(raw_symbol)
            master = master_map.get(canonical, {})
            ranking_market = normalize_market_label(row.get("market"))
            master_market = normalize_market_label(master.get("market"))
            rank_type = (row.get("rank_type") or "").strip().lower()
            source = row.get("source")

            if raw_symbol and str(raw_symbol).strip().upper().startswith("Q") and canonical:
                q_prefix_rows.append(
                    {
                        "symbol": raw_symbol,
                        "canonical_symbol": canonical,
                        "base_date": row.get("base_date"),
                        "market": ranking_market,
                        "rank_type": rank_type,
                    }
                )

            if not is_allowed_ranking_source(rank_type, source):
                legacy_source_rows.append(
                    {
                        "symbol": canonical,
                        "base_date": row.get("base_date"),
                        "market": ranking_market,
                        "rank_type": rank_type,
                        "source": source,
                    }
                )
                continue

            if ranking_market and master_market and not ranking_market_matches_master(ranking_market, master_market):
                market_mismatches.append(
                    {
                        "symbol": canonical,
                        "base_date": row.get("base_date"),
                        "ranking_market": ranking_market,
                        "master_market": master_market,
                        "rank_type": rank_type,
                        "source": source,
                    }
                )
                continue

            price_row = price_map.get(canonical, {})
            enriched.append(
                {
                    **row,
                    **{key: value for key, value in price_row.items() if key not in {"symbol", "base_date"}},
                    "symbol": canonical,
                    "display_symbol": canonical,
                    "canonical_symbol": canonical,
                    "name": row.get("name") or master.get("name") or canonical,
                    "market": ranking_market or master_market,
                    "master_market": master_market,
                    "asset_type": infer_asset_type(row.get("name") or master.get("name"), ranking_market or master_market, canonical),
                    "rank_type": rank_type,
                    "source": source,
                    "ranking_base_date": row.get("base_date"),
                    "price_base_date": price_row.get("base_date"),
                }
            )

        return enriched, market_mismatches, q_prefix_rows, legacy_source_rows

    def _build_price_fallback_rankings(self, price_base_date: str | None, limit: int = 10):
        sections = {
            "volume": {"KOSPI": [], "KOSDAQ": [], "ETF": [], "ETN": []},
            "trading_value": {"KOSPI": [], "KOSDAQ": [], "ETF": [], "ETN": []},
            "market_cap": {"KOSPI": [], "KOSDAQ": [], "ETF": [], "ETN": []},
        }
        if not price_base_date:
            return sections

        master_map = self.fetch_stocks_master_map()
        rows = [self._normalize_price_row_fields(row) for row in self.fetch_price_rows_by_date(price_base_date)]
        enriched_rows = []
        for row in rows:
            canonical = canonicalize_symbol(row.get("symbol"))
            master = master_map.get(canonical, {})
            market = normalize_market_label(master.get("market"))
            name = master.get("name") or canonical
            asset_type = infer_asset_type(name, market, canonical)
            if not canonical or not market:
                continue
            enriched_rows.append(
                {
                    **row,
                    "symbol": canonical,
                    "display_symbol": canonical,
                    "canonical_symbol": canonical,
                    "name": name,
                    "market": market,
                    "asset_type": asset_type,
                    "ranking_base_date": price_base_date,
                    "price_base_date": price_base_date,
                    "source": "VALID_PRICE_FALLBACK",
                }
            )

        market_groups = {
            "KOSPI": [row for row in enriched_rows if row.get("market") == "KOSPI" and is_common_stock_top_eligible(row)],
            "KOSDAQ": [row for row in enriched_rows if row.get("market") == "KOSDAQ" and is_common_stock_top_eligible(row)],
            "ETF": [row for row in enriched_rows if row.get("asset_type") == "ETF"],
            "ETN": [row for row in enriched_rows if row.get("asset_type") == "ETN"],
        }

        for market, items in market_groups.items():
            volume_sorted = sorted(
                [row for row in items if has_minimum_top_data(row)],
                key=lambda item: (-float(item.get("volume") or 0), -float(item.get("trading_value") or 0)),
            )
            trading_sorted = sorted(
                [row for row in items if has_minimum_top_data(row)],
                key=lambda item: (-float(item.get("trading_value") or 0), -float(item.get("volume") or 0)),
            )
            market_cap_sorted = sorted(
                [row for row in items if row.get("market_cap") not in (None, "")],
                key=lambda item: -float(item.get("market_cap") or 0),
            )

            sections["volume"][market] = [
                {**row, "rank_type": "volume", "rank": index + 1}
                for index, row in enumerate(volume_sorted[:limit])
            ]
            sections["trading_value"][market] = [
                {**row, "rank_type": "trading_value", "rank": index + 1}
                for index, row in enumerate(trading_sorted[:limit])
            ]
            sections["market_cap"][market] = [
                {**row, "rank_type": "market_cap", "rank": index + 1}
                for index, row in enumerate(market_cap_sorted[:limit])
            ]
        return sections

    def get_latest_market_rankings(self, report_date: str | None = None, limit: int = 10):
        candidate_dates = self._candidate_ranking_dates(report_date, lookback_days=7)
        latest_valid_price = self.get_latest_valid_price_date(report_date, lookback_days=7)
        price_base_date = latest_valid_price.get("base_date")
        price_map, price_meta = self._build_price_map_for_date(price_base_date)
        master_map = self.fetch_stocks_master_map()
        ranking_rows = self._fetch_market_ranking_rows_for_dates(candidate_dates)
        sections_template = {
            "volume": {"KOSPI": [], "KOSDAQ": [], "ETF": [], "ETN": []},
            "trading_value": {"KOSPI": [], "KOSDAQ": [], "ETF": [], "ETN": []},
            "market_cap": {"KOSPI": [], "KOSDAQ": [], "ETF": [], "ETN": []},
        }

        per_date_rows = {}
        for row in ranking_rows:
            base_date = str(row.get("base_date"))
            per_date_rows.setdefault(base_date, []).append(row)

        selected_date = None
        fallback_used = False
        selected_sections = sections_template
        selected_diagnostics = {
            "market_mismatch_rows": [],
            "q_prefix_rows": [],
            "legacy_source_rows": [],
            "candidate_dates": candidate_dates,
            "ranking_counts": {},
        }

        required_markets = {"KOSPI", "KOSDAQ"}
        for index, base_date in enumerate(candidate_dates):
            rows = per_date_rows.get(base_date, [])
            enriched, market_mismatches, q_prefix_rows, legacy_source_rows = self._enrich_ranking_rows(rows, price_map, master_map)
            sections = {
                "volume": {"KOSPI": [], "KOSDAQ": [], "ETF": [], "ETN": []},
                "trading_value": {"KOSPI": [], "KOSDAQ": [], "ETF": [], "ETN": []},
                "market_cap": {"KOSPI": [], "KOSDAQ": [], "ETF": [], "ETN": []},
            }
            for row in enriched:
                rank_type = row.get("rank_type")
                market = normalize_market_label(row.get("market"))
                if rank_type in sections and market in sections[rank_type]:
                    sections[rank_type][market].append(row)

            for rank_type in sections:
                for market in sections[rank_type]:
                    sections[rank_type][market] = sorted(
                        sections[rank_type][market],
                        key=lambda item: (
                            float(item.get("rank") or 999999),
                            -float(item.get("trading_value") or 0),
                            -float(item.get("volume") or 0),
                        ),
                    )[:limit]

            ranking_counts = {
                f"{rank_type}:{market}": len(rows_)
                for rank_type, market_map in sections.items()
                for market, rows_ in market_map.items()
            }
            has_required_volume = all(sections["volume"].get(market) for market in required_markets)
            selected_date = base_date
            selected_sections = sections
            selected_diagnostics = {
                "market_mismatch_rows": market_mismatches,
                "q_prefix_rows": q_prefix_rows,
                "legacy_source_rows": legacy_source_rows,
                "candidate_dates": candidate_dates,
                "ranking_counts": ranking_counts,
            }
            fallback_used = bool(report_date and base_date != str(report_date)) or index > 0
            if has_required_volume:
                break

        price_fallback_sections = self._build_price_fallback_rankings(price_base_date, limit=limit)
        fallback_applied_sections = []
        for rank_type, market_map in selected_sections.items():
            for market, rows in market_map.items():
                if rows:
                    continue
                fallback_rows = price_fallback_sections.get(rank_type, {}).get(market, [])
                if fallback_rows:
                    selected_sections[rank_type][market] = fallback_rows
                    fallback_applied_sections.append(f"{rank_type}:{market}")

        if fallback_applied_sections:
            fallback_used = True

        ranking_status = "정상"
        if not selected_date:
            ranking_status = "부족"
        elif fallback_used or selected_diagnostics["legacy_source_rows"] or fallback_applied_sections:
            ranking_status = "일부 fallback"
        if selected_diagnostics["market_mismatch_rows"] or selected_diagnostics["q_prefix_rows"]:
            ranking_status = "부족"

        return {
            "ranking_base_date": selected_date,
            "price_base_date": price_meta.get("base_date"),
            "latest_valid_price_date": price_meta.get("base_date"),
            "fallback_used": fallback_used,
            "sections": selected_sections,
            "diagnostics": selected_diagnostics,
            "ranking_status": ranking_status,
            "price_meta": price_meta,
            "fallback_applied_sections": fallback_applied_sections,
        }

    def get_ranking_based_universe(self, report_date: str | None = None, limit: int = 10):
        ranking_bundle = self.get_latest_market_rankings(report_date=report_date, limit=limit)
        rows = []
        for rank_type, market_map in (ranking_bundle.get("sections") or {}).items():
            for market, items in market_map.items():
                for item in items:
                    rows.append(
                        {
                            **item,
                            "source_category": "ranking",
                            "ranking_market": market,
                            "rank_type": rank_type,
                        }
                    )
        return rows

    def get_watchlist_snapshots(self, report_date: str | None = None):
        latest_valid_price = self.get_latest_valid_price_date(report_date, lookback_days=7)
        snapshots = self.fetch_static_universe_stock_snapshot(price_base_date=latest_valid_price.get("base_date"))
        for snapshot in snapshots:
            snapshot["source_category"] = "watchlist"
            snapshot["selected_price_base_date"] = latest_valid_price.get("base_date")
        return {
            "price_base_date": latest_valid_price.get("base_date"),
            "snapshots": snapshots,
            "price_meta": latest_valid_price,
        }

    def fetch_price_rows_by_date(self, base_date: str):
        if not base_date:
            return []
        try:
            return self._execute_paged_query(
                "normalized_stock_prices_daily",
                (
                    "symbol, base_date, open_price, high_price, low_price, close_price, "
                    "volume, trading_value, market_cap, outstanding_shares, available_at"
                ),
                query_mutator=lambda query: query.eq("base_date", base_date),
            )
        except Exception as exc:
            print(f"[WARNING] fetch_price_rows_by_date failed: {exc}")
            return []

    def fetch_raw_price_rows_by_date(self, base_date: str):
        if not base_date:
            return []
        candidate_selects = [
            "symbol, base_date, close, volume, amount, open, high, low",
            "symbol, base_date, close_price, volume, trading_value, open_price, high_price, low_price",
            "symbol, base_date, stck_clpr, acml_vol, acml_tr_pbmn, stck_oprc, stck_hgpr, stck_lwpr",
        ]
        for columns in candidate_selects:
            try:
                rows = self._execute_paged_query(
                    "raw_stock_prices_daily",
                    columns,
                    query_mutator=lambda query: query.eq("base_date", base_date),
                )
                if rows is not None:
                    return rows or []
            except Exception:
                continue
        return []

    def fetch_stocks_master_map(self):
        try:
            rows = self._execute_paged_query(
                "stocks_master",
                "symbol, name, market, is_active, updated_at",
            )
        except Exception as exc:
            print(f"[WARNING] fetch_stocks_master_map failed: {exc}")
            rows = []
        return {canonicalize_symbol(row.get("symbol")): row for row in rows if row.get("symbol")}

    def fetch_price_date_candidates(self, report_base_date: str | None = None, lookback_days: int = 7):
        candidates: list[str] = []
        if report_base_date:
            candidates.append(str(report_base_date))
        try:
            response = (
                self.client.table("normalized_stock_prices_daily")
                .select("base_date")
                .order("base_date", desc=True)
                .limit(max(lookback_days, 7))
                .execute()
            )
            for row in response.data or []:
                base_date = str(row.get("base_date"))
                if base_date and base_date not in candidates:
                    candidates.append(base_date)
        except Exception as exc:
            print(f"[WARNING] fetch_price_date_candidates failed: {exc}")
        return candidates

    def fetch_price_diagnostics(self, report_base_date: str | None = None, lookback_days: int = 7):
        master_map = self.fetch_stocks_master_map()
        latest_normalized = self.fetch_latest_base_date_and_count("normalized_stock_prices_daily")
        latest_raw = self.fetch_latest_base_date_and_count("raw_stock_prices_daily")
        candidate_dates = self.fetch_price_date_candidates(report_base_date, lookback_days=lookback_days)
        evaluated_candidates = []

        for base_date in candidate_dates:
            rows = self.fetch_price_rows_by_date(base_date)
            enriched_rows = []
            market_counts = {}
            valid_close = 0
            valid_volume = 0
            valid_trading_value = 0
            market_not_null = 0
            for row in rows:
                canonical = canonicalize_symbol(row.get("symbol"))
                master = master_map.get(canonical, {})
                market = normalize_market_label(master.get("market"))
                if row.get("close_price") not in (None, "", 0):
                    valid_close += 1
                if row.get("volume") not in (None, "", 0):
                    try:
                        if float(row.get("volume")) > 0:
                            valid_volume += 1
                    except Exception:
                        pass
                if row.get("trading_value") not in (None, "", 0):
                    try:
                        if float(row.get("trading_value")) > 0:
                            valid_trading_value += 1
                    except Exception:
                        pass
                if market:
                    market_not_null += 1
                    market_counts[market] = market_counts.get(market, 0) + 1
                enriched_rows.append(
                    {
                        **row,
                        "symbol": canonical,
                        "canonical_symbol": canonical,
                        "name": (master.get("name") or canonical),
                        "market": market,
                        "asset_type": infer_asset_type(master.get("name"), market, canonical),
                    }
                )

            evaluated_candidates.append(
                {
                    "base_date": base_date,
                    "rows": enriched_rows,
                    "total_rows": len(rows),
                    "valid_close_rows": valid_close,
                    "valid_volume_rows": valid_volume,
                    "valid_trading_value_rows": valid_trading_value,
                    "market_not_null_rows": market_not_null,
                    "market_distribution": market_counts,
                }
            )

        selected = None
        if evaluated_candidates:
            report_candidate = next((item for item in evaluated_candidates if item["base_date"] == str(report_base_date)), None)
            if report_candidate and report_candidate["valid_close_rows"] >= 500:
                selected = report_candidate
            else:
                selected = max(
                    evaluated_candidates,
                    key=lambda item: (item["valid_close_rows"], item["valid_trading_value_rows"], item["base_date"]),
                )

        return {
            "report_base_date": report_base_date,
            "latest_normalized": latest_normalized,
            "latest_raw": latest_raw,
            "candidates": evaluated_candidates,
            "selected": selected,
            "fallback_used": bool(selected and report_base_date and selected["base_date"] != str(report_base_date)),
        }

    def fetch_static_universe_stock_snapshot(self, price_base_date: str | None = None):
        static_universe = self.fetch_static_stock_universe()
        symbols = [canonicalize_symbol(item.get("symbol")) for item in static_universe if item.get("symbol")]
        symbols = [symbol for symbol in symbols if symbol]
        if not symbols:
            return []

        latest_price_base_date = price_base_date or self._get_latest_base_date("normalized_stock_prices_daily")
        price_columns = (
            "symbol, base_date, open_price, high_price, low_price, close_price, "
            "volume, trading_value, market_cap, outstanding_shares, available_at"
        )
        supply_columns = (
            "symbol, base_date, individual_net_buy, foreign_net_buy, institutional_net_buy, "
            "pension_net_buy, corporate_net_buy, foreign_holding_ratio, available_at"
        )
        fundamental_columns = "symbol, base_date, per, pbr, roe, debt_ratio, source, available_at"
        short_columns = "symbol, base_date, short_volume, short_value, short_ratio, source, available_at"
        event_columns = "symbol, base_date, event_type, event_score, sentiment_score, available_at"

        try:
            price_rows = self.fetch_price_rows_by_date(latest_price_base_date)
        except Exception as exc:
            print(f"[WARNING] fetch_static_universe_stock_snapshot prices failed: {exc}")
            price_rows = []

        watchlist_set = set(symbols)
        price_map = {}
        for row in price_rows:
            canonical = canonicalize_symbol(row.get("symbol"))
            if canonical in watchlist_set and canonical not in price_map:
                price_map[canonical] = {**row, "symbol": canonical}
        supply_map = self._pick_rows_matching_price_date(
            symbols,
            price_map,
            self._fetch_rows_for_symbols("normalized_stock_supply_daily", supply_columns, symbols),
        )
        fundamental_map = self._pick_latest_rows_by_symbol(
            self._fetch_rows_for_symbols("normalized_stock_fundamentals_ratios", fundamental_columns, symbols)
        )
        short_map = self.fetch_latest_short_selling(symbols)
        event_map = self.fetch_latest_stock_events(symbols)
        raw_fundamentals_map = self.fetch_latest_fundamentals_raw(symbols)
        ratio_history_map = self.fetch_fundamentals_ratio_history(symbols)
        feature_map = self.fetch_stock_feature_pivot(
            symbols,
            ["return_5d", "moving_avg_5", "moving_avg_20", "volatility_20d", "foreign_flow_zscore"],
        )

        snapshots = []
        for item in static_universe:
            symbol = canonicalize_symbol(item.get("symbol"))
            snapshots.append(
                {
                    "symbol": symbol,
                    "name": item.get("name") or symbol,
                    "market": normalize_market_label(item.get("market")),
                    "source_file": item.get("source_file"),
                    "updated_at": item.get("updated_at"),
                    "price": price_map.get(symbol, {}),
                    "supply": supply_map.get(symbol, {}),
                    "fundamentals": fundamental_map.get(symbol, {}),
                    "fundamentals_raw": raw_fundamentals_map.get(symbol, {}),
                    "fundamentals_history": ratio_history_map.get(symbol, []),
                    "short_selling": short_map.get(symbol, {}),
                    "event": event_map.get(symbol, {}),
                    "features": feature_map.get(symbol, {}),
                }
            )
        return snapshots

    def fetch_full_market_top_volume_stocks(self, limit=5, price_base_date: str | None = None):
        latest_price_base_date = price_base_date or self._get_latest_base_date("normalized_stock_prices_daily")
        result = {
            "base_date": latest_price_base_date,
            "coverage": {
                "covered_symbols": 0,
                "kospi_covered": 0,
                "kosdaq_covered": 0,
                "null_close_rows": 0,
            },
            "rows": [],
        }
        if not latest_price_base_date:
            return result

        master_map = self.fetch_stocks_master_map()

        try:
            price_rows = (
                self.client.table("normalized_stock_prices_daily")
                .select(
                    "symbol, base_date, open_price, high_price, low_price, close_price, "
                    "volume, trading_value, market_cap, outstanding_shares, available_at"
                )
                .eq("base_date", latest_price_base_date)
                .order("volume", desc=True)
                .limit(7000)
                .execute()
                .data
                or []
            )
        except Exception as exc:
            print(f"[WARNING] fetch_full_market_top_volume_stocks prices failed: {exc}")
            return result

        joined_rows = []
        covered = set()
        kospi_covered = set()
        kosdaq_covered = set()
        null_close_rows = 0

        for row in price_rows:
            canonical = canonicalize_symbol(row.get("symbol"))
            master = master_map.get(canonical)
            if not master:
                continue
            market = normalize_market_label(master.get("market"))
            symbol = canonical
            covered.add(symbol)
            if market == "KOSPI":
                kospi_covered.add(symbol)
            elif market == "KOSDAQ":
                kosdaq_covered.add(symbol)
            if row.get("close_price") is None:
                null_close_rows += 1
            asset_type = infer_asset_type(master.get("name"), market, symbol)
            joined_rows.append(
                {
                    **row,
                    "symbol": canonical,
                    "name": master.get("name") or symbol,
                    "market": market,
                    "asset_type": asset_type,
                    "canonical_symbol": canonical,
                }
            )

        result["coverage"] = {
            "covered_symbols": len(covered),
            "kospi_covered": len(kospi_covered),
            "kosdaq_covered": len(kosdaq_covered),
            "null_close_rows": null_close_rows,
        }
        result["rows"] = joined_rows[: max(limit, 5) * 200]
        return result

    def fetch_top_volume_stocks_by_market(self, limit=5, price_base_date: str | None = None):
        full_market = self.fetch_full_market_top_volume_stocks(limit=limit, price_base_date=price_base_date)
        latest_price_base_date = full_market.get("base_date")
        market_rows = full_market.get("rows") or []
        coverage = full_market.get("coverage") or {}

        result = {
            "base_date": latest_price_base_date,
            "coverage": coverage,
            "KOSPI": [],
            "KOSDAQ": [],
            "ETF_ETN": [],
            "duplicates": [],
            "excluded_invalid_rows": [],
            "candidate_counts_before": {"KOSPI": 0, "KOSDAQ": 0, "ETF_ETN": 0},
            "candidate_counts_after": {"KOSPI": 0, "KOSDAQ": 0, "ETF_ETN": 0},
            "empty_reasons": {"KOSPI": [], "KOSDAQ": [], "ETF_ETN": []},
        }

        deduped_rows, duplicates = deduplicate_by_canonical_symbol(market_rows)
        result["duplicates"] = duplicates
        valid_rows = []
        for row in deduped_rows:
            if has_minimum_top_data(row):
                valid_rows.append(row)
            else:
                result["excluded_invalid_rows"].append(
                    {
                        "symbol": row.get("symbol"),
                        "canonical_symbol": row.get("canonical_symbol"),
                        "name": row.get("name"),
                        "market": row.get("market"),
                        "asset_type": row.get("asset_type"),
                        "base_date": row.get("base_date"),
                        "close_price": row.get("close_price"),
                        "volume": row.get("volume"),
                        "trading_value": row.get("trading_value"),
                    }
                )

        for market_name in ("KOSPI", "KOSDAQ"):
            filtered = [
                row for row in valid_rows
                if row.get("market") == market_name and is_common_stock_top_eligible(row)
            ]
            result["candidate_counts_before"][market_name] = len(
                [row for row in deduped_rows if row.get("market") == market_name]
            )
            result["candidate_counts_after"][market_name] = len(filtered)
            result[market_name] = filtered[:limit]
            if not result[market_name]:
                if not result["candidate_counts_before"][market_name]:
                    result["empty_reasons"][market_name].append("selected price table에 해당 market row가 없음")
                elif not result["candidate_counts_after"][market_name]:
                    result["empty_reasons"][market_name].append("asset_type/common stock 필터 후 후보 0건")

        etf_candidates = [row for row in valid_rows if is_etf_etn_top_eligible(row)]
        try:
            master_rows = (
                self.client.table("stocks_master")
                .select("symbol, name, market")
                .execute()
                .data
                or []
            )
            master_map = {row["symbol"]: row for row in master_rows if row.get("symbol")}
            price_rows = (
                self.client.table("normalized_stock_prices_daily")
                .select(
                    "symbol, base_date, open_price, high_price, low_price, close_price, "
                    "volume, trading_value, market_cap, outstanding_shares, available_at"
                )
                .eq("base_date", latest_price_base_date)
                .order("volume", desc=True)
                .limit(7000)
                .execute()
                .data
                or []
            )
            supplemental = []
            for row in price_rows:
                master = master_map.get(row.get("symbol"), {})
                market = master.get("market")
                name = master.get("name") or row.get("symbol")
                asset_type = infer_asset_type(name, market, row.get("symbol"))
                candidate = {
                    **row,
                    "name": name,
                    "market": market,
                    "asset_type": asset_type,
                    "canonical_symbol": canonicalize_symbol(row.get("symbol")),
                }
                if is_etf_etn_top_eligible(candidate):
                    supplemental.append(candidate)
            supplemental, supplemental_duplicates = deduplicate_by_canonical_symbol(supplemental)
            result["duplicates"].extend(supplemental_duplicates)
            etf_candidates.extend([row for row in supplemental if has_minimum_top_data(row)])
        except Exception as exc:
            print(f"[WARNING] fetch_top_volume_stocks_by_market ETF supplement failed: {exc}")

        etf_candidates, etf_duplicates = deduplicate_by_canonical_symbol(etf_candidates)
        result["duplicates"].extend(etf_duplicates)
        result["candidate_counts_before"]["ETF_ETN"] = len(
            [row for row in deduped_rows if row.get("asset_type") in {"ETF", "ETN"}]
        )
        result["candidate_counts_after"]["ETF_ETN"] = len(etf_candidates)
        result["ETF_ETN"] = etf_candidates[:limit]
        if not result["ETF_ETN"]:
            if not result["candidate_counts_before"]["ETF_ETN"]:
                result["empty_reasons"]["ETF_ETN"].append("ETF/ETN 분류 후보 0건")
            elif not result["candidate_counts_after"]["ETF_ETN"]:
                result["empty_reasons"]["ETF_ETN"].append("유효 가격/거래량/거래대금 조건 통과 후보 0건")
        return result

    def fetch_report_readiness(self):
        latest_price_date = self._get_latest_base_date("normalized_stock_prices_daily")
        latest_macro_date = self._get_latest_base_date("normalized_global_macro_daily")
        latest_valid_price = self.get_latest_valid_price_date(latest_macro_date, lookback_days=7)
        watchlist_bundle = self.get_watchlist_snapshots(report_date=latest_macro_date)
        ranking_bundle = self.get_latest_market_rankings(report_date=latest_macro_date, limit=10)
        static_universe = self.fetch_static_stock_universe()
        static_enabled_count = len(static_universe)

        watchlist_snapshots = watchlist_bundle.get("snapshots") or []
        watchlist_hits = 0
        for snapshot in watchlist_snapshots:
            price = snapshot.get("price") or {}
            if price.get("close_price") not in (None, ""):
                watchlist_hits += 1
        watchlist_hit_ratio = (watchlist_hits / len(watchlist_snapshots)) if watchlist_snapshots else 0.0

        full_market = self.fetch_full_market_top_volume_stocks(limit=5, price_base_date=latest_valid_price.get("base_date"))
        coverage = full_market.get("coverage") or {}
        ranking_sections = ranking_bundle.get("sections") or {}
        ranking_diag = ranking_bundle.get("diagnostics") or {}
        ranking_ready = bool(ranking_sections.get("volume", {}).get("KOSPI")) and bool(ranking_sections.get("volume", {}).get("KOSDAQ")) and not ranking_diag.get("market_mismatch_rows") and not ranking_diag.get("q_prefix_rows")
        watchlist_ready = watchlist_hit_ratio >= 0.5

        recent_logs = []
        try:
            logs = (
                self.client.table("pipeline_run_logs")
                .select("job_name, target_date, status, records_processed, error_message")
                .in_(
                    "job_name",
                    [
                        "daily_stock_pipeline",
                        "daily_stock_full_price_pipeline",
                        "daily_ecos_macro_pipeline",
                        "daily_macro_pipeline",
                        "daily_derivatives_pipeline",
                        "daily_feature_generator",
                    ],
                )
                .order("target_date", desc=True)
                .limit(20)
                .execute()
                .data
                or []
            )
            recent_logs = logs
        except Exception as exc:
            print(f"[WARNING] fetch_report_readiness pipeline logs failed: {exc}")

        latest_full_price_processed = 0
        for log in recent_logs:
            if log.get("job_name") == "daily_stock_full_price_pipeline":
                latest_full_price_processed = log.get("records_processed") or 0
                break

        recent_problem_logs = [
            log for log in recent_logs
            if str(log.get("status", "")).upper() in {"WARN", "WARNING", "ERROR", "FAIL", "FAILED"}
        ]

        covered_symbols = coverage.get("covered_symbols") or 0
        minimum_report_ready = (
            bool(latest_macro_date)
            and bool(latest_valid_price.get("base_date"))
            and static_enabled_count > 0
            and watchlist_hits > 0
            and (
                bool(ranking_sections.get("volume", {}).get("KOSPI"))
                or bool(ranking_sections.get("volume", {}).get("KOSDAQ"))
                or bool(ranking_sections.get("trading_value", {}).get("KOSPI"))
                or bool(ranking_sections.get("trading_value", {}).get("KOSDAQ"))
            )
        )
        full_market_coverage_pass = covered_symbols > 0
        if ranking_bundle.get("ranking_status") == "정상":
            ranking_status = "정상"
        elif ranking_bundle.get("ranking_status") == "일부 fallback":
            ranking_status = "일부 fallback"
        else:
            ranking_status = "부족"

        if latest_valid_price.get("valid_rows", 0) > 0:
            price_status = "정상"
        elif latest_valid_price.get("base_date"):
            price_status = "일부 fallback"
        else:
            price_status = "부족"

        return {
            "latest_price_date": latest_price_date,
            "latest_macro_date": latest_macro_date,
            "latest_valid_price_date": latest_valid_price.get("base_date"),
            "minimum_report_ready": minimum_report_ready,
            "ranking_ready": ranking_ready,
            "watchlist_ready": watchlist_ready,
            "full_market_coverage_pass": full_market_coverage_pass,
            "coverage_status": "REFERENCE_ONLY",
            "price_coverage": coverage,
            "static_enabled_count": static_enabled_count,
            "recent_pipeline_logs": recent_logs,
            "recent_problem_logs": recent_problem_logs,
            "latest_full_price_records_processed": latest_full_price_processed,
            "ranking_base_date": ranking_bundle.get("ranking_base_date"),
            "ranking_status": ranking_status,
            "price_status": price_status,
            "watchlist_price_hit_ratio": watchlist_hit_ratio,
            "market_master_status": "정상" if not ranking_diag.get("market_mismatch_rows") else "점검 필요",
            "ranking_diagnostics": ranking_diag,
        }
