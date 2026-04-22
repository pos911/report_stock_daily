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
            # 실제 존재하는 컬럼 위주로 조회 (kospi/kosdaq은 macro_series에서 주로 가져옴)
            req_cols = (
                "base_date, nasdaq, sp500, usdkrw, dxy, us10y, wti, vix, sox"
            )
            results["normalized_global_macro_daily"] = self._fetch_latest_row_by_date(
                "normalized_global_macro_daily",
                latest_global_macro_date,
                select_cols=req_cols
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
