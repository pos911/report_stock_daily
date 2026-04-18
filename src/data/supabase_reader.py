import os
import re
import json
import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.utils import config

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

    def _fetch_and_ffill_timeseries(self, table_name, lookback_days=5):
        """
        Fetches the last N days of data from a table, applies forward fill,
        and returns the most recent record as a dictionary.
        """
        try:
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

    def _fetch_latest_row_by_date(self, table_name: str, base_date: str):
        """지정한 base_date 행을 조회하고 forward fill 후 마지막 레코드를 반환한다."""
        if not base_date:
            return None
        try:
            resp = (
                self.client.table(table_name)
                .select("*")
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

    def get_latest_date(self):
        """
        feature_store_daily 테이블에서 feature_name='volume' 기준으로
        가장 최신 base_date를 반환. (전체 파이프라인의 기준 날짜 단일 소스)
        """
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
        except Exception as e:
            print(f"[WARNING] get_latest_date 실패: {e}")
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

        # 1. normalized_macro_series
        try:
            latest_macro_date = self._get_latest_base_date("normalized_macro_series")
            results["normalized_macro_series"] = self._fetch_latest_row_by_date(
                "normalized_macro_series",
                latest_macro_date,
            )
        except Exception as e:
            print(f"[WARNING] normalized_macro_series 조회 실패: {e}")
            results["normalized_macro_series"] = None

        # 1-b. normalized_global_macro_daily (consumer spec 핵심)
        try:
            latest_global_macro_date = self._get_latest_base_date("normalized_global_macro_daily")
            results["normalized_global_macro_daily"] = self._fetch_latest_row_by_date(
                "normalized_global_macro_daily",
                latest_global_macro_date,
            )
        except Exception as e:
            print(f"[WARNING] normalized_global_macro_daily 조회 실패: {e}")
            results["normalized_global_macro_daily"] = None

        # 2. market_breadth_daily
        try:
            latest_breadth_date = self._get_latest_base_date("market_breadth_daily")
            results["market_breadth_daily"] = self._fetch_latest_row_by_date(
                "market_breadth_daily",
                latest_breadth_date,
            )
        except Exception as e:
            print(f"[WARNING] market_breadth_daily 조회 실패: {e}")
            results["market_breadth_daily"] = None

        # 3. momentum 피처: feature_store_daily에서 symbol='GLOBAL'인 데이터 추출
        try:
            latest_date = self.get_latest_date()
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
        kst_today = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date()
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
            latest_date = self._get_latest_base_date(table_name)
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
            date_rows = (
                self.client.table("normalized_stock_prices_daily")
                .select("base_date")
                .order("base_date", desc=True)
                .limit(12)
                .execute()
            )
            distinct_dates = []
            for row in date_rows.data or []:
                d = row.get("base_date")
                if d and d not in distinct_dates:
                    distinct_dates.append(d)
                if len(distinct_dates) >= 2:
                    break

            if len(distinct_dates) >= 2:
                latest_d, prev_d = distinct_dates[0], distinct_dates[1]
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
            logs_resp = (
                self.client.table("pipeline_run_logs")
                .select("job_name, target_date, status, records_processed, error_message")
                .order("target_date", desc=True)
                .limit(120)
                .execute()
            )
            for row in logs_resp.data or []:
                status = (row.get("status") or "").upper()
                if status in {"WARN", "FAIL", "FAILED", "ERROR"}:
                    log_alerts.append(row)
            log_alerts = log_alerts[:20]
        except Exception as e:
            print(f"[WARNING] pipeline_run_logs 조회 실패: {e}")

        return {
            "as_of_kst_date": kst_today.isoformat(),
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
        latest_date = self.get_latest_date()
        if not latest_date:
            return {"KOSPI": [], "KOSDAQ": [], "ETF": []}

        # 3. volume 상위 추출
        try:
            vol_resp = (
                self.client.table("feature_store_daily")
                .select("symbol, feature_value")
                .eq("base_date", latest_date)
                .eq("feature_name", "volume")
                .execute()
            )
            sorted_vols = sorted(vol_resp.data, key=lambda x: float(x.get("feature_value") or 0), reverse=True)
        except Exception as e:
            print(f"[ERROR] volume 조회 실패: {e}")
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
        타겟 종목 분석 데이터 수집 (Fundamentals Fallback 적용).
        """
        if not target_symbols:
            return {}

        name_map = self._fetch_stock_name_map()
        results = {}
        latest_date = self.get_latest_date()
        if not latest_date:
            return {}

        # 1. Fundamentals (Fallback 적용)
        try:
            try:
                fund_data = (
                    self.client.table("normalized_stock_fundamentals_ratios")
                    .select("symbol, per, pbr, market_cap, base_date")
                    .eq("base_date", latest_date)
                    .in_("symbol", target_symbols)
                    .execute()
                )
            except Exception:
                fund_data = (
                    self.client.table("normalized_stock_fundamentals_ratios")
                    .select("symbol, per, pbr, base_date")
                    .eq("base_date", latest_date)
                    .in_("symbol", target_symbols)
                    .execute()
                )
            results["normalized_stock_fundamentals_ratios"] = self._inject_stock_names(fund_data.data, name_map)
        except Exception as e:
            print(f"[WARNING] Fundamentals 조회 실패: {e}")
            results["normalized_stock_fundamentals_ratios"] = []

        # 2. 기타 분석 테이블
        other_tables_map = {
            "normalized_stock_short_selling": "*",
            "normalized_stock_supply_daily": "symbol, base_date, individual_net_buy, foreign_net_buy, institutional_net_buy, pension_net_buy, corporate_net_buy"
        }
        for table, select_query in other_tables_map.items():
            try:
                data_query = self.client.table(table).select(select_query).eq("base_date", latest_date).in_("symbol", target_symbols).execute()
                results[table] = self._inject_stock_names(data_query.data, name_map)
            except Exception as e:
                print(f"[WARNING] {table} 조회 실패: {e}")
                results[table] = []

        # 3. feature_store_daily
        try:
            feature_query = self.client.table("feature_store_daily").select("*").eq("base_date", latest_date).in_("symbol", target_symbols).execute()
            results["feature_store_daily"] = self._inject_stock_names(feature_query.data, name_map)
        except Exception as e:
            print(f"[WARNING] feature_store_daily 조회 실패: {e}")
            results["feature_store_daily"] = []

        return results
