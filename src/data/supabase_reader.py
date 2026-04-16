import os
import re
import json
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.utils import config

# ETF 판별 정규식 (요구사항 기준)
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
        """
        Fetches all (symbol, name) mappings from stocks_master and caches them.
        """
        try:
            response = self.client.table("stocks_master").select("symbol, name").execute()
            if response.data:
                return {item["symbol"]: item["name"] for item in response.data}
        except Exception as e:
            print(f"Error fetching stock name map: {e}")
        return {}

    def _fetch_and_ffill_timeseries(self, table_name, lookback_days=5):
        """
        Fetches the last N days of data from a table, applies forward fill,
        and returns the most recent record as a dictionary.
        """
        try:
            response = self.client.table(table_name).select("*").order("base_date", desc=True).limit(lookback_days).execute()
            if not response.data:
                return None
            
            df = pd.DataFrame(response.data)
            # Sort from oldest to newest for ffill
            df = df.sort_values("base_date")
            df = df.ffill()
            
            return df.iloc[-1].to_dict()
        except Exception as e:
            print(f"Error fetching/ffilling timeseries for {table_name}: {e}")
            return None

    def _inject_stock_names(self, data, name_map):
        """
        Injects stock names into the data list using a pre-fetched name map.
        Example: {'symbol': '005930'} becomes {'symbol': '005930', 'stock_name': '삼성전자'}
        """
        if not data:
            return []
        
        for item in data:
            symbol = item.get("symbol")
            if symbol and symbol in name_map:
                item["stock_name"] = name_map[symbol]
        return data

    def get_latest_date(self):
        """
        feature_store_daily 테이블에서 feature_name='volume' 기준으로
        가장 최신 base_date를 반환한다. (기준 날짜 단일 소스)
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
        Fetches the latest macro and market breadth data with weekend gap filling.
        매크로 테이블에서 단순 value 외에 _1d_chg, _5d_chg 변화율 피처를 모두 포함하여 반환.
        """
        results = {}

        # market_breadth_daily: 단일 최신 행 반환
        results["market_breadth_daily"] = self._fetch_and_ffill_timeseries("market_breadth_daily")

        # normalized_global_macro_daily: 모든 피처(value, _1d_chg, _5d_chg 포함) 조회
        try:
            macro_resp = (
                self.client
                .table("normalized_global_macro_daily")
                .select("*")
                .order("base_date", desc=True)
                .limit(10)
                .execute()
            )
            if macro_resp.data:
                df = pd.DataFrame(macro_resp.data).sort_values("base_date")
                df = df.ffill()
                results["normalized_global_macro_daily"] = df.iloc[-1].to_dict()
            else:
                results["normalized_global_macro_daily"] = None
        except Exception as e:
            print(f"[WARNING] fetch_macro_and_market_data 실패: {e}")
            results["normalized_global_macro_daily"] = None

        return results

    def fetch_top_volume_stocks(self, limit=10):
        """
        feature_store_daily 기준 최신 날짜(volume)로 거래량 상위 종목을 조회.
        - KOSPI, KOSDAQ, ETF 각각 상위 limit개 추출 (총 limit*3개)
        - ETF 판별: 종목명에 ETF_PATTERN 정규식 매칭 시 강제 분류
        """
        # 1. stocks_master에서 symbol → name, market 맵 구성
        try:
            resp = self.client.table("stocks_master").select("symbol, name, market").execute()
            market_map = {}
            name_map = {}
            if resp.data:
                for item in resp.data:
                    name = item.get("name", "")
                    market = item.get("market", "Unknown")
                    # 정규식 기반 ETF 강제 분류
                    if ETF_PATTERN.search(name):
                        market = "ETF"
                    market_map[item["symbol"]] = market
                    name_map[item["symbol"]] = name
        except Exception as e:
            print(f"[ERROR] stocks_master 조회 실패 (top volume): {e}")
            return {"KOSPI": [], "KOSDAQ": [], "ETF": []}

        # 2. feature_store_daily에서 최신 날짜 취득 (단일 소스)
        latest_date = self.get_latest_date()
        if not latest_date:
            print("[WARNING] fetch_top_volume_stocks: latest_date 조회 실패")
            return {"KOSPI": [], "KOSDAQ": [], "ETF": []}

        # 3. 해당 날짜의 volume 피처 전체 조회 후 float 캐스팅 + 내림차순 정렬
        try:
            vol_resp = (
                self.client
                .table("feature_store_daily")
                .select("symbol, feature_value")
                .eq("base_date", latest_date)
                .eq("feature_name", "volume")
                .execute()
            )
            if not vol_resp.data:
                print(f"[WARNING] fetch_top_volume_stocks: {latest_date} 기준 volume 데이터 없음")
                return {"KOSPI": [], "KOSDAQ": [], "ETF": []}

            # float 캐스팅 후 내림차순 정렬
            sorted_vols = sorted(
                vol_resp.data,
                key=lambda x: float(x.get("feature_value") or 0),
                reverse=True
            )
        except Exception as e:
            print(f"[ERROR] volume 피처 조회 실패: {e}")
            return {"KOSPI": [], "KOSDAQ": [], "ETF": []}

        # 4. 시장별 상위 limit개 분류
        result = {"KOSPI": [], "KOSDAQ": [], "ETF": []}
        for item in sorted_vols:
            sym = item["symbol"]
            market = market_map.get(sym)
            if market not in result:
                continue
            if len(result[market]) < limit:
                result[market].append({
                    "symbol": sym,
                    "stock_name": name_map.get(sym, "Unknown"),
                    "market": market,
                    "volume": float(item["feature_value"]) if item.get("feature_value") is not None else 0.0,
                    "base_date": latest_date,
                })
            # 세 카테고리 모두 꽉 찼으면 조기 종료
            if all(len(v) >= limit for v in result.values()):
                break

        return result

    def fetch_target_stocks_data(self, target_symbols):
        """
        타겟 종목들의 분석 데이터를 수집.
        - normalized_stock_fundamentals_ratios에서 market_cap, per, pbr 포함
        - feature_store_daily에서 _1d_chg, _5d_chg 포함 전체 피처 조회
        """
        if not target_symbols:
            return {}
            
        name_map = self._fetch_stock_name_map()
        results = {}

        # 기준 날짜: feature_store_daily volume 기준 단일 소스
        latest_date = self.get_latest_date()
        if not latest_date:
            print("[WARNING] fetch_target_stocks_data: latest_date 조회 실패, 데이터 수집 불가")
            return {}

        # 분석 테이블 순회 (fundamentals에서 market_cap/per/pbr 포함)
        stock_analysis_tables = [
            "normalized_stock_short_selling",
            "normalized_stock_fundamentals_ratios",  # market_cap, per, pbr 포함
            "normalized_stock_supply_daily",
        ]
        
        for table in stock_analysis_tables:
            try:
                data_query = (
                    self.client
                    .table(table)
                    .select("*")
                    .eq("base_date", latest_date)
                    .in_("symbol", target_symbols)
                    .execute()
                )
                results[table] = self._inject_stock_names(data_query.data, name_map)
            except Exception as e:
                print(f"[WARNING] {table} 조회 실패: {e}")
                results[table] = []

        # feature_store_daily: 전체 피처(value, _1d_chg, _5d_chg 등) 조회
        try:
            feature_query = (
                self.client
                .table("feature_store_daily")
                .select("*")
                .eq("base_date", latest_date)
                .in_("symbol", target_symbols)
                .execute()
            )
            results["feature_store_daily"] = self._inject_stock_names(feature_query.data, name_map)
        except Exception as e:
            print(f"[WARNING] feature_store_daily 조회 실패: {e}")
            results["feature_store_daily"] = []

        return results
