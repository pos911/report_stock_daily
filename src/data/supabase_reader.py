import os
import json
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.utils import config

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

    def fetch_macro_and_market_data(self):
        """
        Fetches the latest macro and market breadth data with weekend gap filling.
        """
        results = {}
        market_tables = [
            "normalized_global_macro_daily",
            "market_breadth_daily"
        ]
        for table in market_tables:
            results[table] = self._fetch_and_ffill_timeseries(table)
        return results

    def fetch_top_volume_stocks(self, limit=5):
        """
        Fetches top volume stocks for KOSPI, KOSDAQ, and ETF based on the latest available data.
        """
        try:
            # Build market map
            resp = self.client.table("stocks_master").select("symbol, name, market").execute()
            market_map = {}
            name_map = {}
            if resp.data:
                for item in resp.data:
                    market_map[item["symbol"]] = item.get("market", "Unknown")
                    name_map[item["symbol"]] = item["name"]
        except Exception as e:
            print(f"Error fetching stock master for top volume: {e}")
            return {"KOSPI": [], "KOSDAQ": [], "ETF": []}

        try:
            # Get latest base_date from a reliable table like fundamentals or breadth
            date_query = self.client.table("normalized_stock_fundamentals_ratios").select("base_date").order("base_date", desc=True).limit(1).execute()
            if not date_query.data:
                return {"KOSPI": [], "KOSDAQ": [], "ETF": []}
            latest_date = date_query.data[0]["base_date"]

            # Assuming volume is in feature_store_daily under "volume" (or "acc_trdvol")
            vol_query = self.client.table("feature_store_daily").select("symbol, feature_value").eq("base_date", latest_date).eq("feature_name", "volume").execute()
            
            result = {"KOSPI": [], "KOSDAQ": [], "ETF": []}
            if not vol_query.data:
                return result
                
            sorted_vols = sorted(vol_query.data, key=lambda x: float(x.get("feature_value", 0)), reverse=True)
            for item in sorted_vols:
                sym = item["symbol"]
                market = market_map.get(sym)
                # Group by market
                if market in result and len(result[market]) < limit:
                    result[market].append({
                        "symbol": sym,
                        "stock_name": name_map.get(sym, "Unknown"),
                        "volume": item["feature_value"]
                    })
            return result
        except Exception as e:
            print(f"Error fetching top volume stocks: {e}")
            return {"KOSPI": [], "KOSDAQ": [], "ETF": []}

    def fetch_target_stocks_data(self, target_symbols):
        """
        Fetches analysis tables specifically for target_symbols.
        """
        if not target_symbols:
            return {}
            
        name_map = self._fetch_stock_name_map()
        results = {}
        
        stock_analysis_tables = [
            "normalized_stock_short_selling",
            "normalized_stock_fundamentals_ratios",
            "normalized_stock_supply_daily"
        ]
        
        for table in stock_analysis_tables:
            try:
                date_query = self.client.table(table).select("base_date").order("base_date", desc=True).limit(1).execute()
                if date_query.data:
                    latest_date = date_query.data[0]["base_date"]
                    data_query = self.client.table(table).select("*").eq("base_date", latest_date).in_("symbol", target_symbols).execute()
                    results[table] = self._inject_stock_names(data_query.data, name_map)
                else:
                    results[table] = []
            except Exception as e:
                print(f"Error fetching {table} for target stocks: {e}")
                results[table] = []

        try:
            date_query = self.client.table("feature_store_daily").select("base_date").order("base_date", desc=True).limit(1).execute()
            if date_query.data:
                latest_date = date_query.data[0]["base_date"]
                feature_query = self.client.table("feature_store_daily").select("*").eq("base_date", latest_date).in_("symbol", target_symbols).execute()
                results["feature_store_daily"] = self._inject_stock_names(feature_query.data, name_map)
            else:
                results["feature_store_daily"] = []
        except Exception as e:
            print(f"Error fetching feature_store_daily for target stocks: {e}")
            results["feature_store_daily"] = []

        return results
