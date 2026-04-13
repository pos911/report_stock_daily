import os
import json
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

class SupabaseReader:
    def __init__(self, config_path="config/api_keys.json"):
        """
        Initialize Supabase client by reading from environment variables or a JSON file.
        Priority: 
        1. Environment Variables (for GitHub Actions)
        2. Local JSON config file
        """
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")

        if not self.url or not self.key:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    # Check for nested 'supabase' key or root keys
                    supabase_config = config.get("supabase", {})
                    self.url = self.url or supabase_config.get("url") or config.get("supabase_url")
                    self.key = self.key or supabase_config.get("service_role_key") or config.get("supabase_key")

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

    def fetch_latest_data(self):
        """
        Fetches the latest data with weekend gap filling and full feature counts.
        Expanded to include stock names and additional analysis tables.
        """
        results = {}
        
        # Build stock name map first
        name_map = self._fetch_stock_name_map()

        # 1. Market-level Time-series (ffill for weekend/holidays)
        market_tables = [
            "normalized_global_macro_daily",
            "market_breadth_daily"
        ]
        for table in market_tables:
            results[table] = self._fetch_and_ffill_timeseries(table)

        # 2. Stock-level Analysis Tables (Latest Snapshot)
        stock_analysis_tables = [
            "normalized_stock_short_selling",
            "normalized_stock_fundamentals_ratios",
            "normalized_stock_supply_daily"
        ]
        
        for table in stock_analysis_tables:
            try:
                # Find the latest date in the table
                date_query = self.client.table(table).select("base_date").order("base_date", desc=True).limit(1).execute()
                if date_query.data:
                    latest_date = date_query.data[0]["base_date"]
                    # Fetch all rows for that date
                    data_query = self.client.table(table).select("*").eq("base_date", latest_date).execute()
                    results[table] = self._inject_stock_names(data_query.data, name_map)
                else:
                    results[table] = []
            except Exception as e:
                print(f"Error fetching {table}: {e}")
                results[table] = []

        # 3. Feature Store (Detailed Quant Features)
        try:
            date_query = self.client.table("feature_store_daily").select("base_date").order("base_date", desc=True).limit(1).execute()
            if date_query.data:
                latest_date = date_query.data[0]["base_date"]
                feature_query = self.client.table("feature_store_daily").select("*").eq("base_date", latest_date).execute()
                results["feature_store_daily"] = self._inject_stock_names(feature_query.data, name_map)
            else:
                results["feature_store_daily"] = []
        except Exception as e:
            print(f"Error fetching feature_store_daily: {e}")
            results["feature_store_daily"] = []

        return results
