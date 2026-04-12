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

    def fetch_latest_data(self):
        """
        Fetches the latest data with weekend gap filling and full feature counts.
        """
        results = {}

        # 1. Macro and Market Breadth (ffill for weekend/holidays)
        for table in ["normalized_global_macro_daily", "market_breadth_daily"]:
            # Fetch last 5 rows to ensure we capture the most recent Friday if it's currently Sunday/Monday
            response = self.client.table(table).select("*").order("base_date", desc=True).limit(5).execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                # Ensure past-to-future order for forward fill
                df = df.sort_values("base_date")
                # Forward fill missing values (captures weekday data to holiday/weekend slots)
                df = df.ffill()
                # Take the latest row (the intended target day)
                results[table] = df.iloc[-1].to_dict()
            else:
                results[table] = None

        # 2. Feature Store (Collect all symbols for the latest date)
        # First, find the latest date
        date_query = self.client.table("feature_store_daily").select("base_date").order("base_date", desc=True).limit(1).execute()
        
        if date_query.data:
            latest_date = date_query.data[0]["base_date"]
            # Fetch ALL rows for that specific date (no limit)
            feature_query = self.client.table("feature_store_daily").select("*").eq("base_date", latest_date).execute()
            results["feature_store_daily"] = feature_query.data if feature_query.data else []
        else:
            results["feature_store_daily"] = []

        return results
