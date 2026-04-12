import os
import json
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

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
        Fetches the latest data from 3 tables:
        - market_breadth_daily
        - normalized_global_macro_daily
        - feature_store_daily
        Returns a dictionary containing the results.
        """
        tables = [
            "market_breadth_daily",
            "normalized_global_macro_daily",
            "feature_store_daily"
        ]
        results = {}

        for table in tables:
            # Most tables in this project use 'base_date' for the daily grain
            response = self.client.table(table).select("*").order("base_date", desc=True).limit(1).execute()
            
            if response.data:
                results[table] = response.data[0]
            else:
                results[table] = None
        
        return results
