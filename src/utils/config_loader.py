import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env if it exists
load_dotenv()

class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Loads config from api_keys.json if it exists."""
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "config" / "api_keys.json"
        
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
                self._config = {}
        else:
            self._config = {}

    def get(self, key, section=None, default=None):
        """
        Retrieves a configuration value.
        Priority: 1. Environment Variable, 2. JSON Config, 3. Default
        
        Example: get("api_key", section="gemini") 
        Will look for:
        1. env: GEMINI_API_KEY
        2. json: config['gemini']['api_key']
        3. json: config['gemini_api_key'] 
        """
        # 1. Environment Variable lookup
        env_key = f"{section.upper()}_{key.upper()}" if section else key.upper()
        env_val = os.getenv(env_key)
        if env_val:
            return env_val

        # 2. JSON Config lookup
        if self._config:
            # Nested lookup
            if section and section in self._config and isinstance(self._config[section], dict):
                val = self._config[section].get(key)
                if val:
                    return val
            
            # Root level fallback (e.g. "supabase_url")
            root_key = f"{section}_{key}" if section else key
            val = self._config.get(root_key)
            if val:
                return val

        return default

# Singleton instance for easy access
config = ConfigLoader()
