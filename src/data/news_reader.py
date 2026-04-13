import os
import json
import requests


def fetch_news_document(config_path="config/api_keys.json"):
    """
    Downloads text from a specific Google Docs export URL.
    Priority: 1. api_keys.json, 2. GOOGLE_DOCS_NEWS_URL env var
    """
    url = None

    # 1. Try api_keys.json first
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            url = config.get("google_docs", {}).get("news_url")

    # 2. Fallback to env var
    if not url:
        url = os.getenv("GOOGLE_DOCS_NEWS_URL")

    if not url:
        print("Warning: Google Docs news URL not found in api_keys.json or env var. Skipping news.")
        return ""

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching news document: {e}")
        return ""
