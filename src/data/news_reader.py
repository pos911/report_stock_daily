import os
import json
import requests


from src.utils import config

def fetch_news_document():
    """
    Downloads text from a specific Google Docs export URL using the unified config loader.
    Priority: 1. Environment Variable, 2. config/api_keys.json
    """
    url = config.get("news_url", section="google_docs")

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
