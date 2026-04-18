import time

import requests

from src.utils import config


NEWS_FETCH_TIMEOUT = 30
NEWS_FETCH_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def fetch_news_document():
    """
    Downloads text from a specific Google Docs export URL using the unified config loader.
    Priority: 1. Environment Variable, 2. config/api_keys.json
    """
    url = config.get("news_url", section="google_docs")

    if not url:
        print("Warning: Google Docs news URL not found in api_keys.json or env var. Skipping news.")
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
