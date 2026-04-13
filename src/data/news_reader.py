import os
import requests
from dotenv import load_dotenv

load_dotenv()

def fetch_news_document():
    """
    Downloads text from a specific Google Docs export URL.
    The URL is expected to be provided via the GOOGLE_DOCS_NEWS_URL environment variable.
    """
    url = os.getenv("GOOGLE_DOCS_NEWS_URL")
    if not url:
        print("Error: GOOGLE_DOCS_NEWS_URL environment variable is not set.")
        return ""

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching news document: {e}")
        return ""
