import requests

def fetch_news_document():
    """
    Downloads text from a specific Google Docs export URL.
    """
    url = "https://docs.google.com/document/export?format=txt&id=1J5iXVQssP45ASr1vqF9bGR7On9JD7ZhWE8GhEtFJb_4"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching news document: {e}")
        return ""
