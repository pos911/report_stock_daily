import html
import os
import re

import requests


NAVER_NEWS_ENDPOINT = "https://openapi.naver.com/v1/search/news.json"
HTML_TAG_RE = re.compile(r"<[^>]+>")


class NaverNewsService:
    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.enabled = bool(self.client_id and self.client_secret)
        self.session = requests.Session()

    def search_news(self, query: str, display: int = 3) -> list[dict]:
        if not self.enabled or not query:
            return []

        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
        params = {
            "query": query,
            "display": max(1, min(display, 10)),
            "sort": "date",
        }
        try:
            response = self.session.get(
                NAVER_NEWS_ENDPOINT,
                headers=headers,
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            items = []
            for item in payload.get("items", []):
                title = self._clean_text(item.get("title"))
                description = self._clean_text(item.get("description"))
                if not title and not description:
                    continue
                items.append(
                    {
                        "title": title,
                        "description": description,
                        "originallink": item.get("originallink"),
                        "pubDate": item.get("pubDate"),
                    }
                )
            return items
        except Exception as exc:
            print(f"[WARNING] Naver news search failed for '{query}': {exc}")
            return []

    @staticmethod
    def _clean_text(value: str | None) -> str:
        if not value:
            return ""
        no_html = HTML_TAG_RE.sub("", value)
        return html.unescape(no_html).strip()

