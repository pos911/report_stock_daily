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

    def search_news(self, query: str, display: int = 10) -> list[dict]:
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
            seen_titles = set()
            for item in payload.get("items", []):
                title = self._clean_text(item.get("title"))
                description = self._clean_text(item.get("description"))
                if not title and not description:
                    continue
                dedupe_key = title or description
                if dedupe_key in seen_titles:
                    continue
                seen_titles.add(dedupe_key)
                items.append(
                    {
                        "title": title,
                        "description": description,
                        "link": item.get("link") or item.get("originallink"),
                        "originallink": item.get("originallink"),
                        "pubDate": item.get("pubDate"),
                    }
                )
            return items
        except Exception as exc:
            print(f"[WARNING] Naver news search failed for '{query}': {exc}")
            return []

    def search_queries(self, queries: list[str], display_per_query: int = 5, max_items: int = 5) -> list[dict]:
        collected: list[dict] = []
        seen = set()
        for query in queries:
            for item in self.search_news(query, display=display_per_query):
                dedupe_key = (
                    item.get("title"),
                    item.get("originallink") or item.get("link"),
                    item.get("pubDate"),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                item["query"] = query
                collected.append(item)
                if len(collected) >= max_items:
                    return collected
        return collected

    @staticmethod
    def _clean_text(value: str | None) -> str:
        if not value:
            return ""
        no_html = HTML_TAG_RE.sub("", value)
        return html.unescape(no_html).strip()

