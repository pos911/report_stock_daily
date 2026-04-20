import time

import requests

from src.utils import config


NEWS_FETCH_TIMEOUT = 30
NEWS_FETCH_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
MAX_NEWS_ITEM_CHARS = 120
MAX_NEWS_CONTEXT_CHARS = 3000


def prepare_news_context(news_text: str) -> str:
    """
    Normalize Google Docs news text into a compact, loss-aware list for LLM input.
    Keeps every distinct paragraph/item, but removes excess whitespace and duplicates.
    """
    if not news_text:
        return ""

    text = news_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    items = []
    seen = set()
    current = []

    def flush():
        if not current:
            return
        item = " ".join(current)
        item = " ".join(item.split())
        item = item.strip(" -•*|\t")
        if not item:
            return
        normalized_key = item.lower()
        if normalized_key in seen:
            return
        seen.add(normalized_key)
        if len(item) > MAX_NEWS_ITEM_CHARS:
            item = item[: MAX_NEWS_ITEM_CHARS - 1].rstrip() + "…"
        items.append(item)

    for raw_line in text.split("\n"):
        line = " ".join(raw_line.strip().split())
        if not line:
            flush()
            current = []
            continue
        if (
            line.startswith(("갱신 시각:", "시간:", "유지 기준:"))
            or line.startswith("{\"type\":\"text\"")
            or line.startswith("📢")
            or set(line) <= {"_", "-", "=", " "}
        ):
            continue
        if not line.startswith("["):
            continue

        if current and not raw_line.startswith((" ", "\t", "-", "•", "*")):
            flush()
            current = []

        current.append(line)

    flush()

    if not items:
        fallback = " ".join(text.split())
        return fallback[:MAX_NEWS_CONTEXT_CHARS].rstrip() + (
            "…" if len(fallback) > MAX_NEWS_CONTEXT_CHARS else ""
        )

    max_item_chars = MAX_NEWS_ITEM_CHARS
    while max_item_chars >= 60:
        rendered_items = []
        for item in items:
            shortened = item
            if len(shortened) > max_item_chars:
                shortened = shortened[: max_item_chars - 1].rstrip() + "…"
            rendered_items.append(f"- {shortened}")

        rendered = "\n".join(rendered_items)
        if len(rendered) <= MAX_NEWS_CONTEXT_CHARS:
            return rendered
        max_item_chars -= 20

    rendered = "\n".join(f"- {item[:59].rstrip()}…" for item in items)
    if len(rendered) <= MAX_NEWS_CONTEXT_CHARS:
        return rendered

    return rendered[:MAX_NEWS_CONTEXT_CHARS].rstrip() + "\n- (뉴스 항목이 많아 헤드라인 중심으로 압축됨)"


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
