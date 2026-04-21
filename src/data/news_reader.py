import time
import re
from datetime import datetime, timedelta, timezone
import requests
from src.utils import config

NEWS_FETCH_TIMEOUT = 60
NEWS_FETCH_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3
MAX_NEWS_ITEM_CHARS = 150
MAX_NEWS_CONTEXT_CHARS = 3000

def prepare_news_context(news_text: str) -> str:
    if not news_text:
        return ""

    text = news_text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    
    items = []
    seen = set()
    current = []
    keep_current = True

    def flush():
        nonlocal current, keep_current
        if not current or not keep_current:
            current = []
            keep_current = True
            return
        item = " ".join(current)
        item = " ".join(item.split()).strip(" -•*|\t")
        if not item:
            return
        normalized_key = item.lower()
        if normalized_key in seen:
            return
        seen.add(normalized_key)
        if len(item) > MAX_NEWS_ITEM_CHARS:
            item = item[: MAX_NEWS_ITEM_CHARS - 1].rstrip() + "…"
        items.append(item)
        current = []
        keep_current = True

    for raw_line in text.split("\n"):
        line = " ".join(raw_line.strip().split())
        if not line:
            flush()
            continue
            
        # 불필요한 메타데이터 스킵
        if line.startswith(("갱신 시각:", "시간:", "유지 기준:")) or line.startswith('{"type":"text"') or line.startswith("📢"):
            continue

        # 새로운 뉴스 항목 시작
        if line.startswith("["):
            flush()
            # 12시간 이내 필터링 로직 (예: [04.21 14:30] 또는 [14:30])
            time_match = re.search(r'\[.*?(\d{1,2})[:/.-](\d{2}).*?\]', line)
            if time_match:
                hour, minute = int(time_match.group(1)), int(time_match.group(2))
                try:
                    target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if target_time > now:
                        target_time -= timedelta(days=1)
                    if (now - target_time) > timedelta(hours=12):
                        keep_current = False # 12시간 지났으면 버림 (Drop)
                except ValueError:
                    pass
            
        if current and not raw_line.startswith((" ", "\t", "-", "•", "*")):
            flush()

        current.append(line)

    flush()

    if not items:
        return ""

    rendered = "\n".join(f"- {item}" for item in items)
    return rendered[:MAX_NEWS_CONTEXT_CHARS]


def fetch_news_document():
    url = config.get("news_url", section="google_docs")
    if not url:
        print("Warning: Google Docs news URL not found. Skipping news.")
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
