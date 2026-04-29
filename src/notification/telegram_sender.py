from typing import List

import requests

from src.utils import config


TELEGRAM_MSG_LIMIT = 4096
TELEGRAM_SEND_TIMEOUT = 30


class TelegramSender:
    def __init__(self):
        self.bot_token = config.get("bot_token", section="telegram")
        self.chat_id = config.get("chat_id", section="telegram")

        if not self.bot_token or not self.chat_id:
            raise ValueError(
                "Telegram bot_token and chat_id must be provided via config/api_keys.json or env vars."
            )

        self.api_base = f"https://api.telegram.org/bot{self.bot_token}"
        self.session = requests.Session()

    def _send_single_message(self, text: str) -> bool:
        url = f"{self.api_base}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        response = self.session.post(url, json=payload, timeout=TELEGRAM_SEND_TIMEOUT)
        if response.status_code == 200:
            return True

        print(f"Telegram API error: {response.status_code} - {response.text}")
        return False

    def send_report(self, report_text: str) -> bool:
        if not report_text or not report_text.strip():
            print("Empty report text, skipping Telegram send.")
            return False

        chunks = self._build_message_chunks(report_text)
        success = True
        for chunk in chunks:
            success = self._send_single_message(chunk) and success
        return success

    @classmethod
    def _build_message_chunks(cls, report_text: str) -> List[str]:
        normalized = cls._normalize_report_text(report_text)
        if len(normalized) <= TELEGRAM_MSG_LIMIT:
            return [normalized]
        return cls._split_text(normalized, TELEGRAM_MSG_LIMIT)

    @staticmethod
    def _normalize_report_text(report_text: str) -> str:
        text = report_text.replace("\r\n", "\n").strip()
        banned_fragments = ("Test_only", "섹션 수:", "Sections:")
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue
            if any(fragment in stripped for fragment in banned_fragments):
                continue
            lines.append(stripped)
        text = "\n".join(lines)
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text.strip()

    @staticmethod
    def _split_text(text: str, limit: int) -> List[str]:
        if len(text) <= limit:
            return [text]

        chunks: List[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining.strip())
                break

            split_pos = remaining.rfind("\n## ", 0, limit)
            if split_pos <= 0:
                split_pos = remaining.rfind("\n[", 0, limit)
            if split_pos <= 0:
                split_pos = remaining.rfind("\n", 0, limit)
            if split_pos <= 0:
                split_pos = limit

            chunks.append(remaining[:split_pos].strip())
            remaining = remaining[split_pos:].lstrip()

        return [chunk for chunk in chunks if chunk]

