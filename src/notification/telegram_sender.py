import os
import json
import requests

# Telegram sendMessage has a 4096 character limit
TELEGRAM_MSG_LIMIT = 4096


from src.utils import config

class TelegramSender:
    def __init__(self):
        """
        Initialize Telegram Bot credentials using the unified config loader.
        Priority: 1. Environment Variables, 2. config/api_keys.json
        """
        self.bot_token = config.get("bot_token", section="telegram")
        self.chat_id = config.get("chat_id", section="telegram")

        if not self.bot_token or not self.chat_id:
            raise ValueError(
                "Telegram bot_token and chat_id must be provided "
                "via config/api_keys.json or env vars."
            )

        self.api_base = f"https://api.telegram.org/bot{self.bot_token}"

    def _send_single_message(self, text, parse_mode="Markdown"):
        """
        Sends a single message via Telegram Bot API sendMessage.
        https://core.telegram.org/bots/api#sendmessage
        """
        url = f"{self.api_base}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return True
        else:
            print(f"Telegram API error: {response.status_code} - {response.text}")
            # Retry without parse_mode (in case markdown breaks the API)
            payload.pop("parse_mode", None)
            retry = requests.post(url, json=payload, timeout=30)
            if retry.status_code == 200:
                return True
            print(f"Telegram retry also failed: {retry.status_code} - {retry.text}")
            return False

    def send_report(self, report_text):
        """
        Sends a full report to Telegram, chunking if it exceeds the 4096 char limit.
        """
        if not report_text:
            print("Empty report text, skipping Telegram send.")
            return False

        chunks = self._split_text(report_text, TELEGRAM_MSG_LIMIT)
        print(f"Sending report to Telegram ({len(chunks)} message(s))...")

        success = True
        for i, chunk in enumerate(chunks, 1):
            ok = self._send_single_message(chunk)
            if ok:
                print(f"  Part {i}/{len(chunks)} sent successfully.")
            else:
                print(f"  Part {i}/{len(chunks)} FAILED.")
                success = False
        return success

    @staticmethod
    def _split_text(text, limit):
        """
        Splits text into chunks of at most `limit` characters,
        preferring to break at newline boundaries for cleaner messages.
        """
        if len(text) <= limit:
            return [text]

        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break

            # Find the last newline within the limit
            split_pos = text.rfind("\n", 0, limit)
            if split_pos == -1:
                # No newline found — hard cut
                split_pos = limit

            chunks.append(text[:split_pos])
            text = text[split_pos:].lstrip("\n")

        return chunks
