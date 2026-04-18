import re
from typing import List

import requests

from src.utils import config


TELEGRAM_MSG_LIMIT = 4096
TELEGRAM_SEND_TIMEOUT = 30
CHUNK_OVERHEAD = 128


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
        self.session = requests.Session()

    def _send_single_message(self, text: str) -> bool:
        """
        Sends a single plain-text message via Telegram Bot API sendMessage.
        Plain text avoids parse_mode-related failures from LLM output.
        """
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
        """
        Sends a formatted report to Telegram, chunking by section when needed.
        """
        if not report_text or not report_text.strip():
            print("Empty report text, skipping Telegram send.")
            return False

        chunks = self._build_message_chunks(report_text)
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

    @classmethod
    def _build_message_chunks(cls, report_text: str) -> List[str]:
        normalized = cls._normalize_report_text(report_text)
        title, generated_at, sections = cls._extract_sections(normalized)

        intro_lines = [title]
        if generated_at:
            intro_lines.append(f"Generated at: {generated_at}")
        intro_lines.append(f"Sections: {len(sections)}")
        intro = "\n".join(intro_lines).strip()

        if not sections:
            return cls._split_text(intro + "\n\n" + normalized, TELEGRAM_MSG_LIMIT)

        max_body_limit = max(512, TELEGRAM_MSG_LIMIT - CHUNK_OVERHEAD)
        section_blocks = []
        for heading, body in sections:
            block = heading if not body else f"{heading}\n{body}"
            section_blocks.extend(cls._split_text(block, max_body_limit))

        total_parts = len(section_blocks)
        chunks = []
        for idx, block in enumerate(section_blocks, 1):
            prefix = f"[Report Part {idx}/{total_parts}]"
            if idx == 1:
                chunk = f"{intro}\n\n{prefix}\n{block}".strip()
            else:
                chunk = f"{prefix}\n{block}".strip()
            chunks.append(chunk)

        return chunks

    @staticmethod
    def _normalize_report_text(report_text: str) -> str:
        text = report_text.replace("\r\n", "\n").strip()
        text = re.sub(r"(?m)^###\s+", "", text)
        text = re.sub(r"(?m)^##\s+", "", text)
        text = re.sub(r"(?m)^#\s+", "", text)
        replacements = (
            ("> **Generated at**:", "Generated at:"),
            (">", ""),
            ("---", ""),
            ("**", ""),
            ("`", ""),
        )
        for src, dest in replacements:
            text = text.replace(src, dest)

        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _extract_sections(text: str):
        lines = text.splitlines()
        title = lines[0].strip() if lines else "Daily Quant Report"
        generated_at = ""
        sections = []

        current_heading = None
        current_body = []

        for line in lines[1:]:
            stripped = line.strip()

            if stripped.startswith("Generated at:"):
                generated_at = stripped.split("Generated at:", 1)[1].strip()
                continue

            if re.match(r"^\d+\.\s", stripped):
                if current_heading is not None:
                    sections.append((current_heading, "\n".join(current_body).strip()))
                current_heading = stripped
                current_body = []
                continue

            if current_heading is None:
                continue

            current_body.append(line.rstrip())

        if current_heading is not None:
            sections.append((current_heading, "\n".join(current_body).strip()))

        return title, generated_at, sections

    @staticmethod
    def _split_text(text: str, limit: int) -> List[str]:
        """
        Splits text into chunks of at most `limit` characters.
        Priority:
        1. Double newline
        2. Single newline
        3. Sentence boundary
        4. Hard cut
        """
        if len(text) <= limit:
            return [text]

        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining.strip())
                break

            split_pos = remaining.rfind("\n\n", 0, limit)
            if split_pos == -1:
                split_pos = remaining.rfind("\n", 0, limit)
            if split_pos == -1:
                split_pos = remaining.rfind(". ", 0, limit)
            if split_pos == -1 or split_pos < int(limit * 0.4):
                split_pos = limit

            chunk = remaining[:split_pos].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_pos:].lstrip("\n ")

        return chunks
