import re
from typing import List

import requests

from src.utils import config


TELEGRAM_MSG_LIMIT = 4096
TELEGRAM_SEND_TIMEOUT = 30
CHUNK_OVERHEAD = 128
SECTION_TITLE_MAP = {
    "0. Data Quality Guardrails": "0. 데이터 품질 점검",
    "1. Market Summary": "1. 시장 요약",
    "2. Top Volume & Smart Money": "2. 거래대금 상위/수급 포착",
    "3. Stock Analysis & Strategy": "3. 종목 분석 및 전략",
}
TEXT_REPLACEMENTS = (
    ("Daily Quant Report", "데일리 퀀트 리포트"),
    ("Regular Report", "정규 리포트"),
    ("Morning Briefing", "오전 브리핑"),
    ("Closing Analysis", "마감 분석"),
    ("Generated at:", "생성 시각:"),
    ("Sections:", "섹션 수:"),
    ("As of (KST):", "기준 시각:"),
    ("Table Freshness (lag_days)", "테이블 최신성(지연 일수)"),
    ("Zero-Volume Guardrail", "거래량 0 종목 점검"),
    ("Pipeline Alerts (recent 3 days)", "최근 3일 파이프라인 경고"),
    ("lag_days=", "지연 "),
    ("Risk-On", "위험자산 선호"),
    ("Risk-Off", "안전자산 선호"),
    ("BUY", "매수"),
    ("HOLD", "보유"),
    ("SELL", "매도"),
    ("WARN", "경고"),
    ("FAILED", "실패"),
    ("FAIL", "실패"),
    ("records=", "처리건수 "),
    ("error=", "오류 "),
    ("Report 작성 시간:", "생성 시각:"),
    ("🔴", "1) 공격 포인트"),
    ("🔵", "2) 보수 포인트"),
    ("⚖️", "3) 최종 결론"),
    ("⚖", "3) 최종 결론"),
    ("긍정적인 포인트:", ""),
    ("보수적인 포인트:", ""),
    ("최종 결론:", ""),
    ("Zero volume guardrail", "거래량 0 종목 점검"),
    ("base_date=", "기준일 "),
)


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
        normalized = cls._humanize_text(normalized)
        title, generated_at, sections = cls._extract_sections(normalized)

        intro_lines = [title]
        if generated_at:
            intro_lines.append(f"생성 시각: {generated_at}")
        intro_lines.append(f"섹션 수: {len(sections)}")
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
            prefix = f"[리포트 {idx}/{total_parts}]"
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

    @classmethod
    def _humanize_text(cls, text: str) -> str:
        lines = text.splitlines()
        humanized_lines = []

        for line in lines:
            updated = line.strip()
            if not updated:
                humanized_lines.append("")
                continue

            if updated.startswith(("Report 작성 시간:", "생성 시각:")):
                continue

            updated = cls._humanize_line(updated)
            if updated:
                humanized_lines.append(updated)

        result = "\n".join(humanized_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    @classmethod
    def _humanize_line(cls, line: str) -> str:
        updated = line

        if updated in SECTION_TITLE_MAP:
            updated = SECTION_TITLE_MAP[updated]

        for src, dest in TEXT_REPLACEMENTS:
            updated = updated.replace(src, dest)

        if "데일리 퀀트 리포트" in updated:
            if "정규 리포트" in updated:
                updated = "데일리 퀀트 리포트 - 정규 리포트"
            elif "오전 브리핑" in updated:
                updated = "데일리 퀀트 리포트 - 오전 브리핑"
            elif "마감 분석" in updated:
                updated = "데일리 퀀트 리포트 - 마감 분석"
            else:
                updated = "데일리 퀀트 리포트"

        updated = cls._format_datetime_strings(updated)
        updated = cls._format_numeric_tokens(updated)
        updated = cls._cleanup_line(updated)
        return updated

    @staticmethod
    def _format_datetime_strings(text: str) -> str:
        text = re.sub(
            r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})(?::\d{2}(?:\.\d+)?)?\+09:00",
            r"\1 \2 (KST)",
            text,
        )
        text = re.sub(
            r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})(?::\d{2})?Z",
            r"\1 \2 (UTC)",
            text,
        )
        return text

    @classmethod
    def _format_numeric_tokens(cls, text: str) -> str:
        text = cls._format_ratio_keywords(text)

        def repl(match):
            token = match.group(0)
            start = match.start()
            end = match.end()
            prev_char = text[start - 1] if start > 0 else ""
            next_char = text[end] if end < len(text) else ""

            if prev_char == "-" or next_char == "-":
                return token
            if prev_char == "(" and next_char == ")" and re.fullmatch(r"\d{6}", token):
                return token

            suffix = ""
            core = token
            if core.endswith("%"):
                suffix = "%"
                core = core[:-1]

            try:
                value = float(core.replace(",", ""))
            except ValueError:
                return token

            if suffix == "%":
                formatted = cls._format_percent_value(value)
            elif prev_char == "(" and next_char == ")":
                formatted = f"{int(round(value)):,}" if float(value).is_integer() else cls._trim_decimal(value, digits=2)
            elif abs(value) >= 1000 or float(value).is_integer():
                formatted = f"{int(round(value)):,}"
            else:
                formatted = cls._trim_decimal(value, digits=2)

            return f"{formatted}{suffix if suffix and '%' not in formatted else ''}"

        return re.sub(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?", repl, text)

    @staticmethod
    def _format_percent_value(value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    @classmethod
    def _format_ratio_keywords(cls, text: str) -> str:
        pattern = re.compile(
            r"((?:수익률|변화율|등락률|상승률|하락률)\s*)([-+]?\d+\.\d+)(?![%\d])"
        )

        def repl(match):
            label = match.group(1)
            value = float(match.group(2))
            if abs(value) <= 1:
                formatted = f"{value * 100:.2f}".rstrip("0").rstrip(".")
                return f"{label}{formatted}%"
            return match.group(0)

        return pattern.sub(repl, text)

    @staticmethod
    def _trim_decimal(value: float, digits: int = 2) -> str:
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")

    @staticmethod
    def _cleanup_line(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"지연 N/A", "지연 확인 불가", text)
        text = text.replace("? ? ?", "")
        text = text.replace("? ?", "")
        text = re.sub(r"\s*-\s*(\d)", r" -\1", text)
        text = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", text)
        text = re.sub(r"\(\s*-\s*(\d[\d,]*)\)", r"(-\1)", text)
        text = re.sub(r"\(\s*(\d[\d,]*)\s*\)", r"(\1)", text)
        text = re.sub(r"1\)\s*공격 포인트\s*", "1) 공격 포인트: ", text)
        text = re.sub(r"2\)\s*보수 포인트\s*", "2) 보수 포인트: ", text)
        text = re.sub(r"3\)\s*최종 결론\s*", "3) 최종 결론: ", text)
        text = re.sub(r":\s*:", ": ", text)
        text = re.sub(r"\s+\)", ")", text)
        return text

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
