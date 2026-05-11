from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

try:
    from google import genai
except Exception:  # pragma: no cover - import guard
    genai = None

from src.utils.config_loader import config


logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("GEMINI_INTERPRETATION_MODEL", "gemini-2.0-flash")
FORBIDDEN_TERMS = (
    "BUY",
    "SELL",
    "HOLD",
    "전체시장 거래대금 Top",
    "전체시장 시총 Top",
    "외국인 선물 순매수",
    "프로그램 매매",
    "실시간 뉴스",
    "공시 발생",
    "목표가",
    "매수 추천",
    "매도 추천",
    "매수추천",
    "매도추천",
)
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")


class GeminiInterpretationError(RuntimeError):
    pass


def _collect_allowed_number_tokens(payload: Any) -> set[str]:
    tokens: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, (int, float)):
            numeric = float(value)
            tokens.add(str(int(numeric)) if numeric.is_integer() else str(numeric))
            tokens.add(f"{numeric:.1f}")
            tokens.add(f"{numeric:.2f}")
            if abs(numeric) <= 1:
                pct = numeric * 100
                tokens.add(f"{pct:.1f}%")
                tokens.add(f"{pct:.2f}%")
            else:
                tokens.add(f"{numeric:.0f}")
        elif isinstance(value, str):
            stripped = value.strip()
            if stripped:
                for token in NUMBER_PATTERN.findall(stripped):
                    tokens.add(token.replace(",", ""))

    visit(payload)
    return {token for token in tokens if token}


def _extract_json_payload(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise GeminiInterpretationError("empty Gemini response")
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise GeminiInterpretationError("Gemini response is not valid JSON")
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise GeminiInterpretationError(f"Gemini JSON parse failed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise GeminiInterpretationError("Gemini response root must be an object")
    return parsed


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", str(text or "").strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _contains_forbidden_term(text: str) -> bool:
    upper = text.upper()
    return any(term.upper() in upper for term in FORBIDDEN_TERMS)


def _sentence_has_unknown_number(sentence: str, allowed_numbers: set[str]) -> bool:
    for token in NUMBER_PATTERN.findall(sentence):
        normalized = token.replace(",", "")
        if normalized and normalized not in allowed_numbers:
            return True
    return False


def _sanitize_text_value(text: str, allowed_numbers: set[str]) -> str:
    kept: list[str] = []
    for sentence in _split_sentences(text):
        if _contains_forbidden_term(sentence):
            continue
        if _sentence_has_unknown_number(sentence, allowed_numbers):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def _sanitize_payload(payload: Any, allowed_numbers: set[str]) -> Any:
    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            sanitized = _sanitize_payload(value, allowed_numbers)
            if sanitized in ("", None, [], {}):
                continue
            cleaned[key] = sanitized
        return cleaned
    if isinstance(payload, list):
        cleaned_list = []
        for value in payload:
            sanitized = _sanitize_payload(value, allowed_numbers)
            if sanitized in ("", None, [], {}):
                continue
            cleaned_list.append(sanitized)
        return cleaned_list
    if isinstance(payload, str):
        return _sanitize_text_value(payload, allowed_numbers)
    return payload


def _build_prompt(session: str, context: dict[str, Any]) -> str:
    session_label = {"morning": "Morning", "regular": "Regular", "closing": "Closing"}[session]
    output_schema = {
        "morning": {
            "scenario_summary": "오늘 장전 시나리오에 대한 1~2문장",
            "aggressive_view": "공격적 관점 1~2문장",
            "conservative_view": "보수적 관점 1~2문장",
            "must_watch": ["장초반 확인 조건 1", "장초반 확인 조건 2"],
        },
        "regular": {
            "view_vs_actual_status": "유지|일부 유지|약화|확인 제한",
            "view_vs_actual_reason": "오전 View와 장중 KIS 거래량/관심종목 반응을 비교한 1~2문장",
            "kis_volume_interpretation": ["종목별 해석 1"],
            "watchlist_comments": {"000660": "종목별 코멘트"},
            "next_checkpoints": ["오후 체크포인트 1"],
        },
        "closing": {
            "market_review_status": "추세 지속|단기 이벤트|혼재|확인 제한",
            "market_review_reason": "오늘 움직임이 지속 가능한지에 대한 1~2문장",
            "key_drivers": ["핵심 키워드 1"],
            "watchlist_review": {"000660": "종목별 마감 진단"},
            "tomorrow_strategy": {
                "aggressive_condition": "공격적 조건",
                "conservative_condition": "보수적 조건",
                "must_check": ["필수 확인 데이터 1"],
            },
        },
    }[session]
    instructions = [
        "당신은 한국 주식시장 Morning/Regular/Closing 리포트 분석가다.",
        "입력 JSON 외의 사실을 만들지 마라.",
        "숫자, 가격, 등락률, rank, score는 새로 만들지 마라.",
        "KIS 거래량 순위는 전체시장 Top이 아니라 KIS API 기반 후보군이다.",
        "현재 데이터에 없는 외국인 선물, 프로그램 매매, 실시간 뉴스, 공시를 언급하지 마라.",
        "투자 추천이 아니라 관찰 조건과 리스크 기준을 제시하라.",
        "각 종목별로 서로 다른 해석을 작성하라.",
        "같은 문장을 반복하지 마라.",
        "응답은 JSON만 반환하라.",
        "BUY / SELL / HOLD / 목표가 / 매수추천 / 매도추천 금지.",
    ]
    return (
        f"{session_label} 리포트 해석 레이어를 작성하라.\n"
        + "\n".join(f"- {line}" for line in instructions)
        + "\n\n출력 JSON 스키마 예시:\n"
        + json.dumps(output_schema, ensure_ascii=False, indent=2)
        + "\n\n입력 JSON:\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


class GeminiInterpretationClient:
    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = api_key or config.get("api_key", section="gemini") or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or DEFAULT_MODEL
        if not self.api_key:
            raise GeminiInterpretationError("Gemini API key not available")
        if genai is None:
            raise GeminiInterpretationError("google genai client not available")
        self.client = genai.Client(api_key=self.api_key)

    def _call(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self.model_name, contents=prompt)
        text = getattr(response, "text", None)
        if not text:
            raise GeminiInterpretationError("empty Gemini text response")
        return text

    def generate(self, session: str, context: dict[str, Any]) -> dict[str, Any]:
        prompt = _build_prompt(session, context)
        parsed = _extract_json_payload(self._call(prompt))
        sanitized = _sanitize_payload(parsed, _collect_allowed_number_tokens(context))
        if not sanitized:
            raise GeminiInterpretationError("sanitized Gemini payload empty")
        return sanitized


def generate_morning_analysis_insight(context: dict[str, Any], client: GeminiInterpretationClient | None = None) -> dict[str, Any]:
    engine = client or GeminiInterpretationClient()
    return engine.generate("morning", context)


def generate_regular_analysis_insight(context: dict[str, Any], client: GeminiInterpretationClient | None = None) -> dict[str, Any]:
    engine = client or GeminiInterpretationClient()
    return engine.generate("regular", context)


def generate_closing_analysis_insight(context: dict[str, Any], client: GeminiInterpretationClient | None = None) -> dict[str, Any]:
    engine = client or GeminiInterpretationClient()
    return engine.generate("closing", context)
