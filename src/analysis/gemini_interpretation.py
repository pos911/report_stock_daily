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
    "뉴스",
    "관련 뉴스",
    "실적 발표 관련 뉴스",
    "공시",
    "수주",
    "계약",
    "외국인 투자자",
    "외국인 선물",
    "프로그램 매매",
    "목표가",
    "매수",
    "매도",
    "BUY",
    "SELL",
    "HOLD",
    "전체시장 거래대금 Top",
    "전체시장 시총 Top",
    "매수 추천",
    "매도 추천",
    "매수추천",
    "매도추천",
    "기술적 반등",
    "업황 및 경쟁사 동향",
    "특정 이슈",
    "단기 전략",
    "시세 차익",
    "연관성",
    "연관성을 확인",
    "연관성을 확인해 볼 필요",
    "변동 요인",
)
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
ANCHOR_PATTERN = re.compile(
    r"(KIS|거래량|거래대금|상승|약화|유지|과열|관찰|확인 제한|리스크|USD/KRW|Nasdaq|SOX|VIX|Brent|WTI|DXY|환율|금리|유가|반도체|2차전지|조선|증권|소재|점수|score|label|라벨|관심종목|ETF)",
    re.IGNORECASE,
)


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


def _build_anchor_terms(context: dict[str, Any]) -> set[str]:
    terms = {
        "KIS",
        "거래량",
        "거래대금",
        "상승",
        "유지",
        "과열",
        "관찰",
        "확인 제한",
        "리스크",
        "USD/KRW",
        "Nasdaq",
        "SOX",
        "VIX",
        "Brent",
        "WTI",
        "DXY",
        "점수",
        "score",
        "label",
        "라벨",
        "관심종목",
        "ETF",
    }
    for section in ("watchlist", "kis_volume_top", "sector_etfs"):
        for row in context.get(section) or []:
            for key in ("symbol", "name", "sector_group", "signal_label"):
                value = str(row.get(key) or "").strip()
                if value:
                    terms.add(value)
    return terms


def _anchor_rule_for_path(path: tuple[str, ...]) -> set[str] | None:
    if not path:
        return None
    root = path[0]
    if root == "view_vs_actual_status":
        return {"유지", "약화", "확인 제한"}
    if root == "market_review_status":
        return {"추세", "단기 이벤트", "혼재", "확인 제한"}
    if root == "key_drivers":
        return {"KIS", "거래량", "거래대금", "환율", "금리", "유가", "반도체", "2차전지", "조선", "증권"}
    if root in {"scenario_summary", "aggressive_view", "conservative_view"}:
        return {"USD/KRW", "Nasdaq", "SOX", "VIX", "Brent", "WTI", "DXY", "거래대금", "ETF", "관심종목", "환율", "금리", "유가"}
    if root in {"must_watch", "next_checkpoints"}:
        return {"KIS", "거래량", "거래대금", "USD/KRW", "SOX", "Nasdaq", "ETF", "관심종목", "확인 제한"}
    if root in {"kis_volume_interpretation"}:
        return {"KIS", "거래량", "순위", "rank", "거래대금", "유지"}
    if root in {"watchlist_comments", "watchlist_review"}:
        return {"거래대금", "상승", "과열", "확인 제한", "KIS", "점수", "score", "label", "라벨", "관찰", "리스크", "유지"}
    if root in {"view_vs_actual_reason", "market_review_reason"}:
        return {"KIS", "거래량", "관심종목", "거래대금", "유지", "약화", "확인 제한"}
    if root in {"aggressive_condition", "conservative_condition", "must_check"}:
        return {"KIS", "거래량", "거래대금", "USD/KRW", "SOX", "Nasdaq", "ETF", "관심종목"}
    return None


def _has_anchor_term(sentence: str, anchor_terms: set[str], required_terms: set[str] | None = None) -> bool:
    normalized = sentence.strip()
    if not normalized:
        return False
    if required_terms and not any(term and term in normalized for term in required_terms):
        return False
    if any(term and term in normalized for term in anchor_terms):
        return True
    return bool(ANCHOR_PATTERN.search(normalized))


def _sanitize_text_value(text: str, allowed_numbers: set[str], anchor_terms: set[str], path: tuple[str, ...]) -> str:
    kept: list[str] = []
    required_terms = _anchor_rule_for_path(path)
    root = path[0] if path else ""
    for sentence in _split_sentences(text):
        if _contains_forbidden_term(sentence):
            continue
        if root in {"scenario_summary", "aggressive_view", "conservative_view", "must_watch"}:
            if "KIS 거래량" in sentence or "거래량 1위" in sentence:
                continue
        if _sentence_has_unknown_number(sentence, allowed_numbers):
            continue
        if not _has_anchor_term(sentence, anchor_terms, required_terms):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def _sanitize_payload(payload: Any, allowed_numbers: set[str], anchor_terms: set[str], path: tuple[str, ...] = ()) -> Any:
    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            sanitized = _sanitize_payload(value, allowed_numbers, anchor_terms, path + (str(key),))
            if sanitized in ("", None, [], {}):
                continue
            cleaned[key] = sanitized
        return cleaned
    if isinstance(payload, list):
        cleaned_list = []
        for index, value in enumerate(payload):
            sanitized = _sanitize_payload(value, allowed_numbers, anchor_terms, path + (str(index),))
            if sanitized in ("", None, [], {}):
                continue
            cleaned_list.append(sanitized)
        return cleaned_list
    if isinstance(payload, str):
        return _sanitize_text_value(payload, allowed_numbers, anchor_terms, path)
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
        "현재 데이터에 없는 뉴스, 공시, 수주, 계약, 외국인 투자자 동향, 외국인 선물, 프로그램 매매를 언급하지 마라.",
        "투자 추천이 아니라 관찰 조건과 리스크 기준을 제시하라.",
        "Morning에서는 단기 전략, 시세차익, 추격 매수 같은 매매성 표현을 쓰지 마라.",
        "Morning에서는 당일 KIS 거래량 후보만으로 시나리오 근거를 만들지 마라.",
        "각 종목별 해석은 반드시 입력 데이터 중 최소 1개와 직접 연결하라.",
        "거래대금 유지 여부, 상승 지속·과열 부담, 현재 데이터로는 확인 제한, KIS 거래량 상위와의 연결성, score/label 기반 리스크 중 하나는 반드시 포함하라.",
        "관련 뉴스 참고, 업황 확인, 특정 이슈, 기술적 반등, 연관성을 확인해 볼 필요 같은 일반론만 쓰지 마라.",
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
        sanitized = _sanitize_payload(parsed, _collect_allowed_number_tokens(context), _build_anchor_terms(context))
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
