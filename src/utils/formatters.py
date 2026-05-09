from __future__ import annotations

import math
from collections.abc import Iterable


NA_TEXT = "미확인"


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return value == ""


def safe_float(value) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_change_rate(value) -> float | None:
    numeric = safe_float(value)
    if numeric is None:
        return None
    if abs(numeric) > 1:
        return numeric / 100
    return numeric


def format_date(value) -> str:
    return NA_TEXT if is_missing(value) else str(value)


def format_index(value) -> str:
    numeric = safe_float(value)
    return NA_TEXT if numeric is None else f"{numeric:,.2f}"


def format_number(value, digits: int = 2) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return NA_TEXT
    return f"{numeric:,.{digits}f}".rstrip("0").rstrip(".")


def format_plain_number(value, digits: int = 2) -> str:
    return format_number(value, digits=digits)


def format_pct(value, digits: int = 2) -> str:
    numeric = safe_change_rate(value)
    if numeric is None:
        return NA_TEXT
    return f"{numeric:+.{digits}%}"


def format_percent(value) -> str:
    return format_pct(value)


def format_rate_percent(value) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return NA_TEXT
    return f"{numeric:.2f}%"


def format_rate_level(value) -> str:
    return format_rate_percent(value)


def format_bp(value) -> str:
    numeric = safe_float(value)
    return NA_TEXT if numeric is None else f"{numeric:+.1f}bp"


def format_spread_bp(value) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return NA_TEXT
    return f"{numeric * 100:+.1f}bp"


def format_yield_spread(value, change_bp=None) -> str:
    spread_text = format_spread_bp(value)
    if change_bp is None:
        return spread_text
    return f"{spread_text} / 전일대비 {format_bp(change_bp)}"


def format_usdkrw(value) -> str:
    numeric = safe_float(value)
    return NA_TEXT if numeric is None else f"1달러 = {numeric:,.2f}원"


def format_volume(value) -> str:
    numeric = safe_float(value)
    return NA_TEXT if numeric is None else f"{int(round(numeric)):,}주"


def format_price(value) -> str:
    numeric = safe_float(value)
    return NA_TEXT if numeric is None else f"{int(round(numeric)):,}원"


def format_trading_value(value) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return NA_TEXT
    eok = numeric / 100_000_000
    return f"{eok:,.0f}억원"


def format_market_cap(value) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return NA_TEXT
    if numeric >= 1_000_000_000_000:
        return f"{numeric / 1_000_000_000_000:.1f}조원"
    return f"{numeric / 100_000_000:,.0f}억원"


def format_outstanding_shares(value) -> str:
    numeric = safe_float(value)
    return NA_TEXT if numeric is None else f"{int(round(numeric)):,}주"


def format_flow_amount(value) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return NA_TEXT
    eok = numeric / 100_000_000
    action = "순매수" if eok >= 0 else "순매도"
    return f"{action} {abs(eok):,.0f}억원"


def format_flow_generic(value) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return NA_TEXT
    action = "순매수" if numeric >= 0 else "순매도"
    return f"{action} {abs(numeric):,.0f}"


def format_multiple(value, suffix: str) -> str:
    numeric = safe_float(value)
    return NA_TEXT if numeric is None else f"{numeric:.1f}{suffix}"


def format_signed_multiple(value, suffix: str) -> str:
    numeric = safe_float(value)
    return NA_TEXT if numeric is None else f"{numeric:+.2f}{suffix}"


def format_ratio_metric(value, suffix: str = "%") -> str:
    numeric = safe_float(value)
    return NA_TEXT if numeric is None else f"{numeric:.1f}{suffix}"


def format_sections_list(values: Iterable[str] | None) -> str:
    cleaned = [str(value).strip() for value in (values or []) if str(value).strip()]
    return ", ".join(cleaned) if cleaned else "없음"


def detect_market_value_anomaly(label: str, value: float | int | str | None) -> str | None:
    numeric = safe_float(value)
    if numeric is None:
        return None
    if numeric <= 0:
        return "일부 지수·종목 가격은 원천 스케일 확인이 필요합니다."
    return None


def detect_stock_price_anomaly(
    symbol: str | None,
    name: str | None,
    price: float | int | str | None,
    *,
    previous_price: float | int | str | None = None,
    data_quality_flag: str | None = None,
    source_consistency_status: str | None = None,
    source_mixed: bool | None = None,
) -> str | None:
    numeric = safe_float(price)
    if numeric is None:
        return None
    if str(data_quality_flag or "").upper() == "SOURCE_MIXED":
        return "일부 지수·종목 가격은 원천 스케일 확인이 필요합니다."
    if str(source_consistency_status or "").upper().startswith("SOURCE_MIXED"):
        return "일부 지수·종목 가격은 원천 스케일 확인이 필요합니다."
    if bool(source_mixed):
        return "일부 지수·종목 가격은 원천 스케일 확인이 필요합니다."
    if numeric <= 0:
        return "일부 지수·종목 가격은 원천 스케일 확인이 필요합니다."
    previous_numeric = safe_float(previous_price)
    if previous_numeric not in (None, 0):
        jump_ratio = abs((numeric / previous_numeric) - 1)
        if jump_ratio >= 0.5:
            return "일부 지수·종목 가격은 원천 스케일 확인이 필요합니다."
    return None


def unique_warnings(messages: Iterable[str | None], limit: int | None = None) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for message in messages:
        text = str(message or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
        if limit is not None and len(results) >= limit:
            break
    return results


def clean_sentence(text: str) -> str:
    value = " ".join(str(text or "").strip().split())
    if not value:
        return ""
    if value.endswith((".", "!", "?")):
        return value
    return f"{value}."


def join_sentences(parts: Iterable[str], limit: int | None = None) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        sentence = clean_sentence(part)
        if not sentence or sentence in seen:
            continue
        seen.add(sentence)
        cleaned.append(sentence)
        if limit is not None and len(cleaned) >= limit:
            break
    return " ".join(cleaned)


def add_section(lines: list[str], number: int, title: str, body: Iterable[str]) -> int:
    body_lines = [str(line) for line in body if str(line).strip()]
    if not body_lines:
        return number
    if lines:
        lines.append("")
    lines.append(f"{number}. {title}")
    lines.extend(body_lines)
    return number + 1
