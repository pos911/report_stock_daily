from __future__ import annotations

from collections.abc import Iterable

from src.utils.formatters import (
    NA_TEXT,
    clean_sentence,
    format_bp,
    format_date,
    format_number,
    format_pct,
    format_price,
    format_sections_list,
    join_sentences,
    safe_change_rate,
    safe_float,
)


DISPLAY_MODE_TEXT = {
    "FULL_MARKET": "FULL_MARKET",
    "KIS_UNIVERSE_ONLY": "KIS_UNIVERSE_ONLY",
    "MACRO_ONLY": "MACRO_ONLY",
}

ALLOWED_SECTION_TEXT = {
    "macro": "macro",
    "us_market": "us_market",
    "kis_volume_top": "kis_volume_top",
    "watchlist_signal": "watchlist_signal",
    "etf_etn": "etf_etn",
    "kr_full_market_trading_value_top": "kr_full_market_trading_value_top",
    "kr_full_market_market_cap_top": "kr_full_market_market_cap_top",
}

BLOCKED_SECTION_TEXT = {
    "kr_full_market_trading_value_top": "전체시장 거래대금 Top",
    "kr_full_market_market_cap_top": "전체시장 시총 Top",
    "kis_volume_top": "KIS 거래량 순위",
    "watchlist_signal": "관심종목 Signal",
    "etf_etn": "ETF/ETN 참고",
}

SECTOR_NAME_MAP = {
    "반도체": "반도체",
    "2차전지": "2차전지",
    "조선": "조선",
    "방산": "방산",
    "금융/증권": "금융/증권",
    "바이오/헬스케어": "바이오/헬스케어",
    "AI전력/인프라": "AI전력/인프라",
    "자동차": "자동차",
    "화장품/소비재": "화장품/소비재",
    "정유화학": "정유화학",
    "원자력": "원자력",
    "반도체 장비": "반도체 장비",
    "인터넷/플랫폼": "인터넷/플랫폼",
    "통신": "통신",
    "게임": "게임",
    "전력/유틸리티": "전력/유틸리티",
    "우주항공": "우주항공",
}

STOCK_NAME_MAP = {
    "000660": "SK하이닉스",
    "005930": "삼성전자",
    "012330": "현대모비스",
    "071050": "한국금융지주",
    "278470": "에이피알",
    "058470": "리노공업",
    "047810": "한국항공우주",
    "012450": "한화에어로스페이스",
    "017670": "SK텔레콤",
    "015760": "한국전력",
}


def build_data_status_section(freshness: dict, readiness: dict, contract_failed_views: list[str] | None = None) -> list[str]:
    allowed = readiness.get("allowed_korean_sections") or []
    blocked = readiness.get("blocked_korean_sections") or []
    lines = [
        f"- 기준일: {format_date(freshness.get('target_date'))}",
        f"- 한국장: {_market_open_text(freshness.get('xkrx_is_open'))}",
        f"- 미국장: {_market_open_text(freshness.get('xnys_is_open'))}",
        f"- 국내 데이터 모드: {DISPLAY_MODE_TEXT.get(readiness.get('display_mode'), readiness.get('display_mode') or '미확인')}",
        f"- 사용 가능: {format_sections_list(_translate_allowed_sections(allowed))}",
        f"- 생략: {format_sections_list(_translate_blocked_sections(blocked))}",
    ]

    note_parts: list[str] = []
    limitation = readiness.get("data_limitation_note")
    if limitation:
        note_parts.append(limitation)
    if freshness.get("carry_forward_fields"):
        note_parts.append("미국 일부 지표는 직전 거래일 기준으로 해석합니다")
    if freshness.get("stale_warnings"):
        note_parts.append(_translate_stale_warning(freshness.get("stale_warnings")))
    if contract_failed_views:
        note_parts.append(f"대체 기준 데이터 사용: {', '.join(contract_failed_views)}")

    if note_parts:
        lines.append(f"- 참고: {join_sentences(note_parts, limit=2)}")
    return lines[:7]


def build_one_line_judgment_section(regime: dict, top_sectors: list[dict], freshness: dict, readiness: dict) -> list[str]:
    market_tone = _market_tone_text(regime)
    lead_sector = _display_sector_name((top_sectors[0] if top_sectors else {}).get("sector_group"))
    second_sector = _display_sector_name((top_sectors[1] if len(top_sectors) > 1 else {}).get("sector_group"))
    sector_text = lead_sector if not second_sector else f"{lead_sector}와 {second_sector}"
    global_driver = _global_driver_summary(regime)
    risk_text = _risk_summary(top_sectors, regime)
    readiness_note = readiness.get("data_limitation_note")

    trimmed_global = _strip_terminal_period(global_driver)
    trimmed_risk = _strip_terminal_period(risk_text or readiness_note or "")

    if freshness.get("xkrx_is_open") is False:
        summary = join_sentences(
            [
                f"한국장은 휴장이라 다음 거래일 준비 관점에서는 {sector_text or '주요 테마'} 점검이 우선입니다",
                f"{trimmed_global} 국내 대응은 관심종목·랭킹 후보 기준으로 선별 확인이 적절합니다",
                trimmed_risk,
            ],
            limit=3,
        )
    else:
        summary = join_sentences(
            [
                f"오늘 한국장은 {market_tone} 쪽에 가깝습니다",
                f"{trimmed_global} {sector_text or '주요 테마'} 중심의 선별 대응이 유리합니다",
                trimmed_risk,
            ],
            limit=3,
        )
    return [summary]


def build_global_market_section(macro: dict) -> list[str]:
    return [
        _market_line("S&P500", macro.get("sp500"), macro.get("sp500_change_value"), macro.get("sp500_change_rate"), _interpret_equity(macro.get("sp500_change_rate"), "미국 대형주 심리")),
        _market_line("Nasdaq", macro.get("nasdaq"), macro.get("nasdaq_change_value"), macro.get("nasdaq_change_rate"), _interpret_equity(macro.get("nasdaq_change_rate"), "성장주 심리")),
        _market_line("SOX", macro.get("sox"), macro.get("sox_change_value"), macro.get("sox_change_rate"), _interpret_equity(macro.get("sox_change_rate"), "국내 반도체 심리")),
        _market_line("VIX", macro.get("vix"), macro.get("vix_change_value"), macro.get("vix_change_rate"), _interpret_vix(macro.get("vix"), macro.get("vix_change_rate"))),
        _market_line("USD/KRW", macro.get("usdkrw"), macro.get("usdkrw_change_value"), macro.get("usdkrw_change_rate"), _interpret_fx(macro.get("usdkrw_change_rate"))),
        _market_line("DXY", macro.get("dxy"), macro.get("dxy_change_value"), macro.get("dxy_change_rate"), _interpret_dxy(macro.get("dxy_change_rate"))),
        _rate_line("US10Y", macro.get("us10y"), macro.get("us10y_change_bp"), _interpret_rate(macro.get("us10y_change_bp"), "성장주 밸류에이션")),
        _rate_line("US3Y", macro.get("us3y"), macro.get("us3y_change_bp"), _interpret_rate(macro.get("us3y_change_bp"), "단기 정책금리 기대")),
        _rate_line("10Y-3Y spread", macro.get("us10y_us3y_spread"), macro.get("us10y_us3y_spread_change_bp"), _interpret_spread(macro.get("us10y_us3y_spread"))),
        _rate_line("KR10Y", macro.get("kr10y"), macro.get("kr10y_change_bp"), "국내 금리 민감 업종 점검"),
        _market_line("Brent", macro.get("brent"), macro.get("brent_change_value"), macro.get("brent_change_rate"), _interpret_oil(macro.get("brent_change_rate"))),
        _market_line("WTI", macro.get("wti"), macro.get("wti_change_value"), macro.get("wti_change_rate"), _interpret_oil(macro.get("wti_change_rate"))),
    ]


def build_korean_impact_section(top_sectors: list[dict], freshness: dict, readiness: dict) -> list[str]:
    lines: list[str] = []
    if readiness.get("display_mode") != "FULL_MARKET":
        lines.append(f"- {readiness.get('data_limitation_note')}")
    if freshness.get("xkrx_is_open") is False:
        lines.append("- 한국장 휴장일이라 국내 해석은 다음 거래일 준비 관점으로 압축합니다.")
    for row in top_sectors[:3]:
        sector_name = _display_sector_name(row.get("sector_group"))
        reason = join_sentences(
            [
                row.get("global_reason"),
                _summarize_etf_status(row),
            ],
            limit=2,
        )
        lines.append(f"- {sector_name}: {row.get('label')} / {reason}")
    return lines


def build_priority_themes_section(top_sectors: list[dict], freshness: dict, readiness: dict) -> list[str]:
    lines: list[str] = []
    point_label = "장중 체크포인트" if freshness.get("xkrx_is_open") else "다음 거래일 확인 포인트"
    for index, row in enumerate(top_sectors[:3], 1):
        lines.append(f"{index}순위 {_display_sector_name(row.get('sector_group'))}")
        lines.append(f"- 판단 라벨: {row.get('label')}")
        lines.append(f"- 글로벌 근거: {join_sentences([row.get('global_reason')], limit=1) or '매크로 해석 중심으로 관찰합니다.'}")
        lines.append(f"- ETF/관심종목 근거: {_theme_evidence_text(row, readiness)}")
        investor_reason = row.get("investor_reason")
        if investor_reason and investor_reason != "Investor flow unavailable":
            lines.append(f"- 수급 또는 거래대금 근거: {clean_sentence(investor_reason)}")
        else:
            lines.append("- 수급 또는 거래대금 근거: 관심종목·랭킹 후보 중심으로 거래대금 흐름을 확인합니다.")
        risk_text = join_sentences([row.get("risk")], limit=1) or "과열 여부를 함께 점검합니다."
        lines.append(f"- 리스크: {risk_text}")
        lines.append(f"- {point_label}: {_checkpoint_text(row.get('intraday_checkpoints') or [], freshness)}")
    return lines


def build_watchlist_section(watchlist_scores: list[dict], freshness: dict) -> list[str]:
    lines: list[str] = []
    display_count = min(5, len(watchlist_scores))
    remaining = max(len(watchlist_scores) - display_count, 0)
    lines.append(f"- 관심종목 {len(watchlist_scores)}개 중 주요 {display_count}개 표시. 나머지 {remaining}개는 snapshot에 저장합니다.")
    point_label = "장중 체크포인트" if freshness.get("xkrx_is_open") else "다음 거래일 확인 포인트"
    decision_label = "Signal label" if freshness.get("xkrx_is_open") else "다음 거래일 참고 판단"
    for row in watchlist_scores[:display_count]:
        name = _display_stock_name(row.get("symbol"), row.get("name"))
        lines.append(f"{name}({row.get('symbol')})")
        lines.append(f"- {decision_label}: {row.get('signal_label')}")
        lines.append(f"- 핵심 근거: {_bullet_join(row.get('quant_reasons') or [], limit=2)}")
        lines.append(f"- 부담 요인: {_bullet_join(row.get('negative_factors') or [], limit=1) or '단기 부담은 제한적입니다.'}")
        interpretation = _bullet_join(row.get("positive_factors") or [], limit=1)
        if interpretation:
            lines.append(f"- 해석: {interpretation}")
        lines.append(f"- {point_label}: {_checkpoint_text(row.get('intraday_checkpoints') or [], freshness)}")
    return lines


def build_risk_section(regime: dict, top_sectors: list[dict], watchlist_scores: list[dict], freshness: dict, readiness: dict) -> list[str]:
    risks: list[str] = []
    if readiness.get("display_mode") != "FULL_MARKET":
        risks.append("[데이터] 국내 전종목 가격 커버리지가 아직 충분하지 않습니다.")
    if freshness.get("stale_warnings"):
        risks.append(f"[데이터] {_translate_stale_warning(freshness.get('stale_warnings'))}.")
    for warning in regime.get("warnings") or []:
        text = _translate_warning(warning)
        if text:
            risks.append(f"[시장] {text}")
    for row in top_sectors:
        for warning in row.get("warnings") or []:
            text = _translate_warning(warning)
            if text:
                risks.append(f"[테마/종목] {text}")
    unique: list[str] = []
    seen: set[str] = set()
    for risk in risks:
        if risk in seen:
            continue
        seen.add(risk)
        unique.append(risk)
    return [f"- {risk}" for risk in unique[:5]]


def build_checkpoints_section(top_sectors: list[dict], freshness: dict, readiness: dict) -> list[str]:
    if freshness.get("xkrx_is_open") is False:
        lines = ["- 한국장 휴장으로 실시간 대응은 없습니다."]
        futures = [
            f"다음 거래일 확인: {', '.join(_display_sector_name(row.get('sector_group')) for row in top_sectors[:2])} 흐름",
            "다음 거래일 확인: SOX/Nasdaq 방향과 USD/KRW 레벨",
        ]
        if readiness.get("display_mode") != "FULL_MARKET":
            futures.append("다음 거래일 확인: 관심종목·랭킹 후보 거래대금 유지 여부")
        lines.extend(f"- {item}" for item in futures[:3] if item)
        return lines

    base_points = [
        "09:30 외국인 선물 방향",
        "10:30 주도 테마 거래대금 유지 여부",
        "12:30 KIS 거래량 상위 종목 지속 여부",
    ]
    for row in top_sectors[:2]:
        base_points.extend(row.get("intraday_checkpoints") or [])
    deduped = []
    seen = set()
    for item in base_points:
        text = clean_sentence(item).rstrip(".")
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(f"- {text}")
    return deduped[:5]


def _theme_evidence_text(row: dict, readiness: dict) -> str:
    if row.get("data_status") in {"STALE", "NO_DATA"}:
        return "대표 ETF 기준일이 오래돼 참고 비중을 낮추고, 관심종목·랭킹 후보 중심으로 제한 해석합니다."
    etf_name = row.get("etf_name") or _display_sector_name(row.get("sector_group"))
    pieces = [etf_name]
    change_rate = safe_change_rate(row.get("change_rate_1d"))
    ratio = safe_float(row.get("trading_value_ratio_20d"))
    if change_rate is not None:
        pieces.append(f"단기 등락률 {format_pct(change_rate)}")
    if ratio is not None:
        pieces.append(f"거래대금 20일 평균 대비 {ratio:.2f}배")
    if row.get("etf_symbol") == "462330":
        pieces.append("레버리지 ETF 급등은 과열 참고 신호로만 봅니다")
    suffix = "관심종목·랭킹 후보 기반 해석입니다." if readiness.get("display_mode") != "FULL_MARKET" else ""
    return clean_sentence(", ".join(pieces)) + (f" {suffix}" if suffix else "")


def _translate_allowed_sections(values: Iterable[str]) -> list[str]:
    return [ALLOWED_SECTION_TEXT.get(value, value) for value in values]


def _translate_blocked_sections(values: Iterable[str]) -> list[str]:
    return [BLOCKED_SECTION_TEXT.get(value, value) for value in values]


def _market_open_text(value: bool | None) -> str:
    if value is True:
        return "개장"
    if value is False:
        return "휴장"
    return "미확인"


def _market_tone_text(regime: dict) -> str:
    label = regime.get("regime_label")
    return {
        "Risk-on": "우호",
        "Mild risk-on": "중립·우호",
        "Neutral": "중립",
        "Cautious": "주의",
        "Risk-off": "주의",
    }.get(label, "중립")


def _display_sector_name(value) -> str:
    text = str(value or "").strip()
    return SECTOR_NAME_MAP.get(text, text or "주요 테마")


def _display_stock_name(symbol, name) -> str:
    symbol_text = str(symbol or "").strip()
    return STOCK_NAME_MAP.get(symbol_text, str(name or symbol_text or "관심종목").strip())


def _global_driver_summary(regime: dict) -> str:
    positives = " ".join(regime.get("positive_drivers") or []).lower()
    negatives = " ".join(regime.get("negative_drivers") or []).lower()
    if "sox" in positives and "nasdaq" in positives:
        return "미국 기술주와 SOX 흐름은 우호적입니다"
    if "sox" in negatives and "nasdaq" in negatives:
        return "미국 기술주와 SOX 흐름은 다소 약합니다"
    if "usdkrw" in negatives or "us10y" in negatives:
        return "환율과 금리 레벨 부담을 함께 점검해야 합니다"
    if regime.get("positive_drivers"):
        return "미국 지표는 대체로 위험선호 쪽에 기울어 있습니다"
    if regime.get("negative_drivers"):
        return "미국 지표는 대체로 보수적 해석이 필요합니다"
    return "글로벌 지표는 방향성이 뚜렷하지 않습니다"


def _risk_summary(top_sectors: list[dict], regime: dict) -> str:
    for row in top_sectors:
        warnings = row.get("warnings") or []
        if any("OVERHEATED_20D" in warning for warning in warnings):
            return "단기 과열 신호는 함께 확인해야 합니다"
        if row.get("data_status") == "STALE":
            return "대표 ETF 기준일이 오래된 섹터는 참고 비중을 낮추는 편이 좋습니다"
    for warning in regime.get("warnings") or []:
        translated = _translate_warning(warning)
        if translated:
            return translated
    return ""


def _market_line(label: str, current, change_value, change_rate, interpretation: str) -> str:
    return f"- {label}: {format_number(current)} / {format_number(change_value)} / {format_pct(change_rate)} / {interpretation}"


def _rate_line(label: str, current, change_bp, interpretation: str) -> str:
    return f"- {label}: {format_number(current)} / {format_bp(change_bp)} / {interpretation}"


def _interpret_equity(change_rate, meaning: str) -> str:
    numeric = safe_change_rate(change_rate)
    if numeric is None:
        return f"{meaning}는 미확인입니다."
    if numeric >= 0.005:
        return f"{meaning}에 우호적입니다."
    if numeric <= -0.005:
        return f"{meaning}에 부담입니다."
    return f"{meaning}는 중립권입니다."


def _interpret_vix(level, change_rate) -> str:
    level_value = safe_float(level)
    rate = safe_change_rate(change_rate)
    if level_value is not None and level_value >= 20:
        return "변동성 부담이 높아졌습니다."
    if rate is not None and rate <= -0.05:
        return "리스크 선호 회복 신호입니다."
    return "변동성은 중립권입니다."


def _interpret_fx(change_rate) -> str:
    rate = safe_change_rate(change_rate)
    if rate is None:
        return "환율 방향은 미확인입니다."
    if rate <= -0.003:
        return "원화 강세로 국내 수급에는 우호적입니다."
    if rate >= 0.003:
        return "원화 약세로 성장주 변동성 확대 가능성이 있습니다."
    return "환율 영향은 중립입니다."


def _interpret_dxy(change_rate) -> str:
    rate = safe_change_rate(change_rate)
    if rate is None:
        return "달러 방향은 미확인입니다."
    if rate >= 0.003:
        return "달러 강세 부담이 있습니다."
    if rate <= -0.003:
        return "달러 부담은 다소 완화됐습니다."
    return "달러 방향은 중립입니다."


def _interpret_rate(change_bp, meaning: str) -> str:
    numeric = safe_float(change_bp)
    if numeric is None:
        return f"{meaning}은 미확인입니다."
    if numeric <= -5:
        return f"{meaning}에는 우호적입니다."
    if numeric >= 5:
        return f"{meaning}에는 부담입니다."
    return f"{meaning}은 중립입니다."


def _interpret_spread(spread) -> str:
    numeric = safe_float(spread)
    if numeric is None:
        return "장단기 금리차 해석은 미확인입니다."
    if numeric > 0.25:
        return "정상 곡선 구간으로 성장 기대와 기간 프리미엄을 함께 봅니다."
    if numeric >= -0.05:
        return "경기와 정책 기대가 혼재된 구간입니다."
    return "역전 구간이라 경기 둔화 우려를 금리·달러와 함께 확인해야 합니다."


def _interpret_oil(change_rate) -> str:
    rate = safe_change_rate(change_rate)
    if rate is None:
        return "유가 방향은 미확인입니다."
    if rate >= 0.02:
        return "원가 부담 확대 가능성이 있습니다."
    if rate <= -0.02:
        return "원가 부담 완화 가능성이 있습니다."
    return "유가 영향은 중립입니다."


def _translate_stale_warning(value) -> str:
    text = str(value or "")
    if text.startswith("sector_etf:"):
        return "대표 ETF 일부의 기준일이 오래됐습니다"
    if text.startswith("watchlist:"):
        return "관심종목 일부의 기준일이 오래됐습니다"
    return clean_sentence(text).rstrip(".")


def _translate_warning(value) -> str:
    text = str(value or "")
    replacements = {
        "sp500 change rate anomaly": "S&P500 변화율 스케일은 재확인이 필요합니다.",
        "nasdaq change rate anomaly": "Nasdaq 변화율 스케일은 재확인이 필요합니다.",
        "sox change rate anomaly": "SOX 변화율 스케일은 재확인이 필요합니다.",
        "vix change rate anomaly": "VIX 변화율 스케일은 재확인이 필요합니다.",
        "usdkrw invalid": "USD/KRW 원천값 점검이 필요합니다.",
        "brent out of sanity range": "Brent 원천값 범위 점검이 필요합니다.",
        "sp500 out of sanity range": "S&P500 원천값 범위 점검이 필요합니다.",
        "OVERHEATED_20D": "최근 20일 상승폭이 커 과열 부담을 함께 봐야 합니다.",
        "Speculative ETF excluded": "레버리지 ETF 급등은 주근거가 아닌 과열 참고 신호로만 봅니다.",
        "ETF evidence excluded because data is stale or missing": "대표 ETF 기준일이 오래돼 정량 해석 비중을 낮춥니다.",
        "ETF stale but usable": "대표 ETF 기준일이 다소 늦어 보조 신호로만 활용합니다.",
    }
    lowered = text.lower()
    for source, target in replacements.items():
        if source.lower() in lowered:
            return target
    if "missing_required_data" in lowered:
        return "핵심 데이터 일부가 지연돼 보수적 해석이 필요합니다."
    if "market breadth missing" in lowered:
        return "시장 breadth 확인이 지연됐습니다."
    if "foreign flow missing" in lowered or "institutional flow missing" in lowered:
        return "투자자 수급 확인이 제한적입니다."
    return clean_sentence(text)


def _summarize_etf_status(row: dict) -> str:
    if row.get("data_status") == "STALE":
        return "대표 ETF 기준일이 오래돼 참고 비중을 낮춥니다."
    if row.get("data_status") == "STALE_BUT_USABLE":
        return "대표 ETF는 보조 신호로만 활용합니다."
    return clean_sentence(row.get("etf_reason") or "ETF 흐름을 보조 신호로 확인합니다.")


def _checkpoint_text(items: Iterable[str], freshness: dict) -> str:
    if freshness.get("xkrx_is_open") is False:
        return "휴장으로 실시간 대응 없음 / 다음 거래일 확인: 관심종목·랭킹 후보 흐름 점검"
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return "09:30 외국인 선물 방향과 거래대금 유지 여부 확인"
    return ", ".join(dict.fromkeys(cleaned))


def _strip_terminal_period(text: str) -> str:
    return str(text or "").strip().rstrip(".")


def _bullet_join(items: Iterable[str], limit: int = 2) -> str:
    cleaned = [clean_sentence(item).rstrip(".") for item in items if clean_sentence(item)]
    if not cleaned:
        return ""
    deduped = list(dict.fromkeys(cleaned))
    return ", ".join(deduped[:limit])
