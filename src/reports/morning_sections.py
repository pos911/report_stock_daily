from __future__ import annotations

from collections.abc import Iterable

from src.utils.formatters import NA_TEXT


SECTOR_NAME_MAP = {
    "Semiconductor": "반도체",
    "Semiconductor Equipment": "반도체 장비",
    "Battery": "2차전지",
    "Shipbuilding": "조선",
    "Defense": "방산",
    "Financials": "금융/증권",
    "Healthcare": "바이오/헬스케어",
    "Consumer": "화장품/소비재",
    "Automobile": "자동차",
    "AI Power": "AI전력/인프라",
    "Energy Chemicals": "정유화학",
    "Nuclear": "원자력",
    "Robotics": "로봇",
    "Telecom": "통신",
    "Gaming": "게임",
    "Aerospace": "항공우주",
    "Internet": "인터넷",
    "Utilities": "유틸리티",
}

STOCK_NAME_MAP = {
    "000660": "SK하이닉스",
    "005930": "삼성전자",
    "012330": "현대모비스",
}


def build_data_status_section(freshness: dict, bundle: dict) -> list[str]:
    lines = ["1. 데이터 상태"]
    lines.append(f"- 기준일: {freshness.get('target_date') or NA_TEXT}")
    lines.append(f"- 한국장: {_market_open_text(freshness.get('xkrx_is_open'))}")
    lines.append(f"- 미국장: {_market_open_text(freshness.get('xnys_is_open'))}")
    lines.append(f"- 매크로 기준일: {freshness.get('latest_macro_date') or NA_TEXT}")
    lines.append(f"- 주식 가격 기준일: {freshness.get('latest_stock_price_date') or NA_TEXT}")
    lines.append(f"- 랭킹 기준일: {freshness.get('latest_ranking_date') or NA_TEXT}")
    lines.append(f"- 수급 기준일: {freshness.get('latest_supply_date') or NA_TEXT}")
    lines.append(f"- breadth 기준일: {freshness.get('latest_breadth_date') or NA_TEXT}")
    lines.append(f"- 섹터 ETF coverage: {_coverage_text(freshness.get('sector_etf_coverage_status'))}")
    lines.append(f"- 관심종목 coverage: {_coverage_text(freshness.get('watchlist_coverage_status'))}")
    if bundle.get("contract_failed_views"):
        lines.append(f"- 주의: 리포트 contract view 일부 미조회로 fallback 데이터를 사용했습니다. ({', '.join(bundle.get('contract_failed_views') or [])})")
    if freshness.get("carry_forward_fields"):
        lines.append(f"- 주의: 미국 지표 일부는 직전 거래일 carry-forward ({', '.join(freshness.get('carry_forward_fields') or [])})")
    if freshness.get("stale_warnings"):
        lines.append(f"- 주의: {_friendly_stale_warning(freshness.get('stale_warnings'))}")
    if freshness.get("missing_required_data"):
        lines.append(f"- 주의: {freshness.get('missing_required_data')}")
    return lines


def build_one_line_judgment_section(regime: dict, top_sectors: list[dict], freshness: dict) -> list[str]:
    tone = regime.get("market_tone") or "데이터 부족"
    leading_sector = top_sectors[0] if top_sectors else {}
    leading_sector_name = _display_sector_name(leading_sector.get("sector_group"))
    global_clause = _build_global_clause(regime)
    etf_clause = _build_etf_clause(leading_sector)
    flow_clause = _build_flow_or_risk_clause(top_sectors, freshness, regime)

    if not freshness.get("xkrx_is_open"):
        opening = "한국장은 휴장입니다. 오늘 직접 대응보다 다음 거래일 시나리오를 점검하는 편이 더 중요합니다."
    elif freshness.get("missing_required_data") or bundle_warns(freshness):
        opening = f"오늘 한국장은 {tone} 쪽으로 보되 단정적으로 해석하기보다 보수적으로 접근할 필요가 있습니다."
    else:
        opening = f"오늘 한국장은 {tone} 쪽에 가깝습니다."

    if freshness.get("carry_forward_fields") and not freshness.get("xnys_is_open"):
        opening += " 미국 지표는 직전 거래일 기준으로 해석해야 합니다."

    details = [clause for clause in [global_clause, etf_clause, flow_clause] if clause]
    if leading_sector_name != "미확인" and not freshness.get("xkrx_is_open"):
        details.insert(1, f"다음 거래일 우선 관찰 테마는 {leading_sector_name}입니다.")
    sentence = " ".join([opening] + details[:4]).strip()
    return ["2. 오늘의 한 줄 판단", sentence]


def build_global_market_section(macro: dict) -> list[str]:
    return [
        "3. 야간 글로벌 시장",
        _market_line("S&P500", macro.get("sp500"), macro.get("sp500_change_value"), macro.get("sp500_change_rate"), _interpret_equity(macro.get("sp500_change_rate"), "미국 대형주 위험선호")),
        _market_line("Nasdaq", macro.get("nasdaq"), macro.get("nasdaq_change_value"), macro.get("nasdaq_change_rate"), _interpret_equity(macro.get("nasdaq_change_rate"), "성장주 선호")),
        _market_line("SOX", macro.get("sox"), macro.get("sox_change_value"), macro.get("sox_change_rate"), _interpret_equity(macro.get("sox_change_rate"), "한국 반도체 심리")),
        _market_line("VIX", macro.get("vix"), macro.get("vix_change_value"), macro.get("vix_change_rate"), _interpret_vix(macro.get("vix"), macro.get("vix_change_rate"))),
        _market_line("USD/KRW", macro.get("usdkrw"), macro.get("usdkrw_change_value"), macro.get("usdkrw_change_rate"), _interpret_fx(macro.get("usdkrw_change_rate"))),
        _market_line("DXY", macro.get("dxy"), macro.get("dxy_change_value"), macro.get("dxy_change_rate"), _interpret_equity(macro.get("dxy_change_rate"), "달러 방향")),
        _rate_line("US10Y", macro.get("us10y"), macro.get("us10y_change_bp"), _interpret_rate(macro.get("us10y_change_bp"), "성장주 밸류에이션")),
        _rate_line("US3Y", macro.get("us3y"), macro.get("us3y_change_bp"), _interpret_rate(macro.get("us3y_change_bp"), "단기 정책 금리 기대")),
        _rate_line("10Y-3Y spread", macro.get("us10y_us3y_spread"), macro.get("us10y_us3y_spread_change_bp"), "장단기 금리차로 정책·성장 기대 확인"),
        _rate_line("KR10Y", macro.get("kr10y"), macro.get("kr10y_change_bp"), "국내 금리 민감 업종 점검"),
        _market_line("Brent", macro.get("brent"), macro.get("brent_change_value"), macro.get("brent_change_rate"), _interpret_oil(macro.get("brent_change_rate"))),
        _market_line("WTI", macro.get("wti"), macro.get("wti_change_value"), macro.get("wti_change_rate"), _interpret_oil(macro.get("wti_change_rate"))),
    ]


def build_korean_impact_section(top_sectors: list[dict], freshness: dict) -> list[str]:
    lines = ["4. 한국장 예상 영향"]
    if not freshness.get("xkrx_is_open"):
        lines.append("- 한국장 휴장으로 국내 섹터 판단은 축소하고 글로벌/다음 거래일 관찰 포인트 위주로 정리합니다.")
        for row in top_sectors[:3]:
            lines.append(f"- {_display_sector_name(row.get('sector_group'))}: {row.get('label')} / {_best_reason(row)}")
        return lines
    if not top_sectors:
        lines.append("- 데이터 부족")
        return lines
    for row in top_sectors[:5]:
        lines.append(f"- {_display_sector_name(row.get('sector_group'))}: {row.get('label')} / {_best_reason(row)}")
    return lines


def build_priority_themes_section(top_sectors: list[dict], freshness: dict) -> list[str]:
    lines = ["5. 오늘 우선 관찰 테마"]
    if not top_sectors:
        lines.append("- 데이터 부족")
        return lines
    for index, row in enumerate(top_sectors[:3], 1):
        lines.append(f"{index}순위 {_display_sector_name(row.get('sector_group'))}")
        lines.append(f"- 판단 라벨: {row.get('label')}")
        lines.append(f"- 글로벌 근거: {_joined_text(row.get('global_reason'))}")
        lines.append(f"- ETF 근거: {_etf_reason_text(row)}")
        lines.append(f"- 모멘텀: {_joined_text(row.get('leading_stock_reason'))}")
        lines.append(f"- 수급: {_investor_reason_text(row.get('investor_reason'))}")
        lines.append(f"- 리스크: {_joined_text(row.get('risk'))}")
        lines.append(f"- 장중 체크포인트: {_checkpoint_text(row.get('intraday_checkpoints') or ['09:30 거래대금 확인'], freshness)}")
        lines.append(f"- 데이터 상태: {row.get('data_status') or NA_TEXT}")
    return lines


def build_watchlist_section(watchlist_scores: list[dict], freshness: dict) -> list[str]:
    lines = ["6. 관심종목 장전 점검"]
    if not watchlist_scores:
        lines.append("- WARNING: watchlist empty")
        return lines

    display_count = min(max(3, len(watchlist_scores)), 6)
    remaining = max(len(watchlist_scores) - display_count, 0)
    lines.append(f"- 관심종목 {len(watchlist_scores)}개 중 주요 {display_count}개 표시. 나머지 {remaining}개는 snapshot에 저장합니다.")
    for row in watchlist_scores[:display_count]:
        lines.append(f"{_display_stock_name(row.get('symbol'), row.get('name'))}({row.get('symbol')})")
        lines.append(f"- 장전 판단: {row.get('label')}")
        lines.append(f"- 퀀트 근거: {_joined_items(row.get('quant_reasons') or ['데이터 부족'])}")
        lines.append(f"- 우호 요인: {_joined_items((row.get('positive_factors') or ['없음'])[:3])}")
        lines.append(f"- 부담 요인: {_joined_items((row.get('negative_factors') or ['없음'])[:3])}")
        lines.append(f"- 장중 체크포인트: {_checkpoint_text(row.get('intraday_checkpoints') or ['09:30 거래대금 확인'], freshness)}")
    return lines


def build_risk_section(regime: dict, top_sectors: list[dict], watchlist_scores: list[dict], freshness: dict) -> list[str]:
    lines = ["7. 오늘의 리스크"]
    risks = list(regime.get("warnings") or [])
    risks.extend(warning for row in top_sectors for warning in (row.get("warnings") or []))
    if freshness.get("watchlist_coverage_status") not in {None, "PASS", "정상"}:
        risks.append("watchlist coverage warning")
    if freshness.get("sector_etf_coverage_status") not in {None, "PASS", "정상"}:
        risks.append("sector ETF coverage warning")
    if not watchlist_scores:
        risks.append("watchlist empty")

    unique = []
    seen = set()
    for risk in risks:
        text = _translate_warning(risk)
        if text and text not in seen:
            seen.add(text)
            unique.append(text)

    for risk in unique[:5]:
        lines.append(f"- {risk}")
    if len(lines) == 1:
        lines.append("- 현재 추가 리스크 경고는 제한적입니다.")
    return lines


def build_checkpoints_section(top_sectors: list[dict], freshness: dict) -> list[str]:
    lines = ["8. 장중 확인 포인트"]
    if not freshness.get("xkrx_is_open"):
        lines.append("- 한국장 휴장으로 장중 체크포인트는 없습니다.")
        future_items = []
        if top_sectors:
            future_items.append(f"다음 거래일 확인: {', '.join(_display_sector_name(row.get('sector_group')) for row in top_sectors[:2])} 흐름")
        if freshness.get("sector_etf_coverage_status") not in {None, "PASS", "정상"}:
            future_items.append("다음 거래일 확인: 반도체·2차전지 외 대표 ETF 데이터 정상화 여부")
        future_items.append("다음 거래일 확인: SOX/Nasdaq 흐름과 USD/KRW 방향")
        for item in future_items[:3]:
            lines.append(f"- {item}")
        return lines
    checkpoints = [
        "09:30 외국인 KOSPI200 선물 방향",
        "10:30 주도 섹터 거래대금 유지 여부",
        "12:30 아침 주도 테마 유지 여부",
    ]
    for row in top_sectors[:2]:
        checkpoints.extend(row.get("intraday_checkpoints") or [])
    if freshness.get("carry_forward_fields"):
        checkpoints.append("미국 carry-forward 신호와 국내 현물 수급이 다르게 움직이는지 확인")

    seen = set()
    for checkpoint in checkpoints:
        if checkpoint not in seen:
            seen.add(checkpoint)
            lines.append(f"- {checkpoint}")
    return lines[:8]


def bundle_warns(freshness: dict) -> bool:
    return bool(
        freshness.get("missing_required_data")
        or freshness.get("stale_warnings")
        or freshness.get("sector_etf_coverage_status") not in {None, "PASS", "정상"}
        or freshness.get("watchlist_coverage_status") not in {None, "PASS", "정상"}
    )


def _build_global_clause(regime: dict) -> str:
    positives = regime.get("positive_drivers") or []
    negatives = regime.get("negative_drivers") or []
    if positives:
        return f"글로벌 지표는 {_driver_summary(positives[:2])}로 위험선호에 힘을 실었습니다."
    if negatives:
        return f"글로벌 지표는 {_driver_summary(negatives[:2])}로 리스크 관리가 우선입니다."
    return "글로벌 지표는 방향성이 강하지 않아 중립 해석이 적절합니다."


def _build_etf_clause(leading_sector: dict) -> str:
    if not leading_sector:
        return "섹터 ETF 주근거는 제한적입니다."
    name = _display_sector_name(leading_sector.get("sector_group"))
    if leading_sector.get("data_status") in {"STALE", "NO_DATA"}:
        return f"{name}은 대표 ETF 데이터가 stale 상태라 ETF 기반 정량 판단을 제한적으로만 봐야 합니다."
    etf_reason = _joined_text(leading_sector.get("etf_reason"))
    risk_note = ""
    if any("Speculative ETF excluded" in warning for warning in (leading_sector.get("warnings") or [])):
        risk_note = " 레버리지 ETF 급등은 주근거가 아닌 과열 참고 신호로만 해석합니다."
    return f"{name} ETF는 {etf_reason}{risk_note}"


def _build_flow_or_risk_clause(top_sectors: list[dict], freshness: dict, regime: dict) -> str:
    for row in top_sectors:
        investor_reason = row.get("investor_reason")
        if investor_reason and investor_reason != "Investor flow unavailable":
            return f"수급 측면에서는 {_investor_reason_text(investor_reason)}"
    if freshness.get("stale_warnings"):
        return f"리스크 측면에서는 {freshness.get('stale_warnings')} 경고가 있어 추격 해석은 보수적으로 봐야 합니다."
    warnings = regime.get("warnings") or []
    if warnings:
        return f"리스크 측면에서는 {_translate_warning(warnings[0])} 이슈가 있어 장 초반 확인이 필요합니다."
    return "리스크 측면에서는 장 초반 거래대금 유지 여부를 먼저 확인하는 편이 안전합니다."


def _market_open_text(value) -> str:
    if value is True:
        return "개장"
    if value is False:
        return "휴장"
    return "미확인"


def _coverage_text(value) -> str:
    if value in {"PASS", "정상"}:
        return "정상"
    if value in {"WARN", "주의"}:
        return "일부 stale"
    if value in {"FAIL", "누락"}:
        return "누락"
    return value or "미확인"


def _market_line(label: str, current, change_value, change_rate, interpretation: str) -> str:
    if current in (None, ""):
        return f"- {label}: 데이터 없음 / 데이터 없음 / 데이터 없음 / {interpretation}"
    change_text = _format_number(change_value)
    rate_text = _format_pct(change_rate)
    return f"- {label}: {_format_number(current)} / {change_text} / {rate_text} / {interpretation}"


def _rate_line(label: str, current, change_bp, interpretation: str) -> str:
    if current in (None, ""):
        return f"- {label}: 데이터 없음 / 데이터 없음 / {interpretation}"
    bp_text = f"{float(change_bp):+.1f}bp" if isinstance(change_bp, (int, float)) else "데이터 없음"
    return f"- {label}: {_format_number(current)} / {bp_text} / {interpretation}"


def _interpret_equity(change_rate, meaning: str) -> str:
    if isinstance(change_rate, (int, float)):
        if change_rate > 0:
            return f"{meaning}에 우호적입니다."
        if change_rate < 0:
            return f"{meaning}에 부담입니다."
    return f"{meaning}은 중립입니다."


def _interpret_vix(level, change_rate) -> str:
    if isinstance(level, (int, float)) and level >= 20:
        return "변동성 부담이 높아졌습니다."
    if isinstance(change_rate, (int, float)) and change_rate <= -0.05:
        return "리스크 선호 회복 신호입니다."
    return "변동성은 중립권입니다."


def _interpret_fx(change_rate) -> str:
    if isinstance(change_rate, (int, float)):
        if change_rate <= -0.003:
            return "원화 강세로 외국인 수급에는 우호적입니다."
        if change_rate >= 0.003:
            return "원화 약세로 성장주 변동성에 주의가 필요합니다."
    return "환율 영향은 중립입니다."


def _interpret_rate(change_bp, meaning: str) -> str:
    neutral_particle = "는" if str(meaning).endswith("기대") else "은"
    if isinstance(change_bp, (int, float)):
        if change_bp <= -5:
            return f"{meaning}에 우호적입니다."
        if change_bp >= 5:
            return f"{meaning}에 부담입니다."
    return f"{meaning}{neutral_particle} 중립입니다."


def _interpret_oil(change_rate) -> str:
    if isinstance(change_rate, (int, float)):
        if change_rate >= 0.02:
            return "유가 상승 부담이 커졌습니다."
        if change_rate <= -0.02:
            return "원가 부담 완화 가능성이 있습니다."
    return "유가 영향은 중립입니다."


def _display_sector_name(value) -> str:
    text = str(value or "").strip()
    return SECTOR_NAME_MAP.get(text, text or "미확인")


def _best_reason(row: dict) -> str:
    for candidate in [row.get("global_reason"), row.get("etf_reason"), row.get("leading_stock_reason")]:
        text = _joined_text(candidate)
        if text and text not in {"ETF 중립", "모멘텀 확인 제한"}:
            return text
    return "정량 근거가 제한적입니다."


def _etf_reason_text(row: dict) -> str:
    if row.get("data_status") in {"STALE", "NO_DATA"}:
        return "대표 ETF 데이터가 stale 상태라 ETF 기반 정량 판단은 제한적입니다."
    reason = _joined_text(row.get("etf_reason"))
    if any("Speculative ETF excluded" in warning for warning in (row.get("warnings") or [])):
        reason += " 레버리지 ETF 급등은 주근거가 아닌 과열 참고 신호로만 해석합니다."
    return reason


def _investor_reason_text(value) -> str:
    if value == "Investor flow unavailable" or not value:
        return "수급 데이터 부족"
    return _join_fragments(_split_fragments(_translate_text(value)))


def _driver_summary(drivers: Iterable[str]) -> str:
    return ", ".join(_translate_driver(driver) for driver in drivers)


def _translate_driver(text: str) -> str:
    translated = (
        str(text)
        .replace("S&P500 change", "S&P500")
        .replace("Nasdaq change", "Nasdaq")
        .replace("SOX change", "SOX")
        .replace("USDKRW change", "USD/KRW")
        .replace("KOSPI foreign net buy", "KOSPI 외국인 순매수")
        .replace("KOSPI institutional net buy", "KOSPI 기관 순매수")
        .replace("KOSPI foreign net sell", "KOSPI 외국인 순매도")
        .replace("KOSPI institutional net sell", "KOSPI 기관 순매도")
    )
    return translated


def _translate_text(value) -> str:
    if not value:
        return "정량 근거 제한"
    text = str(value)
    replacements = {
        "Macro fit neutral": "매크로 적합도는 중립입니다.",
        "SOX strength supports semiconductors": "SOX 강세가 반도체에 우호적입니다.",
        "Nasdaq support": "나스닥 강세가 성장주 심리를 지지합니다.",
        "SOX weakness hurts semiconductors": "SOX 약세가 반도체에 부담입니다.",
        "Growth sentiment helps batteries": "성장주 심리가 2차전지에 우호적입니다.",
        "Rates down supports growth assets": "금리 하락이 성장 자산에 우호적입니다.",
        "20-day overheating warning": "20일 과열 경고가 있습니다.",
        "KRW weakness can support exporters": "원화 약세가 수출주에 일부 우호적입니다.",
        "Risk-on can help financial turnover": "위험선호가 금융 거래대금에 우호적입니다.",
        "Nasdaq support helps healthcare sentiment": "나스닥 강세가 바이오 심리를 지지합니다.",
        "Rates down supports healthcare valuations": "금리 하락이 바이오 밸류에이션에 우호적입니다.",
        "Lower oil burden can help chemicals": "유가 부담 완화가 화학 업종에 우호적입니다.",
        "ETF neutral": "ETF 흐름은 중립입니다.",
        "ETF excluded from signal": "레버리지/특수 ETF라 주신호에서 제외했습니다.",
        "ETF data stale or missing": "ETF 데이터 stale 또는 누락입니다.",
        "ETF 1D momentum positive": "ETF 1일 모멘텀이 양호합니다.",
        "ETF 1D momentum negative": "ETF 1일 모멘텀이 약합니다.",
        "ETF 20D trend reflected": "ETF 20일 추세가 반영됐습니다.",
        "ETF trading value expanded": "ETF 거래대금이 20일 평균을 웃돌았습니다.",
        "ETF trading value subdued": "ETF 거래대금이 평균보다 약합니다.",
        "Sector-related names appear in market rankings": "관련 대표 종목이 시장 랭킹에 진입했습니다.",
        "Watchlist names confirm turnover": "관심종목 거래대금이 확인됩니다.",
        "Leading stock confirmation limited": "대표 종목 확인 근거는 제한적입니다.",
        "Foreign flow positive": "외국인 수급이 우호적입니다.",
        "Foreign flow negative": "외국인 수급이 부담입니다.",
        "Institutional flow positive": "기관 수급이 우호적입니다.",
        "Institutional flow negative": "기관 수급이 부담입니다.",
        "Risk manageable": "과열 부담은 제한적입니다.",
        "Near 52-week high": "52주 고가권 부담이 있습니다.",
        "Speculative ETF excluded": "투기성 ETF는 주판단에서 제외했습니다.",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _translate_warning(value) -> str:
    if not value:
        return ""
    text = str(value)
    replacements = {
        "watchlist empty": "관심종목 데이터가 비어 있습니다.",
        "watchlist coverage warning": "관심종목 coverage 경고가 있습니다.",
        "sector ETF coverage warning": "섹터 ETF coverage 경고가 있습니다.",
        "ETF stale but usable": "ETF 데이터가 다소 오래돼 참고 비중을 낮췄습니다.",
        "ETF evidence excluded because data is stale or missing": "ETF 데이터 stale 또는 누락으로 주근거에서 제외했습니다.",
        "Excluded from primary sector signal": "레버리지/특수 ETF는 주근거에서 제외했습니다.",
        "Speculative ETF excluded": "레버리지 ETF 급등은 과열 참고 신호로만 봅니다.",
        "OVERHEATED_20D": "20일 급등 과열 경고가 있습니다.",
        "contract fallback used": "리포트 contract view 일부 미조회로 fallback 데이터를 사용했습니다.",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    if text.endswith("missing"):
        return text.replace(" missing", " 데이터 누락")
    return text


def _friendly_stale_warning(value) -> str:
    text = str(value or "").strip()
    if text.startswith("sector_etf:"):
        symbols = text.split(":", 1)[1].strip()
        return f"섹터 ETF 일부가 stale 상태입니다. ({symbols})"
    if text.startswith("watchlist:"):
        symbols = text.split(":", 1)[1].strip()
        return f"관심종목 일부가 stale 또는 누락 상태입니다. ({symbols})"
    return text


def _format_number(value) -> str:
    if value in (None, ""):
        return "데이터 없음"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.2f}".rstrip("0").rstrip(".")
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _format_pct(value) -> str:
    if value in (None, ""):
        return "데이터 없음"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:+.2%}"


def _split_fragments(value) -> list[str]:
    if not value:
        return []
    parts = []
    for chunk in str(value).replace(";", ".").split("."):
        text = chunk.strip()
        if text:
            parts.append(_clean_sentence(text))
    return parts


def _join_fragments(parts: list[str]) -> str:
    cleaned = []
    seen = set()
    for part in parts:
        text = _clean_sentence(part)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    if not cleaned:
        return "정량 근거 제한"
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} {cleaned[1]}"
    return f"{cleaned[0]} {cleaned[1]} 또한 {cleaned[2]}"


def _joined_items(items: list[str]) -> str:
    return _join_fragments([_translate_text(item) for item in items if item])


def _joined_text(value) -> str:
    return _join_fragments(_split_fragments(_translate_text(value)))


def _clean_sentence(text: str) -> str:
    value = str(text).strip()
    value = value.replace("입니다.가", "입니다. ").replace("습니다.로", "습니다. ")
    value = value.replace("..", ".")
    if not value.endswith((".", "!", "?")):
        value += "."
    return value


def _checkpoint_text(items: list[str], freshness: dict) -> str:
    if not freshness.get("xkrx_is_open"):
        future_points = ["한국장 휴장으로 장중 체크포인트 없음"]
        translated = [_clean_checkpoint(item) for item in items if item]
        if translated:
            future_points.append(f"다음 거래일 확인: {translated[0]}")
        return " / ".join(future_points[:2])
    return ", ".join(_clean_checkpoint(item) for item in items if item)


def _clean_checkpoint(text: str) -> str:
    return str(text).replace("09:30 ", "").replace("10:30 ", "").replace("12:30 ", "").strip()


def _display_stock_name(symbol, name) -> str:
    text_symbol = str(symbol or "").strip()
    if text_symbol in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[text_symbol]
    return str(name or text_symbol or "미확인").strip()
