from __future__ import annotations

from typing import Any


CORE_PRIORITY = {
    "071050": 1,
    "005930": 2,
    "000660": 3,
    "278470": 4,
    "058470": 5,
    "012330": 6,
    "047810": 7,
    "012450": 8,
}

SECTOR_MAP = {
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
}


def build_watchlist_morning_scores(watchlist_snapshot: list[dict], regime: dict, sector_impacts: list[dict]) -> list[dict]:
    sector_map = {row.get("sector_group"): row for row in sector_impacts}
    scored = []
    for row in watchlist_snapshot:
        sector_key = _normalize_sector_group(row.get("sector_group"))
        score_bundle = _score_watchlist_row(row, regime, sector_map.get(sector_key))
        scored.append({**row, **score_bundle})
    scored.sort(
        key=lambda item: (
            1 if item.get("data_status") == "FRESH" else 0,
            -int(item.get("core_priority") or 999),
            float((item.get("sector_impact_score") or 0)),
            float(item.get("score") or 0),
            float(item.get("trading_value_ratio_20d") or 0),
        ),
        reverse=True,
    )
    return scored


def _score_watchlist_row(row: dict, regime: dict, sector_impact: dict | None) -> dict:
    if row.get("data_status") in {"DATA_MISSING", "NO_DATA"} or row.get("close_price") in (None, ""):
        return {
            "score": 0.0,
            "label": "데이터 부족",
            "quant_reasons": ["가격 데이터 부족"],
            "positive_factors": ["정량 해석 제한"],
            "negative_factors": ["관심종목 데이터 부족"],
            "intraday_checkpoints": ["데이터 갱신 여부 확인"],
            "core_priority": CORE_PRIORITY.get(str(row.get("symbol") or ""), 999),
            "sector_impact_score": float((sector_impact or {}).get("score") or 0),
        }

    price_score, price_bundle = _price_momentum_score(row)
    liquidity_score, liquidity_bundle = _liquidity_score(row)
    investor_score, investor_bundle = _investor_score(row)
    value_score, value_bundle = _value_quality_score(row)
    risk_score, risk_bundle = _risk_penalty(row)
    macro_fit_score, macro_bundle = _macro_fit_score(regime, sector_impact)

    total = (
        price_score * 0.25
        + liquidity_score * 0.20
        + investor_score * 0.20
        + value_score * 0.15
        + risk_score * 0.10
        + macro_fit_score * 0.10
    )
    core_bonus = 4 if str(row.get("symbol") or "") in CORE_PRIORITY else 0
    total += core_bonus
    label = _label(total)

    evidence = _dedupe(
        price_bundle["evidence"]
        + liquidity_bundle["evidence"]
        + investor_bundle["evidence"]
        + value_bundle["evidence"]
    )
    positive = _dedupe(
        price_bundle["positive"]
        + liquidity_bundle["positive"]
        + investor_bundle["positive"]
        + value_bundle["positive"]
        + macro_bundle["positive"]
    )
    negative = _dedupe(
        price_bundle["negative"]
        + liquidity_bundle["negative"]
        + investor_bundle["negative"]
        + value_bundle["negative"]
        + risk_bundle["negative"]
        + macro_bundle["negative"]
    )
    checkpoints = _build_intraday_checkpoints(row, sector_impact)
    return {
        "score": round(total, 1),
        "label": label,
        "quant_reasons": evidence[:3] or ["정량 근거 제한"],
        "positive_factors": positive[:3] or ["정량 해석 제한"],
        "negative_factors": negative[:3] or ["뚜렷한 부담 요인은 제한적입니다."],
        "intraday_checkpoints": checkpoints,
        "core_priority": CORE_PRIORITY.get(str(row.get("symbol") or ""), 999),
        "sector_impact_score": float((sector_impact or {}).get("score") or 0),
    }


def _price_momentum_score(row: dict):
    score = 50.0
    evidence = []
    positive = []
    negative = []
    for field, weight, label in (("return_5d", 30, "5일"), ("return_20d", 25, "20일"), ("return_60d", 20, "60일")):
        value = _to_float(row.get(field))
        if value is None:
            continue
        score += max(min(value * weight, 15), -15)
        evidence.append(f"{label} 수익률 {value:+.2%}")
        if value > 0:
            positive.append(f"{label} 주가 흐름이 상승 쪽으로 유지되고 있습니다.")
        elif value < 0:
            negative.append(f"{label} 주가 흐름이 약해 단기 반전 확인이 필요합니다.")
    near_high = _to_float(row.get("near_52w_high_pct"))
    if near_high is not None and near_high >= 95:
        evidence.append(f"52주 고가 대비 {near_high:.1f}%")
        negative.append("52주 고가권에 가까워 추격 부담이 있습니다.")
        score -= 5
    return max(0.0, min(100.0, score)), {"evidence": evidence, "positive": positive, "negative": negative}


def _liquidity_score(row: dict):
    score = 50.0
    evidence = []
    positive = []
    negative = []
    ratio = _to_float(row.get("trading_value_ratio_20d"))
    if ratio is not None:
        evidence.append(f"거래대금 20일 평균 대비 {ratio:.2f}배")
        if ratio >= 2:
            score += 20
            positive.append("거래대금이 크게 늘어 수급 집중 여부를 확인하기 좋습니다.")
        elif ratio >= 1.2:
            score += 10
            positive.append("거래대금이 평균보다 개선돼 관심이 유지되고 있습니다.")
        elif ratio < 0.8:
            score -= 10
            negative.append("거래대금이 평균보다 약해 추세 신뢰도가 낮습니다.")
    return max(0.0, min(100.0, score)), {"evidence": evidence, "positive": positive, "negative": negative}


def _investor_score(row: dict):
    score = 50.0
    evidence = []
    positive = []
    negative = []
    for field, label, pos_text, neg_text, delta in (
        ("foreign_net_buy", "외국인", "외국인 수급이 유지되면 주도주 확인에 유리합니다.", "외국인 매도가 이어지면 단기 탄력이 둔화될 수 있습니다.", 12),
        ("institutional_net_buy", "기관", "기관 수급이 받쳐주면 눌림 구간 방어에 도움이 됩니다.", "기관 매도가 겹치면 변동성이 커질 수 있습니다.", 8),
    ):
        value = _to_float(row.get(field))
        if value is None:
            continue
        direction = "순매수" if value > 0 else "순매도" if value < 0 else "보합"
        evidence.append(f"{label} {direction}")
        if value > 0:
            score += delta
            positive.append(pos_text)
        elif value < 0:
            score -= delta
            negative.append(neg_text)
    return max(0.0, min(100.0, score)), {"evidence": evidence, "positive": positive, "negative": negative}


def _value_quality_score(row: dict):
    score = 50.0
    evidence = []
    positive = []
    negative = []
    roe = _to_float(row.get("roe"))
    debt = _to_float(row.get("debt_ratio"))
    if roe is not None:
        evidence.append(f"ROE {roe:.1f}%")
        if roe >= 10:
            score += 10
            positive.append("수익성이 받쳐주면 단기 수급 이후에도 추세 유지 가능성이 높습니다.")
    if debt is not None:
        evidence.append(f"부채비율 {debt:.1f}%")
        if debt >= 150:
            score -= 10
            negative.append("재무 부담이 높아 변동성 확대 구간에서 약해질 수 있습니다.")
    return max(0.0, min(100.0, score)), {"evidence": evidence, "positive": positive, "negative": negative}


def _risk_penalty(row: dict):
    score = 50.0
    negative = []
    short_ratio = _to_float(row.get("short_ratio"))
    if short_ratio is not None and short_ratio >= 5:
        score -= 15
        negative.append("공매도 비중이 높아 단기 차익실현 압력이 커질 수 있습니다.")
    ret20 = _to_float(row.get("return_20d"))
    if ret20 is not None and ret20 >= 0.30:
        score -= 10
        negative.append("최근 20일 급등폭이 커 추격 부담을 함께 봐야 합니다.")
    return max(0.0, min(100.0, score)), {"negative": negative}


def _macro_fit_score(regime: dict, sector_impact: dict | None):
    score = 50.0
    positive = []
    negative = []
    tone = regime.get("market_tone")
    sector_label = (sector_impact or {}).get("label")
    sector_name = (sector_impact or {}).get("sector_group")
    if tone == "우호":
        score += 10
        positive.append("시장 전반의 위험선호가 유지되면 탄력 확장이 가능합니다.")
    elif tone == "주의":
        score -= 10
        negative.append("시장 전반이 보수적이면 종목별 추격보다 확인이 우선입니다.")
    if sector_impact:
        if sector_label == "우호":
            score += 10
            positive.append(f"{sector_name} 섹터 흐름이 우호적이라 종목 해석도 상대적으로 편합니다.")
        elif sector_label in {"주의", "비우호"}:
            score -= 10
            negative.append(f"{sector_name} 섹터 근거가 약해 종목 단독 접근은 보수적으로 봐야 합니다.")
    return max(0.0, min(100.0, score)), {"positive": positive, "negative": negative}


def _build_intraday_checkpoints(row: dict, sector_impact: dict | None) -> list[str]:
    sector_name = (sector_impact or {}).get("sector_group") or row.get("sector_group")
    if sector_name:
        checkpoints = [f"{sector_name} 거래대금과 수급 지속 여부"]
    else:
        checkpoints = ["관련 섹터 거래대금과 수급 지속 여부"]
    ratio = _to_float(row.get("trading_value_ratio_20d"))
    if ratio is not None and ratio >= 2:
        checkpoints.append("개장 후 거래대금이 빠르게 꺾이지 않는지 확인")
    if sector_impact and sector_impact.get("label") in {"우호", "중립~우호"}:
        checkpoints.append(f"{sector_name} 대표 ETF와 동행하는지 확인")
    return checkpoints[:3]


def _label(score: float) -> str:
    if score >= 75:
        return "우호"
    if score >= 60:
        return "중립~우호"
    if score >= 45:
        return "중립"
    if score >= 30:
        return "주의"
    return "데이터 부족"


def _dedupe(items: list[str]) -> list[str]:
    results = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
    return results


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_sector_group(value: Any) -> str:
    text = str(value or "").strip()
    return SECTOR_MAP.get(text, text)
