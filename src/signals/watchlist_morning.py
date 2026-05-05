from __future__ import annotations

from typing import Any


def build_watchlist_morning_scores(watchlist_snapshot: list[dict], regime: dict, sector_impacts: list[dict]) -> list[dict]:
    sector_map = {row.get("sector_group"): row for row in sector_impacts}
    scored = []
    for row in watchlist_snapshot:
        score_bundle = _score_watchlist_row(row, regime, sector_map.get(row.get("sector_group")))
        scored.append({**row, **score_bundle})
    scored.sort(key=lambda item: (float(item.get("score") or 0), item.get("data_status") == "FRESH"), reverse=True)
    return scored


def _score_watchlist_row(row: dict, regime: dict, sector_impact: dict | None) -> dict:
    if row.get("data_status") in {"DATA_MISSING", "NO_DATA"} or row.get("close_price") in (None, ""):
        return {
            "score": 0.0,
            "label": "데이터 부족",
            "quant_reasons": ["가격 데이터 부족"],
            "positive_factors": [],
            "negative_factors": ["관심종목 데이터 부족"],
            "intraday_checkpoints": ["데이터 갱신 여부 확인"],
        }

    price_score, price_reasons = _price_momentum_score(row)
    liquidity_score, liquidity_reasons = _liquidity_score(row)
    investor_score, investor_reasons = _investor_score(row)
    value_score, value_reasons = _value_quality_score(row)
    risk_score, risk_reasons = _risk_penalty(row)
    macro_fit_score, macro_reasons = _macro_fit_score(regime, sector_impact)

    total = (
        price_score * 0.25
        + liquidity_score * 0.20
        + investor_score * 0.20
        + value_score * 0.15
        + risk_score * 0.10
        + macro_fit_score * 0.10
    )
    label = _label(total)
    positive = price_reasons["positive"] + liquidity_reasons["positive"] + investor_reasons["positive"] + value_reasons["positive"] + macro_reasons["positive"]
    negative = price_reasons["negative"] + liquidity_reasons["negative"] + investor_reasons["negative"] + value_reasons["negative"] + risk_reasons["negative"] + macro_reasons["negative"]
    checkpoints = _build_intraday_checkpoints(row, sector_impact)
    return {
        "score": round(total, 1),
        "label": label,
        "quant_reasons": [reason for reason in positive[:3] + negative[:3] if reason] or ["정량 근거 제한"],
        "positive_factors": positive,
        "negative_factors": negative,
        "intraday_checkpoints": checkpoints,
    }


def _price_momentum_score(row: dict):
    score = 50.0
    positive = []
    negative = []
    for field, weight, label in (("return_5d", 30, "5일 수익률"), ("return_20d", 25, "20일 수익률"), ("return_60d", 20, "60일 수익률")):
        value = _to_float(row.get(field))
        if value is None:
            continue
        score += max(min(value * weight, 15), -15)
        if value > 0:
            positive.append(f"{label}이 플러스입니다.")
        elif value < 0:
            negative.append(f"{label}이 마이너스입니다.")
    near_high = _to_float(row.get("near_52w_high_pct"))
    if near_high is not None and near_high >= 95:
        negative.append("52주 고가권 추격 부담이 있습니다.")
        score -= 5
    return max(0.0, min(100.0, score)), {"positive": positive, "negative": negative}


def _liquidity_score(row: dict):
    score = 50.0
    positive = []
    negative = []
    ratio = _to_float(row.get("trading_value_ratio_20d"))
    if ratio is not None:
        if ratio >= 2:
            score += 20
            positive.append("거래대금이 20일 평균을 크게 웃돕니다.")
        elif ratio >= 1.2:
            score += 10
            positive.append("거래대금이 평균보다 개선됐습니다.")
        elif ratio < 0.8:
            score -= 10
            negative.append("거래대금이 평균보다 약합니다.")
    return max(0.0, min(100.0, score)), {"positive": positive, "negative": negative}


def _investor_score(row: dict):
    score = 50.0
    positive = []
    negative = []
    for field, pos_text, neg_text, delta in (
        ("foreign_net_buy", "외국인 순매수가 유입됐습니다.", "외국인 순매도가 우세합니다.", 12),
        ("institutional_net_buy", "기관 순매수가 유입됐습니다.", "기관 순매도가 우세합니다.", 8),
    ):
        value = _to_float(row.get(field))
        if value is None:
            continue
        if value > 0:
            score += delta
            positive.append(pos_text)
        elif value < 0:
            score -= delta
            negative.append(neg_text)
    return max(0.0, min(100.0, score)), {"positive": positive, "negative": negative}


def _value_quality_score(row: dict):
    score = 50.0
    positive = []
    negative = []
    roe = _to_float(row.get("roe"))
    debt = _to_float(row.get("debt_ratio"))
    per = _to_float(row.get("per"))
    pbr = _to_float(row.get("pbr"))
    if roe is not None and roe >= 10:
        score += 10
        positive.append("ROE가 양호합니다.")
    if debt is not None and debt >= 150:
        score -= 10
        negative.append("부채비율이 높습니다.")
    if per is not None and per <= 0:
        score -= 10
        negative.append("PER 해석력이 낮습니다.")
    if pbr is not None and pbr <= 0:
        score -= 5
        negative.append("PBR 해석력이 낮습니다.")
    return max(0.0, min(100.0, score)), {"positive": positive, "negative": negative}


def _risk_penalty(row: dict):
    score = 50.0
    negative = []
    short_ratio = _to_float(row.get("short_ratio"))
    if short_ratio is not None and short_ratio >= 5:
        score -= 15
        negative.append("공매도 비중이 높습니다.")
    ret20 = _to_float(row.get("return_20d"))
    if ret20 is not None and ret20 >= 0.30:
        score -= 10
        negative.append("20일 급등 과열 부담이 있습니다.")
    return max(0.0, min(100.0, score)), {"negative": negative}


def _macro_fit_score(regime: dict, sector_impact: dict | None):
    score = 50.0
    positive = []
    negative = []
    tone = regime.get("market_tone")
    if tone == "우호":
        score += 10
        positive.append("아침 레짐이 우호적입니다.")
    elif tone == "주의":
        score -= 10
        negative.append("아침 레짐이 보수적입니다.")
    if sector_impact:
        if sector_impact.get("label") == "우호":
            score += 10
            positive.append("섹터 영향 점수가 우호적입니다.")
        elif sector_impact.get("label") in {"주의", "비우호"}:
            score -= 10
            negative.append("섹터 영향 점수가 보수적입니다.")
    return max(0.0, min(100.0, score)), {"positive": positive, "negative": negative}


def _build_intraday_checkpoints(row: dict, sector_impact: dict | None) -> list[str]:
    checkpoints = ["시가 직후 거래대금과 호가 흐름 확인"]
    ratio = _to_float(row.get("trading_value_ratio_20d"))
    if ratio is not None and ratio >= 2:
        checkpoints.append("장 초반 거래대금이 유지되는지 확인")
    if sector_impact:
        checkpoints.extend(sector_impact.get("intraday_checkpoints") or [])
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


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
