from __future__ import annotations

from src.utils.formatters import safe_float


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


def build_watchlist_morning_scores(watchlist_snapshot: list[dict], regime: dict, sector_impacts: list[dict]) -> list[dict]:
    sector_map = {str(row.get("sector_group") or ""): row for row in sector_impacts}
    results = []
    for row in watchlist_snapshot:
        sector_impact = sector_map.get(str(row.get("sector_group") or ""))
        score_row = _score_row(row, regime, sector_impact)
        results.append({**row, **score_row})
    results.sort(
        key=lambda item: (
            1 if item.get("data_status") == "FRESH" else 0,
            1 if not item.get("source_mixed") else 0,
            -CORE_PRIORITY.get(str(item.get("symbol") or ""), 999),
            float(item.get("score") or 0),
            float(item.get("sector_impact_score") or 0),
            float(item.get("trading_value_ratio_20d") or 0),
        ),
        reverse=True,
    )
    return results


def _score_row(row: dict, regime: dict, sector_impact: dict | None) -> dict:
    source_mixed = bool(row.get("source_mixed"))
    stale_days = _to_int(row.get("stale_days"))
    data_status = str(row.get("data_status") or "").upper()

    if data_status in {"DATA_MISSING", "NO_DATA"} or row.get("close_price") in (None, ""):
        return {
            "score": 0.0,
            "signal_label": "판단 유보",
            "quant_reasons": ["핵심 가격과 거래대금 확인이 제한적입니다."],
            "positive_factors": ["데이터 갱신 이후 다시 확인하는 편이 좋습니다."],
            "negative_factors": ["가격과 수급 기준이 충분하지 않습니다."],
            "intraday_checkpoints": ["다음 거래일 기준 데이터 갱신 여부 확인"],
            "core_priority": CORE_PRIORITY.get(str(row.get("symbol") or ""), 999),
            "sector_impact_score": float((sector_impact or {}).get("score") or 0),
        }

    score = 50.0
    quant_reasons: list[str] = []
    positives: list[str] = []
    negatives: list[str] = []

    score += _momentum_score(row, quant_reasons, positives, negatives, source_mixed)
    score += _liquidity_score(row, quant_reasons, positives, negatives)
    score += _investor_score(row, quant_reasons, positives, negatives)
    score += _quality_score(row, quant_reasons, positives, negatives)
    score += _risk_penalty(row, negatives)
    score += _macro_fit(regime, sector_impact, positives, negatives)

    if str(row.get("symbol") or "") in CORE_PRIORITY:
        score += 4

    if source_mixed:
        negatives.append("가격 이력 원천이 혼합되어 모멘텀 판단은 보수적으로 봅니다.")
        score = min(score, 58.0)
    if stale_days is not None and stale_days > 0:
        negatives.append("기준일이 하루 이상 지나 보조 신호로만 봅니다.")
        score = min(score, 60.0)
    if data_status == "STALE_BUT_USABLE":
        score = min(score, 60.0)

    label = _label(score, source_mixed=source_mixed, data_status=data_status)

    return {
        "score": round(max(0.0, min(100.0, score)), 1),
        "signal_label": label,
        "quant_reasons": _dedupe(quant_reasons)[:3],
        "positive_factors": _dedupe(positives)[:2],
        "negative_factors": _dedupe(negatives)[:3],
        "intraday_checkpoints": _build_checkpoints(row, sector_impact),
        "core_priority": CORE_PRIORITY.get(str(row.get("symbol") or ""), 999),
        "sector_impact_score": float((sector_impact or {}).get("score") or 0),
    }


def _momentum_score(row: dict, quant_reasons: list[str], positives: list[str], negatives: list[str], source_mixed: bool) -> float:
    delta = 0.0
    if source_mixed:
        quant_reasons.append("가격 이력 원천 혼합으로 단기·중기 수익률은 참고에서 제외합니다.")
        return 0.0

    for field, label, weight in (
        ("return_5d", "5일 수익률", 16),
        ("return_20d", "20일 수익률", 12),
        ("return_60d", "60일 수익률", 8),
    ):
        value = safe_float(row.get(field))
        if value is None:
            continue
        quant_reasons.append(f"{label} {value:+.2%}")
        if value > 0:
            delta += min(value * weight * 100, 12)
        elif value < 0:
            delta += max(value * weight * 100, -12)

    return_5d = safe_float(row.get("return_5d"))
    if return_5d is not None and return_5d > 0:
        positives.append("단기 주가 흐름이 상승 쪽으로 유지되고 있습니다.")
    elif return_5d is not None and return_5d < 0:
        negatives.append("단기 주가 흐름이 약해 반전 확인이 필요합니다.")
    return delta


def _liquidity_score(row: dict, quant_reasons: list[str], positives: list[str], negatives: list[str]) -> float:
    ratio = safe_float(row.get("trading_value_ratio_20d"))
    if ratio is None:
        return 0.0
    quant_reasons.append(f"거래대금 20일 평균 대비 {ratio:.2f}배")
    if ratio >= 2:
        positives.append("거래대금이 크게 늘어 수급 집중 여부를 확인하기 좋습니다.")
        return 12.0
    if ratio >= 1.2:
        positives.append("거래대금이 평균보다 개선돼 관심이 유지되고 있습니다.")
        return 8.0
    if ratio < 0.8:
        negatives.append("거래대금이 평균보다 약해 추세 신뢰도가 낮습니다.")
        return -8.0
    return 0.0


def _investor_score(row: dict, quant_reasons: list[str], positives: list[str], negatives: list[str]) -> float:
    delta = 0.0
    foreign = safe_float(row.get("foreign_net_buy"))
    inst = safe_float(row.get("institutional_net_buy"))
    if foreign is not None:
        quant_reasons.append("외국인 순매수" if foreign > 0 else "외국인 순매도" if foreign < 0 else "외국인 보합")
        if foreign > 0:
            positives.append("외국인 수급이 유지되면 주도주 확인에 유리합니다.")
            delta += 8
        elif foreign < 0:
            negatives.append("외국인 매도가 이어지면 단기 탄력이 둔화될 수 있습니다.")
            delta -= 8
    if inst is not None:
        quant_reasons.append("기관 순매수" if inst > 0 else "기관 순매도" if inst < 0 else "기관 보합")
        if inst > 0:
            positives.append("기관 수급이 받쳐주면 눌림 구간 방어력이 높아질 수 있습니다.")
            delta += 6
        elif inst < 0:
            negatives.append("기관 매도가 겹치면 변동성이 커질 수 있습니다.")
            delta -= 6
    return delta


def _quality_score(row: dict, quant_reasons: list[str], positives: list[str], negatives: list[str]) -> float:
    delta = 0.0
    roe = safe_float(row.get("roe"))
    debt = safe_float(row.get("debt_ratio"))
    if roe is not None:
        quant_reasons.append(f"ROE {roe:.1f}%")
        if roe >= 10:
            positives.append("수익성이 받쳐주면 단기 수급 이후에도 추세 유지 가능성이 높습니다.")
            delta += 5
    if debt is not None and debt >= 150:
        negatives.append("재무 부담이 높아 변동성 확대 구간에서 약해질 수 있습니다.")
        delta -= 5
    return delta


def _risk_penalty(row: dict, negatives: list[str]) -> float:
    delta = 0.0
    short_ratio = safe_float(row.get("short_ratio"))
    ret20 = safe_float(row.get("return_20d"))
    if short_ratio is not None and short_ratio >= 5:
        negatives.append("공매도 비중이 높아 단기 차익실현 압력이 커질 수 있습니다.")
        delta -= 8
    if ret20 is not None and ret20 >= 0.30:
        negatives.append("최근 20일 상승폭이 커 추격 부담이 있습니다.")
        delta -= 6
    return delta


def _macro_fit(regime: dict, sector_impact: dict | None, positives: list[str], negatives: list[str]) -> float:
    delta = 0.0
    regime_label = regime.get("regime_label")
    if regime_label in {"Risk-on", "Mild risk-on"}:
        delta += 4
    elif regime_label == "Risk-off":
        negatives.append("전반적인 위험회피 구간에서는 추격보다 확인 대응이 유리합니다.")
        delta -= 4

    if sector_impact:
        label = sector_impact.get("label")
        sector_name = sector_impact.get("sector_group")
        if label == "우호":
            positives.append(f"{sector_name} 섹터 흐름이 받쳐주면 종목 해석도 우호적입니다.")
            delta += 5
        elif label == "주의":
            negatives.append(f"{sector_name} 섹터 흐름이 약해 종목 대응도 보수적으로 보는 편이 좋습니다.")
            delta -= 5
    return delta


def _build_checkpoints(row: dict, sector_impact: dict | None) -> list[str]:
    sector_name = (sector_impact or {}).get("sector_group") or row.get("sector_group") or "관심 섹터"
    return [
        f"{sector_name} 거래대금과 수급 지속 여부",
        "개장 후 가격 흐름이 강세를 유지하는지 확인",
    ]


def _label(score: float, source_mixed: bool = False, data_status: str | None = None) -> str:
    if data_status in {"DATA_MISSING", "NO_DATA"}:
        return "판단 유보"
    if source_mixed:
        if score >= 45:
            return "관찰"
        return "판단 유보"
    if data_status == "STALE_BUT_USABLE":
        if score >= 60:
            return "보유·관찰"
        if score >= 45:
            return "관망"
        if score >= 30:
            return "리스크 관리 후보"
        return "판단 유보"
    if score >= 75:
        return "강한 모멘텀 후보"
    if score >= 60:
        return "보유·관찰"
    if score >= 45:
        return "관망"
    if score >= 30:
        return "리스크 관리 후보"
    return "판단 유보"


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


def _to_int(value) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
