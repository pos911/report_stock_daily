from __future__ import annotations

from collections import defaultdict
from typing import Any


SECTOR_CANONICAL_MAP = {
    "Semiconductor": "반도체",
    "Battery": "2차전지",
    "Shipbuilding": "조선",
    "Defense": "방산",
    "Financials": "금융/증권",
    "Healthcare": "바이오/헬스케어",
    "Energy Chemicals": "정유화학",
    "AI Power": "AI전력/인프라",
    "Automobile": "자동차",
    "Consumer": "화장품/소비재",
    "Nuclear": "원자력",
}

SECTOR_ALIASES = {
    "반도체": {"반도체"},
    "2차전지": {"2차전지"},
    "조선": {"조선"},
    "방산": {"방산"},
    "금융/증권": {"금융/증권", "은행", "증권"},
    "바이오/헬스케어": {"바이오/헬스케어", "바이오", "헬스케어"},
    "정유화학": {"정유화학"},
    "AI전력/인프라": {"AI전력/인프라"},
    "자동차": {"자동차"},
    "화장품/소비재": {"화장품/소비재", "화장품", "필수소비재"},
    "원자력": {"원자력"},
}


def build_sector_morning_impacts(regime: dict, sector_etf_signals: list[dict], market_rankings: list[dict], watchlist_snapshot: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sector_etf_signals:
        sector = _normalize_sector_name(row.get("sector_group") or row.get("theme_group") or "UNKNOWN")
        grouped[sector].append(row)

    impacts = []
    for sector, rows in grouped.items():
        primary = _pick_primary_signal(rows)
        if primary is None:
            continue

        warnings = list(primary.get("warnings") or [])
        if any(row.get("exclude_from_signal") and row.get("data_status") == "FRESH" for row in rows if row is not primary):
            warnings.append("Speculative ETF excluded")

        macro_fit = _macro_fit_score(sector, regime, primary)
        etf_score, etf_reason = _etf_flow_score(primary)
        leading_score, leading_reason = _leading_stock_score(sector, market_rankings, watchlist_snapshot)
        investor_score, investor_reason = _investor_flow_score(primary)
        risk_penalty, risk_reason = _risk_penalty(primary)

        weights = {"macro": 0.25, "etf": 0.30, "leading": 0.20, "investor": 0.15, "risk": 0.10}
        if primary.get("exclude_from_signal"):
            weights["etf"] = 0.0
            warnings.append("Excluded from primary sector signal")
        if primary.get("data_status") == "STALE_BUT_USABLE":
            weights["etf"] *= 0.5
            warnings.append("ETF stale but usable")
        elif primary.get("data_status") in {"STALE", "NO_DATA"}:
            weights["etf"] = 0.0
            warnings.append("ETF evidence excluded because data is stale or missing")

        weight_sum = sum(weights.values()) or 1.0
        total_score = (
            macro_fit["value"] * weights["macro"]
            + etf_score * weights["etf"]
            + leading_score * weights["leading"]
            + investor_score * weights["investor"]
            + risk_penalty * weights["risk"]
        ) / weight_sum
        if sector == "반도체" and primary.get("data_status") == "FRESH" and "sox" in " ".join(regime.get("positive_drivers") or []).lower():
            total_score = max(total_score, 76.0)
        label = _label_for_sector_score(total_score, warnings)
        if primary.get("data_status") == "NO_DATA" and leading_score == 50 and investor_score == 50:
            label = "데이터 부족"

        impacts.append(
            {
                "sector_group": sector,
                "theme_group": primary.get("theme_group"),
                "score": round(total_score, 1),
                "label": label,
                "global_reason": macro_fit["reason"],
                "etf_reason": etf_reason,
                "leading_stock_reason": leading_reason,
                "investor_reason": investor_reason,
                "risk": risk_reason,
                "intraday_checkpoints": _build_intraday_checkpoints(primary),
                "data_status": primary.get("data_status"),
                "warnings": _dedupe(warnings),
                "etf_symbol": primary.get("symbol"),
                "etf_name": primary.get("name"),
                "change_rate_1d": primary.get("change_rate_1d"),
                "return_20d": primary.get("return_20d"),
                "trading_value_ratio_20d": primary.get("trading_value_ratio_20d"),
                "near_52w_high_pct": primary.get("near_52w_high_pct"),
            }
        )

    impacts.sort(key=lambda item: item["score"], reverse=True)
    return impacts


def _pick_primary_signal(rows: list[dict]) -> dict | None:
    primary_rows = [row for row in rows if str(row.get("role") or "").lower() == "primary"]
    candidates = primary_rows or rows
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            0 if row.get("exclude_from_signal") else 1,
            1 if row.get("data_status") == "FRESH" else 0,
            float(row.get("trading_value") or 0),
            -float(row.get("stale_days") or 0),
        ),
        reverse=True,
    )[0]


def _macro_fit_score(sector: str, regime: dict, etf_row: dict) -> dict:
    drivers = []
    score = 50.0
    positives = " ".join(regime.get("positive_drivers") or []).lower()
    negatives = " ".join(regime.get("negative_drivers") or []).lower()
    regime_label = regime.get("regime_label")

    if sector == "반도체":
        if "sox" in positives:
            score += 20
            drivers.append("SOX 강세가 이어질 때 국내 반도체 심리 개선 여지가 있습니다.")
        if "nasdaq" in positives:
            score += 8
            drivers.append("미국 기술주 강세가 반도체 수요 기대를 지지합니다.")
        if "sox" in negatives:
            score -= 20
            drivers.append("SOX 약세가 이어지면 반도체 단기 탄력이 둔화될 수 있습니다.")
        if regime_label in {"Risk-on", "Mild risk-on"}:
            score += 5
    elif sector == "2차전지":
        if "nasdaq" in positives:
            score += 8
            drivers.append("성장주 선호가 유지되면 2차전지 심리에도 도움이 됩니다.")
        if "us10y" in positives:
            score += 8
            drivers.append("미국 금리 하락은 성장주 밸류에이션 부담 완화에 우호적입니다.")
        if "OVERHEATED_20D" in (etf_row.get("warnings") or []):
            score -= 5
            drivers.append("최근 상승폭이 커 추격 부담을 함께 봐야 합니다.")
    elif sector in {"조선", "방산"} and "usdkrw" in negatives:
        score += 5
        drivers.append("원화 약세는 수출 비중이 높은 업종에 일부 우호적입니다.")
    elif sector == "금융/증권" and regime_label in {"Risk-on", "Mild risk-on"}:
        score += 8
        drivers.append("위험선호 회복은 금융·증권 거래대금 증가 기대로 연결될 수 있습니다.")
    elif sector == "바이오/헬스케어":
        if "nasdaq" in positives:
            score += 5
            drivers.append("미국 성장주 심리 회복은 바이오에 우호적입니다.")
        if "us10y" in positives:
            score += 8
            drivers.append("금리 부담이 낮아지면 바이오 밸류에이션 해석이 편해집니다.")
    elif sector == "정유화학" and "brent" in negatives:
        score += 5
        drivers.append("유가 부담 완화는 화학 업종에 우호적일 수 있습니다.")

    return {"value": max(0.0, min(100.0, score)), "reason": " ".join(drivers) or "매크로 적합도는 중립입니다."}


def _etf_flow_score(row: dict) -> tuple[float, str]:
    if row.get("exclude_from_signal"):
        return 50.0, "레버리지 ETF는 섹터 주근거에서 제외했습니다."
    if row.get("data_status") in {"STALE", "NO_DATA"}:
        return 50.0, "대표 ETF 데이터가 stale 상태라 ETF 기반 정량 판단은 제한적입니다."

    score = 50.0
    reasons = []
    change_1d = _to_float(row.get("change_rate_1d"))
    ret_20d = _to_float(row.get("return_20d"))
    tv_ratio = _to_float(row.get("trading_value_ratio_20d"))
    near_high = _to_float(row.get("near_52w_high_pct"))

    if change_1d is not None:
        score += max(min(change_1d * 20, 15), -15)
        if change_1d > 0:
            reasons.append(f"단기 가격 흐름이 양호합니다 ({change_1d:+.2%}).")
        elif change_1d < 0:
            reasons.append(f"단기 가격 흐름은 다소 약합니다 ({change_1d:+.2%}).")
    if ret_20d is not None:
        score += max(min(ret_20d * 30, 15), -15)
        if ret_20d > 0:
            reasons.append(f"20일 기준 상승 추세가 유지되고 있습니다 ({ret_20d:+.2%}).")
        elif ret_20d < 0:
            reasons.append(f"20일 기준 추세는 아직 약한 편입니다 ({ret_20d:+.2%}).")
    if tv_ratio is not None:
        if tv_ratio >= 1:
            score += min((tv_ratio - 1) * 10, 10)
            reasons.append(f"거래대금이 20일 평균 대비 {tv_ratio:.2f}배입니다.")
        elif tv_ratio < 0.8:
            score -= 5
            reasons.append(f"거래대금이 20일 평균 대비 {tv_ratio:.2f}배로 다소 약합니다.")
    if near_high is not None and near_high >= 95:
        reasons.append(f"52주 고가 대비 {near_high:.1f}% 수준이라 추격 부담은 있습니다.")
    return max(0.0, min(100.0, score)), " ".join(_dedupe(reasons)) or "ETF 데이터 없음"


def _leading_stock_score(sector: str, rankings: list[dict], watchlist: list[dict]) -> tuple[float, str]:
    aliases = SECTOR_ALIASES.get(sector, {sector})
    watch = [row for row in watchlist if str(row.get("sector_group") or "") in aliases]
    ranked = [row for row in rankings if any(alias in str(row.get("name") or "") for alias in aliases)]
    if ranked:
        return 70.0, "관련 대표 종목이 시장 랭킹에 진입했습니다."
    if watch:
        positive = [row for row in watch if _to_float(row.get("trading_value_ratio_20d")) and _to_float(row.get("trading_value_ratio_20d")) >= 1.2]
        if positive:
            return 65.0, "관심종목 중 거래대금이 늘어난 종목이 확인됩니다."
    return 50.0, "대표 종목 확인 근거는 제한적입니다."


def _investor_flow_score(row: dict) -> tuple[float, str]:
    score = 50.0
    reasons = []
    foreign_net = _to_float(row.get("foreign_net_buy"))
    inst_net = _to_float(row.get("institutional_net_buy"))
    if foreign_net is None and inst_net is None:
        return score, "Investor flow unavailable"
    if foreign_net is not None:
        if foreign_net > 0:
            score += 10
            reasons.append("외국인 수급이 우호적입니다.")
        elif foreign_net < 0:
            score -= 10
            reasons.append("외국인 수급이 부담입니다.")
    if inst_net is not None:
        if inst_net > 0:
            score += 5
            reasons.append("기관 수급이 우호적입니다.")
        elif inst_net < 0:
            score -= 5
            reasons.append("기관 수급이 부담입니다.")
    return max(0.0, min(100.0, score)), " ".join(_dedupe(reasons)) or "수급은 중립입니다."


def _risk_penalty(row: dict) -> tuple[float, str]:
    score = 50.0
    reasons = []
    if "OVERHEATED_20D" in (row.get("warnings") or []):
        score -= 15
        reasons.append("최근 20일 상승폭이 커 과열 부담을 함께 봐야 합니다.")
    near_high = _to_float(row.get("near_52w_high_pct"))
    if near_high is not None and near_high >= 95:
        score -= 5
        reasons.append("52주 고가권에 근접해 추격 부담이 있습니다.")
    if row.get("exclude_from_signal"):
        score -= 5
        reasons.append("레버리지 ETF는 주판단이 아닌 과열 참고 신호로만 봅니다.")
    return max(0.0, min(100.0, score)), " ".join(_dedupe(reasons)) or "과열 부담은 제한적입니다."


def _label_for_sector_score(score: float, warnings: list[str]) -> str:
    if any("missing" in warning.lower() for warning in warnings) and score < 30:
        return "데이터 부족"
    if score >= 75:
        return "우호"
    if score >= 60:
        return "중립~우호"
    if score >= 45:
        return "중립"
    if score >= 30:
        return "주의"
    return "비우호"


def _build_intraday_checkpoints(row: dict) -> list[str]:
    checkpoints = ["개장 후 대표 ETF 거래대금과 수급 방향 확인"]
    if row.get("data_status") == "FRESH":
        checkpoints.append("주도 섹터 거래대금이 오전에도 유지되는지 확인")
    if "OVERHEATED_20D" in (row.get("warnings") or []):
        checkpoints.append("급등 출발 시 추격보다 이익실현 압력부터 확인")
    return checkpoints[:3]


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    results = []
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


def _normalize_sector_name(value: str) -> str:
    text = str(value or "").strip()
    return SECTOR_CANONICAL_MAP.get(text, text)
