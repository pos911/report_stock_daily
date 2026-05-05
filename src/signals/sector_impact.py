from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_sector_morning_impacts(regime: dict, sector_etf_signals: list[dict], market_rankings: list[dict], watchlist_snapshot: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sector_etf_signals:
        sector = row.get("sector_group") or row.get("theme_group") or "UNKNOWN"
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
        risk_penalty, risk_reason = _risk_penalty(sector, primary)

        weights = {
            "macro": 0.25,
            "etf": 0.30,
            "leading": 0.20,
            "investor": 0.15,
            "risk": 0.10,
        }
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
        if (
            "semiconductor" in str(sector or "").lower()
            and primary.get("data_status") == "FRESH"
            and "sox" in " ".join(regime.get("positive_drivers") or []).lower()
        ):
            total_score = max(total_score, 78.0)
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
                "intraday_checkpoints": _build_intraday_checkpoints(sector, primary),
                "data_status": primary.get("data_status"),
                "warnings": warnings,
                "etf_symbol": primary.get("symbol"),
                "etf_name": primary.get("name"),
            }
        )

    impacts.sort(key=lambda item: item["score"], reverse=True)
    return impacts


def _pick_primary_signal(rows: list[dict]) -> dict | None:
    primary_rows = [row for row in rows if str(row.get("role") or "").lower() == "primary"]
    candidates = primary_rows or rows
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda row: (
            0 if row.get("exclude_from_signal") else 1,
            1 if row.get("data_status") == "FRESH" else 0,
            float(row.get("trading_value") or 0),
            -float(row.get("stale_days") or 0),
        ),
        reverse=True,
    )
    return ordered[0]


def _macro_fit_score(sector: str, regime: dict, etf_row: dict) -> dict:
    drivers = []
    score = 50.0
    sector_name = str(sector or "").lower()
    regime_label = regime.get("regime_label")
    positives = " ".join(regime.get("positive_drivers") or []).lower()
    negatives = " ".join(regime.get("negative_drivers") or []).lower()

    if "semiconductor" in sector_name:
        if "sox" in positives:
            score += 25
            drivers.append("SOX 강세가 반도체에 우호적입니다.")
        if "nasdaq" in positives:
            score += 10
            drivers.append("나스닥 강세가 성장주 심리를 지지합니다.")
        if "sox" in negatives:
            score -= 25
            drivers.append("SOX 약세가 반도체에 부담입니다.")
        if regime_label in {"Risk-on", "Mild risk-on"}:
            score += 5
    elif "battery" in sector_name:
        if "nasdaq" in positives:
            score += 10
            drivers.append("나스닥 강세가 2차전지에 우호적입니다.")
        if "us10y" in positives:
            score += 10
            drivers.append("미국 금리 하락이 성장주 밸류에이션에 우호적입니다.")
        if "OVERHEATED_20D" in (etf_row.get("warnings") or []):
            score -= 5
            drivers.append("20일 과열 경고가 있습니다.")
    elif "ship" in sector_name or "defense" in sector_name:
        if "usdkrw" in negatives:
            score += 5
            drivers.append("원화 약세가 수출주에 일부 우호적입니다.")
    elif "financial" in sector_name or "bank" in sector_name or "brokerage" in sector_name:
        if regime_label in {"Risk-on", "Mild risk-on"}:
            score += 8
            drivers.append("위험선호가 금융 거래대금에 우호적입니다.")
    elif "health" in sector_name or "bio" in sector_name:
        if "nasdaq" in positives:
            score += 5
            drivers.append("나스닥 강세가 바이오 심리를 지지합니다.")
        if "us10y" in positives:
            score += 10
            drivers.append("금리 하락이 바이오 밸류에이션에 우호적입니다.")
    elif "energy" in sector_name or "chemical" in sector_name:
        if "brent" in negatives:
            score += 5
            drivers.append("유가 부담 완화가 화학 업종에 우호적입니다.")

    score = max(0.0, min(100.0, score))
    return {"value": score, "reason": ". ".join(drivers) or "매크로 적합도는 중립입니다."}


def _etf_flow_score(row: dict) -> tuple[float, str]:
    if row.get("exclude_from_signal"):
        return 50.0, "레버리지/특수 ETF라 주신호에서 제외했습니다."
    if row.get("data_status") in {"STALE", "NO_DATA"}:
        return 50.0, "대표 ETF 데이터가 stale 상태라 ETF 기반 정량 판단은 제한적입니다."
    score = 50.0
    reasons = []
    change_1d = _to_float(row.get("change_rate_1d"))
    ret_20d = _to_float(row.get("return_20d"))
    tv_ratio = _to_float(row.get("trading_value_ratio_20d"))
    if change_1d is not None:
        if change_1d > 0:
            score += min(change_1d * 20, 15)
            reasons.append("ETF 1일 모멘텀이 양호합니다.")
        elif change_1d < 0:
            score += max(change_1d * 20, -15)
            reasons.append("ETF 1일 모멘텀이 약합니다.")
    if ret_20d is not None:
        score += max(min(ret_20d * 30, 15), -15)
        reasons.append("ETF 20일 추세가 반영됐습니다.")
    if tv_ratio is not None:
        if tv_ratio >= 2:
            score += 10
            reasons.append("ETF 거래대금이 20일 평균을 웃돌았습니다.")
        elif tv_ratio < 0.8:
            score -= 5
            reasons.append("ETF 거래대금이 평균보다 약합니다.")
    return max(0.0, min(100.0, score)), ". ".join(reasons) or "ETF 흐름은 중립입니다."


def _leading_stock_score(sector: str, rankings: list[dict], watchlist: list[dict]) -> tuple[float, str]:
    sector_name = str(sector or "").lower()
    ranked = [row for row in rankings if sector_name and sector_name in str(row.get("name") or "").lower()]
    watch = [row for row in watchlist if sector_name and sector_name in str(row.get("sector_group") or "").lower()]
    if ranked:
        return 70.0, "관련 대표 종목이 시장 랭킹에 진입했습니다."
    if watch:
        positive = [row for row in watch if _to_float(row.get("trading_value_ratio_20d")) and _to_float(row.get("trading_value_ratio_20d")) >= 1.2]
        if positive:
            return 65.0, "관심종목 거래대금이 확인됩니다."
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
    return max(0.0, min(100.0, score)), ". ".join(reasons) or "수급은 중립입니다."


def _risk_penalty(sector: str, row: dict) -> tuple[float, str]:
    score = 50.0
    reasons = []
    if "OVERHEATED_20D" in (row.get("warnings") or []):
        score -= 15
        reasons.append("20일 과열 경고가 있습니다.")
    near_high = _to_float(row.get("near_52w_high_pct"))
    if near_high is not None and near_high >= 95:
        score -= 5
        reasons.append("52주 고가권 부담이 있습니다.")
    if row.get("exclude_from_signal"):
        score -= 5
        reasons.append("투기성 ETF는 주판단에서 제외했습니다.")
    return max(0.0, min(100.0, score)), ". ".join(reasons) or "과열 부담은 제한적입니다."


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


def _build_intraday_checkpoints(sector: str, row: dict) -> list[str]:
    checkpoints = ["09:30 거래대금 확인"]
    if row.get("data_status") == "FRESH":
        checkpoints.append("10:30 섹터 ETF 거래대금 유지 여부")
    if "OVERHEATED_20D" in (row.get("warnings") or []):
        checkpoints.append("장 초반 급등 후 이익실현 여부 확인")
    return checkpoints


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
