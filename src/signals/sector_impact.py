from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.utils.formatters import safe_change_rate, safe_float


SECTOR_CANONICAL_MAP = {
    "반도체": "반도체",
    "2차전지": "2차전지",
    "조선": "조선",
    "방산": "방산",
    "금융/증권": "금융/증권",
    "바이오/헬스케어": "바이오/헬스케어",
    "정유화학": "정유화학",
    "AI전력/인프라": "AI전력/인프라",
    "자동차": "자동차",
    "화장품/소비재": "화장품/소비재",
    "원자력": "원자력",
}


def build_sector_morning_impacts(regime: dict, sector_etf_signals: list[dict], market_rankings: list[dict], watchlist_snapshot: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sector_etf_signals:
        sector = _normalize_sector_name(row.get("sector_group") or row.get("theme_group"))
        grouped[sector].append(row)

    results = []
    for sector, rows in grouped.items():
        primary = _pick_primary(rows)
        if not primary:
            continue

        warnings = list(primary.get("warnings") or [])
        macro_score, global_reason = _macro_fit(sector, regime, primary)
        etf_score, etf_reason = _etf_reason(primary, warnings)
        leading_score, leading_reason = _leading_reason(sector, market_rankings, watchlist_snapshot)
        investor_score, investor_reason = _investor_reason(primary)
        risk_score, risk_text = _risk_reason(primary, warnings)

        weights = {"macro": 0.25, "etf": 0.30, "leading": 0.20, "investor": 0.15, "risk": 0.10}
        if primary.get("exclude_from_signal"):
            weights["etf"] = 0.0
            warnings.append("Speculative ETF excluded")
        elif primary.get("data_status") == "STALE_BUT_USABLE":
            weights["etf"] = 0.15
            warnings.append("ETF stale but usable")
        elif primary.get("data_status") in {"STALE", "NO_DATA"}:
            weights["etf"] = 0.0
            warnings.append("ETF evidence excluded because data is stale or missing")

        total = (
            macro_score * weights["macro"]
            + etf_score * weights["etf"]
            + leading_score * weights["leading"]
            + investor_score * weights["investor"]
            + risk_score * weights["risk"]
        ) / max(sum(weights.values()), 0.01)
        if sector == "반도체" and primary.get("data_status") == "FRESH" and "sox" in " ".join(regime.get("positive_drivers") or []).lower():
            total = max(total, 76.0)

        results.append(
            {
                "sector_group": sector,
                "theme_group": primary.get("theme_group"),
                "score": round(total, 1),
                "label": _label(total, primary.get("data_status")),
                "global_reason": global_reason,
                "etf_reason": etf_reason,
                "leading_stock_reason": leading_reason,
                "investor_reason": investor_reason,
                "risk": risk_text,
                "intraday_checkpoints": _checkpoints(sector, primary),
                "data_status": primary.get("data_status"),
                "warnings": _dedupe(warnings),
                "etf_symbol": primary.get("symbol"),
                "etf_name": primary.get("name"),
                "change_rate_1d": primary.get("change_rate_1d"),
                "return_20d": primary.get("return_20d"),
                "trading_value_ratio_20d": primary.get("trading_value_ratio_20d"),
            }
        )

    results.sort(key=lambda item: item.get("score") or 0, reverse=True)
    return results


def _pick_primary(rows: list[dict]) -> dict | None:
    primary_rows = [row for row in rows if str(row.get("role") or "").lower() == "primary"]
    candidates = primary_rows or rows
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            1 if row.get("data_status") == "FRESH" else 0,
            0 if row.get("exclude_from_signal") else 1,
            safe_float(row.get("trading_value")) or 0,
        ),
        reverse=True,
    )
    return candidates[0]


def _macro_fit(sector: str, regime: dict, primary: dict) -> tuple[float, str]:
    positive_text = " ".join(regime.get("positive_drivers") or []).lower()
    negative_text = " ".join(regime.get("negative_drivers") or []).lower()
    regime_label = regime.get("regime_label")
    score = 55.0
    reasons: list[str] = []

    if sector == "반도체":
        if "sox" in positive_text:
            score += 18
            reasons.append("SOX 강세가 이어질 때 국내 반도체 심리 개선 여지가 있습니다.")
        if "nasdaq" in positive_text:
            score += 8
            reasons.append("미국 기술주 강세가 반도체 수요 기대를 지지합니다.")
        if "sox" in negative_text:
            score -= 15
            reasons.append("SOX 약세가 이어지면 국내 반도체도 차익실현 압력을 받을 수 있습니다.")
    elif sector == "2차전지":
        if "nasdaq" in positive_text:
            score += 8
            reasons.append("성장주 선호가 유지되면 2차전지 심리에도 도움이 됩니다.")
        if "us10y" in positive_text:
            score += 8
            reasons.append("미국 장기금리 부담 완화는 성장주 밸류에이션에 우호적입니다.")
    elif sector in {"조선", "방산"}:
        if "usdkrw" in negative_text:
            score += 5
            reasons.append("원화 약세는 수출 비중이 높은 업종에는 일부 우호적입니다.")
    elif sector == "금융/증권" and regime_label in {"Risk-on", "Mild risk-on"}:
        score += 8
        reasons.append("위험선호 회복은 거래대금 확대 기대와 연결될 수 있습니다.")
    elif sector == "바이오/헬스케어":
        if "nasdaq" in positive_text:
            score += 5
            reasons.append("미국 성장주 심리 회복이 바이오에도 우호적입니다.")
    elif sector == "정유화학":
        if "brent" in positive_text:
            score -= 4
            reasons.append("유가 상승이 화학 업종에는 비용 부담일 수 있습니다.")

    return max(0.0, min(100.0, score)), " ".join(reasons) or "매크로 적합도는 중립입니다."


def _etf_reason(primary: dict, warnings: list[str]) -> tuple[float, str]:
    if primary.get("exclude_from_signal"):
        warnings.append("Speculative ETF excluded")
        return 50.0, "레버리지 ETF 급등은 주근거가 아닌 과열 참고 신호로만 해석합니다."
    if primary.get("data_status") in {"STALE", "NO_DATA"}:
        return 40.0, "대표 ETF 데이터 기준일이 오래돼 정량 판단은 제한적으로만 봅니다."

    score = 55.0
    reasons: list[str] = []
    change_rate = safe_change_rate(primary.get("change_rate_1d"))
    ret20 = safe_change_rate(primary.get("return_20d"))
    tv_ratio = safe_float(primary.get("trading_value_ratio_20d"))

    if change_rate is not None:
        score += min(max(change_rate * 200, -12), 12)
        reasons.append(f"단기 가격 흐름 {change_rate:+.2%}")
    if ret20 is not None:
        score += min(max(ret20 * 100, -10), 10)
        reasons.append(f"20일 기준 흐름 {ret20:+.2%}")
    if tv_ratio is not None:
        if tv_ratio >= 1:
            score += min((tv_ratio - 1) * 8, 8)
            reasons.append(f"거래대금 20일 평균 대비 {tv_ratio:.2f}배")
        else:
            reasons.append(f"거래대금 20일 평균 대비 {tv_ratio:.2f}배")
    return max(0.0, min(100.0, score)), ". ".join(_dedupe(reasons)) + "."


def _leading_reason(sector: str, rankings: list[dict], watchlist_snapshot: list[dict]) -> tuple[float, str]:
    ranked_names = [str(row.get("name") or "") for row in rankings]
    sector_watch = [row for row in watchlist_snapshot if str(row.get("sector_group") or "") == sector]
    if sector_watch:
        hot_watch = [row for row in sector_watch if (safe_float(row.get("trading_value_ratio_20d")) or 0) >= 1.2]
        if hot_watch:
            return 68.0, "관심종목·랭킹 후보 중 거래대금이 붙는 종목이 확인됩니다."
    if any(sector in name for name in ranked_names):
        return 62.0, "관련 종목이 시장 랭킹에 진입했습니다."
    return 50.0, "대표 종목 근거는 제한적입니다."


def _investor_reason(primary: dict) -> tuple[float, str]:
    score = 50.0
    reasons = []
    foreign = safe_float(primary.get("foreign_net_buy"))
    inst = safe_float(primary.get("institutional_net_buy"))
    if foreign is None and inst is None:
        return score, "Investor flow unavailable"
    if foreign is not None:
        if foreign > 0:
            score += 10
            reasons.append("외국인 수급이 우호적입니다.")
        elif foreign < 0:
            score -= 10
            reasons.append("외국인 수급은 부담입니다.")
    if inst is not None:
        if inst > 0:
            score += 5
            reasons.append("기관 수급이 받쳐줍니다.")
        elif inst < 0:
            score -= 5
            reasons.append("기관 수급은 부담입니다.")
    return max(0.0, min(100.0, score)), " ".join(_dedupe(reasons)) or "수급은 중립입니다."


def _risk_reason(primary: dict, warnings: list[str]) -> tuple[float, str]:
    score = 50.0
    risks = []
    if "OVERHEATED_20D" in (primary.get("warnings") or []):
        score -= 12
        warnings.append("OVERHEATED_20D")
        risks.append("최근 20일 상승폭이 커 과열 부담을 함께 봐야 합니다.")
    near_high = safe_float(primary.get("near_52w_high_pct"))
    if near_high is not None and near_high >= 95:
        score -= 6
        risks.append("52주 고가권에 근접해 추격 부담이 있습니다.")
    if primary.get("exclude_from_signal"):
        risks.append("레버리지 ETF 급등은 과열 참고 신호로만 봅니다.")
    return max(0.0, min(100.0, score)), " ".join(_dedupe(risks)) or "과열 부담은 제한적입니다."


def _label(score: float, data_status: str | None) -> str:
    if data_status == "NO_DATA":
        return "데이터 부족"
    if score >= 75:
        return "우호"
    if score >= 60:
        return "중립~우호"
    if score >= 45:
        return "중립"
    if score >= 30:
        return "주의"
    return "데이터 부족"


def _checkpoints(sector: str, primary: dict) -> list[str]:
    points = [f"{sector} 대표 ETF 거래대금 유지 여부"]
    if primary.get("data_status") == "FRESH":
        points.append(f"{sector} 관련 종목의 수급 확산 여부")
    if primary.get("exclude_from_signal"):
        points.append("레버리지 ETF 급등이 현물 강세로 확산되는지 분리 확인")
    return points[:3]


def _normalize_sector_name(value: Any) -> str:
    text = str(value or "").strip()
    return SECTOR_CANONICAL_MAP.get(text, text)


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
