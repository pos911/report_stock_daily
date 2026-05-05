from __future__ import annotations

from typing import Any


def build_global_morning_regime(macro_snapshot: dict, freshness: dict) -> dict:
    score = 0.0
    positive_drivers: list[str] = []
    negative_drivers: list[str] = []
    neutral_drivers: list[str] = []
    warnings: list[str] = []

    us_weight = 0.5 if (not freshness.get("xnys_is_open") or freshness.get("carry_forward_fields")) else 1.0

    def add_driver(
        label: str,
        value: Any,
        positive_threshold: float | None,
        negative_threshold: float | None,
        positive_score: float,
        negative_score: float,
        weight: float = 1.0,
    ) -> None:
        nonlocal score
        numeric = _to_float(value)
        if numeric is None:
            warnings.append(f"{label} missing")
            return
        if positive_threshold is not None and numeric >= positive_threshold:
            score += positive_score * weight
            positive_drivers.append(f"{label} {numeric:+.2f}")
        elif negative_threshold is not None and numeric <= negative_threshold:
            score += negative_score * weight
            negative_drivers.append(f"{label} {numeric:+.2f}")
        else:
            neutral_drivers.append(f"{label} {numeric:+.2f}")

    add_driver("S&P500 change", macro_snapshot.get("sp500_change_rate"), 0.005, -0.005, 1.0, -1.0, us_weight)
    add_driver("Nasdaq change", macro_snapshot.get("nasdaq_change_rate"), 0.007, -0.007, 1.0, -1.0, us_weight)
    add_driver("SOX change", macro_snapshot.get("sox_change_rate"), 0.01, -0.01, 2.0, -2.0, us_weight)

    vix_level = _to_float(macro_snapshot.get("vix"))
    vix_change = _to_float(macro_snapshot.get("vix_change_rate"))
    if vix_level is None and vix_change is None:
        warnings.append("VIX missing")
    elif (vix_level is not None and vix_level <= 15) or (vix_change is not None and vix_change <= -0.05):
        score += 1.0 * us_weight
        positive_drivers.append("VIX calm")
    elif (vix_level is not None and vix_level >= 20) or (vix_change is not None and vix_change >= 0.05):
        score -= 1.0 * us_weight
        negative_drivers.append("VIX elevated")
    else:
        neutral_drivers.append("VIX neutral")

    usdkrw_change = _to_float(macro_snapshot.get("usdkrw_change_rate"))
    if usdkrw_change is None:
        warnings.append("USDKRW change missing")
    elif usdkrw_change <= -0.003:
        score += 1.0
        positive_drivers.append(f"USDKRW {usdkrw_change:+.2%}")
    elif usdkrw_change >= 0.003:
        score -= 1.0
        negative_drivers.append(f"USDKRW {usdkrw_change:+.2%}")
    else:
        neutral_drivers.append(f"USDKRW {usdkrw_change:+.2%}")

    dxy_change = _to_float(macro_snapshot.get("dxy_change_rate"))
    if dxy_change is None:
        warnings.append("DXY change missing")
    elif dxy_change <= -0.003:
        score += 0.5 * us_weight
        positive_drivers.append(f"DXY {dxy_change:+.2%}")
    elif dxy_change >= 0.003:
        score -= 0.5 * us_weight
        negative_drivers.append(f"DXY {dxy_change:+.2%}")
    else:
        neutral_drivers.append(f"DXY {dxy_change:+.2%}")

    us10y_bp = _to_float(macro_snapshot.get("us10y_change_bp"))
    if us10y_bp is None:
        warnings.append("US10Y change missing")
    elif us10y_bp <= -5:
        score += 1.0 * us_weight
        positive_drivers.append(f"US10Y {us10y_bp:+.1f}bp")
    elif us10y_bp >= 5:
        score -= 1.0 * us_weight
        negative_drivers.append(f"US10Y {us10y_bp:+.1f}bp")
    else:
        neutral_drivers.append(f"US10Y {us10y_bp:+.1f}bp")

    us3y_bp = _to_float(macro_snapshot.get("us3y_change_bp"))
    if us3y_bp is None:
        warnings.append("US3Y change missing")
    elif us3y_bp <= -5:
        score += 0.5 * us_weight
        positive_drivers.append(f"US3Y {us3y_bp:+.1f}bp")
    elif us3y_bp >= 5:
        score -= 0.5 * us_weight
        negative_drivers.append(f"US3Y {us3y_bp:+.1f}bp")
    else:
        neutral_drivers.append(f"US3Y {us3y_bp:+.1f}bp")

    brent_change = _to_float(macro_snapshot.get("brent_change_rate"))
    if brent_change is None:
        warnings.append("Brent missing")
    elif brent_change >= 0.02:
        score -= 1.0
        negative_drivers.append(f"Brent {brent_change:+.2%}")
    elif brent_change <= -0.02:
        score += 0.5
        positive_drivers.append(f"Brent {brent_change:+.2%}")
    else:
        neutral_drivers.append(f"Brent {brent_change:+.2%}")

    breadth_ratio = _to_float(macro_snapshot.get("advancing_ratio"))
    if breadth_ratio is None:
        warnings.append("Market breadth missing")
    elif breadth_ratio >= 0.55:
        score += 1.0
        positive_drivers.append(f"Breadth {breadth_ratio:.1%}")
    elif breadth_ratio <= 0.45:
        score -= 1.0
        negative_drivers.append(f"Breadth {breadth_ratio:.1%}")
    else:
        neutral_drivers.append(f"Breadth {breadth_ratio:.1%}")

    kospi_foreign = _to_float(macro_snapshot.get("kospi_foreign_net_buy"))
    if kospi_foreign is None:
        warnings.append("KOSPI foreign flow missing")
    elif kospi_foreign > 0:
        score += 1.0
        positive_drivers.append("KOSPI foreign net buy")
    elif kospi_foreign < 0:
        score -= 1.0
        negative_drivers.append("KOSPI foreign net sell")
    else:
        neutral_drivers.append("KOSPI foreign flat")

    kospi_inst = _to_float(macro_snapshot.get("kospi_institutional_net_buy"))
    if kospi_inst is None:
        warnings.append("KOSPI institutional flow missing")
    elif kospi_inst > 0:
        score += 0.5
        positive_drivers.append("KOSPI institutional net buy")
    elif kospi_inst < 0:
        score -= 0.5
        negative_drivers.append("KOSPI institutional net sell")
    else:
        neutral_drivers.append("KOSPI institutional flat")

    if score >= 5:
        regime_label = "Risk-on"
        market_tone = "우호"
    elif score >= 2:
        regime_label = "Mild risk-on"
        market_tone = "중립~우호"
    elif score >= -1:
        regime_label = "Neutral"
        market_tone = "중립"
    elif score >= -4.5:
        regime_label = "Cautious"
        market_tone = "주의"
    else:
        regime_label = "Risk-off"
        market_tone = "주의"

    if freshness.get("missing_required_data"):
        warnings.append(f"missing_required_data: {freshness.get('missing_required_data')}")
    if not positive_drivers and not negative_drivers and warnings:
        market_tone = "데이터 부족"

    return {
        "score": round(score, 2),
        "regime_label": regime_label,
        "market_tone": market_tone,
        "positive_drivers": positive_drivers,
        "negative_drivers": negative_drivers,
        "neutral_drivers": neutral_drivers,
        "warnings": warnings,
        "one_line_summary_inputs": {
            "us_weight": us_weight,
            "missing_required_data": freshness.get("missing_required_data"),
            "positive_driver_count": len(positive_drivers),
            "negative_driver_count": len(negative_drivers),
        },
    }


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
