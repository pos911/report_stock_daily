from __future__ import annotations

import json
from pathlib import Path

from src.reports.morning_sections import (
    build_checkpoints_section,
    build_data_status_section,
    build_global_market_section,
    build_korean_impact_section,
    build_one_line_judgment_section,
    build_priority_themes_section,
    build_risk_section,
    build_watchlist_section,
)
from src.signals.morning_regime import build_global_morning_regime
from src.signals.sector_impact import build_sector_morning_impacts
from src.signals.watchlist_morning import build_watchlist_morning_scores


def generate_morning_brief(bundle: dict, report_date: str) -> dict:
    freshness = bundle["freshness"]
    macro = bundle["macro"]
    sector_etfs = bundle["sector_etfs"]
    watchlist = bundle["watchlist"]
    rankings = bundle["rankings"]

    regime = build_global_morning_regime(macro, freshness)
    sector_impacts = build_sector_morning_impacts(regime, sector_etfs, rankings, watchlist)
    eligible_sectors = [
        row for row in sector_impacts
        if row.get("data_status") != "STALE"
        or (row.get("score", 0) >= 60 and row.get("leading_stock_reason") != "대표 종목 확인 근거는 제한적입니다.")
    ]
    top_sectors = eligible_sectors[:3]
    watchlist_scores = build_watchlist_morning_scores(watchlist, regime, sector_impacts)

    sections = [
        build_data_status_section(freshness, bundle),
        build_one_line_judgment_section(regime, top_sectors, freshness),
        build_global_market_section(macro),
        build_korean_impact_section(top_sectors, freshness),
        build_priority_themes_section(top_sectors, freshness),
        build_watchlist_section(watchlist_scores, freshness),
        build_risk_section(regime, top_sectors, watchlist_scores, freshness),
        build_checkpoints_section(top_sectors, freshness),
    ]
    lines = [f"[Morning Brief | {report_date}]"]
    for section in sections:
        lines.append("")
        lines.extend(section)
    report_text = "\n".join(lines).strip() + "\n"

    snapshot = {
        "report_date": report_date,
        "regime_label": regime.get("regime_label"),
        "regime_score": regime.get("score"),
        "positive_drivers": regime.get("positive_drivers") or ["positive driver unavailable"],
        "negative_drivers": regime.get("negative_drivers") or ["negative driver unavailable"],
        "top_sectors": top_sectors,
        "sector_etf_signals": sector_etfs or [{"warning": "sector etf signals unavailable"}],
        "watchlist_morning_scores": watchlist_scores or [{"warning": "watchlist morning scores unavailable"}],
        "risk_flags": [line[2:] for line in build_risk_section(regime, top_sectors, watchlist_scores, freshness)[1:]] or ["risk flags unavailable"],
        "intraday_checkpoints": [line[2:] for line in build_checkpoints_section(top_sectors, freshness)[1:]] or ["intraday checkpoints unavailable"],
        "data_freshness_manifest": freshness,
        "watchlist_coverage_status": freshness.get("watchlist_coverage_status"),
    }
    return {"report_text": report_text, "snapshot": snapshot}


def save_morning_snapshot(project_root: Path, report_date: str, snapshot: dict) -> Path:
    output_dir = project_root / "outputs" / "snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"morning_{report_date}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
