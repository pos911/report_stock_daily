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
from src.utils.formatters import add_section


def generate_morning_brief(bundle: dict, report_date: str) -> dict:
    freshness = bundle.get("freshness") or {}
    macro = bundle.get("macro") or {}
    sector_etfs = bundle.get("sector_etfs") or []
    watchlist = bundle.get("watchlist") or []
    rankings = bundle.get("rankings") or []
    readiness = bundle.get("readiness") or {}
    contract_failed_views = bundle.get("contract_failed_views") or []

    regime = build_global_morning_regime(macro, freshness)
    sector_impacts = build_sector_morning_impacts(regime, sector_etfs, rankings, watchlist)
    top_sectors = [row for row in sector_impacts if row.get("label") != "데이터 부족"][:3]
    watchlist_scores = build_watchlist_morning_scores(watchlist, regime, sector_impacts)

    lines = [f"[Morning Brief | {report_date}]"]
    section_no = 1
    section_no = add_section(lines, section_no, "데이터 상태", build_data_status_section(freshness, readiness, contract_failed_views))
    section_no = add_section(lines, section_no, "오늘의 한 줄 판단", build_one_line_judgment_section(regime, top_sectors, freshness, readiness))
    section_no = add_section(lines, section_no, "야간 글로벌 시장", build_global_market_section(macro))
    section_no = add_section(lines, section_no, "한국장 예상 영향", build_korean_impact_section(top_sectors, freshness, readiness))
    section_no = add_section(lines, section_no, "오늘 우선 관찰 테마", build_priority_themes_section(top_sectors, freshness, readiness))
    watchlist_title = "관심종목 장전 점검" if freshness.get("xkrx_is_open") else "관심종목 다음 거래일 점검"
    section_no = add_section(lines, section_no, watchlist_title, build_watchlist_section(watchlist_scores, freshness))
    section_no = add_section(lines, section_no, "오늘의 리스크", build_risk_section(regime, top_sectors, watchlist_scores, freshness, readiness))
    checkpoint_title = "장중 확인 포인트" if freshness.get("xkrx_is_open") else "다음 거래일 확인 포인트"
    section_no = add_section(lines, section_no, checkpoint_title, build_checkpoints_section(top_sectors, freshness, readiness))

    report_text = "\n".join(lines).strip() + "\n"

    snapshot = {
        "report_date": report_date,
        "regime_label": regime.get("regime_label"),
        "regime_score": regime.get("score"),
        "positive_drivers": regime.get("positive_drivers") or ["중립"],
        "negative_drivers": regime.get("negative_drivers") or ["중립"],
        "top_sectors": top_sectors or [{"label": "중립", "sector_group": "주요 테마"}],
        "sector_etf_signals": sector_etfs or [{"label": "미확인"}],
        "watchlist_morning_scores": watchlist_scores or [{"signal_label": "판단 유보"}],
        "risk_flags": [line[2:] for line in build_risk_section(regime, top_sectors, watchlist_scores, freshness, readiness)] or ["리스크 요인 점검"],
        "intraday_checkpoints": [line[2:] for line in build_checkpoints_section(top_sectors, freshness, readiness)] or ["체크포인트 없음"],
        "data_freshness_manifest": freshness,
        "watchlist_coverage_status": freshness.get("watchlist_coverage_status"),
        "stockdata_readiness": readiness,
        "report_allowed_sections": readiness.get("report_allowed_sections") or [],
        "report_blocked_sections": readiness.get("report_blocked_sections") or [],
        "kr_full_market_price_ready": readiness.get("kr_full_market_price_ready"),
        "kis_volume_ranking_ready": readiness.get("kis_volume_ranking_ready"),
        "kis_universe_ready": readiness.get("kis_universe_ready"),
        "display_mode": readiness.get("display_mode"),
        "data_limitation_note": readiness.get("data_limitation_note"),
    }
    return {"report_text": report_text, "snapshot": snapshot}


def save_morning_snapshot(project_root: Path, report_date: str, snapshot: dict) -> Path:
    output_dir = project_root / "outputs" / "snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"morning_{report_date}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
