from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from zoneinfo import ZoneInfo


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.analysis.gemini_analyzer import GeminiAnalyzer
from src.data.supabase_reader import SupabaseReader
from src.notification.telegram_sender import TelegramSender
from src.reports.morning_report import generate_morning_brief, save_morning_snapshot
from src.services.supabase_stockdata_reader import SupabaseStockDataReader
from src.utils.formatters import (
    NA_TEXT,
    add_section,
    format_number,
    format_pct,
    format_price,
    format_sections_list,
    join_sentences,
    safe_change_rate,
    safe_float,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_REPORT_TYPES = ("morning", "regular", "closing")
KST = ZoneInfo("Asia/Seoul")


def _parse_args():
    parser = argparse.ArgumentParser(description="Daily Quant Report Generator")
    parser.add_argument("--type", dest="report_type", default="regular")
    parser.add_argument("--date", dest="report_date", help="Report date in YYYYMMDD or YYYY-MM-DD (KST basis).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-send", action="store_true")
    args = parser.parse_args()
    report_type = (args.report_type or "regular").strip().lower()
    if report_type not in VALID_REPORT_TYPES:
        parser.error(f"invalid choice: '{args.report_type}' (choose from {', '.join(VALID_REPORT_TYPES)})")
    args.report_type = report_type
    return args


def _get_now_kst() -> datetime.datetime:
    return datetime.datetime.now(KST)


def _normalize_report_date(report_date: str | None, now_kst: datetime.datetime) -> str:
    if not report_date:
        return now_kst.date().isoformat()
    text = str(report_date).strip()
    if len(text) == 8 and text.isdigit():
        return datetime.datetime.strptime(text, "%Y%m%d").date().isoformat()
    return datetime.date.fromisoformat(text).isoformat()


def _should_skip_all_markets(calendar_status: dict) -> bool:
    return calendar_status.get("report_market_mode") == "SKIP_ALL_MARKETS_CLOSED"


def _should_include_kr_sections(calendar_status: dict) -> bool:
    return calendar_status.get("report_market_mode") in {"FULL_REPORT", "KOREA_ONLY", "CALENDAR_UNKNOWN"}


def _should_include_us_sections(calendar_status: dict) -> bool:
    return calendar_status.get("report_market_mode") in {"FULL_REPORT", "US_ONLY", "CALENDAR_UNKNOWN"}


def _build_market_closed_skip_text(report_type: str, now_kst: datetime.datetime, calendar_status: dict) -> str:
    return "\n".join(
        [
            f"SKIPPED_REPORT_MARKET_CLOSED | type={report_type}",
            f"- generated_at: {now_kst.strftime('%Y-%m-%d %H:%M KST')}",
            f"- report_date: {calendar_status.get('report_date')}",
            f"- XKRX: closed ({calendar_status.get('xkrx_reason')}) / prev {calendar_status.get('xkrx_previous_trading_day') or NA_TEXT}",
            f"- XNYS: closed ({calendar_status.get('xnys_reason')}) / prev {calendar_status.get('xnys_previous_trading_day') or NA_TEXT}",
            "- status: SKIPPED_REPORT_MARKET_CLOSED",
        ]
    )


def _save_report(report_type: str, report_content: str, now_kst: datetime.datetime) -> Path:
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    file_path = reports_dir / f"daily_quant_report_{report_type}_{now_kst.strftime('%Y%m%d_%H%M')}.md"
    file_path.write_text(report_content, encoding="utf-8")
    logger.info("Report saved: %s", file_path)
    return file_path


def _interpret_us_10y_3y_spread(us10y, us3y) -> dict | None:
    us10y_value = safe_float(us10y)
    us3y_value = safe_float(us3y)
    if us10y_value is None or us3y_value is None:
        return None
    spread_bp = (us10y_value - us3y_value) * 100
    if spread_bp >= 25:
        regime = "mildly_positive"
        plain = "정상 곡선 구간으로 성장 기대와 기간 프리미엄을 함께 봅니다."
    elif spread_bp > -25:
        regime = "flat"
        plain = "경기와 정책 기대가 혼재된 평탄 구간입니다."
    elif spread_bp > -75:
        regime = "mildly_inverted"
        plain = "완만한 역전 구간으로 경기 둔화 우려를 함께 봅니다."
    else:
        regime = "deeply_inverted"
        plain = "깊은 역전 구간으로 위험자산 해석은 보수적으로 보는 편이 좋습니다."
    return {
        "us10y": us10y_value,
        "us3y": us3y_value,
        "spread_bp": spread_bp,
        "regime": regime,
        "plain_korean_summary": plain,
    }


def _score_watchlist_snapshot(snapshot: dict, ranking_lookup: dict, macro: dict) -> dict:
    price = snapshot.get("price") or {}
    supply = snapshot.get("supply") or {}
    features = snapshot.get("features") or {}
    score = 0

    close_price = safe_float(price.get("close_price"))
    ma5 = safe_float(features.get("moving_avg_5"))
    ma20 = safe_float(features.get("moving_avg_20"))
    return_5d = safe_float(features.get("return_5d"))
    volatility = safe_float(features.get("volatility_20d"))
    foreign_z = safe_float(features.get("foreign_flow_zscore"))
    short_ratio = safe_float((snapshot.get("short_selling") or {}).get("short_ratio"))
    ranking = ranking_lookup.get(snapshot.get("symbol"), {})

    if return_5d is not None:
        score += 1 if return_5d > 0 else -1 if return_5d < 0 else 0
    if close_price is not None and ma5 is not None:
        score += 1 if close_price >= ma5 else -1
    if close_price is not None and ma20 is not None:
        score += 1 if close_price >= ma20 else -1
    if ranking.get("trading_value_rank") is not None and int(ranking.get("trading_value_rank")) <= 5:
        score += 1
    foreign_flow = safe_float(supply.get("foreign_net_buy"))
    if foreign_flow not in (None, 0):
        score += 1 if foreign_flow > 0 else -1
    if foreign_z is not None and foreign_z >= 1:
        score += 1
    if volatility is not None and volatility >= 0.05:
        score -= 1
    if short_ratio is not None and short_ratio >= 5:
        score -= 1
    if safe_float(macro.get("usdkrw")) is not None and safe_float(macro.get("usdkrw")) >= 1450:
        score -= 1

    if close_price is None:
        label = "판단 유보"
    elif score >= 4:
        label = "비중확대 후보"
    elif score >= 1:
        label = "보유·관찰"
    elif score <= -2:
        label = "리스크 축소 후보"
    else:
        label = "관망"

    return {
        "total_signal_score": score,
        "price_momentum_score": score,
        "volume_trading_score": 0,
        "supply_score": 0,
        "valuation_score": 0,
        "risk_score": 0,
        "macro_fit_score": 0,
        "confidence": "medium" if close_price is not None else "low",
        "label": label,
        "strategy_memo": "개장 초반 거래대금과 수급 유지 여부 확인",
        "signal_model_version": "v0.1_unbacktested",
    }


def _format_watchlist_section(prepared_snapshots: list[dict], title: str = "## Watchlist Summary") -> list[str]:
    lines = [title, f"- Watchlist {len(prepared_snapshots):,} items", "_Source contract: `report_watchlist_snapshot_view`_"]
    if not prepared_snapshots:
        lines.append("- watchlist empty")
        return lines
    for snapshot in prepared_snapshots[:5]:
        lines.append(f"- {snapshot.get('name')}({snapshot.get('symbol')})")
    return lines


def _safe_get_analyzer() -> GeminiAnalyzer | None:
    try:
        return GeminiAnalyzer()
    except Exception as exc:
        logger.warning("GeminiAnalyzer initialization failed: %s", exc)
        return None


def _build_simple_non_morning_report(report_type: str, report_date: str, bundle: dict) -> str:
    readiness = bundle.get("readiness") or {}
    freshness = bundle.get("freshness") or {}
    macro = bundle.get("macro") or {}
    rankings = bundle.get("rankings") or []
    watchlist = bundle.get("watchlist") or []

    title = "Regular Brief" if report_type == "regular" else "Closing Brief"
    lines = [f"[{title} | {report_date}]"]
    section_no = 1

    state_body = [
        f"- 한국장: {'개장' if freshness.get('xkrx_is_open') else '휴장'}",
        f"- 미국장: {'개장' if freshness.get('xnys_is_open') else '휴장'}",
        f"- 국내 데이터 모드: {readiness.get('display_mode') or '미확인'}",
        f"- 사용 가능: {format_sections_list(readiness.get('allowed_korean_sections') or [])}",
        f"- 생략: {format_sections_list(_translate_blocked(readiness.get('blocked_korean_sections') or []))}",
    ]
    section_no = add_section(lines, section_no, "시장 상태" if report_type == "regular" else "마감 데이터 상태", state_body)

    summary_title = "장중 핵심 요약" if report_type == "regular" else "마감 요약"
    summary_body = [
        f"- KOSPI: {format_number(macro.get('kospi'))}",
        f"- KOSDAQ: {format_number(macro.get('kosdaq'))}",
        f"- USD/KRW: {format_number(macro.get('usdkrw'))}",
        f"- 한 줄 판단: {_build_session_summary(report_type, macro, readiness)}",
    ]
    section_no = add_section(lines, section_no, summary_title, summary_body)

    limitation_body = []
    if readiness.get("display_mode") != "FULL_MARKET":
        limitation_body.append(f"- {readiness.get('data_limitation_note')}")
    if readiness.get("display_mode") == "FULL_MARKET":
        limitation_body.append("- 전체시장 거래대금·시총 Top 해석이 가능합니다.")
    section_no = add_section(lines, section_no, "국내 데이터 범위", limitation_body)

    if "kis_volume_top" in (readiness.get("report_allowed_sections") or []):
        volume_rows = [
            row for row in rankings
            if row.get("rank_type") == "volume" and row.get("source") == "KIS"
        ][:5]
        volume_body = [
            f"- {row.get('name') or row.get('symbol')}({row.get('symbol')}), {row.get('market')}, rank {row.get('rank')}, 거래량 {format_number(row.get('volume'), 0)}"
            for row in volume_rows
        ]
        volume_title = "KIS 거래량 순위 기준" if report_type == "regular" else "KIS 거래량 순위 기준 마감 점검"
        section_no = add_section(lines, section_no, volume_title, volume_body)

    if "watchlist_signal" in (readiness.get("report_allowed_sections") or []):
        signal_body = []
        for row in watchlist[:5]:
            derived = _derive_watchlist_signal(row)
            parts = [f"- {row.get('name') or row.get('symbol')}({row.get('symbol')})"]
            if row.get("close_price") is not None:
                parts.append(format_price(row.get("close_price")))
            change_rate = safe_change_rate(row.get("change_rate_1d"))
            if change_rate is not None:
                parts.append(format_pct(change_rate))
            label = row.get("signal_label") or row.get("label") or derived["label"]
            if label:
                parts.append(label)
            score_value = row.get("signal_score")
            if score_value is None:
                score_value = derived["score"]
            if score_value is not None:
                parts.append(f"score {safe_float(score_value):.1f}")
            signal_body.append(" | ".join(parts))
        title_text = "관심종목·랭킹 후보 Signal" if report_type == "regular" else "관심종목·후보군 마감 점검"
        section_no = add_section(lines, section_no, title_text, signal_body)

    if readiness.get("kr_trading_value_ranking_ready"):
        tv_rows = [row for row in rankings if row.get("rank_type") == "trading_value"][:5]
        tv_body = [
            f"- {row.get('name') or row.get('symbol')}({row.get('symbol')}), {row.get('market')}, {format_number(row.get('trading_value'), 0)}"
            for row in tv_rows
        ]
        section_no = add_section(lines, section_no, "전체시장 거래대금 Top", tv_body)

    if readiness.get("kr_market_cap_ranking_ready"):
        mc_rows = [row for row in rankings if row.get("rank_type") == "market_cap"][:5]
        mc_body = [
            f"- {row.get('name') or row.get('symbol')}({row.get('symbol')}), {row.get('market')}, {format_number(row.get('market_cap'), 0)}"
            for row in mc_rows
        ]
        section_no = add_section(lines, section_no, "전체시장 시총 Top", mc_body)

    checkpoint_title = "오후 체크포인트" if report_type == "regular" else "다음 거래일 체크포인트"
    checkpoint_body = _build_non_morning_checkpoints(report_type, readiness)
    section_no = add_section(lines, section_no, checkpoint_title, checkpoint_body)

    return "\n".join(lines).strip() + "\n"


def _build_session_summary(report_type: str, macro: dict, readiness: dict) -> str:
    usdkrw = safe_float(macro.get("usdkrw"))
    kospi = safe_float(macro.get("kospi"))
    kosdaq = safe_float(macro.get("kosdaq"))
    parts = []
    if kospi is not None and kosdaq is not None:
        parts.append("국내 지수 흐름과 환율을 함께 보는 구간입니다")
    if usdkrw is not None and usdkrw >= 1450:
        parts.append("환율 부담이 커지면 성장주 대응은 보수적으로 보는 편이 좋습니다")
    if "kis_volume_top" in (readiness.get("report_allowed_sections") or []):
        parts.append("KIS 거래량 상위의 지속성과 관심종목 Signal 변화가 핵심입니다")
    if report_type == "closing":
        parts.append("마감 기준으로는 다음 거래일 후보군 압축이 중요합니다")
    return join_sentences(parts, limit=2) or "핵심 지표 중심으로 점검합니다."


def _build_non_morning_checkpoints(report_type: str, readiness: dict) -> list[str]:
    if report_type == "closing":
        return [
            "- 미국장 확인 항목과 환율 방향 재점검",
            "- KIS ranking 후보 지속 여부 확인",
            "- 관심종목 리스크와 다음 거래일 대응 조건 정리",
        ]
    checkpoints = [
        "- 환율과 외국인 선물 방향 확인",
        "- KIS 거래량 상위 지속 여부 확인",
        "- 관심종목 Signal 변화 확인",
    ]
    if readiness.get("display_mode") != "FULL_MARKET":
        checkpoints.append("- 전체시장 Top 대신 관심종목·랭킹 후보 흐름에 집중")
    return checkpoints[:4]


def _derive_watchlist_signal(row: dict) -> dict:
    score = 50.0
    change_rate = safe_change_rate(row.get("change_rate_1d"))
    trading_ratio = safe_float(row.get("trading_value_ratio_20d"))
    foreign_flow = safe_float(row.get("foreign_net_buy"))
    inst_flow = safe_float(row.get("institutional_net_buy"))

    if change_rate is not None:
        score += max(min(change_rate * 150, 12), -12)
    if trading_ratio is not None:
        if trading_ratio >= 1.5:
            score += 10
        elif trading_ratio < 0.8:
            score -= 6
    if foreign_flow is not None:
        score += 6 if foreign_flow > 0 else -6 if foreign_flow < 0 else 0
    if inst_flow is not None:
        score += 4 if inst_flow > 0 else -4 if inst_flow < 0 else 0

    if score >= 75:
        label = "비중확대 후보"
    elif score >= 60:
        label = "보유·관찰"
    elif score >= 45:
        label = "관망"
    elif score >= 30:
        label = "리스크 축소 후보"
    else:
        label = "판단 유보"
    return {"score": max(0.0, min(100.0, score)), "label": label}


def _translate_blocked(values: list[str]) -> list[str]:
    mapping = {
        "kr_full_market_trading_value_top": "전체시장 거래대금 Top",
        "kr_full_market_market_cap_top": "전체시장 시총 Top",
    }
    return [mapping.get(value, value) for value in values]


def run_report(report_type: str, now_kst: datetime.datetime, report_date: str | None = None, send_enabled: bool = True):
    base_reader = SupabaseReader()
    reader = SupabaseStockDataReader(base_reader=base_reader)
    analyzer = _safe_get_analyzer()
    normalized_report_date = _normalize_report_date(report_date, now_kst)
    calendar_status = base_reader.fetch_market_calendar_status(normalized_report_date)
    logger.info("calendar_status=%s", calendar_status)

    if _should_skip_all_markets(calendar_status):
        skip_text = _build_market_closed_skip_text(report_type, now_kst, calendar_status)
        logger.info("\n%s", skip_text)
        return

    bundle = reader.get_report_contract_bundle(report_type=report_type, target_date=normalized_report_date)
    logger.info("stockdata_readiness=%s", bundle.get("readiness") or {})
    logger.info("Gemini analyzer active=%s", bool(analyzer))

    if report_type == "morning":
        result = generate_morning_brief(bundle, normalized_report_date)
        report_content = result["report_text"]
        snapshot_path = save_morning_snapshot(project_root, normalized_report_date, result["snapshot"])
        logger.info("Morning snapshot saved: %s", snapshot_path)
    else:
        report_content = _build_simple_non_morning_report(report_type, normalized_report_date, bundle)

    _save_report(report_type, report_content, now_kst)

    if not send_enabled:
        logger.info("Dry-run or no-send mode enabled. Telegram sending skipped.")
        return

    try:
        sender = TelegramSender()
        sender.send_report(report_content)
    except Exception as exc:
        logger.warning("Telegram send failed (non-fatal): %s", exc)


def main():
    args = _parse_args()
    now_kst = _get_now_kst()
    send_enabled = not (args.dry_run or args.no_send)
    logger.info("=== Daily Report Pipeline start [type=%s dry_run=%s] ===", args.report_type, not send_enabled)
    run_report(args.report_type, now_kst, report_date=args.report_date, send_enabled=send_enabled)


if __name__ == "__main__":
    main()
