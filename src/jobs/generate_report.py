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

from src.notification.telegram_sender import TelegramSender
from src.reports.morning_report import generate_morning_brief, save_morning_snapshot
from src.services.supabase_stockdata_reader import SupabaseStockDataReader
from src.data.supabase_reader import SupabaseReader
from src.analysis.gemini_analyzer import GeminiAnalyzer
from src.utils.formatters import NA_TEXT, is_missing


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_REPORT_TYPES = ("morning", "regular", "closing")
KST = ZoneInfo("Asia/Seoul")
SIGNAL_MODEL_VERSION = "v0.1_unbacktested"


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


def _should_include_kr_sections(calendar_status: dict) -> bool:
    return calendar_status.get("report_market_mode") in {"FULL_REPORT", "KOREA_ONLY", "CALENDAR_UNKNOWN"}


def _should_include_us_sections(calendar_status: dict) -> bool:
    return calendar_status.get("report_market_mode") in {"FULL_REPORT", "US_ONLY", "CALENDAR_UNKNOWN"}


def _should_skip_all_markets(calendar_status: dict) -> bool:
    return calendar_status.get("report_market_mode") == "SKIP_ALL_MARKETS_CLOSED"


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


def _safe_float(value):
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interpret_us_10y_3y_spread(us10y, us3y) -> dict | None:
    us10y_value = _safe_float(us10y)
    us3y_value = _safe_float(us3y)
    if us10y_value is None or us3y_value is None:
        return None

    spread_bp = (us10y_value - us3y_value) * 100
    if spread_bp >= 25:
        regime = "mildly_positive"
        plain = "정상 범위의 양(+)의 금리차입니다."
    elif spread_bp > -25:
        regime = "flat"
        plain = "장단기 금리차가 평평합니다."
    elif spread_bp > -75:
        regime = "mildly_inverted"
        plain = "완만한 역전 구간입니다."
    else:
        regime = "deeply_inverted"
        plain = "강한 역전 구간입니다."
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
    signal = 0

    close_price = _safe_float(price.get("close_price"))
    ma5 = _safe_float(features.get("moving_avg_5"))
    ma20 = _safe_float(features.get("moving_avg_20"))
    return_5d = _safe_float(features.get("return_5d"))
    volatility = _safe_float(features.get("volatility_20d"))
    foreign_z = _safe_float(features.get("foreign_flow_zscore"))
    short_ratio = _safe_float((snapshot.get("short_selling") or {}).get("short_ratio"))
    ranking = ranking_lookup.get(snapshot.get("symbol"), {})

    if return_5d is not None:
        signal += 1 if return_5d > 0 else -1 if return_5d < 0 else 0
    if close_price is not None and ma5 is not None:
        signal += 1 if close_price >= ma5 else -1
    if close_price is not None and ma20 is not None:
        signal += 1 if close_price >= ma20 else -1
    if ranking.get("trading_value_rank") is not None and int(ranking.get("trading_value_rank")) <= 5:
        signal += 1
    if _safe_float(supply.get("foreign_net_buy")) not in (None, 0):
        signal += 1 if _safe_float(supply.get("foreign_net_buy")) > 0 else -1
    if foreign_z is not None and foreign_z >= 1:
        signal += 1
    if volatility is not None and volatility >= 0.05:
        signal -= 1
    if short_ratio is not None and short_ratio >= 5:
        signal -= 1
    if _safe_float(macro.get("usdkrw")) is not None and _safe_float(macro.get("usdkrw")) >= 1450:
        signal -= 1

    if close_price is None:
        label = "데이터 부족"
    elif signal >= 4:
        label = "비중확대 후보"
    elif signal >= 1:
        label = "보유/관찰"
    elif signal <= -2:
        label = "리스크 축소 후보"
    else:
        label = "관망"

    return {
        "total_signal_score": signal,
        "price_momentum_score": signal,
        "volume_trading_score": 0,
        "supply_score": 0,
        "valuation_score": 0,
        "risk_score": 0,
        "macro_fit_score": 0,
        "confidence": "medium" if close_price is not None else "low",
        "label": label,
        "strategy_memo": "장 초반 거래대금 유지 여부 확인",
        "signal_model_version": SIGNAL_MODEL_VERSION,
    }


def _format_watchlist_section(prepared_snapshots: list[dict], title: str = "## Watchlist Summary") -> list[str]:
    lines = [title, f"- Watchlist {len(prepared_snapshots):,} items", "_Source contract: `report_watchlist_snapshot_view`_"]
    if not prepared_snapshots:
        lines.append("- WARNING: watchlist empty")
        return lines
    for snapshot in prepared_snapshots[:5]:
        lines.append(f"- {snapshot.get('name')}({snapshot.get('symbol')})")
    return lines


def _save_report(report_type: str, report_content: str, now_kst: datetime.datetime) -> Path:
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    file_path = reports_dir / f"daily_quant_report_{report_type}_{now_kst.strftime('%Y%m%d_%H%M')}.md"
    file_path.write_text(report_content, encoding="utf-8")
    logger.info("Report saved: %s", file_path)
    return file_path


def _build_simple_non_morning_report(report_type: str, report_date: str, bundle: dict) -> str:
    readiness = bundle.get("readiness") or {}
    allowed_sections = readiness.get("report_allowed_sections") or []
    blocked_sections = readiness.get("report_blocked_sections") or []
    
    freshness = bundle.get("freshness") or {}
    macro = bundle.get("macro") or {}
    rankings = bundle.get("rankings") or []
    watchlist = bundle.get("watchlist") or []

    lines = [
        f"[{report_type.title()} Brief | {report_date}]",
        "",
        "1. 시장 상태",
        f"- 한국장: {'개장' if freshness.get('xkrx_is_open') else '휴장'}",
        f"- 미국장: {'개장' if freshness.get('xnys_is_open') else '휴장'}",
        f"- 국내 전종목 가격 커버리지: {'충족' if readiness.get('kr_full_market_price_ready') else '미충족'}",
        f"- 사용 가능: {', '.join(allowed_sections) if allowed_sections else '없음'}",
        f"- 생략: {', '.join(blocked_sections) if blocked_sections else '없음'}",
        "",
        "2. 주요 지표",
        f"- KOSPI: {macro.get('kospi') or NA_TEXT}",
        f"- KOSDAQ: {macro.get('kosdaq') or NA_TEXT}",
        f"- USD/KRW: {macro.get('usdkrw') or NA_TEXT}",
        f"- US 10Y-3Y: {macro.get('us10y_us3y_spread_bp') or NA_TEXT}bp",
        "",
    ]
    
    # 3. 국내 데이터 범위
    if not readiness.get('kr_full_market_price_ready'):
        lines.extend([
            "3. 국내 데이터 범위",
            "- 국내 전종목 가격 커버리지가 부족해 거래대금·시총 기준 전체시장 Top은 생략합니다.",
            "- 거래량 상위는 KIS ranking 기준으로만 참고합니다.",
            ""
        ])

    # 4. KIS 거래량 순위 기준
    if "kis_volume_top" in allowed_sections:
        kis_rankings = [r for r in rankings if r.get("source") == "KIS" and r.get("rank_type") == "volume"]
        if kis_rankings:
            lines.append("4. KIS 거래량 순위 기준")
            for r in kis_rankings[:5]:
                symbol = r.get("symbol")
                name = r.get("name") or r.get("stock_name") or symbol
                val = r.get("rank_value")
                if val is not None:
                    try:
                        val_str = f"{float(val):,.0f}"
                        lines.append(f"- {name}({symbol}): {val_str}")
                    except (ValueError, TypeError):
                        lines.append(f"- {name}({symbol})")
                else:
                    lines.append(f"- {name}({symbol})")
            lines.append("")

    # 5. 관심종목·랭킹 후보 Signal
    if "watchlist_signal" in allowed_sections:
        if watchlist:
            lines.append("5. 관심종목·랭킹 후보 기반 Signal")
            for w in watchlist[:5]:
                name = w.get("name") or w.get("symbol")
                price = w.get("close_price")
                change = w.get("change_rate_1d")
                score = w.get("signal_score")
                label = w.get("label")
                
                parts = [f"- {name}"]
                if price is not None:
                    try:
                        parts.append(f"{float(price):,.0f}원")
                    except (ValueError, TypeError):
                        pass
                if change is not None:
                    try:
                        parts.append(f"({float(change):+.2%})")
                    except (ValueError, TypeError):
                        pass
                if label:
                    parts.append(f"| {label}")
                if score is not None:
                    parts.append(f"(Score: {score})")
                
                lines.append(" ".join(parts))
            lines.append("")

    # 6. 비고
    if blocked_sections:
        lines.append(f"※ 생략된 섹션: {', '.join(blocked_sections)}")
    
    return "\n".join(lines).strip() + "\n"


def _safe_get_analyzer() -> GeminiAnalyzer | None:
    try:
        return GeminiAnalyzer()
    except Exception as exc:
        logger.warning("GeminiAnalyzer initialization failed: %s", exc)
        return None


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

    bundle = reader.get_report_contract_bundle(report_type="morning", target_date=normalized_report_date)
    readiness = bundle.get("readiness") or {}
    logger.info("stockdata_readiness=%s", readiness)

    if report_type == "morning":
        result = generate_morning_brief(bundle, normalized_report_date)
        report_content = result["report_text"]
        snapshot_path = save_morning_snapshot(project_root, normalized_report_date, result["snapshot"])
        logger.info("Morning snapshot saved: %s", snapshot_path)
        logger.info("Gemini call count and purpose=0 / []") # Will update this when integrated
        logger.info("Naver call count=0")
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
