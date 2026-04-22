import argparse
import datetime
import json
import logging
import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.analysis.gemini_analyzer import GeminiAnalyzer
from src.data.supabase_reader import SupabaseReader
from src.notification.telegram_sender import TelegramSender


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_REPORT_TYPES = ("morning", "closing", "regular")


def _parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Daily Quant Report Generator")
    parser.add_argument(
        "--type",
        dest="report_type",
        default="regular",
        help="리포트 유형: morning(07:00), closing(15:30), regular(기타). 기본값: regular",
    )
    args = parser.parse_args()
    
    rt = args.report_type.strip().lower()
    if rt in ("mornig", "morining"):
        rt = "morning"
    if rt not in VALID_REPORT_TYPES:
        parser.error(f"invalid choice: '{args.report_type}' (choose from {', '.join(VALID_REPORT_TYPES)})")
    
    args.report_type = rt
    return args


def _validate_top_volume(top_volume_data: dict) -> bool:
    """Return True if at least one market bucket has data."""
    if not top_volume_data:
        return False
    return any(isinstance(v, list) and len(v) > 0 for v in top_volume_data.values())


def _validate_target_stocks(target_stocks_data: dict) -> bool:
    """Return True if at least one target-stock table has data."""
    if not target_stocks_data:
        return False
    return any(isinstance(v, list) and len(v) > 0 for v in target_stocks_data.values())


def _render_data_guardrails_md(data_guardrails: dict) -> str:
    """Convert guardrail summary into deterministic markdown."""
    if not data_guardrails:
        return "_Guardrail data unavailable._"

    as_of = data_guardrails.get("as_of_kst_datetime", "N/A")
    lag_map = data_guardrails.get("lag_days_by_table") or {}
    zero = data_guardrails.get("zero_volume_guardrail") or {}
    alerts = data_guardrails.get("pipeline_alert_logs") or []

    lag_lines = []
    for table_name in sorted(lag_map.keys()):
        lag = lag_map.get(table_name)
        lag_text = "N/A" if lag is None else str(lag)
        lag_lines.append(f"- `{table_name}`: lag_days={lag_text}")
    lag_block = "\n".join(lag_lines) if lag_lines else "- (no lag data)"

    latest_zero = zero.get("latest_zero_volume_pct")
    prev_zero = zero.get("previous_zero_volume_pct")
    delta_zero = zero.get("delta_pct")
    latest_base = zero.get("latest_base_date", "N/A")
    prev_base = zero.get("previous_base_date", "N/A")

    def _pct_text(value):
        return "N/A" if value is None else f"{value}%"

    zero_line = (
        f"- latest={_pct_text(latest_zero)} (base_date={latest_base}), "
        f"prev={_pct_text(prev_zero)} (base_date={prev_base}), "
        f"delta={_pct_text(delta_zero)}"
    )

    if alerts:
        alert_lines = []
        for item in alerts[:10]:
            error_message = item.get("error_message")
            error_suffix = f" | error={error_message}" if error_message else ""
            occurrences = item.get("occurrences")
            occurrences_suffix = f" | occurrences={occurrences}" if occurrences else ""
            alert_lines.append(
                f"- `{item.get('target_date')}` | `{item.get('job_name')}` | "
                f"`{item.get('status')}` | records={item.get('records_processed')}"
                f"{occurrences_suffix}{error_suffix}"
            )
        alerts_block = "\n".join(alert_lines)
    else:
        alerts_block = "- 최근 3일 WARN/FAIL 없음"

    return (
        f"**As of (KST)**: {as_of}\n\n"
        f"### Table Freshness (lag_days)\n{lag_block}\n\n"
        f"### Zero-Volume Guardrail\n{zero_line}\n\n"
        f"### Pipeline Alerts (recent 3 days)\n{alerts_block}\n"
    )


def _extract_md_section(text: str, heading: str) -> str:
    """Extract body for a markdown heading like '### 뉴스 요약'."""
    if not text:
        return ""

    lines = text.splitlines()
    target = f"### {heading}"
    collecting = False
    collected = []

    for line in lines:
        stripped = line.strip()
        if stripped == target:
            collecting = True
            continue
        if collecting and stripped.startswith("### "):
            break
        if collecting:
            collected.append(line.rstrip())

    return "\n".join(collected).strip()


def _build_telegram_warning(data_guardrails: dict) -> str:
    """Return a short one-line warning only when guardrails look abnormal."""
    if not data_guardrails:
        return "데이터 점검 필요: 데이터 품질 정보를 불러오지 못했습니다."

    issues = []
    lag_map = data_guardrails.get("lag_days_by_table") or {}
    alerts = data_guardrails.get("pipeline_alert_logs") or []
    zero = data_guardrails.get("zero_volume_guardrail") or {}

    critical_tables = (
        "normalized_stock_prices_daily",
        "normalized_stock_supply_daily",
        "normalized_global_macro_daily",
        "feature_store_daily",
    )
    stale_tables = [
        table_name
        for table_name in critical_tables
        if isinstance(lag_map.get(table_name), int) and lag_map.get(table_name, 0) > 1
    ]
    if stale_tables:
        issues.append(f"지연 테이블 {', '.join(stale_tables[:3])}")

    if alerts:
        issues.append(f"파이프라인 경고 {len(alerts)}건")

    latest_zero = zero.get("latest_zero_volume_pct")
    delta_zero = zero.get("delta_pct")
    if isinstance(latest_zero, (int, float)) and latest_zero >= 10:
        issues.append(f"거래량 0 종목 비율 {latest_zero:.1f}%")
    if isinstance(delta_zero, (int, float)) and abs(delta_zero) >= 20:
        issues.append(f"거래량 0 종목 비율 급변 {delta_zero:+.1f}%p")

    if not issues:
        return ""

    return "데이터 점검 필요: " + ", ".join(issues[:3])


def _build_telegram_report(
    report_label: str,
    generation_time_str: str,
    market_summary_md: str,
    stock_analysis_md: str,
    data_guardrails: dict,
) -> str:
    """Create a Telegram-first message focused on financial analysis."""
    market_one_liner = _extract_md_section(market_summary_md, "시장 한줄 요약")
    market_points = _extract_md_section(market_summary_md, "핵심 포인트")
    news_summary = _extract_md_section(market_summary_md, "뉴스 요약")
    investment_implications = _extract_md_section(market_summary_md, "투자 시사점")
    market_view = _extract_md_section(market_summary_md, "오늘의 시장 판단")
    target_stock_analysis = _extract_md_section(stock_analysis_md, "관심 종목 분석")
    warning_line = _build_telegram_warning(data_guardrails)

    blocks = [
        f"# 데일리 퀀트 리포트 - {report_label}",
        f"> **Generated at**: {generation_time_str}",
        "",
        "## 1. 시장 분석",
    ]

    if market_view:
        blocks.extend(["", f"### 오늘의 시장 판단\n{market_view}"])
    if market_one_liner:
        blocks.extend(["", f"### 시장 한줄 요약\n{market_one_liner}"])

    blocks.extend(["", "## 2. 주요 시황 및 뉴스"])
    if market_points:
        blocks.extend(["", f"### 핵심 포인트\n{market_points}"])
    if news_summary:
        blocks.extend(["", f"### 주요 뉴스\n{news_summary}"])
    if investment_implications:
        blocks.extend(["", f"### 투자 시사점\n{investment_implications}"])

    blocks.extend(["", "## 3. 지정 종목 투자 분석"])
    if target_stock_analysis:
        blocks.extend(["", target_stock_analysis])
    else:
        blocks.extend(["", "_지정 종목 분석 데이터가 없어 이번 메시지에서는 제외했습니다._"])

    if warning_line:
        blocks.extend(["", "---", "", f"주의: {warning_line}"])

    return "\n".join(blocks).strip() + "\n"


def main():
    args = _parse_args()
    report_type = args.report_type
    kst_tz = datetime.timezone(datetime.timedelta(hours=9))
    now_kst = datetime.datetime.now(kst_tz)

    logger.info(f"=== Daily Report Pipeline 시작 [type={report_type}] ===")

    target_stocks_path = project_root / "config" / "target_stocks.json"
    target_symbols = []
    if target_stocks_path.exists():
        with open(target_stocks_path, "r", encoding="utf-8") as f:
            targets = json.load(f)
            target_symbols = [t["symbol"] for t in targets if t.get("enabled", True)]

    try:
        reader = SupabaseReader()
        analyzer = GeminiAnalyzer()
    except Exception as exc:
        logger.error(f"모듈 초기화 실패: {exc}")
        return

    logger.info("1. 매크로/시장 폭 데이터 수집 중...")
    macro_market_data = reader.fetch_macro_and_market_data()
    macro_data = (
        macro_market_data.get("normalized_global_macro_daily")
        or macro_market_data.get("normalized_macro_series")
    )
    if not macro_data:
        logger.error(
            "매크로 데이터를 수집하지 못했습니다. 시장 요약 품질이 저하될 수 있습니다."
        )

    logger.info("2. 거래대금 상위 종목 데이터 수집 중 (KOSPI/KOSDAQ/ETF 각 10개)...")
    top_volume_data = reader.fetch_top_volume_stocks(limit=10)
    if not _validate_top_volume(top_volume_data):
        logger.error("거래대금 상위 종목 데이터를 수집하지 못했습니다. (Clean Skip 대상)")
        top_volume_data = None

    logger.info(f"3. 관심 종목 데이터 수집 중 ({len(target_symbols)}개)...")
    if target_symbols:
        target_stocks_data = reader.fetch_target_stocks_data(target_symbols)
        if not _validate_target_stocks(target_stocks_data):
            logger.error("관심 종목 데이터를 수집하지 못했습니다. (Clean Skip 대상)")
            target_stocks_data = {}
    else:
        target_stocks_data = {}

    logger.info("4. 뉴스 문서 수집 중...")
    raw_news_text = reader.fetch_news_document()
    news_text = reader.prepare_news_context(raw_news_text)

    logger.info("5. 데이터 품질 가드레일 점검 중...")
    data_guardrails = reader.fetch_data_quality_guardrails()

    generation_time_str = now_kst.strftime("%Y-%m-%d %H:%M (KST)")

    type_label_map = {
        "morning": "오전 브리핑",
        "closing": "마감 분석",
        "regular": "정규 리포트",
    }
    report_label = type_label_map.get(report_type, "데일리 리포트")

    report_content = (
        f"# 데일리 퀀트 리포트 - {report_label}\n"
        f"> **Generated at**: {generation_time_str}\n\n"
    )

    logger.info("STEP 1: 시장/뉴스 요약 생성 중...")
    market_summary_md = analyzer.generate_market_summary(
        macro_data=macro_data,
        market_breadth=macro_market_data.get("market_breadth_daily"),
        momentum_data=macro_market_data.get("momentum"),
        data_guardrails=data_guardrails,
        news_text=news_text,
        korean_market_snapshot=macro_market_data.get("normalized_global_macro_daily"),
        generation_time=generation_time_str,
        report_type=report_type,
    )
    report_content += f"## 1. 시장 요약 및 뉴스\n\n{market_summary_md.strip()}\n\n---\n\n"

    stock_analysis_md = ""
    if target_stocks_data:
        logger.info("STEP 2: 종목 분석 생성 중...")
        top_volume_md = ""
        if top_volume_data:
            logger.info("STEP 2-1: 거래대금 상위 종목 요약 생성 중...")
            top_volume_md = analyzer.generate_top_volume_analysis(
                top_volume_data=top_volume_data,
                market_summary=market_summary_md,
                report_type=report_type,
            )

        logger.info("STEP 2-2: 관심 종목 배치 분석 생성 중...")
        stock_analysis_md = analyzer.generate_batched_stock_analysis(
            market_summary=market_summary_md,
            target_stocks_data=target_stocks_data,
            macro_market_data=macro_market_data,
            data_guardrails=data_guardrails,
            generation_time=generation_time_str,
            report_type=report_type,
        )
        combined_stock_section = "\n\n".join(
            part.strip() for part in (top_volume_md, stock_analysis_md) if part and part.strip()
        )
        report_content += f"## 2. 거래대금 상위 종목 및 관심 종목 분석\n\n{combined_stock_section.strip()}\n\n---\n\n"
    else:
        logger.warning("관심 종목 데이터 부재로 STEP 2를 건너뜁니다.")

    guardrails_md = _render_data_guardrails_md(data_guardrails)
    report_content += f"## 3. 데이터 품질 점검\n\n{guardrails_md}\n"

    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    file_name = f"daily_quant_report_{report_type}_{now_kst.strftime('%Y%m%d_%H%M')}.md"
    file_path = reports_dir / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"리포트 저장 완료: {file_path}")

    try:
        telegram_report = _build_telegram_report(
            report_label=report_label,
            generation_time_str=generation_time_str,
            market_summary_md=market_summary_md,
            stock_analysis_md=stock_analysis_md,
            data_guardrails=data_guardrails,
        )
        sender = TelegramSender()
        sent = sender.send_report(telegram_report)
        if sent:
            logger.info("텔레그램 발송 성공.")
        else:
            logger.warning("텔레그램 발송 요청 완료, 전달 실패.")
    except Exception as exc:
        logger.warning(f"텔레그램 발송 실패 (non-fatal): {exc}")


if __name__ == "__main__":
    main()
