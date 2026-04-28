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
from src.services.naver_news_service import NaverNewsService
from src.services.supabase_stockdata_reader import SupabaseStockDataReader
from src.utils.formatters import (
    NA_TEXT,
    format_date,
    format_flow_amount,
    format_flow_generic,
    format_index,
    format_market_cap,
    format_multiple,
    format_outstanding_shares,
    format_percent,
    format_plain_number,
    format_price,
    format_rate_percent,
    format_ratio_metric,
    format_trading_value,
    format_usdkrw,
    format_volume,
    is_missing,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_REPORT_TYPES = ("morning", "closing", "regular")


def _parse_args():
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
    if not top_volume_data:
        return False
    return any(isinstance(v, list) and len(v) > 0 for k, v in top_volume_data.items() if k != "base_date")


def _validate_target_stocks(target_stocks_data: dict) -> bool:
    if not target_stocks_data:
        return False
    return any(isinstance(v, list) and len(v) > 0 for v in target_stocks_data.values())


def _render_data_guardrails_md(data_guardrails: dict) -> str:
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


def _safe_get_analyzer() -> GeminiAnalyzer | None:
    try:
        return GeminiAnalyzer()
    except Exception as exc:
        logger.warning(f"Gemini analyzer unavailable, rule-based fallback will be used: {exc}")
        return None


def _extract_news_headlines(news_text: str, limit: int = 3) -> list[str]:
    headlines = []
    for line in (news_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            headlines.append(stripped[2:].strip())
        elif stripped:
            headlines.append(stripped)
        if len(headlines) >= limit:
            break
    return headlines


def _safe_float(value):
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_supply_value(value) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return NA_TEXT
    if abs(numeric) >= 1_000_000:
        return format_flow_amount(numeric)
    return format_flow_generic(numeric)


def _format_market_flow_value(value) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return NA_TEXT
    if abs(numeric) < 1_000_000:
        label = "순매수" if numeric >= 0 else "순매도"
        return f"{label} {numeric:+,.0f}억원"
    return format_flow_amount(numeric)


def _relative_position(price_value, ma_value):
    price_num = _safe_float(price_value)
    ma_num = _safe_float(ma_value)
    if price_num is None or ma_num is None or ma_num == 0:
        return None
    return (price_num / ma_num) - 1.0


def _rule_based_us_summary(macro: dict, news_text: str) -> str:
    sp500 = _safe_float(macro.get("sp500_change_rate"))
    nasdaq = _safe_float(macro.get("nasdaq_change_rate"))
    sox = _safe_float(macro.get("sox"))
    vix = _safe_float(macro.get("vix"))
    us10y = _safe_float(macro.get("us10y"))
    dxy = _safe_float(macro.get("dxy"))
    wti = _safe_float(macro.get("wti"))
    headlines = _extract_news_headlines(news_text, limit=2)

    line1 = (
        f"S&P500은 {format_index(macro.get('sp500'))} ({format_percent(macro.get('sp500_change_rate'))}), "
        f"NASDAQ은 {format_index(macro.get('nasdaq'))} ({format_percent(macro.get('nasdaq_change_rate'))})로 마감했습니다."
    )

    if sp500 is not None and nasdaq is not None:
        if sp500 > 0 and nasdaq > 0:
            tone = "대형주와 성장주가 함께 강한 흐름을 보인 세션입니다."
        elif sp500 < 0 and nasdaq < 0:
            tone = "대형주와 성장주가 함께 눌리며 위험선호가 약해진 세션입니다."
        else:
            tone = "대형주와 성장주 흐름이 엇갈리며 섹터별 차별화가 나타난 세션입니다."
    else:
        tone = "미국 증시는 지수별로 혼재된 신호를 보였습니다."

    line2 = (
        f"미국 10년물은 {format_rate_percent(us10y)}, DXY는 {format_plain_number(dxy)}, "
        f"WTI는 {format_plain_number(wti)}이며 VIX는 {format_plain_number(vix)} 수준입니다. {tone}"
    )
    if sox is not None:
        line2 += f" SOX는 {format_plain_number(sox)}로 반도체 투자심리 확인이 필요합니다."

    if headlines:
        line3 = f"해외 뉴스에서는 {headlines[0]}"
        if len(headlines) > 1:
            line3 += f" / {headlines[1]}"
        line3 += " 이슈가 시장 해석의 중심입니다."
    else:
        line3 = "해외 뉴스 요약 입력이 제한적이어서 수치 중심으로 미국 시장을 해석했습니다."

    return "\n".join([f"- {line1}", f"- {line2}", f"- {line3}"])


def _generate_us_market_summary(macro: dict, news_text: str, analyzer: GeminiAnalyzer | None) -> str:
    if analyzer:
        summary_input = {
            "sp500": macro.get("sp500"),
            "sp500_change_rate": macro.get("sp500_change_rate"),
            "nasdaq": macro.get("nasdaq"),
            "nasdaq_change_rate": macro.get("nasdaq_change_rate"),
            "sox": macro.get("sox"),
            "vix": macro.get("vix"),
            "us10y": macro.get("us10y"),
            "dxy": macro.get("dxy"),
            "wti": macro.get("wti"),
            "brent": macro.get("brent"),
            "gold": macro.get("gold"),
            "copper": macro.get("copper"),
        }
        try:
            generated = analyzer.generate_morning_us_summary(summary_input, news_text)
            if generated:
                lines = [line.strip() for line in generated.splitlines() if line.strip()]
                return "\n".join(f"- {line.lstrip('- ').strip()}" for line in lines[:3])
        except Exception as exc:
            logger.warning(f"Gemini US summary fallback to rule-based: {exc}")
    return _rule_based_us_summary(macro, news_text)


def _build_market_impact_lists(macro: dict, derivatives: dict) -> tuple[list[str], list[str], list[str]]:
    positives = []
    burdens = []
    watchpoints = []

    nasdaq_chg = _safe_float(macro.get("nasdaq_change_rate"))
    sp500_chg = _safe_float(macro.get("sp500_change_rate"))
    kospi_chg = _safe_float(macro.get("kospi_change_rate"))
    kosdaq_chg = _safe_float(macro.get("kosdaq_change_rate"))
    dxy = _safe_float(macro.get("dxy"))
    usdkrw = _safe_float(macro.get("usdkrw"))
    us10y = _safe_float(macro.get("us10y"))
    kr10y = _safe_float(macro.get("kr10y"))
    vix = _safe_float(macro.get("vix"))
    wti = _safe_float(macro.get("wti"))
    brent = _safe_float(macro.get("brent"))
    sox = _safe_float(macro.get("sox"))
    night_ret = _safe_float(derivatives.get("night_futures_return"))

    if nasdaq_chg is not None and nasdaq_chg > 0:
        positives.append(f"NASDAQ이 {format_percent(nasdaq_chg)}로 강세여서 성장주 심리에는 우호적입니다.")
    if sp500_chg is not None and sp500_chg > 0:
        positives.append(f"S&P500이 {format_percent(sp500_chg)}로 마감해 대형주 전반의 위험선호는 유지됐습니다.")
    if sox is not None:
        watchpoints.append(f"SOX는 {format_plain_number(sox)}로 반도체 주도력의 지속 여부를 확인할 필요가 있습니다.")
    if kospi_chg is not None and kospi_chg > 0:
        positives.append(f"전일 KOSPI가 {format_percent(kospi_chg)}로 마감해 국내 대형주 수급은 완전히 꺾이지 않았습니다.")
    if kosdaq_chg is not None and kosdaq_chg < 0:
        burdens.append(f"전일 KOSDAQ이 {format_percent(kosdaq_chg)}로 약세였던 만큼 중소형 성장주 변동성은 경계가 필요합니다.")
    if usdkrw is not None and usdkrw >= 1450:
        burdens.append(f"원/달러 환율이 {format_usdkrw(usdkrw)} 수준이라 외국인 위험자산 선호에는 부담입니다.")
    if dxy is not None and dxy >= 100:
        burdens.append(f"DXY가 {format_plain_number(dxy)}로 높아 달러 강세 압력이 이어지고 있습니다.")
    if us10y is not None and us10y >= 4.3:
        burdens.append(f"미국 10년물이 {format_rate_percent(us10y)} 수준으로 높아 밸류에이션 부담이 남아 있습니다.")
    if kr10y is not None and us10y is not None:
        watchpoints.append(
            f"한국 10년물 {format_rate_percent(kr10y)}와 미국 10년물 {format_rate_percent(us10y)}의 금리 격차가 자금 민감도를 좌우할 수 있습니다."
        )
    commodity_value = wti if wti is not None else brent
    commodity_label = "WTI" if wti is not None else "Brent"
    if commodity_value is not None and commodity_value >= 90:
        burdens.append(f"{commodity_label}가 {format_plain_number(commodity_value)}로 높아 에너지/물가 부담을 자극할 수 있습니다.")
    if vix is not None and vix <= 18:
        positives.append(f"VIX가 {format_plain_number(vix)}로 낮아 패닉성 위험회피는 제한적입니다.")
    elif vix is not None and vix >= 22:
        burdens.append(f"VIX가 {format_plain_number(vix)}로 높아 장중 변동성 확대 가능성이 있습니다.")
    if night_ret is not None:
        tone = "우호적" if night_ret > 0 else "부담" if night_ret < 0 else "중립"
        watchpoints.append(f"야간선물 수익률은 {format_percent(night_ret)}로 개장 체감에는 {tone} 신호입니다.")

    foreign_kospi = _safe_float(macro.get("kospi_foreign_net_buy"))
    inst_kospi = _safe_float(macro.get("kospi_institutional_net_buy"))
    if foreign_kospi is not None:
        watchpoints.append(
            f"KOSPI 외국인은 {_format_market_flow_value(foreign_kospi)}, 기관은 {_format_market_flow_value(inst_kospi)}로 전일 현물 수급의 방향을 확인할 수 있습니다."
        )

    if not positives:
        positives.append("미국 지수와 국내 수급 신호가 혼재해 뚜렷한 상방 우위는 아직 제한적입니다.")
    if not burdens:
        burdens.append("금리·달러·원자재 중 단일 팩터가 시장을 압도하는 수준은 아니어서 과도한 비관은 경계가 필요합니다.")
    if not watchpoints:
        watchpoints.append("개장 직후 외국인 현물/선물 방향과 반도체 주도력 지속 여부를 우선 확인해야 합니다.")

    return positives[:3], burdens[:3], watchpoints[:3]


def _format_news_reason_from_titles(news_items: list[dict]) -> str:
    if not news_items:
        return ""
    first = news_items[0]
    title = first.get("title") or ""
    desc = first.get("description") or ""
    if title and desc:
        return f"{title} 이슈가 부각되며 관련 관심이 유입됐습니다."
    if title:
        return f"{title} 관련 뉴스가 부각됐습니다."
    return ""


def _build_top_volume_reason(stock: dict, naver_service: NaverNewsService, analyzer: GeminiAnalyzer | None) -> str:
    stock_name = stock.get("name") or stock.get("symbol") or ""
    upper_name = stock_name.upper()
    if any(keyword in upper_name for keyword in ("KODEX", "TIGER", "KBSTAR", "SOL", "ACE", "ARIRANG", "KOSEF", "ETF", "ETN", "인버스", "레버리지")):
        return "ETF/ETN 거래대금 상위는 방향성 단정보다 단기 트레이딩 수요와 변동성 확대 신호로 해석하는 편이 적절합니다."
    news_items = naver_service.search_news(stock_name, display=3)
    if news_items:
        if analyzer:
            try:
                summarized = analyzer.summarize_news_reason(stock_name, news_items)
                if summarized:
                    return summarized
            except Exception as exc:
                logger.warning(f"Gemini reason fallback for {stock_name}: {exc}")
        title_reason = _format_news_reason_from_titles(news_items)
        if title_reason:
            return title_reason

    trading_value = stock.get("trading_value")
    volume = stock.get("volume")
    if not is_missing(trading_value):
        return f"거래대금 {format_trading_value(trading_value)} 수준으로 시장 관심이 집중됐습니다."
    if not is_missing(volume):
        return f"거래량 {format_volume(volume)}로 시장 내 회전이 크게 나타났습니다."
    return "시장 관심 집중으로 거래량 상위권에 진입했습니다."


def _build_quant_comment(snapshot: dict) -> str:
    price = snapshot.get("price") or {}
    supply = snapshot.get("supply") or {}
    fundamentals = snapshot.get("fundamentals") or {}
    features = snapshot.get("features") or {}

    comments = []
    return_5d = _safe_float(features.get("return_5d"))
    volatility = _safe_float(features.get("volatility_20d"))
    foreign_z = _safe_float(features.get("foreign_flow_zscore"))
    foreign_holding = _safe_float(supply.get("foreign_holding_ratio"))
    per = _safe_float(fundamentals.get("per"))
    pbr = _safe_float(fundamentals.get("pbr"))
    close_price = _safe_float(price.get("close_price"))
    ma5 = _safe_float(features.get("moving_avg_5"))
    ma20 = _safe_float(features.get("moving_avg_20"))

    rel_ma5 = _relative_position(close_price, ma5)
    rel_ma20 = _relative_position(close_price, ma20)

    if return_5d is not None:
        if return_5d >= 0.05:
            comments.append(f"5일 수익률이 {return_5d * 100:.1f}%로 강한 단기 모멘텀을 보입니다.")
        elif return_5d <= -0.05:
            comments.append(f"5일 수익률이 {return_5d * 100:.1f}%로 약해 단기 추세 둔화에 유의해야 합니다.")
    if rel_ma5 is not None and rel_ma20 is not None:
        if rel_ma5 > 0 and rel_ma20 > 0:
            comments.append("종가가 MA5와 MA20을 모두 상회해 추세는 우호적입니다.")
        elif rel_ma5 < 0 and rel_ma20 < 0:
            comments.append("종가가 MA5와 MA20을 모두 하회해 추세 복원 여부를 먼저 확인해야 합니다.")
    if volatility is not None:
        if volatility >= 0.04:
            comments.append(f"20일 변동성이 {volatility * 100:.1f}%로 높아 추격 매수보다는 분할 접근이 적절합니다.")
        elif volatility <= 0.02:
            comments.append(f"20일 변동성이 {volatility * 100:.1f}%로 낮아 추세 추종 부담은 상대적으로 제한적입니다.")
    if foreign_z is not None:
        if foreign_z >= 1:
            comments.append(f"외국인 수급 z-score가 {foreign_z:.2f}로 높아 수급 확인 신호는 우호적입니다.")
        elif foreign_z <= -1:
            comments.append(f"외국인 수급 z-score가 {foreign_z:.2f}로 약해 가격 대비 수급 확인은 보수적으로 볼 필요가 있습니다.")
    if foreign_holding is not None:
        comments.append(f"외국인 보유율은 {foreign_holding:.1f}% 수준입니다.")
    if per is not None or pbr is not None:
        ratio_parts = []
        if per is not None:
            ratio_parts.append(f"PER {per:.1f}배")
        if pbr is not None:
            ratio_parts.append(f"PBR {pbr:.1f}배")
        comments.append(" / ".join(ratio_parts) + "로 절대 배수 해석은 업종 맥락과 함께 봐야 합니다.")

    if not comments:
        return "수치 입력이 제한적이어서 가격·수급·밸류에이션을 추가 확인한 뒤 대응하는 편이 안전합니다."

    return " ".join(comments[:3])


def _format_static_snapshot(snapshot: dict) -> str:
    price = snapshot.get("price") or {}
    supply = snapshot.get("supply") or {}
    fundamentals = snapshot.get("fundamentals") or {}
    short_row = snapshot.get("short_selling") or {}
    event = snapshot.get("event") or {}
    features = snapshot.get("features") or {}

    price_base_date = format_date(price.get("base_date"))
    supply_base_date = format_date(supply.get("base_date"))
    supply_date_note = ""
    if supply_base_date != NA_TEXT and price_base_date != NA_TEXT and supply_base_date != price_base_date:
        supply_date_note = f" (수급 기준일 상이: {supply_base_date})"

    event_text = " / ".join(
        [
            str(event.get("event_type", NA_TEXT) or NA_TEXT),
            str(event.get("event_score", NA_TEXT) if event.get("event_score") is not None else NA_TEXT),
            str(event.get("sentiment_score", NA_TEXT) if event.get("sentiment_score") is not None else NA_TEXT),
        ]
    )

    return "\n".join(
        [
            f"- 종가: {format_price(price.get('close_price'))} / 거래량: {format_volume(price.get('volume'))} / 거래대금: {format_trading_value(price.get('trading_value'))} / 시총: {format_market_cap(price.get('market_cap'))}",
            f"- 상장주식수: {format_outstanding_shares(price.get('outstanding_shares'))} / 기준일: {price_base_date}",
            (
                f"- 수급: 개인 {_format_supply_value(supply.get('individual_net_buy'))}, "
                f"외국인 {_format_supply_value(supply.get('foreign_net_buy'))}, "
                f"기관 {_format_supply_value(supply.get('institutional_net_buy'))}, "
                f"외국인 보유율 {format_ratio_metric(supply.get('foreign_holding_ratio'))}{supply_date_note}"
            ),
            (
                f"- 밸류에이션: PER {format_multiple(fundamentals.get('per'), '배')}, "
                f"PBR {format_multiple(fundamentals.get('pbr'), '배')}, "
                f"ROE {format_ratio_metric(fundamentals.get('roe'))}, "
                f"부채비율 {format_ratio_metric(fundamentals.get('debt_ratio'))}"
            ),
            (
                f"- 퀀트: 5일 수익률 {format_percent((_safe_float(features.get('return_5d')) * 100) if _safe_float(features.get('return_5d')) is not None else None)}, "
                f"MA5 {format_index(features.get('moving_avg_5'))}, MA20 {format_index(features.get('moving_avg_20'))}, "
                f"변동성 {format_ratio_metric((_safe_float(features.get('volatility_20d')) * 100) if _safe_float(features.get('volatility_20d')) is not None else None)}, "
                f"외국인 수급 z-score {format_signed_multiple(features.get('foreign_flow_zscore'), '')}"
            ),
            f"- 공매도 비중: {format_ratio_metric(short_row.get('short_ratio'))}",
            f"- 공시 이벤트: {event_text}",
            f"- 코멘트: {_build_quant_comment(snapshot)}",
        ]
    )


def format_signed_multiple(value, suffix: str = "") -> str:
    return NA_TEXT if is_missing(value) else f"{float(value):+.2f}{suffix}"


def _build_morning_report(
    bundle: dict,
    news_text: str,
    analyzer: GeminiAnalyzer | None,
    now_kst: datetime.datetime,
) -> str:
    macro = bundle.get("macro") or {}
    breadth = bundle.get("breadth") or {}
    derivatives = bundle.get("derivatives") or {}
    top_volume = bundle.get("top_volume") or {}
    static_snapshots = bundle.get("static_snapshots") or []
    macro_base_date = format_date(macro.get("base_date"))
    breadth_base_date = format_date(breadth.get("base_date"))
    derivatives_base_date = format_date(derivatives.get("base_date"))
    latest_price_base_date = format_date(bundle.get("latest_price_base_date"))
    now_text = now_kst.strftime("%Y-%m-%d %H:%M KST")

    us_market_summary = _generate_us_market_summary(macro, news_text, analyzer)
    positives, burdens, watchpoints = _build_market_impact_lists(macro, derivatives)
    naver_service = NaverNewsService()

    lines = [
        f"# Morning Market Brief | 기준일: {macro_base_date} / 작성시각: {now_text}",
        "",
        "## 1. 미국 시장 정리",
        f"_기준일: {macro_base_date} (`normalized_global_macro_daily`)_",
        f"- S&P500: {format_index(macro.get('sp500'))} ({format_percent(macro.get('sp500_change_rate'))})",
        f"- NASDAQ: {format_index(macro.get('nasdaq'))} ({format_percent(macro.get('nasdaq_change_rate'))})",
        f"- SOX: {format_index(macro.get('sox'))}",
        f"- VIX: {format_plain_number(macro.get('vix'))}",
        f"- 미국 10년물: {format_rate_percent(macro.get('us10y'))}",
        f"- DXY: {format_plain_number(macro.get('dxy'))}",
        f"- WTI: {format_plain_number(macro.get('wti'))} (Supabase 적재값 기준)",
        f"- Brent: {format_plain_number(macro.get('brent'))} (Supabase 적재값 기준)",
        f"- Gold: {format_plain_number(macro.get('gold'))} (Supabase 적재값 기준)",
        f"- Copper: {format_plain_number(macro.get('copper'))} (Supabase 적재값 기준)",
        f"- 요약:\n{us_market_summary}",
        "",
        "## 2. 한국 시장 영향 전망",
        f"_기준일: {macro_base_date} (`normalized_global_macro_daily`), 보조: {derivatives_base_date} (`normalized_derivatives_daily`)_",
        "- 긍정 요인:",
    ]
    lines.extend([f"  - {item}" for item in positives])
    lines.extend(["- 부담 요인:"])
    lines.extend([f"  - {item}" for item in burdens])
    lines.extend(["- 오늘 관전 포인트:"])
    lines.extend([f"  - {item}" for item in watchpoints])
    lines.extend(
        [
            f"- 원/달러: {format_usdkrw(macro.get('usdkrw'))}",
            f"- 한국 10년물: {format_rate_percent(macro.get('kr10y'))}",
            f"- 미국 10년물: {format_rate_percent(macro.get('us10y'))}",
            f"- KOSPI 외국인: {_format_market_flow_value(macro.get('kospi_foreign_net_buy'))} / 기관: {_format_market_flow_value(macro.get('kospi_institutional_net_buy'))}",
            "",
            "## 3. 전일 한국 시장 요약",
            f"_기준일: {macro_base_date} (`normalized_global_macro_daily`), 시장 체력: {breadth_base_date} (`market_breadth_daily`)_",
            f"- KOSPI: {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))})",
            f"- KOSDAQ: {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))})",
            f"- KOSPI 수급: 개인 {_format_market_flow_value(macro.get('kospi_individual_net_buy'))}, 외국인 {_format_market_flow_value(macro.get('kospi_foreign_net_buy'))}, 기관 {_format_market_flow_value(macro.get('kospi_institutional_net_buy'))}",
            f"- KOSDAQ 수급: 개인 {_format_market_flow_value(macro.get('kosdaq_individual_net_buy'))}, 외국인 {_format_market_flow_value(macro.get('kosdaq_foreign_net_buy'))}, 기관 {_format_market_flow_value(macro.get('kosdaq_institutional_net_buy'))}",
            f"- 상승/하락/보합: 상승 {breadth.get('advances', NA_TEXT)}개, 하락 {breadth.get('declines', NA_TEXT)}개, 보합 {breadth.get('unchanged', NA_TEXT)}개",
            f"- 상승 거래량 / 하락 거래량: {format_volume(breadth.get('advancing_volume'))} / {format_volume(breadth.get('declining_volume'))}",
            "",
            "## 4. 전일 거래량 상위 종목",
            f"_기준일: {latest_price_base_date} (`normalized_stock_prices_daily` + `stocks_master`)_",
        ]
    )

    for market_key in ("KOSPI", "KOSDAQ", "ETF"):
        lines.append(f"[{market_key} Top 5]")
        market_rows = top_volume.get(market_key) or []
        if not market_rows:
            if market_key in ("KOSPI", "KOSDAQ") and top_volume.get("unclassified_count"):
                lines.append(
                    f"- 데이터 없음 (`stocks_master.market`가 DYNAMIC 등으로 적재된 종목이 많아 {market_key} 공식 분류가 제한됨)"
                )
            else:
                lines.append("- 데이터 없음")
            lines.append("")
            continue
        for idx, stock in enumerate(market_rows, 1):
            reason = _build_top_volume_reason(stock, naver_service, analyzer)
            lines.append(
                f"{idx}) {stock.get('name', stock.get('symbol'))}({stock.get('symbol')}) | "
                f"종가 {format_price(stock.get('close_price'))} | 거래량 {format_volume(stock.get('volume'))} | "
                f"거래대금 {format_trading_value(stock.get('trading_value'))}"
            )
            lines.append(f"   - 주목 사유: {reason}")
        lines.append("")

    lines.extend(
        [
            "## 5. Static 관심종목 요약",
            "_기준 universe: `static_stock_universe.enabled = true`_",
        ]
    )
    if not static_snapshots:
        lines.append("- 데이터 없음")
    else:
        for idx, snapshot in enumerate(static_snapshots, 1):
            lines.extend(
                [
                    f"{idx}) {snapshot.get('name')}({snapshot.get('symbol')})",
                    f"- 시장: {snapshot.get('market', NA_TEXT)}",
                    _format_static_snapshot(snapshot),
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def _run_morning_report(now_kst: datetime.datetime) -> None:
    reader = SupabaseReader()
    stockdata_reader = SupabaseStockDataReader(reader)
    analyzer = _safe_get_analyzer()

    logger.info("Morning report: Supabase 공식 테이블 데이터 수집 중...")
    bundle = stockdata_reader.fetch_morning_bundle(top_n=5)
    logger.info("Morning report: 해외 뉴스 문서 수집 중...")
    raw_news_text = reader.fetch_news_document()
    news_text = reader.prepare_news_context(raw_news_text)

    report_content = _build_morning_report(bundle, news_text, analyzer, now_kst)

    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    file_name = f"daily_quant_report_morning_{now_kst.strftime('%Y%m%d_%H%M')}.md"
    file_path = reports_dir / file_name
    file_path.write_text(report_content, encoding="utf-8")
    logger.info(f"Morning report 저장 완료: {file_path}")

    try:
        sender = TelegramSender()
        sent = sender.send_report(report_content)
        if sent:
            logger.info("Morning report 텔레그램 발송 성공.")
        else:
            logger.warning("Morning report 텔레그램 발송 요청 완료, 전달 실패.")
    except Exception as exc:
        logger.warning(f"Morning report 텔레그램 발송 실패 (non-fatal): {exc}")


def _run_existing_report_flow(report_type: str, now_kst: datetime.datetime) -> None:
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
        logger.error("매크로 데이터를 수집하지 못했습니다. 시장 요약 품질이 저하될 수 있습니다.")

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
    file_path.write_text(report_content, encoding="utf-8")
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


def main():
    args = _parse_args()
    report_type = args.report_type
    kst_tz = datetime.timezone(datetime.timedelta(hours=9))
    now_kst = datetime.datetime.now(kst_tz)
    logger.info(f"=== Daily Report Pipeline 시작 [type={report_type}] ===")

    if report_type == "morning":
        _run_morning_report(now_kst)
        return

    _run_existing_report_flow(report_type, now_kst)


if __name__ == "__main__":
    main()
