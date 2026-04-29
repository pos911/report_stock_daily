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
from src.services.naver_news_service import NaverNewsService
from src.utils.formatters import (
    NA_TEXT,
    format_date,
    format_flow_amount,
    format_index,
    format_market_cap,
    format_multiple,
    format_outstanding_shares,
    format_percent,
    format_plain_number,
    format_price,
    format_rate_percent,
    format_ratio_metric,
    format_signed_multiple,
    format_trading_value,
    format_usdkrw,
    format_volume,
    is_missing,
)
from src.utils.market_assets import (
    canonicalize_symbol,
    display_symbol,
    extract_theme_keywords,
    infer_asset_type,
    label_for_column,
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
    args = parser.parse_args()
    report_type = (args.report_type or "regular").strip().lower()
    if report_type not in VALID_REPORT_TYPES:
        parser.error(f"invalid choice: '{args.report_type}' (choose from {', '.join(VALID_REPORT_TYPES)})")
    args.report_type = report_type
    return args


def _get_now_kst() -> datetime.datetime:
    return datetime.datetime.now(KST)


def _get_regular_slot_label(now_kst: datetime.datetime) -> str:
    if 10 <= now_kst.hour < 11:
        return "오전 10:30 점검"
    if 12 <= now_kst.hour < 13:
        return "오후 12:30 점검"
    if 14 <= now_kst.hour < 15:
        return "오후 14:30 점검"
    return "장중 정기 점검"


def _safe_get_analyzer() -> GeminiAnalyzer | None:
    try:
        return GeminiAnalyzer()
    except Exception as exc:
        logger.warning(f"Gemini analyzer unavailable, rule-based fallback will be used: {exc}")
        return None


def _safe_float(value):
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_debug_lines(report_content: str) -> str:
    banned_fragments = ("Test_only", "섹션 수:", "Sections:")
    lines = []
    for line in report_content.splitlines():
        if any(fragment in line for fragment in banned_fragments):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip() + "\n"


def _truncate_text(text: str, max_len: int = 60) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _extract_docs_matches(news_text: str, keywords: list[str], max_matches: int = 3) -> list[str]:
    matches = []
    if not news_text:
        return matches
    lowered_keywords = [keyword.lower() for keyword in keywords if keyword]
    for line in news_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(keyword in lowered for keyword in lowered_keywords):
            matches.append(stripped)
        if len(matches) >= max_matches:
            break
    return matches


def _build_market_data_status(readiness: dict) -> str:
    if not readiness.get("minimum_report_ready"):
        return "위험"
    if readiness.get("full_market_coverage_pass"):
        return "정상"
    return "경고"


def _build_header_notes(readiness: dict, diagnostics: dict) -> list[str]:
    notes = []
    if readiness.get("coverage_status") == "PARTIAL":
        notes.append("전체시장 거래량 상위 분석은 커버리지 부족으로 참고용")
    if (readiness.get("price_coverage") or {}).get("covered_symbols", 0) <= 2000:
        notes.append("covered_symbols <= 2000")
    if (readiness.get("latest_full_price_records_processed") or 0) <= 2000:
        notes.append("daily_stock_full_price_pipeline full run 미완료 가능성")
    if diagnostics.get("supply_unit_needs_review"):
        notes.append("수급 단위 확인 필요")
    if diagnostics.get("valuation_zero_needs_review"):
        notes.append("밸류에이션 0값 점검 필요")
    if diagnostics.get("short_ratio_needs_review"):
        notes.append("공매도 비중 점검 필요")
    return notes


def _build_header_lines(title: str, now_kst: datetime.datetime, readiness: dict, diagnostics: dict) -> list[str]:
    latest_price_date = format_date(readiness.get("latest_price_date"))
    latest_macro_date = format_date(readiness.get("latest_macro_date"))
    coverage = (readiness.get("price_coverage") or {}).get("covered_symbols", 0)
    status = _build_market_data_status(readiness)
    notes = _build_header_notes(readiness, diagnostics)
    lines = [
        f"# {title}",
        f"- 작성시각: {now_kst.strftime('%Y-%m-%d %H:%M KST')}",
        "- 수치 기준: Supabase StockData 최신 적재값",
        f"- 가격 기준일: {latest_price_date}",
        f"- 매크로 기준일: {latest_macro_date}",
        f"- minimum_report_ready: {'true' if readiness.get('minimum_report_ready') else 'false'}",
        f"- full_market_coverage_pass: {'true' if readiness.get('full_market_coverage_pass') else 'false'}",
        f"- coverage_status: {readiness.get('coverage_status') or 'LIMITED'}",
        f"- 데이터 점검: {status}",
        f"- 전체시장 커버리지: {readiness.get('coverage_status') or 'LIMITED'}",
        f"- 전체시장 가격 커버리지: {coverage:,}종목",
        f"- Static 관심종목: {readiness.get('static_enabled_count', 0):,}개",
        f"- daily_stock_full_price_pipeline 최신 처리건수: {(readiness.get('latest_full_price_records_processed') or 0):,}",
        f"- 최근 3일 파이프라인 경고/실패: {len(readiness.get('recent_problem_logs') or []):,}건",
    ]
    for note in notes:
        lines.append(f"- 점검 메모: {note}")
    return lines


def _diagnose_supply_unit(snapshot: dict) -> dict:
    supply = snapshot.get("supply") or {}
    price = snapshot.get("price") or {}
    close_price = _safe_float(price.get("close_price"))
    trading_value = _safe_float(price.get("trading_value"))
    values = [
        _safe_float(supply.get("individual_net_buy")),
        _safe_float(supply.get("foreign_net_buy")),
        _safe_float(supply.get("institutional_net_buy")),
        _safe_float(supply.get("pension_net_buy")),
        _safe_float(supply.get("corporate_net_buy")),
    ]
    nonzero_values = [abs(value) for value in values if value not in (None, 0)]
    if not nonzero_values:
        return {"unit": "unknown", "needs_review": False}

    share_like = 0
    amount_like = 0
    if close_price and trading_value:
        for value in nonzero_values:
            share_ratio = (value * close_price) / trading_value
            amount_ratio = value / trading_value
            if 0.01 <= share_ratio <= 20:
                share_like += 1
            if 0.001 <= amount_ratio <= 5:
                amount_like += 1

    if share_like > amount_like:
        return {"unit": "shares", "needs_review": False}
    if amount_like > share_like:
        return {"unit": "krw", "needs_review": False}
    return {"unit": "unknown", "needs_review": True}


def _format_supply_value(value, unit_diag: dict) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return NA_TEXT
    unit = unit_diag.get("unit")
    if unit == "shares":
        label = "순매수" if numeric >= 0 else "순매도"
        return f"{label} {abs(int(round(numeric))):,}주"
    if unit == "krw":
        return format_flow_amount(numeric)
    return f"{numeric:+,.0f} (단위 확인 필요)"


def _diagnose_fundamentals(snapshot: dict) -> dict:
    fundamentals = snapshot.get("fundamentals") or {}
    raw = snapshot.get("fundamentals_raw") or {}
    history = snapshot.get("fundamentals_history") or []

    metrics = {
        "per": _safe_float(fundamentals.get("per")),
        "pbr": _safe_float(fundamentals.get("pbr")),
        "roe": _safe_float(fundamentals.get("roe")),
        "debt_ratio": _safe_float(fundamentals.get("debt_ratio")),
    }
    current_all_zero = all(value in (0, None) for value in metrics.values())
    history_all_zero = False
    if history:
        recent_rows = history[:3]
        history_all_zero = all(
            all(_safe_float(row.get(key)) in (0, None) for key in ("per", "pbr", "roe", "debt_ratio"))
            for row in recent_rows
        )
    raw_all_empty = all(
        _safe_float(raw.get(key)) in (0, None)
        for key in ("revenue", "operating_income", "net_income", "total_assets", "total_liabilities", "total_equity")
    ) if raw else False

    def should_flag(metric_name: str, value) -> bool:
        if value is None:
            return False
        if metric_name in {"per", "pbr"} and value <= 0:
            return True
        if metric_name in {"roe", "debt_ratio"} and value == 0 and (current_all_zero or history_all_zero or raw_all_empty):
            return True
        if value == 0 and current_all_zero:
            return True
        return False

    display = {}
    suspicious = False
    for key, value in metrics.items():
        if should_flag(key, value):
            display[key] = "N/A(점검필요)"
            suspicious = True
        elif value is None:
            display[key] = NA_TEXT
        elif key in {"per", "pbr"}:
            display[key] = format_multiple(value, "배")
        else:
            display[key] = format_ratio_metric(value)

    return {
        "display": display,
        "needs_review": suspicious,
        "raw_all_empty": raw_all_empty,
        "history_all_zero": history_all_zero,
    }


def _diagnose_short_selling(snapshot: dict) -> dict:
    short_row = snapshot.get("short_selling") or {}
    ratio = _safe_float(short_row.get("short_ratio"))
    short_value = _safe_float(short_row.get("short_value"))
    short_volume = _safe_float(short_row.get("short_volume"))

    if ratio is None:
        ratio_text = "비중 N/A"
        needs_review = bool((short_value or 0) > 0 or (short_volume or 0) > 0)
    elif ratio == 0 and ((short_value or 0) > 0 or (short_volume or 0) > 0):
        ratio_text = "비중 N/A(점검필요)"
        needs_review = True
    else:
        ratio_text = f"비중 {format_ratio_metric(ratio)}"
        needs_review = False

    parts = []
    if short_value is not None and short_value > 0:
        parts.append(f"거래금액 {format_trading_value(short_value)}")
    if short_volume is not None and short_volume > 0:
        parts.append(f"거래량 {format_volume(short_volume)}")
    parts.append(ratio_text)
    note = ""
    if needs_review:
        note = "공매도 금액·수량은 확인되나 비중값 부재로 방향성 해석은 제한적"
    return {"summary": " / ".join(parts), "needs_review": needs_review, "note": note}


def _build_quant_comment(snapshot: dict, unit_diag: dict, fundamentals_diag: dict, short_diag: dict) -> str:
    price = snapshot.get("price") or {}
    features = snapshot.get("features") or {}
    supply = snapshot.get("supply") or {}
    comments = []

    return_5d = _safe_float(features.get("return_5d"))
    volatility = _safe_float(features.get("volatility_20d"))
    foreign_z = _safe_float(features.get("foreign_flow_zscore"))
    foreign_holding = _safe_float(supply.get("foreign_holding_ratio"))
    close_price = _safe_float(price.get("close_price"))
    ma5 = _safe_float(features.get("moving_avg_5"))
    ma20 = _safe_float(features.get("moving_avg_20"))

    if return_5d is not None:
        if return_5d > 0:
            comments.append(f"5일 수익률은 {return_5d * 100:.1f}%로 단기 모멘텀은 우호적입니다.")
        elif return_5d < 0:
            comments.append(f"5일 수익률은 {return_5d * 100:.1f}%로 최근 흐름은 약합니다.")
    if close_price and ma5 and ma20:
        if close_price > ma5 and close_price > ma20:
            comments.append("종가는 MA5·MA20 위에 있어 추세는 상대적으로 안정적입니다.")
        elif close_price < ma5 and close_price < ma20:
            comments.append("종가는 MA5·MA20 아래에 있어 추세 복원 확인이 먼저입니다.")
    if volatility is not None and volatility >= 0.04:
        comments.append(f"20일 변동성은 {volatility * 100:.1f}%로 높아 추격 매수는 신중할 필요가 있습니다.")
    if foreign_z is not None and foreign_z >= 1:
        comments.append(f"외국인 수급 z-score {foreign_z:.2f}로 수급 확인 신호는 우호적입니다.")
    elif foreign_z is not None and foreign_z <= -1:
        comments.append(f"외국인 수급 z-score {foreign_z:.2f}로 수급 확인은 약한 편입니다.")
    if foreign_holding is not None:
        comments.append(f"외국인 보유율은 {foreign_holding:.1f}%입니다.")
    if unit_diag.get("needs_review"):
        comments.append("수급 단위가 확정되지 않아 절대 규모 해석은 제한적입니다.")
    if fundamentals_diag.get("needs_review"):
        comments.append("밸류 지표는 0값 이상 여부 점검 전까지 판단 근거에서 제외하는 편이 안전합니다.")
    if short_diag.get("needs_review"):
        comments.append(short_diag["note"])
    if not comments:
        return "추가 해석 신호가 제한적이어서 가격·수급의 후속 확인이 필요합니다."
    return " ".join(comments[:4])


def _build_news_queries(stock: dict) -> list[str]:
    name = stock.get("name") or stock.get("stock_name") or stock.get("symbol") or ""
    asset_type = stock.get("asset_type") or "UNKNOWN"
    queries = [
        f"{name} 주가",
        f"{name} 특징주",
        f"{name} 실적",
    ]
    theme_keywords = extract_theme_keywords(name)
    if asset_type in {"ETF", "ETN"}:
        queries.extend([f"{name} ETF", f"{name} ETN"])
        for keyword in theme_keywords:
            queries.append(f"{keyword} ETF")
            queries.append(f"{keyword} ETN")
    return list(dict.fromkeys(query for query in queries if query.strip()))


def _build_theme_fallback_reason(stock: dict, docs_matches: list[str], event: dict | None) -> str:
    name = stock.get("name") or stock.get("symbol") or ""
    asset_type = stock.get("asset_type") or "UNKNOWN"
    themes = extract_theme_keywords(name)
    if docs_matches:
        return f"Google Docs 뉴스 컨텍스트에서 `{_truncate_text(docs_matches[0], 70)}` 이슈가 포착됐습니다."
    if event and event.get("event_type"):
        return f"공시 이벤트 `{event.get('event_type')}`가 확인돼 거래가 붙었습니다."
    if asset_type in {"ETF", "ETN"} and themes:
        return f"`{'/'.join(themes[:2])}` 관련 테마 변동성이 해당 상품 거래를 자극했습니다."
    if themes:
        return f"`{'/'.join(themes[:2])}` 관련 뉴스·테마 기대가 거래를 자극한 것으로 보입니다."
    if stock.get("trading_value"):
        return f"거래대금 {format_trading_value(stock.get('trading_value'))} 규모로 회전율이 높아진 종목입니다."
    return "개별 뉴스 매칭은 약하지만 당일 거래대금 집중으로 상위권에 진입했습니다."


def _generate_top_volume_reason(
    stock: dict,
    naver_service: NaverNewsService,
    analyzer: GeminiAnalyzer | None,
    news_text: str,
    event_map: dict,
) -> str:
    queries = _build_news_queries(stock)
    news_items = naver_service.search_queries(queries, display_per_query=5, max_items=5)
    docs_matches = _extract_docs_matches(
        news_text,
        [stock.get("name") or "", stock.get("display_symbol") or "", *extract_theme_keywords(stock.get("name"))],
    )
    event = event_map.get(stock.get("canonical_symbol")) or event_map.get(stock.get("display_symbol")) or {}

    if analyzer and (news_items or docs_matches or event):
        try:
            context = {
                "name": stock.get("name"),
                "symbol": stock.get("display_symbol"),
                "market": stock.get("market"),
                "asset_type": stock.get("asset_type"),
                "close_price": stock.get("close_price"),
                "volume": stock.get("volume"),
                "trading_value": stock.get("trading_value"),
                "naver_news": news_items[:5],
                "event": {
                    "event_type": event.get("event_type"),
                    "event_score": event.get("event_score"),
                    "sentiment_score": event.get("sentiment_score"),
                } if event else {},
                "docs_matches": docs_matches[:3],
            }
            reason = analyzer.generate_top_volume_reason(context)
            if reason:
                return reason
        except Exception as exc:
            logger.warning(f"Gemini top-volume reason fallback for {stock.get('name')}: {exc}")

    if news_items:
        first = news_items[0]
        title = first.get("title") or first.get("description") or ""
        source = first.get("query") or "네이버 뉴스"
        return f"`{_truncate_text(title, 65)}` 관련 이슈가 {source} 검색에서 먼저 포착됐습니다."

    return _build_theme_fallback_reason(stock, docs_matches, event)


def _prepare_static_snapshots(static_snapshots: list[dict]) -> tuple[list[dict], dict]:
    prepared = []
    diagnostics = {
        "supply_unit_needs_review": [],
        "valuation_zero_needs_review": [],
        "short_ratio_needs_review": [],
    }
    for snapshot in static_snapshots:
        supply_unit = _diagnose_supply_unit(snapshot)
        fundamentals_diag = _diagnose_fundamentals(snapshot)
        short_diag = _diagnose_short_selling(snapshot)
        quant_comment = _build_quant_comment(snapshot, supply_unit, fundamentals_diag, short_diag)
        prepared_snapshot = {
            **snapshot,
            "supply_unit_diag": supply_unit,
            "fundamentals_diag": fundamentals_diag,
            "short_diag": short_diag,
            "quant_comment": quant_comment,
        }
        prepared.append(prepared_snapshot)
        if supply_unit.get("needs_review"):
            diagnostics["supply_unit_needs_review"].append(snapshot)
        if fundamentals_diag.get("needs_review"):
            diagnostics["valuation_zero_needs_review"].append(snapshot)
        if short_diag.get("needs_review"):
            diagnostics["short_ratio_needs_review"].append(snapshot)
    return prepared, diagnostics


def _prepare_top_volume_event_map(reader: SupabaseReader, top_volume: dict) -> dict:
    symbols = []
    for key in ("KOSPI", "KOSDAQ", "ETF_ETN"):
        for row in top_volume.get(key) or []:
            symbols.append(canonicalize_symbol(row.get("symbol")))
    if not symbols:
        return {}
    return reader.fetch_latest_stock_events(list(dict.fromkeys(symbols)))


def _format_top_volume_sections(
    top_volume: dict,
    readiness: dict,
    naver_service: NaverNewsService,
    analyzer: GeminiAnalyzer | None,
    news_text: str,
    top_volume_event_map: dict,
) -> list[str]:
    coverage = (top_volume.get("coverage") or {}).get("covered_symbols", 0)
    lines = [
        "## 거래량 상위 종목",
        f"_기준일: {format_date(top_volume.get('base_date'))} (`normalized_stock_prices_daily`, Supabase 최신 적재 기준)_",
        f"- 전체시장 커버리지: {readiness.get('coverage_status') or 'LIMITED'}",
        f"- 전체시장 가격 커버리지: {coverage:,}종목",
    ]
    if coverage <= 2000:
        lines.append("- 전체시장 거래량 상위 분석은 커버리지 부족으로 참고용입니다.")

    for market_key, section_name in (("KOSPI", "KOSPI Top 5"), ("KOSDAQ", "KOSDAQ Top 5"), ("ETF_ETN", "ETF/ETN Top 5")):
        lines.append(f"[{section_name}]")
        rows = top_volume.get(market_key) or []
        if not rows:
            lines.append("- 데이터 없음")
            lines.append("")
            continue
        for index, stock in enumerate(rows[:5], 1):
            reason = _generate_top_volume_reason(stock, naver_service, analyzer, news_text, top_volume_event_map)
            lines.append(
                f"{index}) {stock.get('name')}({stock.get('display_symbol')}) | "
                f"종가 {format_price(stock.get('close_price'))} | 거래량 {format_volume(stock.get('volume'))} | "
                f"거래대금 {format_trading_value(stock.get('trading_value'))}"
            )
            lines.append(f"   - 주목 사유: {reason}")
        lines.append("")
    return lines


def _format_static_section(static_snapshots: list[dict]) -> list[str]:
    lines = [
        "## Static 관심종목 점검",
        f"- Static 관심종목: {len(static_snapshots):,}개",
        "_기준 universe: `static_stock_universe.enabled = true`_",
    ]
    if not static_snapshots:
        lines.append("- 데이터 없음")
        return lines

    for index, snapshot in enumerate(static_snapshots, 1):
        price = snapshot.get("price") or {}
        supply = snapshot.get("supply") or {}
        fundamentals = snapshot.get("fundamentals") or {}
        event = snapshot.get("event") or {}
        features = snapshot.get("features") or {}
        supply_unit = snapshot.get("supply_unit_diag") or {}
        fundamentals_diag = snapshot.get("fundamentals_diag") or {}
        short_diag = snapshot.get("short_diag") or {}
        lines.extend(
            [
                f"{index}) {snapshot.get('name')}({snapshot.get('symbol')})",
                f"- 시장: {snapshot.get('market', NA_TEXT)} / 가격 기준일: {format_date(price.get('base_date'))}",
                f"- 종가: {format_price(price.get('close_price'))} / 거래량: {format_volume(price.get('volume'))} / 거래대금: {format_trading_value(price.get('trading_value'))} / 시가총액: {format_market_cap(price.get('market_cap'))}",
                f"- 상장주식수: {format_outstanding_shares(price.get('outstanding_shares'))}",
                f"- 수급: 개인 {_format_supply_value(supply.get('individual_net_buy'), supply_unit)}, 외국인 {_format_supply_value(supply.get('foreign_net_buy'), supply_unit)}, 기관 {_format_supply_value(supply.get('institutional_net_buy'), supply_unit)}, 외국인 보유율 {format_ratio_metric(supply.get('foreign_holding_ratio'))} (수급 기준일: {format_date(supply.get('base_date'))})",
                f"- 밸류에이션: PER {fundamentals_diag['display']['per']}, PBR {fundamentals_diag['display']['pbr']}, ROE {fundamentals_diag['display']['roe']}, 부채비율 {fundamentals_diag['display']['debt_ratio']} (밸류 기준일: {format_date(fundamentals.get('base_date'))})",
                f"- 퀀트: 5일 수익률 {format_percent((_safe_float(features.get('return_5d')) or 0) * 100) if _safe_float(features.get('return_5d')) is not None else NA_TEXT}, MA5 {format_index(features.get('moving_avg_5'))}, MA20 {format_index(features.get('moving_avg_20'))}, 변동성 {format_ratio_metric((_safe_float(features.get('volatility_20d')) or 0) * 100) if _safe_float(features.get('volatility_20d')) is not None else NA_TEXT}, 외국인 수급 z-score {format_signed_multiple(features.get('foreign_flow_zscore'), '')} (퀀트 기준일: {format_date(features.get('base_date'))})",
                f"- 공매도: {short_diag['summary']} (공매도 기준일: {format_date((snapshot.get('short_selling') or {}).get('base_date'))})",
                f"- 공시 이벤트: {event.get('event_type') or NA_TEXT} / {event.get('event_score') if event.get('event_score') is not None else NA_TEXT} / {event.get('sentiment_score') if event.get('sentiment_score') is not None else NA_TEXT}",
                f"- 퀀트 해석: {snapshot.get('quant_comment')}",
            ]
        )
        if short_diag.get("note"):
            lines.append(f"- 공매도 메모: {short_diag['note']}")
        lines.append("")

    lines.append("- 일부 수급/밸류 데이터는 최신 기준일이 가격 기준일과 다를 수 있음")
    return lines


def _collect_issue_rows(readiness: dict, top_volume: dict, prepared_snapshots: list[dict], diagnostics: dict) -> list[dict]:
    issue_rows: list[dict] = []
    if not readiness.get("full_market_coverage_pass"):
        issue_rows.append(
            {
                "category": "full_market_coverage",
                "name": "전체 시장 커버리지",
                "symbol": "-",
                "base_date": readiness.get("latest_price_date"),
                "details": f"covered_symbols={((readiness.get('price_coverage') or {}).get('covered_symbols', 0))}, records_processed={readiness.get('latest_full_price_records_processed', 0)}",
            }
        )
    for duplicate in top_volume.get("duplicates") or []:
        issue_rows.append(
            {
                "category": "duplicate_symbol",
                "name": duplicate.get("name"),
                "symbol": "/".join(duplicate.get("symbols") or []),
                "base_date": duplicate.get("base_date"),
                "details": f"duplicate canonical symbol -> {duplicate.get('canonical_symbol')}",
            }
        )
    for row in (top_volume.get("excluded_invalid_rows") or [])[:10]:
        issue_rows.append(
            {
                "category": "invalid_top_volume_row",
                "name": row.get("name"),
                "symbol": row.get("symbol"),
                "base_date": row.get("base_date"),
                "details": f"close_price={row.get('close_price')}, volume={row.get('volume')}, trading_value={row.get('trading_value')}",
            }
        )
    for snapshot in diagnostics.get("supply_unit_needs_review") or []:
        issue_rows.append(
            {
                "category": "supply_unit_unknown",
                "name": snapshot.get("name"),
                "symbol": snapshot.get("symbol"),
                "base_date": (snapshot.get("supply") or {}).get("base_date"),
                "details": "net_buy 단위 추정 불가",
            }
        )
    for snapshot in diagnostics.get("valuation_zero_needs_review") or []:
        fundamentals = snapshot.get("fundamentals") or {}
        issue_rows.append(
            {
                "category": "valuation_zero_issue",
                "name": snapshot.get("name"),
                "symbol": snapshot.get("symbol"),
                "base_date": fundamentals.get("base_date"),
                "details": f"per={fundamentals.get('per')}, pbr={fundamentals.get('pbr')}, roe={fundamentals.get('roe')}, debt_ratio={fundamentals.get('debt_ratio')}",
            }
        )
    for snapshot in diagnostics.get("short_ratio_needs_review") or []:
        short_row = snapshot.get("short_selling") or {}
        issue_rows.append(
            {
                "category": "short_ratio_issue",
                "name": snapshot.get("name"),
                "symbol": snapshot.get("symbol"),
                "base_date": short_row.get("base_date"),
                "details": f"short_value={short_row.get('short_value')}, short_volume={short_row.get('short_volume')}, short_ratio={short_row.get('short_ratio')}",
            }
        )
    return issue_rows


def _render_fix_instruction_file(now_kst: datetime.datetime, issue_rows: list[dict]) -> Path | None:
    if not issue_rows:
        return None

    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    path = reports_dir / f"stockdata_fix_instruction_{now_kst.strftime('%Y%m%d_%H%M')}.md"

    problem_summaries = []
    for category in dict.fromkeys(row["category"] for row in issue_rows):
        summary_map = {
            "full_market_coverage": "full price coverage 미완료 또는 PARTIAL 상태",
            "duplicate_symbol": "Q prefix symbol 중복",
            "invalid_top_volume_row": "Top 5 후보 중 최소 데이터 조건 미충족 row 존재",
            "supply_unit_unknown": "normalized_stock_supply_daily net_buy 단위 불명확",
            "valuation_zero_issue": "fundamentals ratios 0값 대량 또는 유효성 점검 필요",
            "short_ratio_issue": "short_value/short_volume 대비 short_ratio 0/null 불일치",
        }
        problem_summaries.append(f"- {summary_map.get(category, category)}")

    evidence_rows = [
        "| category | name | symbol | base_date | details |",
        "|---|---|---|---|---|",
    ]
    for row in issue_rows[:40]:
        evidence_rows.append(
            f"| {row['category']} | {row['name']} | {row['symbol']} | {row['base_date']} | {row['details']} |"
        )

    sql_block = """```sql
-- asset_type 미분류 종목 확인
SELECT market, COUNT(*) 
FROM stocks_master
GROUP BY market
ORDER BY market;

-- Q prefix 중복 symbol 확인
SELECT
    REGEXP_REPLACE(symbol, '^Q', '') AS canonical_symbol,
    ARRAY_AGG(symbol ORDER BY symbol) AS symbols,
    COUNT(*) AS dup_count
FROM normalized_stock_prices_daily
GROUP BY REGEXP_REPLACE(symbol, '^Q', ''), base_date
HAVING COUNT(*) > 1;

-- fundamentals ratios 0값 대량 확인
SELECT
    base_date,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (
        WHERE COALESCE(per, 0) = 0
          AND COALESCE(pbr, 0) = 0
          AND COALESCE(roe, 0) = 0
          AND COALESCE(debt_ratio, 0) = 0
    ) AS zero_ratio_rows
FROM normalized_stock_fundamentals_ratios
GROUP BY base_date
ORDER BY base_date DESC
LIMIT 10;

-- short_value > 0 인데 short_ratio 0/null 인 row
SELECT symbol, base_date, short_value, short_volume, short_ratio
FROM normalized_stock_short_selling
WHERE (COALESCE(short_value, 0) > 0 OR COALESCE(short_volume, 0) > 0)
  AND (short_ratio IS NULL OR short_ratio = 0)
ORDER BY base_date DESC
LIMIT 100;

-- full market coverage 확인
WITH latest_price AS (
    SELECT MAX(base_date) AS base_date
    FROM normalized_stock_prices_daily
)
SELECT COUNT(DISTINCT p.symbol) AS covered_symbols
FROM normalized_stock_prices_daily p
JOIN stocks_master m
  ON m.symbol = p.symbol
WHERE p.base_date = (SELECT base_date FROM latest_price)
  AND m.market IN ('KOSPI', 'KOSDAQ');
```"""

    content = "\n".join(
        [
            "# StockData 데이터 적재/스키마 수정 요청",
            "",
            "대상 저장소:",
            "- https://github.com/pos911/StockData.git",
            "",
            "문제 요약:",
            *problem_summaries,
            "",
            "증거:",
            *evidence_rows,
            "",
            "StockData 수정 지시:",
            "1. `stocks_master`에 `asset_type` 필드를 추가하거나 기존 분류를 정비하라.",
            "2. `market`과 `asset_type`을 분리하라.",
            "3. `Q530036`과 `530036`처럼 동일 상품이 중복 적재되지 않도록 `canonical_symbol`, `display_symbol`, `source_symbol` 체계를 도입하라.",
            "4. `normalized_stock_prices_daily`에 prefix만 다른 중복 row가 올라오지 않게 upsert key를 재검토하라.",
            "5. `static_stock_universe`, `stocks_master`, `normalized_*` 테이블 간 symbol join 기준을 명확히 하라.",
            "6. `normalized_stock_supply_daily`의 net_buy 계열 컬럼 단위를 spec과 코드에 명확히 정의하라.",
            "7. `normalized_stock_fundamentals_ratios`는 수집 실패 시 0이 아니라 null로 적재하라.",
            "8. `normalized_stock_fundamentals` 원천값과 ratio 정합성 검증 로직을 추가하라.",
            "9. `normalized_stock_short_selling.short_ratio`가 미수집/계산불가이면 0이 아니라 null로 적재하라.",
            "10. `short_value > 0` 또는 `short_volume > 0`인데 `short_ratio = 0`인 row를 경고로 남겨라.",
            "11. `daily_stock_full_price_pipeline.records_processed > 2000`을 full market 완료 기준으로 유지하되 부분완료 시 `WARN` 또는 `PARTIAL` 상태를 남겨라.",
            "12. `pipeline_run_logs`에 `PARTIAL` 상태를 명확히 기록하라.",
            "13. `supabase_stockdata_spec.md`에 단위, asset_type, symbol normalization 정책을 반영하라.",
            "",
            "StockData 검증 SQL:",
            sql_block,
            "",
            "주의:",
            "- 이 문서는 report_stock_daily가 생성한 수정 요청 문서이며, report 저장소가 DB를 직접 수정하지 않는다.",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def _build_report_bundle(reader: SupabaseReader) -> dict:
    top_volume = reader.fetch_top_volume_stocks_by_market(limit=20)
    static_snapshots = reader.fetch_static_universe_stock_snapshot()
    return {
        "readiness": reader.fetch_report_readiness(),
        "macro": reader.fetch_latest_global_macro_snapshot(),
        "breadth": reader.fetch_latest_market_breadth(),
        "derivatives": reader.fetch_latest_derivatives_snapshot(),
        "static_snapshots": static_snapshots,
        "top_volume": top_volume,
    }


def _rule_based_us_summary(macro: dict, news_text: str) -> list[str]:
    news_matches = _extract_docs_matches(news_text, ["S&P500", "NASDAQ", "미국", "연준"], max_matches=2)
    lines = [
        f"S&P500 {format_index(macro.get('sp500'))} ({format_percent(macro.get('sp500_change_rate'))}), NASDAQ {format_index(macro.get('nasdaq'))} ({format_percent(macro.get('nasdaq_change_rate'))}) 기준 미국 증시는 혼조 흐름입니다.",
        f"미국 10년물 {format_rate_percent(macro.get('us10y'))}, DXY {format_plain_number(macro.get('dxy'))}, VIX {format_plain_number(macro.get('vix'))}를 보면 금리와 달러 부담은 남아 있습니다.",
        news_matches[0] if news_matches else "해외 뉴스는 Google Docs 컨텍스트 기준 핵심 문장만 반영했습니다.",
    ]
    return [f"- {line}" for line in lines[:3]]


def _generate_us_market_summary(macro: dict, news_text: str, analyzer: GeminiAnalyzer | None) -> list[str]:
    if analyzer:
        try:
            generated = analyzer.generate_morning_us_summary(
                {
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
                },
                news_text,
            )
            lines = [line.strip().lstrip("- ").strip() for line in (generated or "").splitlines() if line.strip()]
            if lines:
                return [f"- {line}" for line in lines[:3]]
        except Exception as exc:
            logger.warning(f"Gemini US summary fallback to rule-based: {exc}")
    return _rule_based_us_summary(macro, news_text)


def _build_market_impact_lists(macro: dict) -> tuple[list[str], list[str], list[str]]:
    positives, burdens, watchpoints = [], [], []
    if (_safe_float(macro.get("nasdaq_change_rate")) or 0) > 0:
        positives.append(f"NASDAQ가 {format_percent(macro.get('nasdaq_change_rate'))}로 마감해 성장주 심리는 완전히 꺾이지 않았습니다.")
    if (_safe_float(macro.get("sp500_change_rate")) or 0) > 0:
        positives.append(f"S&P500이 {format_percent(macro.get('sp500_change_rate'))}로 마감해 미국 대형주 전반의 위험선호는 유지됐습니다.")
    if (_safe_float(macro.get("usdkrw")) or 0) >= 1450:
        burdens.append(f"원/달러 환율이 {format_usdkrw(macro.get('usdkrw'))} 수준이라 외국인 위험선호에는 부담입니다.")
    if (_safe_float(macro.get("us10y")) or 0) >= 4.3:
        burdens.append(f"미국 10년물이 {format_rate_percent(macro.get('us10y'))}로 높아 밸류에이션 부담이 이어질 수 있습니다.")
    if (_safe_float(macro.get("wti")) or 0) >= 90:
        burdens.append(f"유가가 {format_plain_number(macro.get('wti'))} 수준으로 높아 원가 부담 점검이 필요합니다.")
    watchpoints.append(f"KOSPI {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))}), KOSDAQ {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))}) 흐름을 함께 보세요.")
    watchpoints.append(f"KOSPI 외국인 {format_flow_amount(macro.get('kospi_foreign_net_buy')) if _safe_float(macro.get('kospi_foreign_net_buy')) is not None else NA_TEXT}, 기관 {format_flow_amount(macro.get('kospi_institutional_net_buy')) if _safe_float(macro.get('kospi_institutional_net_buy')) is not None else NA_TEXT} 수급도 확인 포인트입니다.")
    if not positives:
        positives.append("미국 지수의 위험선호 신호는 남아 있으나 강도는 제한적입니다.")
    if not burdens:
        burdens.append("거시 변수의 즉각적인 위험회피 신호는 제한적이지만 과도한 추격은 경계가 필요합니다.")
    return positives[:3], burdens[:3], watchpoints[:3]


def _infer_market_judgment(macro: dict, breadth: dict, readiness: dict) -> str:
    score = 0
    for field in ("kospi_change_rate", "kosdaq_change_rate", "sp500_change_rate", "nasdaq_change_rate"):
        value = _safe_float(macro.get(field))
        if value is not None:
            score += 1 if value > 0 else -1 if value < 0 else 0
    advances = _safe_float(breadth.get("advances"))
    declines = _safe_float(breadth.get("declines"))
    if advances is not None and declines is not None:
        score += 1 if advances > declines else -1 if advances < declines else 0
    if not readiness.get("full_market_coverage_pass"):
        score = max(-1, min(1, score))
    if score >= 2:
        return "우호"
    if score <= -2:
        return "부담"
    return "중립"


def _build_morning_report(bundle: dict, now_kst: datetime.datetime, analyzer: GeminiAnalyzer | None, news_text: str, prepared_snapshots: list[dict], diagnostics: dict, top_volume_event_map: dict) -> str:
    macro = bundle["macro"]
    breadth = bundle["breadth"]
    derivatives = bundle["derivatives"]
    readiness = bundle["readiness"]
    top_volume = bundle["top_volume"]
    positives, burdens, watchpoints = _build_market_impact_lists(macro)
    naver_service = NaverNewsService()

    lines = _build_header_lines("Morning Market Brief", now_kst, readiness, diagnostics)
    lines.extend(
        [
            "",
            "## 미국 시장 정리",
            f"_기준일: {format_date(macro.get('base_date'))} (`normalized_global_macro_daily`)_",
            f"- {label_for_column('sp500')}: {format_index(macro.get('sp500'))} ({format_percent(macro.get('sp500_change_rate'))})",
            f"- {label_for_column('nasdaq')}: {format_index(macro.get('nasdaq'))} ({format_percent(macro.get('nasdaq_change_rate'))})",
            f"- SOX: {format_index(macro.get('sox'))}",
            f"- VIX: {format_plain_number(macro.get('vix'))}",
            f"- 미국 10년물: {format_rate_percent(macro.get('us10y'))}",
            f"- DXY: {format_plain_number(macro.get('dxy'))}",
            f"- WTI / Brent / Gold / Copper: {format_plain_number(macro.get('wti'))} / {format_plain_number(macro.get('brent'))} / {format_plain_number(macro.get('gold'))} / {format_plain_number(macro.get('copper'))}",
            "- 미국 시장 핵심 요약:",
        ]
    )
    lines.extend(_generate_us_market_summary(macro, news_text, analyzer))
    lines.extend(
        [
            "",
            "## 한국 시장 영향 전망",
            f"- 원/달러: {format_usdkrw(macro.get('usdkrw'))}",
            f"- 한국 10년물 / 미국 10년물: {format_rate_percent(macro.get('kr10y'))} / {format_rate_percent(macro.get('us10y'))}",
            f"- 전일 KOSPI / KOSDAQ: {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))}) / {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))})",
            "- 긍정 요인:",
        ]
    )
    lines.extend([f"  - {item}" for item in positives])
    lines.append("- 부담 요인:")
    lines.extend([f"  - {item}" for item in burdens])
    lines.append("- 오늘 관전 포인트:")
    lines.extend([f"  - {item}" for item in watchpoints])
    lines.extend(
        [
            "",
            "## 전일 한국 시장 요약",
            f"- KOSPI: {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))})",
            f"- KOSDAQ: {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))})",
            f"- KOSPI 수급: 개인 {format_flow_amount(macro.get('kospi_individual_net_buy')) if _safe_float(macro.get('kospi_individual_net_buy')) is not None else NA_TEXT}, 외국인 {format_flow_amount(macro.get('kospi_foreign_net_buy')) if _safe_float(macro.get('kospi_foreign_net_buy')) is not None else NA_TEXT}, 기관 {format_flow_amount(macro.get('kospi_institutional_net_buy')) if _safe_float(macro.get('kospi_institutional_net_buy')) is not None else NA_TEXT}",
            f"- KOSDAQ 수급: 개인 {format_flow_amount(macro.get('kosdaq_individual_net_buy')) if _safe_float(macro.get('kosdaq_individual_net_buy')) is not None else NA_TEXT}, 외국인 {format_flow_amount(macro.get('kosdaq_foreign_net_buy')) if _safe_float(macro.get('kosdaq_foreign_net_buy')) is not None else NA_TEXT}, 기관 {format_flow_amount(macro.get('kosdaq_institutional_net_buy')) if _safe_float(macro.get('kosdaq_institutional_net_buy')) is not None else NA_TEXT}",
            f"- 상승/하락/보합: 상승 {breadth.get('advances', NA_TEXT)}개 / 하락 {breadth.get('declines', NA_TEXT)}개 / 보합 {breadth.get('unchanged', NA_TEXT)}개",
            f"- 상승 거래량 / 하락 거래량: {format_volume(breadth.get('advancing_volume'))} / {format_volume(breadth.get('declining_volume'))}",
            f"- {label_for_column('kospi200_futures')}: {format_index(derivatives.get('kospi200_futures'))} / 베이시스 {format_plain_number(derivatives.get('futures_basis')) if _safe_float(derivatives.get('futures_basis')) is not None else NA_TEXT} / 야간선물 수익률 {format_percent(derivatives.get('night_futures_return')) if _safe_float(derivatives.get('night_futures_return')) not in (None, 0) else NA_TEXT}",
            "",
        ]
    )
    lines.extend(_format_top_volume_sections(top_volume, readiness, naver_service, analyzer, news_text, top_volume_event_map))
    lines.append("")
    lines.extend(_format_static_section(prepared_snapshots))
    return _clean_debug_lines("\n".join(lines).strip())


def _build_regular_report(bundle: dict, now_kst: datetime.datetime, analyzer: GeminiAnalyzer | None, news_text: str, prepared_snapshots: list[dict], diagnostics: dict, top_volume_event_map: dict) -> str:
    macro = bundle["macro"]
    breadth = bundle["breadth"]
    derivatives = bundle["derivatives"]
    readiness = bundle["readiness"]
    top_volume = bundle["top_volume"]
    naver_service = NaverNewsService()
    judgment = _infer_market_judgment(macro, breadth, readiness)

    lines = _build_header_lines(f"Intraday Market Brief | {_get_regular_slot_label(now_kst)}", now_kst, readiness, diagnostics)
    lines.extend(
        [
            "",
            "## 1. 장중 매크로 점검",
            f"- 원/달러: {format_usdkrw(macro.get('usdkrw'))}",
            f"- DXY: {format_plain_number(macro.get('dxy'))}",
            f"- 미국 10년물 / 한국 10년물: {format_rate_percent(macro.get('us10y'))} / {format_rate_percent(macro.get('kr10y'))}",
            f"- WTI / Brent / Gold / Copper: {format_plain_number(macro.get('wti'))} / {format_plain_number(macro.get('brent'))} / {format_plain_number(macro.get('gold'))} / {format_plain_number(macro.get('copper'))}",
            f"- {label_for_column('kospi200_futures')}: {format_index(derivatives.get('kospi200_futures'))} / 야간선물 수익률 {format_percent(derivatives.get('night_futures_return')) if _safe_float(derivatives.get('night_futures_return')) not in (None, 0) else NA_TEXT}",
            "",
            "## 2. 지수 흐름 점검",
            f"- KOSPI: {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))})",
            f"- KOSDAQ: {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))})",
            f"- S&P500 / NASDAQ: {format_index(macro.get('sp500'))} ({format_percent(macro.get('sp500_change_rate'))}) / {format_index(macro.get('nasdaq'))} ({format_percent(macro.get('nasdaq_change_rate'))})",
            f"- 상승/하락/보합: 상승 {breadth.get('advances', NA_TEXT)}개 / 하락 {breadth.get('declines', NA_TEXT)}개 / 보합 {breadth.get('unchanged', NA_TEXT)}개",
            f"- 시장 판단: {judgment}",
            "",
            "## 3. 거래량 상위 종목 기반 시장 상황",
            f"- 판단 메모: 전체시장 거래량 상위는 `{readiness.get('coverage_status') or 'LIMITED'}` 커버리지 기준으로 해석합니다.",
            "",
        ]
    )
    lines.extend(_format_top_volume_sections(top_volume, readiness, naver_service, analyzer, news_text, top_volume_event_map))
    lines.append("")
    lines.extend(["## 4. Static 관심종목 점검"])
    lines.extend(_format_static_section(prepared_snapshots)[1:])
    lines.extend(
        [
            "",
            "## 5. 요약 판단",
            f"- 종합 판단: {judgment}",
            "- 행동 전략: 우호면 선별 접근, 중립이면 관망·분할, 부담이면 추격주의 관점이 적절합니다.",
        ]
    )
    return _clean_debug_lines("\n".join(lines).strip())


def _build_closing_report(bundle: dict, now_kst: datetime.datetime, analyzer: GeminiAnalyzer | None, news_text: str, prepared_snapshots: list[dict], diagnostics: dict, top_volume_event_map: dict) -> str:
    macro = bundle["macro"]
    breadth = bundle["breadth"]
    derivatives = bundle["derivatives"]
    readiness = bundle["readiness"]
    top_volume = bundle["top_volume"]
    naver_service = NaverNewsService()
    judgment = _infer_market_judgment(macro, breadth, readiness)

    lines = _build_header_lines("Closing Market Brief", now_kst, readiness, diagnostics)
    lines.extend(
        [
            "",
            "## 1. 마감 지수와 수급",
            f"- KOSPI 마감: {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))})",
            f"- KOSDAQ 마감: {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))})",
            f"- KOSPI 수급: 개인 {format_flow_amount(macro.get('kospi_individual_net_buy')) if _safe_float(macro.get('kospi_individual_net_buy')) is not None else NA_TEXT}, 외국인 {format_flow_amount(macro.get('kospi_foreign_net_buy')) if _safe_float(macro.get('kospi_foreign_net_buy')) is not None else NA_TEXT}, 기관 {format_flow_amount(macro.get('kospi_institutional_net_buy')) if _safe_float(macro.get('kospi_institutional_net_buy')) is not None else NA_TEXT}",
            f"- KOSDAQ 수급: 개인 {format_flow_amount(macro.get('kosdaq_individual_net_buy')) if _safe_float(macro.get('kosdaq_individual_net_buy')) is not None else NA_TEXT}, 외국인 {format_flow_amount(macro.get('kosdaq_foreign_net_buy')) if _safe_float(macro.get('kosdaq_foreign_net_buy')) is not None else NA_TEXT}, 기관 {format_flow_amount(macro.get('kosdaq_institutional_net_buy')) if _safe_float(macro.get('kosdaq_institutional_net_buy')) is not None else NA_TEXT}",
            f"- 시장 폭: 상승 {breadth.get('advances', NA_TEXT)}개 / 하락 {breadth.get('declines', NA_TEXT)}개 / 보합 {breadth.get('unchanged', NA_TEXT)}개",
            "",
            "## 2. 마감 매크로와 파생 체크",
            f"- 원/달러: {format_usdkrw(macro.get('usdkrw'))}",
            f"- 미국 10년물 / 한국 10년물: {format_rate_percent(macro.get('us10y'))} / {format_rate_percent(macro.get('kr10y'))}",
            f"- DXY / VIX / SOX: {format_plain_number(macro.get('dxy'))} / {format_plain_number(macro.get('vix'))} / {format_index(macro.get('sox'))}",
            f"- {label_for_column('kospi200_futures')}: {format_index(derivatives.get('kospi200_futures'))} / 베이시스 {format_plain_number(derivatives.get('futures_basis')) if _safe_float(derivatives.get('futures_basis')) is not None else NA_TEXT} / 야간선물 수익률 {format_percent(derivatives.get('night_futures_return')) if _safe_float(derivatives.get('night_futures_return')) not in (None, 0) else NA_TEXT}",
            "",
            "## 3. 거래량 상위 종목 테마",
            f"- 마감 판단: {judgment}",
            "",
        ]
    )
    lines.extend(_format_top_volume_sections(top_volume, readiness, naver_service, analyzer, news_text, top_volume_event_map))
    lines.append("")
    lines.extend(["## 4. Static 관심종목 종가/수급/퀀트/밸류"])
    lines.extend(_format_static_section(prepared_snapshots)[1:])
    lines.extend(
        [
            "",
            "## 5. 다음 거래일 체크포인트",
            "- 환율과 미국 장 마감 방향이 국내 위험선호를 유지시키는지 확인",
            "- 외국인 수급이 가격 흐름을 따라오는지, 혹은 가격과 엇갈리는지 확인",
            "- 거래량 상위 종목이 단기 순환매인지, 특정 테마 확산인지 구분해 추격 여부를 결정",
        ]
    )
    return _clean_debug_lines("\n".join(lines).strip())


def _save_report(report_type: str, report_content: str, now_kst: datetime.datetime) -> Path:
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    file_path = reports_dir / f"daily_quant_report_{report_type}_{now_kst.strftime('%Y%m%d_%H%M')}.md"
    file_path.write_text(report_content, encoding="utf-8")
    logger.info(f"Report saved: {file_path}")
    return file_path


def run_report(report_type: str, now_kst: datetime.datetime):
    reader = SupabaseReader()
    analyzer = _safe_get_analyzer()
    logger.info("Supabase official tables bundle loading...")
    bundle = _build_report_bundle(reader)
    logger.info("Google Docs news loading...")
    news_text = reader.prepare_news_context(reader.fetch_news_document())

    prepared_snapshots, static_diagnostics = _prepare_static_snapshots(bundle["static_snapshots"])
    top_volume_event_map = _prepare_top_volume_event_map(reader, bundle["top_volume"])
    issue_rows = _collect_issue_rows(bundle["readiness"], bundle["top_volume"], prepared_snapshots, static_diagnostics)
    fix_instruction_path = _render_fix_instruction_file(now_kst, issue_rows)

    diagnostics_summary = {
        "supply_unit_needs_review": len(static_diagnostics["supply_unit_needs_review"]),
        "valuation_zero_needs_review": len(static_diagnostics["valuation_zero_needs_review"]),
        "short_ratio_needs_review": len(static_diagnostics["short_ratio_needs_review"]),
    }
    if fix_instruction_path:
        logger.info(f"StockData fix instruction generated: {fix_instruction_path}")

    if report_type == "morning":
        report_content = _build_morning_report(bundle, now_kst, analyzer, news_text, prepared_snapshots, diagnostics_summary, top_volume_event_map)
    elif report_type == "closing":
        report_content = _build_closing_report(bundle, now_kst, analyzer, news_text, prepared_snapshots, diagnostics_summary, top_volume_event_map)
    else:
        report_content = _build_regular_report(bundle, now_kst, analyzer, news_text, prepared_snapshots, diagnostics_summary, top_volume_event_map)

    _save_report(report_type, report_content, now_kst)
    try:
        sender = TelegramSender()
        sender.send_report(report_content)
    except Exception as exc:
        logger.warning(f"Telegram send failed (non-fatal): {exc}")


def main():
    args = _parse_args()
    now_kst = _get_now_kst()
    logger.info(f"=== Daily Report Pipeline start [type={args.report_type}] ===")
    run_report(args.report_type, now_kst)


if __name__ == "__main__":
    main()

