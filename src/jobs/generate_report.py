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
    format_bp,
    format_index,
    format_market_cap,
    format_multiple,
    format_outstanding_shares,
    format_percent,
    format_plain_number,
    format_price,
    format_rate_percent,
    format_rate_level,
    format_ratio_metric,
    format_signed_multiple,
    format_spread_bp,
    format_trading_value,
    format_usdkrw,
    format_volume,
    is_missing,
)
from src.utils.market_assets import canonicalize_symbol, extract_theme_keywords, label_for_column


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_REPORT_TYPES = ("morning", "regular", "closing")
KST = ZoneInfo("Asia/Seoul")
STATIC_SAMPLE_SYMBOLS = {"005930", "000660", "058470", "071050", "278470"}
SIGNAL_MODEL_VERSION = "v0.1_unbacktested"


def _parse_args():
    parser = argparse.ArgumentParser(description="Daily Quant Report Generator")
    parser.add_argument("--type", dest="report_type", default="regular")
    parser.add_argument("--date", dest="report_date", help="Report date in YYYYMMDD or YYYY-MM-DD (KST basis).")
    parser.add_argument("--dry-run", action="store_true", help="Generate report only and skip Telegram sending.")
    parser.add_argument("--no-send", action="store_true", help="Generate report only and skip Telegram sending.")
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
        logger.warning("Gemini analyzer unavailable, rule-based fallback will be used: %s", exc)
        return None


def _safe_float(value):
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_report_text(report_content: str) -> str:
    banned = ("Test_only", "섹션 수:", "Sections:")
    lines = []
    for line in report_content.splitlines():
        if any(fragment in line for fragment in banned):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip() + "\n"


def _should_include_kr_sections(calendar_status: dict) -> bool:
    return calendar_status.get("report_market_mode") in {"FULL_REPORT", "KOREA_ONLY", "CALENDAR_UNKNOWN"}


def _should_include_us_sections(calendar_status: dict) -> bool:
    return calendar_status.get("report_market_mode") in {"FULL_REPORT", "US_ONLY", "CALENDAR_UNKNOWN"}


def _should_skip_all_markets(calendar_status: dict) -> bool:
    return calendar_status.get("report_market_mode") == "SKIP_ALL_MARKETS_CLOSED"


def _build_market_mode_banner(calendar_status: dict) -> str:
    mode = calendar_status.get("report_market_mode")
    if mode == "KOREA_ONLY":
        return "시장 기준: 한국장 개장 / 미국장 휴장"
    if mode == "US_ONLY":
        return "시장 기준: 한국장 휴장 / 미국장 개장"
    if mode == "SKIP_ALL_MARKETS_CLOSED":
        return "시장 기준: 한국장·미국장 모두 휴장"
    if mode == "CALENDAR_UNKNOWN":
        return "시장 기준: calendar fallback 사용"
    return "시장 기준: 한국장 개장 / 미국장 개장"


def _build_market_closed_skip_text(report_type: str, now_kst: datetime.datetime, calendar_status: dict) -> str:
    lines = [
        f"SKIPPED_REPORT_MARKET_CLOSED | type={report_type}",
        f"- 작성시각: {now_kst.strftime('%Y-%m-%d %H:%M KST')}",
        f"- report_date: {calendar_status.get('report_date')}",
        f"- XKRX: closed ({calendar_status.get('xkrx_reason')}) / prev {calendar_status.get('xkrx_previous_trading_day') or NA_TEXT}",
        f"- XNYS: closed ({calendar_status.get('xnys_reason')}) / prev {calendar_status.get('xnys_previous_trading_day') or NA_TEXT}",
        "- status: SKIPPED_REPORT_MARKET_CLOSED",
    ]
    return "\n".join(lines)


def _truncate_text(text: str, max_len: int = 110) -> str:
    stripped = (text or "").strip()
    if len(stripped) <= max_len:
        return stripped
    return stripped[: max_len - 1].rstrip() + "…"


def _extract_docs_matches(news_text: str, keywords: list[str], max_matches: int = 3) -> list[str]:
    if not news_text:
        return []
    lowered_keywords = [keyword.lower() for keyword in keywords if keyword]
    matches = []
    for line in news_text.splitlines():
        stripped = line.strip()
        if stripped and any(keyword in stripped.lower() for keyword in lowered_keywords):
            matches.append(stripped)
        if len(matches) >= max_matches:
            break
    return matches


def _format_market_flow(value) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return NA_TEXT
    if numeric == 0:
        return "0"
    return format_flow_amount(numeric)


def _interpret_us_10y_3y_spread(us10y, us3y) -> dict | None:
    us10y_value = _safe_float(us10y)
    us3y_value = _safe_float(us3y)
    if us10y_value is None or us3y_value is None:
        return None

    spread_bp = (us10y_value - us3y_value) * 100
    if spread_bp >= 75:
        regime = "steep_positive"
        plain = "장기금리가 단기금리보다 꽤 높은 정상 곡선입니다."
        market_implication = "경기 회복 기대 또는 장기 인플레이션·기간프리미엄이 반영될 가능성이 있습니다."
        equity_implication = "금융주에는 우호적일 수 있지만 장기금리 급등이면 성장주 밸류에는 부담입니다."
        watchpoint = "금리차만이 아니라 10년물 레벨, 달러, VIX, 주가지수 흐름을 함께 보아야 합니다."
    elif spread_bp >= 25:
        regime = "mildly_positive"
        plain = "완만한 정상 곡선으로 경기침체 신호는 약한 편입니다."
        market_implication = "다만 금리 레벨 자체가 높으면 위험자산에는 중립 또는 부담일 수 있습니다."
        equity_implication = "성장주보다 실적과 현금흐름이 확인되는 종목 선호가 자연스럽습니다."
        watchpoint = "금리차 자체보다 높은 절대 금리 수준과 달러 강도를 함께 확인해야 합니다."
    elif spread_bp > -25:
        regime = "flat"
        plain = "단기와 장기 금리 차가 거의 없어 향후 금리 경로 기대가 엇갈리는 구간입니다."
        market_implication = "방향성 확신이 낮아 방어주·배당주·저변동 전략이 상대적으로 편할 수 있습니다."
        equity_implication = "방향성 베팅보다 실적 확인과 리스크 관리가 우선입니다."
        watchpoint = "달러, VIX, 지수 추세와 함께 해석해야 의미가 커집니다."
    elif spread_bp > -75:
        regime = "mildly_inverted"
        plain = "단기금리가 장기금리보다 높은 역전 구간입니다."
        market_implication = "긴축 부담과 향후 경기 둔화 기대가 일부 반영됐을 가능성이 있습니다."
        equity_implication = "경기민감주 추격은 신중하고 대형 퀄리티·현금흐름 중심 접근이 유리합니다."
        watchpoint = "금리차만으로 침체를 단정하지 말고 달러, VIX, 실적 흐름과 함께 봐야 합니다."
    else:
        regime = "deeply_inverted"
        plain = "강한 역전 구간으로 정책금리 인하 기대 또는 경기 우려가 크게 반영된 상태일 수 있습니다."
        market_implication = "고위험 성장주보다 방어·현금흐름 중심 자산이 더 중요해질 수 있습니다."
        equity_implication = "레버리지성 종목 추격보다 리스크 예산 관리가 우선입니다."
        watchpoint = "절대 금리 수준, 달러, VIX와 같이 보지 않으면 해석이 과도해질 수 있습니다."

    return {
        "us10y": us10y_value,
        "us3y": us3y_value,
        "spread_bp": spread_bp,
        "regime": regime,
        "plain_korean_summary": plain,
        "market_implication": market_implication,
        "equity_implication": equity_implication,
        "watchpoint": watchpoint,
    }


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
    nonzero = [abs(value) for value in values if value not in (None, 0)]
    if not nonzero:
        return {"unit": "unknown", "needs_review": False}

    share_like = 0
    amount_like = 0
    if close_price and trading_value:
        for value in nonzero:
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


def _format_stock_supply(value, unit_diag: dict) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return NA_TEXT
    if unit_diag.get("unit") == "shares":
        label = "순매수" if numeric >= 0 else "순매도"
        return f"{label} {abs(int(round(numeric))):,}주"
    if unit_diag.get("unit") == "krw":
        return format_flow_amount(numeric)
    return f"{numeric:+,.0f} (단위 확인 필요)"


def _diagnose_fundamentals(snapshot: dict) -> dict:
    fundamentals = snapshot.get("fundamentals") or {}
    raw = snapshot.get("fundamentals_raw") or {}
    history = snapshot.get("fundamentals_history") or []
    metrics = {key: _safe_float(fundamentals.get(key)) for key in ("per", "pbr", "roe", "debt_ratio")}
    current_all_zero = all(value in (0, None) for value in metrics.values())
    history_all_zero = False
    if history:
        history_all_zero = all(
            all(_safe_float(row.get(key)) in (0, None) for key in ("per", "pbr", "roe", "debt_ratio"))
            for row in history[:3]
        )
    raw_all_zero = bool(raw) and all(
        _safe_float(raw.get(key)) in (0, None)
        for key in ("revenue", "operating_income", "net_income", "total_assets", "total_liabilities", "total_equity")
    )

    display = {}
    needs_review = False
    for key, value in metrics.items():
        invalid = False
        if value is None:
            display[key] = NA_TEXT
            continue
        if key in {"per", "pbr"} and value <= 0:
            invalid = True
        if key in {"roe", "debt_ratio"} and value == 0 and (current_all_zero or history_all_zero or raw_all_zero):
            invalid = True
        if invalid:
            display[key] = "N/A(점검필요)"
            needs_review = True
        elif key in {"per", "pbr"}:
            display[key] = format_multiple(value, "배")
        else:
            display[key] = format_ratio_metric(value)
    return {"display": display, "needs_review": needs_review}


def _diagnose_short_selling(snapshot: dict) -> dict:
    short_row = snapshot.get("short_selling") or {}
    price = snapshot.get("price") or {}
    short_ratio = _safe_float(short_row.get("short_ratio"))
    short_value = _safe_float(short_row.get("short_value"))
    short_volume = _safe_float(short_row.get("short_volume"))
    trading_value = _safe_float(price.get("trading_value"))
    volume = _safe_float(price.get("volume"))

    if short_ratio is None and short_value and trading_value:
        short_ratio = (short_value / trading_value) * 100
    volume_ratio = None
    if short_volume and volume:
        volume_ratio = (short_volume / volume) * 100

    if short_ratio is None:
        ratio_text = "비중 N/A"
        needs_review = bool((short_value or 0) > 0 or (short_volume or 0) > 0)
    elif short_ratio == 0 and ((short_value or 0) > 0 or (short_volume or 0) > 0):
        ratio_text = "비중 N/A(점검필요)"
        needs_review = True
    else:
        ratio_text = f"비중 {format_ratio_metric(short_ratio)}"
        needs_review = False

    parts = []
    if short_value is not None and short_value > 0:
        parts.append(f"거래금액 {format_trading_value(short_value)}")
    if short_volume is not None and short_volume > 0:
        parts.append(f"거래량 {format_volume(short_volume)}")
    parts.append(ratio_text)
    if volume_ratio is not None:
        parts.append(f"거래량 기준 {format_ratio_metric(volume_ratio)}")
    note = "공매도 금액·수량은 확인되나 비중값 부재로 방향성 해석은 제한적" if needs_review else ""
    return {"summary": " / ".join(parts), "needs_review": needs_review, "note": note}


def _build_quant_comment(snapshot: dict) -> str:
    features = snapshot.get("features") or {}
    supply = snapshot.get("supply") or {}
    fundamentals_diag = snapshot.get("fundamentals_diag") or {"display": {}}
    parts = []

    return_5d = _safe_float(features.get("return_5d"))
    vol_20d = _safe_float(features.get("volatility_20d"))
    zscore = _safe_float(features.get("foreign_flow_zscore"))
    foreign_holding = _safe_float(supply.get("foreign_holding_ratio"))
    per_display = fundamentals_diag.get("display", {}).get("per", NA_TEXT)
    pbr_display = fundamentals_diag.get("display", {}).get("pbr", NA_TEXT)

    if return_5d is not None:
        parts.append("최근 5일 수익률은 양호" if return_5d > 0 else "최근 5일 수익률은 부진" if return_5d < 0 else "최근 5일 수익률은 중립")
    if vol_20d is not None:
        parts.append("20일 변동성이 높아 추격 매수는 신중" if vol_20d >= 0.05 else "20일 변동성은 과도하지 않음")
    if zscore is not None:
        if zscore >= 1:
            parts.append("외국인 수급 강도는 우호적")
        elif zscore <= -1:
            parts.append("외국인 수급 강도는 부담")
    if foreign_holding is not None and foreign_holding >= 10:
        parts.append("외국인 보유율은 비교적 안정적")
    if per_display == "N/A(점검필요)" or pbr_display == "N/A(점검필요)":
        parts.append("밸류에이션 수치는 점검 전까지 판단 유보")
    if not parts:
        return "현재 공개된 가격·수급·밸류 지표만으로는 방향성을 단정하기보다 관망이 적절합니다."
    return _truncate_text(". ".join(parts) + ".")


def _build_ranking_signal_lookup(ranking_bundle: dict) -> dict:
    lookup = {}
    for rank_type, market_map in (ranking_bundle.get("sections") or {}).items():
        for market, rows in market_map.items():
            for row in rows:
                symbol = canonicalize_symbol(row.get("symbol"))
                if not symbol:
                    continue
                bucket = lookup.setdefault(symbol, {})
                bucket[f"{rank_type}_rank"] = row.get("rank")
                bucket[f"{rank_type}_market"] = market
                bucket[f"{rank_type}_source"] = row.get("source")
    return lookup


def _score_watchlist_snapshot(snapshot: dict, ranking_lookup: dict, macro: dict) -> dict:
    price = snapshot.get("price") or {}
    supply = snapshot.get("supply") or {}
    features = snapshot.get("features") or {}
    fundamentals_diag = snapshot.get("fundamentals_diag") or {"display": {}}
    short_diag = snapshot.get("short_diag") or {}
    symbol = canonicalize_symbol(snapshot.get("symbol"))
    ranking = ranking_lookup.get(symbol, {})

    close_price = _safe_float(price.get("close_price"))
    ma5 = _safe_float(features.get("moving_avg_5"))
    ma20 = _safe_float(features.get("moving_avg_20"))
    return_5d = _safe_float(features.get("return_5d"))
    volatility = _safe_float(features.get("volatility_20d"))
    foreign_z = _safe_float(features.get("foreign_flow_zscore"))
    foreign_net = _safe_float(supply.get("foreign_net_buy"))
    inst_net = _safe_float(supply.get("institutional_net_buy"))
    short_ratio = _safe_float((snapshot.get("short_selling") or {}).get("short_ratio"))

    price_score_raw = 0
    price_components = 0
    if return_5d is not None:
        price_components += 1
        price_score_raw += 1 if return_5d > 0 else -1 if return_5d < 0 else 0
    if close_price is not None and ma20 is not None:
        price_components += 1
        price_score_raw += 1 if close_price >= ma20 else -1
    if close_price is not None and ma5 is not None:
        price_components += 1
        price_score_raw += 1 if close_price >= ma5 else -1
    price_momentum_score = max(-2, min(2, price_score_raw))

    volume_score_raw = 0
    volume_components = 0
    for key in ("volume_rank", "trading_value_rank"):
        rank = ranking.get(key)
        if rank is not None:
            volume_components += 1
            volume_score_raw += 1 if int(rank) <= 5 else 0
    volume_trading_score = max(-2, min(2, volume_score_raw))

    supply_score_raw = 0
    supply_components = 0
    if foreign_z is not None:
        supply_components += 1
        supply_score_raw += 1 if foreign_z >= 1 else -1 if foreign_z <= -1 else 0
    if foreign_net is not None:
        supply_components += 1
        supply_score_raw += 1 if foreign_net > 0 else -1 if foreign_net < 0 else 0
    if inst_net is not None:
        supply_components += 1
        supply_score_raw += 1 if inst_net > 0 else -1 if inst_net < 0 else 0
    supply_score = max(-2, min(2, supply_score_raw))

    valuation_score_raw = 0
    valuation_components = 0
    if fundamentals_diag.get("display", {}).get("per") not in {NA_TEXT, "N/A(점검필요)"}:
        valuation_components += 1
        valuation_score_raw += 1
    if fundamentals_diag.get("display", {}).get("pbr") not in {NA_TEXT, "N/A(점검필요)"}:
        valuation_components += 1
    valuation_score = max(-1, min(1, valuation_score_raw))

    risk_score_raw = 0
    risk_components = 0
    if volatility is not None:
        risk_components += 1
        risk_score_raw += -2 if volatility >= 0.08 else -1 if volatility >= 0.05 else 0
    if short_ratio is not None:
        risk_components += 1
        risk_score_raw += -1 if short_ratio >= 5 else 0
    elif short_diag.get("needs_review"):
        risk_components += 1
        risk_score_raw += -1
    risk_score = max(-2, min(0, risk_score_raw))

    macro_fit_score_raw = 0
    macro_components = 0
    market = snapshot.get("market")
    usdkrw = _safe_float(macro.get("usdkrw"))
    us10y = _safe_float(macro.get("us10y"))
    if market == "KOSPI" and usdkrw is not None:
        macro_components += 1
        macro_fit_score_raw += 1 if usdkrw < 1450 else 0
    if market == "KOSDAQ" and us10y is not None and us10y >= 4.3:
        macro_components += 1
        macro_fit_score_raw -= 1
    macro_fit_score = max(-1, min(1, macro_fit_score_raw))

    total_signal_score = price_momentum_score + volume_trading_score + supply_score + valuation_score + risk_score + macro_fit_score
    available_components = sum(
        1 for count in (price_components, volume_components, supply_components, valuation_components, risk_components, macro_components) if count > 0
    )
    positive_components = sum(1 for score in (price_momentum_score, volume_trading_score, supply_score, valuation_score, macro_fit_score) if score > 0)
    negative_components = sum(1 for score in (price_momentum_score, supply_score, risk_score, macro_fit_score) if score < 0)

    if available_components >= 5 and abs(positive_components - negative_components) >= 2:
        confidence = "high"
    elif available_components >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    if available_components <= 1:
        label = "판단 유보"
    elif total_signal_score >= 4 and confidence != "low":
        label = "비중확대 후보"
    elif total_signal_score >= 1:
        label = "보유/관찰"
    elif total_signal_score <= -2:
        label = "리스크 축소 후보"
    else:
        label = "관망"

    strategy_parts = []
    if volatility is not None and volatility >= 0.05:
        strategy_parts.append("추격 진입보다 분할 접근")
    if total_signal_score >= 4:
        strategy_parts.append("거래량 유지 여부 확인 후 비중 확대 검토")
    elif total_signal_score <= -2:
        strategy_parts.append("손익보다 변동성 관리 우선")
    else:
        strategy_parts.append("확인 신호가 더 쌓일 때까지 관찰")
    if foreign_z is not None and foreign_z < 0:
        strategy_parts.append("외국인 수급 반전 여부 점검")
    strategy_memo = _truncate_text(". ".join(dict.fromkeys(strategy_parts)) + ".")

    return {
        "price_momentum_score": price_momentum_score,
        "volume_trading_score": volume_trading_score,
        "supply_score": supply_score,
        "valuation_score": valuation_score,
        "risk_score": risk_score,
        "macro_fit_score": macro_fit_score,
        "total_signal_score": total_signal_score,
        "confidence": confidence,
        "label": label,
        "strategy_memo": strategy_memo,
        "signal_model_version": SIGNAL_MODEL_VERSION,
        "ranking": ranking,
    }


def _build_news_queries(stock: dict) -> list[str]:
    name = stock.get("name") or ""
    asset_type = stock.get("asset_type") or "UNKNOWN"
    queries = [f"{name} 주가", f"{name} 특징주", f"{name} 실적"]
    if asset_type in {"ETF", "ETN"}:
        themes = extract_theme_keywords(name)
        for theme in themes[:3]:
            queries.extend([f"{theme} ETF", f"{theme} ETN", f"{theme} 테마"])
    return list(dict.fromkeys(query for query in queries if query))


def _build_fallback_reason(stock: dict, docs_matches: list[str], event_row: dict | None) -> str:
    name = stock.get("name") or ""
    themes = extract_theme_keywords(name)
    asset_type = stock.get("asset_type") or "UNKNOWN"
    if docs_matches:
        return _truncate_text(docs_matches[0], 100)
    if event_row and event_row.get("event_type"):
        return f"최근 공시 이벤트({event_row.get('event_type')})가 거래 관심을 자극한 것으로 보입니다."
    if asset_type in {"ETF", "ETN"} and themes:
        return f"{', '.join(themes[:2])} 관련 변동성이 상품 거래를 자극한 것으로 보입니다."
    if themes:
        return f"{', '.join(themes[:2])} 테마 연관성으로 거래가 집중된 것으로 보입니다."
    return "최근 거래 집중이 확인돼 단기 수급 유입 가능성을 점검할 필요가 있습니다."


def _generate_ranked_reason(
    stock: dict,
    naver_service: NaverNewsService,
    analyzer: GeminiAnalyzer | None,
    news_text: str,
    event_map: dict,
    gemini_tracker: dict,
) -> str:
    queries = _build_news_queries(stock)
    news_items = []
    if naver_service.enabled:
        try:
            news_items = naver_service.search_queries(queries, display_per_query=5, max_items=5)
        except Exception as exc:
            logger.warning("Naver news lookup failed for %s: %s", stock.get("name"), exc)

    keywords = [stock.get("name") or "", stock.get("display_symbol") or stock.get("symbol") or ""]
    keywords.extend(extract_theme_keywords(stock.get("name") or ""))
    docs_matches = _extract_docs_matches(news_text, keywords, max_matches=3)
    event_row = event_map.get(canonicalize_symbol(stock.get("symbol"))) if event_map else None

    if analyzer and (news_items or docs_matches or event_row):
        try:
            gemini_tracker["count"] += 1
            gemini_tracker["purposes"].append(f"rank_reason:{stock.get('rank_type')}:{stock.get('display_symbol')}")
            reason = analyzer.generate_top_volume_reason(
                {
                    "symbol": stock.get("display_symbol") or stock.get("symbol"),
                    "name": stock.get("name"),
                    "market": stock.get("market"),
                    "asset_type": stock.get("asset_type"),
                    "rank_type": stock.get("rank_type"),
                    "rank": stock.get("rank"),
                    "source": stock.get("source"),
                    "ranking_base_date": stock.get("ranking_base_date"),
                    "price_base_date": stock.get("price_base_date"),
                    "close_price": stock.get("close_price"),
                    "volume": stock.get("volume"),
                    "trading_value": stock.get("trading_value"),
                    "market_cap": stock.get("market_cap"),
                    "change_rate": stock.get("change_rate"),
                    "recent_news": [
                        {
                            "title": item.get("title"),
                            "summary": item.get("description"),
                            "pubDate": item.get("pubDate"),
                            "origin": item.get("originallink") or item.get("link"),
                            "query": item.get("query"),
                        }
                        for item in news_items[:5]
                    ],
                    "event": {
                        "event_type": event_row.get("event_type"),
                        "event_score": event_row.get("event_score"),
                        "sentiment_score": event_row.get("sentiment_score"),
                    }
                    if event_row
                    else None,
                    "docs_matches": docs_matches[:3],
                }
            )
            if reason:
                return _truncate_text(reason.strip().lstrip("- "), 100)
        except Exception as exc:
            logger.warning("Gemini ranking reason fallback for %s: %s", stock.get("name"), exc)

    if news_items:
        title = (news_items[0].get("title") or "").replace("<b>", "").replace("</b>", "")
        return _truncate_text(f"{title} 관련 보도가 확인돼 거래 집중 배경으로 해석됩니다.", 100)

    return _build_fallback_reason(stock, docs_matches, event_row)


def _prepare_watchlist_snapshots(static_snapshots: list[dict]) -> tuple[list[dict], dict]:
    prepared = []
    diagnostics = {
        "supply_unit_needs_review": False,
        "valuation_zero_needs_review": False,
        "short_ratio_needs_review": False,
        "watchlist_missing_prices": [],
    }
    for snapshot in static_snapshots:
        supply_diag = _diagnose_supply_unit(snapshot)
        fundamentals_diag = _diagnose_fundamentals(snapshot)
        short_diag = _diagnose_short_selling(snapshot)
        quant_comment = _build_quant_comment({**snapshot, "fundamentals_diag": fundamentals_diag})
        price = snapshot.get("price") or {}

        if supply_diag.get("needs_review"):
            diagnostics["supply_unit_needs_review"] = True
        if fundamentals_diag.get("needs_review"):
            diagnostics["valuation_zero_needs_review"] = True
        if short_diag.get("needs_review"):
            diagnostics["short_ratio_needs_review"] = True
        if price.get("close_price") in (None, ""):
            diagnostics["watchlist_missing_prices"].append(snapshot)

        prepared.append(
            {
                **snapshot,
                "supply_diag": supply_diag,
                "fundamentals_diag": fundamentals_diag,
                "short_diag": short_diag,
                "quant_comment": quant_comment,
            }
        )
    return prepared, diagnostics


def _attach_signal_scores(prepared_snapshots: list[dict], ranking_bundle: dict, macro: dict) -> list[dict]:
    ranking_lookup = _build_ranking_signal_lookup(ranking_bundle)
    enriched = []
    for snapshot in prepared_snapshots:
        signal_score = _score_watchlist_snapshot(snapshot, ranking_lookup, macro)
        enriched.append({**snapshot, "signal_score": signal_score})
    return enriched


def _build_reader_bundle(reader: SupabaseReader, report_type: str, report_date: str, calendar_status: dict) -> dict:
    mode = calendar_status.get("report_market_mode")
    macro = reader.fetch_latest_global_macro_snapshot()
    report_base_date = report_date or macro.get("base_date")
    if mode == "US_ONLY":
        ranking_bundle = {
            "ranking_base_date": None,
            "price_base_date": None,
            "latest_valid_price_date": None,
            "fallback_used": False,
            "sections": {"volume": {}, "trading_value": {}, "market_cap": {}},
            "diagnostics": {"market_mismatch_rows": [], "q_prefix_rows": [], "legacy_source_rows": [], "ranking_counts": {}},
            "ranking_status": "한국장 휴장",
            "price_meta": {},
            "fallback_applied_sections": [],
        }
        watchlist_bundle = {"price_base_date": None, "snapshots": [], "price_meta": {}}
        readiness = {
            "latest_macro_date": macro.get("base_date"),
            "latest_valid_price_date": None,
            "market_master_status": "한국장 휴장",
            "ranking_status": "한국장 휴장",
            "price_status": "한국장 휴장",
            "watchlist_price_hit_ratio": 0.0,
            "minimum_report_ready": bool(macro),
            "ranking_ready": False,
            "watchlist_ready": False,
        }
        breadth = {}
        derivatives = {}
    else:
        ranking_bundle = reader.get_latest_market_rankings(report_date=report_base_date, limit=10)
        watchlist_bundle = reader.get_watchlist_snapshots(report_date=report_base_date)
        readiness = reader.fetch_report_readiness()
        breadth = reader.fetch_latest_market_breadth()
        derivatives = reader.fetch_latest_derivatives_snapshot()
    yield_curve = _interpret_us_10y_3y_spread(macro.get("us10y"), macro.get("us3y"))
    if yield_curve:
        macro["yield_curve_regime"] = yield_curve.get("regime")
    readiness["yield_curve"] = yield_curve
    readiness["calendar_status"] = calendar_status
    return {
        "report_type": report_type,
        "macro": macro,
        "breadth": breadth,
        "derivatives": derivatives,
        "readiness": readiness,
        "ranking_bundle": ranking_bundle,
        "watchlist_bundle": watchlist_bundle,
        "calendar_status": calendar_status,
    }


def _build_data_status(readiness: dict) -> str:
    if not readiness.get("minimum_report_ready"):
        return "위험"
    if readiness.get("ranking_ready") and readiness.get("watchlist_ready"):
        return "정상"
    return "경고"


def _log_report_diagnostics(bundle: dict, prepared_snapshots: list[dict], gemini_tracker: dict):
    ranking_bundle = bundle["ranking_bundle"]
    watchlist_bundle = bundle["watchlist_bundle"]
    readiness = bundle["readiness"]
    ranking_diag = ranking_bundle.get("diagnostics") or {}
    logger.info("ranking_base_date=%s", ranking_bundle.get("ranking_base_date"))
    logger.info("price_base_date=%s", ranking_bundle.get("price_base_date"))
    logger.info("latest_valid_price_date=%s", readiness.get("latest_valid_price_date"))
    logger.info("market_master_status=%s", readiness.get("market_master_status"))
    logger.info("ranking_status=%s", readiness.get("ranking_status"))
    logger.info("watchlist_price_hit_ratio=%.3f", readiness.get("watchlist_price_hit_ratio") or 0.0)
    logger.info("yield_curve=%s", readiness.get("yield_curve"))
    logger.info("ranking_counts=%s", ranking_diag.get("ranking_counts", {}))
    logger.info("fallback_applied_sections=%s", ranking_bundle.get("fallback_applied_sections", []))
    logger.info("market_mismatch_rows=%s", len(ranking_diag.get("market_mismatch_rows", [])))
    logger.info("q_prefix_rows=%s", len(ranking_diag.get("q_prefix_rows", [])))
    if ranking_diag.get("q_prefix_rows"):
        logger.error("FAIL_SYMBOL_NORMALIZATION q_prefix_rows=%s", ranking_diag.get("q_prefix_rows"))
    for snapshot in prepared_snapshots:
        if snapshot.get("symbol") in STATIC_SAMPLE_SYMBOLS:
            price = snapshot.get("price") or {}
            logger.info(
                "watchlist_sample symbol=%s price_base_date=%s close_price=%s volume=%s trading_value=%s market=%s name=%s",
                snapshot.get("symbol"),
                price.get("base_date"),
                price.get("close_price"),
                price.get("volume"),
                price.get("trading_value"),
                snapshot.get("market"),
                snapshot.get("name"),
            )
    logger.info("watchlist_count=%s", len(watchlist_bundle.get("snapshots") or []))
    logger.info("Gemini call count and purpose=%s / %s", gemini_tracker["count"], gemini_tracker["purposes"])


def _build_stockdata_fix_text(bundle: dict, diagnostics: dict) -> str:
    ranking_bundle = bundle["ranking_bundle"]
    watchlist_bundle = bundle["watchlist_bundle"]
    ranking_diag = ranking_bundle.get("diagnostics") or {}
    lines = ["pos911/StockData 레포에서 다음 문제를 수정하라."]
    if ranking_diag.get("market_mismatch_rows"):
        lines.append("- normalized_market_rankings_daily와 stocks_master.market 불일치 row를 정리하라.")
    if ranking_diag.get("q_prefix_rows"):
        lines.append("- normalized_market_rankings_daily 또는 normalized_stock_prices_daily에 Q-prefix symbol이 남아 있지 않도록 canonical 6자리 숫자로 정규화하라.")
    if ranking_diag.get("legacy_source_rows"):
        lines.append("- rank_type이 trading_value/market_cap인 row에 source='KIS'가 들어오지 않도록 legacy source를 차단하라.")
    if ranking_bundle.get("fallback_applied_sections"):
        lines.append(f"- ranking row 부족으로 가격 fallback이 사용된 섹션: {', '.join(ranking_bundle.get('fallback_applied_sections', []))}")
    if diagnostics.get("watchlist_missing_prices"):
        samples = ", ".join(f"{item.get('name')}({item.get('symbol')})" for item in diagnostics["watchlist_missing_prices"][:5])
        lines.append(f"- static 관심종목 가격 누락 샘플: {samples}")
    if diagnostics.get("valuation_zero_needs_review"):
        lines.append("- normalized_stock_fundamentals_ratios에서 수집 실패 시 0 대신 null 적재로 바꾸고 ratio 0값 대량 발생 여부를 검증하라.")
    if diagnostics.get("short_ratio_needs_review"):
        lines.append("- normalized_stock_short_selling에서 short_value/short_volume가 양수인데 short_ratio가 0/null인 row를 점검하라.")
    if diagnostics.get("supply_unit_needs_review"):
        lines.append("- normalized_stock_supply_daily net_buy 계열 컬럼 단위를 spec과 코드에 명확히 정의하라.")
    lines.append(
        f"- 최신 ranking_base_date={ranking_bundle.get('ranking_base_date')}, price_base_date={watchlist_bundle.get('price_base_date')} 기준으로 rank/price join 품질을 재점검하라."
    )
    return "\n".join(lines)


def _build_header_lines(title: str, now_kst: datetime.datetime, readiness: dict, ranking_bundle: dict, diagnostics: dict, calendar_status: dict) -> list[str]:
    yield_curve = readiness.get("yield_curve")
    lines = [
        f"# {title}",
        f"- 작성시각: {now_kst.strftime('%Y-%m-%d %H:%M KST')}",
        "- 수치 기준: Supabase StockData 공식 테이블",
        f"- report_market_mode: {calendar_status.get('report_market_mode')}",
        f"- { _build_market_mode_banner(calendar_status) }",
        f"- 랭킹 기준일: {format_date(ranking_bundle.get('ranking_base_date'))}",
        f"- 가격 기준일: {format_date(ranking_bundle.get('price_base_date'))}",
        f"- 매크로 기준일: {format_date(readiness.get('latest_macro_date'))}",
        "- 관심종목 기준: static_stock_universe.enabled=true",
        f"- XKRX: {'open' if calendar_status.get('xkrx_is_open') else 'closed'} / {calendar_status.get('xkrx_reason') or NA_TEXT} / prev {calendar_status.get('xkrx_previous_trading_day') or NA_TEXT}",
        f"- XNYS: {'open' if calendar_status.get('xnys_is_open') else 'closed'} / {calendar_status.get('xnys_reason') or NA_TEXT} / prev {calendar_status.get('xnys_previous_trading_day') or NA_TEXT}",
        f"- market master 상태: {readiness.get('market_master_status')}",
        f"- 랭킹 데이터 상태: {readiness.get('ranking_status')}",
        f"- 가격 데이터 상태: {readiness.get('price_status')}",
        f"- watchlist price hit ratio: {readiness.get('watchlist_price_hit_ratio', 0):.1%}",
        f"- 데이터 점검: {_build_data_status(readiness)}",
    ]
    if calendar_status.get("calendar_fallback_used"):
        lines.append("- calendar_fallback_used: true")
    if yield_curve:
        lines.append(
            f"- 미국 금리: 10년물 {format_rate_level(yield_curve.get('us10y'))}, 3년물 {format_rate_level(yield_curve.get('us3y'))}, 10Y-3Y 스프레드 {format_spread_bp(yield_curve.get('spread_bp'))}"
        )
    if ranking_bundle.get("fallback_used"):
        lines.append("- 랭킹 데이터는 최근 기준일 fallback이 포함돼 있습니다.")
    if ranking_bundle.get("fallback_applied_sections"):
        lines.append(f"- 가격 fallback 사용 섹션: {', '.join(ranking_bundle.get('fallback_applied_sections', []))}")
    if diagnostics["supply_unit_needs_review"]:
        lines.append("- 수급 단위 확인 필요")
    if diagnostics["valuation_zero_needs_review"]:
        lines.append("- 밸류에이션 0값 점검 필요")
    if diagnostics["short_ratio_needs_review"]:
        lines.append("- 공매도 비중 점검 필요")
    return lines


def _generate_us_market_summary(macro: dict, news_text: str, analyzer: GeminiAnalyzer | None, gemini_tracker: dict) -> list[str]:
    if analyzer:
        try:
            gemini_tracker["count"] += 1
            gemini_tracker["purposes"].append("us_market_summary")
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
            logger.warning("Gemini US summary fallback to rule-based: %s", exc)

    docs_matches = _extract_docs_matches(news_text, ["S&P500", "NASDAQ", "미국", "증시"], max_matches=1)
    return [
        f"- S&P500 {format_index(macro.get('sp500'))} ({format_percent(macro.get('sp500_change_rate'))}), NASDAQ {format_index(macro.get('nasdaq'))} ({format_percent(macro.get('nasdaq_change_rate'))}) 기준 미국 증시 방향을 우선 확인합니다.",
        f"- 미국 10년물 {format_rate_percent(macro.get('us10y'))}, DXY {format_plain_number(macro.get('dxy'))}, VIX {format_plain_number(macro.get('vix'))}를 보면 금리·달러 부담 수준을 함께 볼 필요가 있습니다.",
        f"- {docs_matches[0]}" if docs_matches else "- 해외 뉴스는 Google Docs 컨텍스트 기준 문장만 반영했습니다.",
    ]


def _build_market_impact_lists(macro: dict) -> tuple[list[str], list[str], list[str]]:
    positives, burdens, watchpoints = [], [], []
    if (_safe_float(macro.get("nasdaq_change_rate")) or 0) > 0:
        positives.append(f"NASDAQ가 {format_percent(macro.get('nasdaq_change_rate'))}로 마감해 성장주 심리에는 우호적입니다.")
    if (_safe_float(macro.get("sp500_change_rate")) or 0) > 0:
        positives.append(f"S&P500이 {format_percent(macro.get('sp500_change_rate'))}로 마감해 미국 대형주 전반의 투자심리가 급격히 꺾이지는 않았습니다.")
    if (_safe_float(macro.get("usdkrw")) or 0) >= 1450:
        burdens.append(f"원/달러 환율이 {format_usdkrw(macro.get('usdkrw'))} 수준이라 외국인 수급에는 부담입니다.")
    if (_safe_float(macro.get("us10y")) or 0) >= 4.3:
        burdens.append(f"미국 10년물이 {format_rate_percent(macro.get('us10y'))}로 높아 밸류에이션 부담이 이어질 수 있습니다.")
    watchpoints.append(f"KOSPI {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))}), KOSDAQ {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))}) 흐름을 함께 봐야 합니다.")
    watchpoints.append(f"KOSPI 외국인 {_format_market_flow(macro.get('kospi_foreign_net_buy'))}, 기관 {_format_market_flow(macro.get('kospi_institutional_net_buy'))} 수급을 확인할 필요가 있습니다.")
    if not positives:
        positives.append("미국 지수의 위험 선호 신호는 남아 있으나 강도는 제한적입니다.")
    if not burdens:
        burdens.append("거시 변수의 급격한 위험회피 신호는 제한적이지만 과도한 추격은 경계가 필요합니다.")
    return positives[:3], burdens[:3], watchpoints[:3]


def _infer_market_judgment(macro: dict, breadth: dict) -> str:
    score = 0
    for field in ("kospi_change_rate", "kosdaq_change_rate", "sp500_change_rate", "nasdaq_change_rate"):
        value = _safe_float(macro.get(field))
        if value is not None:
            score += 1 if value > 0 else -1 if value < 0 else 0
    advances = _safe_float(breadth.get("advances"))
    declines = _safe_float(breadth.get("declines"))
    if advances is not None and declines is not None:
        score += 1 if advances > declines else -1 if advances < declines else 0
    if score >= 2:
        return "우호"
    if score <= -2:
        return "부담"
    return "중립"


def _format_ranked_market_section(
    title: str,
    rows: list[dict],
    ranking_bundle: dict,
    naver_service: NaverNewsService,
    analyzer: GeminiAnalyzer | None,
    news_text: str,
    event_map: dict,
    gemini_tracker: dict,
) -> list[str]:
    lines = [f"[{title}]"]
    if not rows:
        lines.append("- 생성 불가: 해당 랭킹 데이터가 없어 fallback까지 확인했지만 후보를 만들지 못했습니다.")
        return lines

    seen_reasons = set()
    for index, stock in enumerate(rows[:5], 1):
        reason = _generate_ranked_reason(stock, naver_service, analyzer, news_text, event_map, gemini_tracker)
        if reason in seen_reasons:
            reason = _build_fallback_reason(stock, [], event_map.get(canonicalize_symbol(stock.get("symbol"))) if event_map else None)
        seen_reasons.add(reason)
        rank_label = f"{stock.get('rank')}" if stock.get("rank") is not None else index
        lines.append(
            f"{rank_label}) {stock.get('name')}({stock.get('display_symbol') or stock.get('symbol')}) | 종가 {format_price(stock.get('close_price'))} | 거래량 {format_volume(stock.get('volume'))} | 거래대금 {format_trading_value(stock.get('trading_value'))} | source {stock.get('source') or NA_TEXT}"
        )
        lines.append(f"   - 주목 사유: {reason}")
    return lines


def _format_volume_sections(
    ranking_bundle: dict,
    naver_service: NaverNewsService,
    analyzer: GeminiAnalyzer | None,
    news_text: str,
    event_map: dict,
    gemini_tracker: dict,
) -> list[str]:
    sections = ranking_bundle.get("sections", {})
    volume_map = sections.get("volume", {})
    lines = [
        "## 거래량 상위 종목",
        f"_랭킹 기준일: {format_date(ranking_bundle.get('ranking_base_date'))}_",
    ]
    for market in ("KOSPI", "KOSDAQ", "ETF", "ETN"):
        lines.extend(
            _format_ranked_market_section(
                f"{market} 거래량 Top 5",
                volume_map.get(market) or [],
                ranking_bundle,
                naver_service,
                analyzer,
                news_text,
                event_map,
                gemini_tracker,
            )
        )
        lines.append("")
    return lines


def _format_trading_value_sections(
    ranking_bundle: dict,
    naver_service: NaverNewsService,
    analyzer: GeminiAnalyzer | None,
    news_text: str,
    event_map: dict,
    gemini_tracker: dict,
) -> list[str]:
    sections = ranking_bundle.get("sections", {})
    trading_map = sections.get("trading_value", {})
    lines = [
        "## 거래대금 상위 종목",
        f"_랭킹 기준일: {format_date(ranking_bundle.get('ranking_base_date'))}_",
    ]
    for market in ("KOSPI", "KOSDAQ"):
        lines.extend(
            _format_ranked_market_section(
                f"{market} 거래대금 Top 5",
                trading_map.get(market) or [],
                ranking_bundle,
                naver_service,
                analyzer,
                news_text,
                event_map,
                gemini_tracker,
            )
        )
        lines.append("")
    return lines


def _format_watchlist_section(prepared_snapshots: list[dict], title: str = "## Static 관심종목 요약") -> list[str]:
    lines = [title, f"- Static 관심종목: {len(prepared_snapshots):,}개", "_기준 universe: `static_stock_universe.enabled = true`_"]
    for index, snapshot in enumerate(prepared_snapshots, 1):
        price = snapshot.get("price") or {}
        supply = snapshot.get("supply") or {}
        fundamentals = snapshot.get("fundamentals") or {}
        features = snapshot.get("features") or {}
        event = snapshot.get("event") or {}
        supply_diag = snapshot.get("supply_diag") or {}
        fundamentals_diag = snapshot.get("fundamentals_diag") or {}
        short_diag = snapshot.get("short_diag") or {}
        signal_score = snapshot.get("signal_score") or {}
        return_5d = _safe_float(features.get("return_5d"))
        return_5d_text = format_percent(return_5d * 100) if return_5d is not None else NA_TEXT
        lines.extend(
            [
                f"{index}) {snapshot.get('name')}({snapshot.get('symbol')})",
                f"- 시장: {snapshot.get('market', NA_TEXT)} / 가격 기준일: {format_date(price.get('base_date'))}",
                f"- 종가: {format_price(price.get('close_price'))} / 거래량: {format_volume(price.get('volume'))} / 거래대금: {format_trading_value(price.get('trading_value'))} / 시가총액: {format_market_cap(price.get('market_cap'))}",
                f"- 상장주식수: {format_outstanding_shares(price.get('outstanding_shares'))}",
                f"- 수급: 개인 {_format_stock_supply(supply.get('individual_net_buy'), supply_diag)}, 외국인 {_format_stock_supply(supply.get('foreign_net_buy'), supply_diag)}, 기관 {_format_stock_supply(supply.get('institutional_net_buy'), supply_diag)} (stock_supply_date: {format_date(supply.get('base_date'))})",
                f"- 외국인 보유율: {format_ratio_metric(supply.get('foreign_holding_ratio'))} (foreign_holding_date: {format_date(supply.get('base_date'))})",
                f"- 밸류에이션: PER {fundamentals_diag['display']['per']}, PBR {fundamentals_diag['display']['pbr']}, ROE {fundamentals_diag['display']['roe']}, 부채비율 {fundamentals_diag['display']['debt_ratio']} (기준일: {format_date(fundamentals.get('base_date'))})",
                f"- 퀀트: 5일 수익률 {return_5d_text}, MA5 {format_index(features.get('moving_avg_5'))}, MA20 {format_index(features.get('moving_avg_20'))}, 20일 변동성 {format_ratio_metric(features.get('volatility_20d'))}, 외국인 수급 z-score {format_signed_multiple(features.get('foreign_flow_zscore'), '')} (feature date: {format_date(features.get('base_date'))})",
                f"- Signal Score: {signal_score.get('total_signal_score', NA_TEXT)} / confidence {signal_score.get('confidence', NA_TEXT)} / 라벨 {signal_score.get('label', NA_TEXT)}",
                f"- 신호 구성: 가격 {signal_score.get('price_momentum_score', NA_TEXT)}, 거래 {signal_score.get('volume_trading_score', NA_TEXT)}, 수급 {signal_score.get('supply_score', NA_TEXT)}, 밸류 {signal_score.get('valuation_score', NA_TEXT)}, 리스크 {signal_score.get('risk_score', NA_TEXT)}, 매크로 적합도 {signal_score.get('macro_fit_score', NA_TEXT)}",
                f"- 공매도: {short_diag['summary']} (기준일: {format_date((snapshot.get('short_selling') or {}).get('base_date'))})",
                f"- 공시 이벤트: {event.get('event_type') or NA_TEXT} / {event.get('event_score') if event.get('event_score') is not None else NA_TEXT} / {event.get('sentiment_score') if event.get('sentiment_score') is not None else NA_TEXT}",
                f"- 해석: {snapshot.get('quant_comment')}",
                f"- 전략 메모: {signal_score.get('strategy_memo', NA_TEXT)}",
            ]
        )
        if short_diag.get("note"):
            lines.append(f"- 공매도 메모: {short_diag.get('note')}")
        lines.append("")
    if prepared_snapshots:
        lines.append("- 일부 수급/밸류/피처 데이터는 가격 기준일과 다를 수 있음")
    return lines


def _build_data_status_summary(readiness: dict, ranking_bundle: dict, diagnostics: dict) -> list[str]:
    ranking_diag = ranking_bundle.get("diagnostics") or {}
    lines = [
        "## 데이터 상태 요약",
        f"- 랭킹 기준일: {format_date(ranking_bundle.get('ranking_base_date'))}",
        f"- 가격 기준일: {format_date(ranking_bundle.get('price_base_date'))}",
        f"- 랭킹 상태: {readiness.get('ranking_status')}",
        f"- 가격 상태: {readiness.get('price_status')}",
        f"- watchlist price hit ratio: {readiness.get('watchlist_price_hit_ratio', 0):.1%}",
        f"- market mismatch rows: {len(ranking_diag.get('market_mismatch_rows', []))}",
        f"- q-prefix rows: {len(ranking_diag.get('q_prefix_rows', []))}",
        f"- signal model: {SIGNAL_MODEL_VERSION}",
    ]
    if readiness.get("yield_curve"):
        lines.append(f"- yield curve regime: {readiness['yield_curve'].get('regime')}")
    if ranking_bundle.get("fallback_applied_sections"):
        lines.append(f"- fallback 적용 섹션: {', '.join(ranking_bundle.get('fallback_applied_sections', []))}")
    if diagnostics["watchlist_missing_prices"]:
        lines.append(f"- 가격 누락 관심종목: {len(diagnostics['watchlist_missing_prices'])}개")
    return lines


def _collect_ranking_event_map(reader: SupabaseReader, ranking_bundle: dict) -> dict:
    symbols = []
    for rank_type_map in (ranking_bundle.get("sections") or {}).values():
        for rows in rank_type_map.values():
            for row in rows:
                symbols.append(canonicalize_symbol(row.get("symbol")))
    return reader.fetch_latest_stock_events(list(dict.fromkeys(symbols))) if symbols else {}


def _build_morning_report(bundle: dict, now_kst: datetime.datetime, analyzer: GeminiAnalyzer | None, news_text: str, prepared_snapshots: list[dict], diagnostics: dict, event_map: dict, gemini_tracker: dict) -> str:
    macro = bundle["macro"]
    breadth = bundle["breadth"]
    readiness = bundle["readiness"]
    ranking_bundle = bundle["ranking_bundle"]
    calendar_status = bundle["calendar_status"]
    mode = calendar_status.get("report_market_mode")
    yield_curve = readiness.get("yield_curve")
    positives, burdens, watchpoints = _build_market_impact_lists(macro)
    naver_service = NaverNewsService()
    if analyzer is None and mode in {"KOREA_ONLY", "US_ONLY"}:
        naver_service.enabled = False
    lines = _build_header_lines("Morning Market Brief", now_kst, readiness, ranking_bundle, diagnostics, calendar_status)
    if _should_include_us_sections(calendar_status):
        lines.extend(
            [
                "",
                "## 미국시장/매크로",
                f"- {label_for_column('sp500')}: {format_index(macro.get('sp500'))} ({format_percent(macro.get('sp500_change_rate'))})",
                f"- {label_for_column('nasdaq')}: {format_index(macro.get('nasdaq'))} ({format_percent(macro.get('nasdaq_change_rate'))})",
                f"- SOX: {format_index(macro.get('sox'))}",
                f"- VIX: {format_plain_number(macro.get('vix'))}",
                f"- 미국 10년물: {format_rate_percent(macro.get('us10y'))}",
                f"- DXY: {format_plain_number(macro.get('dxy'))}",
                f"- WTI / Brent / Gold / Copper: {format_plain_number(macro.get('wti'))} / {format_plain_number(macro.get('brent'))} / {format_plain_number(macro.get('gold'))} / {format_plain_number(macro.get('copper'))}",
            ]
        )
        if mode == "KOREA_ONLY":
            lines.append(f"- 미국장은 휴장으로 신규 미국 지수 데이터는 없습니다. 직전 미국 거래일 {calendar_status.get('xnys_previous_trading_day') or NA_TEXT} 기준 참고용입니다.")
        else:
            lines.extend(_generate_us_market_summary(macro, news_text, analyzer, gemini_tracker))
    if yield_curve:
        lines.extend(
            [
                f"- 미국 금리: 10년물 {format_rate_level(yield_curve.get('us10y'))}, 3년물 {format_rate_level(yield_curve.get('us3y'))}, 10Y-3Y 스프레드 {format_spread_bp(yield_curve.get('spread_bp'))}",
                f"- 금리차 해석: {yield_curve.get('plain_korean_summary')} {yield_curve.get('equity_implication')}",
            ]
        )
    if _should_include_kr_sections(calendar_status):
        lines.extend(["", "## 한국시장 예상 영향", "- 긍정 요인:"])
        lines.extend([f"  - {item}" for item in positives])
        lines.append("- 부담 요인:")
        lines.extend([f"  - {item}" for item in burdens])
        lines.append("- 오늘 관전 포인트:")
        lines.extend([f"  - {item}" for item in watchpoints])
        lines.append("")
        lines.extend(_format_watchlist_section(prepared_snapshots))
        lines.append("")
        lines.extend(_format_volume_sections(ranking_bundle, naver_service, analyzer, news_text, event_map, gemini_tracker))
    else:
        lines.extend(["", "## 한국시장", "- 한국장은 휴장으로 국내 종목·랭킹 섹션은 생략합니다."])
    lines.append("")
    lines.extend(_build_data_status_summary(readiness, ranking_bundle, diagnostics))
    return _clean_report_text("\n".join(lines))


def _build_regular_report(bundle: dict, now_kst: datetime.datetime, analyzer: GeminiAnalyzer | None, news_text: str, prepared_snapshots: list[dict], diagnostics: dict, event_map: dict, gemini_tracker: dict) -> str:
    macro = bundle["macro"]
    breadth = bundle["breadth"]
    derivatives = bundle["derivatives"]
    readiness = bundle["readiness"]
    ranking_bundle = bundle["ranking_bundle"]
    calendar_status = bundle["calendar_status"]
    mode = calendar_status.get("report_market_mode")
    yield_curve = readiness.get("yield_curve")
    naver_service = NaverNewsService()
    if analyzer is None and mode in {"KOREA_ONLY", "US_ONLY"}:
        naver_service.enabled = False
    judgment = _infer_market_judgment(macro, breadth)
    lines = _build_header_lines(f"Intraday Market Brief | {_get_regular_slot_label(now_kst)}", now_kst, readiness, ranking_bundle, diagnostics, calendar_status)
    lines.extend(
        [
            "",
            "## 장중 지수/매크로 변화",
            f"- 원/달러: {format_usdkrw(macro.get('usdkrw'))}",
            f"- DXY: {format_plain_number(macro.get('dxy'))}",
            f"- 미국 10년물 / 한국 10년물: {format_rate_percent(macro.get('us10y'))} / {format_rate_percent(macro.get('kr10y'))}",
            f"- 미국 3년물 / 10Y-3Y 스프레드: {format_rate_level(macro.get('us3y'))} / {format_spread_bp((yield_curve or {}).get('spread_bp')) if yield_curve else NA_TEXT}",
            f"- KOSPI: {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))})",
            f"- KOSDAQ: {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))})",
            f"- {label_for_column('kospi200_futures')}: {format_index(derivatives.get('kospi200_futures'))}",
            f"- 시장 폭: 상승 {breadth.get('advances', NA_TEXT)}개 / 하락 {breadth.get('declines', NA_TEXT)}개 / 보합 {breadth.get('unchanged', NA_TEXT)}개",
            f"- 시장 판단: {judgment}",
            "",
        ]
    )
    if yield_curve:
        lines.extend([f"- 금리차 해석: {yield_curve.get('plain_korean_summary')} {yield_curve.get('watchpoint')}", ""])
    if mode == "KOREA_ONLY":
        lines.extend(["- 미국장은 휴장으로 신규 미국 지수 해석은 생략합니다.", ""])
    if _should_include_kr_sections(calendar_status):
        lines.extend(_format_volume_sections(ranking_bundle, naver_service, analyzer, news_text, event_map, gemini_tracker))
        lines.append("")
        lines.extend(_format_trading_value_sections(ranking_bundle, naver_service, analyzer, news_text, event_map, gemini_tracker))
        lines.append("")
        lines.extend(_format_watchlist_section(prepared_snapshots, title="## 관심종목 변동"))
    else:
        lines.extend(["## 한국시장", "- 한국장은 휴장으로 국내 종목·랭킹·Signal Score 섹션을 생략합니다."])
    lines.append("")
    lines.extend(_build_data_status_summary(readiness, ranking_bundle, diagnostics))
    return _clean_report_text("\n".join(lines))


def _build_closing_report(bundle: dict, now_kst: datetime.datetime, analyzer: GeminiAnalyzer | None, news_text: str, prepared_snapshots: list[dict], diagnostics: dict, event_map: dict, gemini_tracker: dict) -> str:
    macro = bundle["macro"]
    breadth = bundle["breadth"]
    derivatives = bundle["derivatives"]
    readiness = bundle["readiness"]
    ranking_bundle = bundle["ranking_bundle"]
    calendar_status = bundle["calendar_status"]
    mode = calendar_status.get("report_market_mode")
    yield_curve = readiness.get("yield_curve")
    naver_service = NaverNewsService()
    if analyzer is None and mode in {"KOREA_ONLY", "US_ONLY"}:
        naver_service.enabled = False
    judgment = _infer_market_judgment(macro, breadth)
    lines = _build_header_lines("Closing Market Brief", now_kst, readiness, ranking_bundle, diagnostics, calendar_status)
    lines.extend(
        [
            "",
            "## 마감 지수/매크로",
            f"- KOSPI 마감: {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))})",
            f"- KOSDAQ 마감: {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))})",
            f"- 시장 수급: 개인 {_format_market_flow(macro.get('kospi_individual_net_buy'))}, 외국인 {_format_market_flow(macro.get('kospi_foreign_net_buy'))}, 기관 {_format_market_flow(macro.get('kospi_institutional_net_buy'))}",
            f"- KOSDAQ 수급: 개인 {_format_market_flow(macro.get('kosdaq_individual_net_buy'))}, 외국인 {_format_market_flow(macro.get('kosdaq_foreign_net_buy'))}, 기관 {_format_market_flow(macro.get('kosdaq_institutional_net_buy'))}",
            f"- 원/달러: {format_usdkrw(macro.get('usdkrw'))}",
            f"- 미국 10년물 / 한국 10년물: {format_rate_percent(macro.get('us10y'))} / {format_rate_percent(macro.get('kr10y'))}",
            f"- 미국 3년물 / 10Y-3Y 스프레드: {format_rate_level(macro.get('us3y'))} / {format_spread_bp((yield_curve or {}).get('spread_bp')) if yield_curve else NA_TEXT}",
            f"- DXY / VIX / SOX: {format_plain_number(macro.get('dxy'))} / {format_plain_number(macro.get('vix'))} / {format_index(macro.get('sox'))}",
            f"- {label_for_column('kospi200_futures')}: {format_index(derivatives.get('kospi200_futures'))}",
            f"- 마감 판단: {judgment}",
            "",
        ]
    )
    if yield_curve:
        lines.extend([f"- 금리차 해석: {yield_curve.get('plain_korean_summary')} {yield_curve.get('market_implication')}", ""])
    if mode == "KOREA_ONLY":
        lines.extend(["- 미국장은 휴장으로 신규 미국 증시 해석은 생략하고, 직전 미국 거래일 수치만 참고합니다.", ""])
    if _should_include_kr_sections(calendar_status):
        lines.extend(_format_volume_sections(ranking_bundle, naver_service, analyzer, news_text, event_map, gemini_tracker))
        lines.append("")
        lines.extend(_format_trading_value_sections(ranking_bundle, naver_service, analyzer, news_text, event_map, gemini_tracker))
        lines.append("")
        lines.extend(_format_watchlist_section(prepared_snapshots, title="## 관심종목 종가/수급/공매도/밸류 점검"))
        lines.extend(
            [
                "",
                "## 다음 거래일 체크포인트",
                "- 환율과 미국 증시 마감 방향이 국내 위험선호를 유지시키는지 확인",
                "- 거래량/거래대금 상위 종목이 단기 순환매인지, 특정 테마 확산인지 구분",
                "- 관심종목의 가격 흐름과 외국인 수급이 동행하는지 점검",
                "",
            ]
        )
    else:
        lines.extend(["## 한국시장", "- 한국장은 휴장으로 국내 종목·랭킹·수급 섹션을 생략합니다.", ""])
    lines.extend(_build_data_status_summary(readiness, ranking_bundle, diagnostics))
    return _clean_report_text("\n".join(lines))


def _save_report(report_type: str, report_content: str, now_kst: datetime.datetime) -> Path:
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    file_path = reports_dir / f"daily_quant_report_{report_type}_{now_kst.strftime('%Y%m%d_%H%M')}.md"
    file_path.write_text(report_content, encoding="utf-8")
    logger.info("Report saved: %s", file_path)
    return file_path


def run_report(report_type: str, now_kst: datetime.datetime, report_date: str | None = None, send_enabled: bool = True):
    reader = SupabaseReader()
    normalized_report_date = _normalize_report_date(report_date, now_kst)
    calendar_status = reader.fetch_market_calendar_status(normalized_report_date)
    logger.info("calendar_status=%s", calendar_status)

    if _should_skip_all_markets(calendar_status):
        skip_text = _build_market_closed_skip_text(report_type, now_kst, calendar_status)
        logger.info("SKIPPED_REPORT_MARKET_CLOSED")
        logger.info("\n%s", skip_text)
        if not send_enabled:
            logger.info("DRY RUN: Telegram send skipped because all relevant markets are closed")
        else:
            logger.info("Telegram send skipped: all relevant markets closed")
        return

    analyzer = _safe_get_analyzer()
    if not send_enabled and calendar_status.get("report_market_mode") in {"KOREA_ONLY", "US_ONLY"}:
        analyzer = None
        logger.info("Dry-run partial-market mode: Gemini disabled to conserve quota")
    logger.info("Supabase official tables bundle loading...")
    bundle = _build_reader_bundle(reader, report_type, normalized_report_date, calendar_status)
    mode = calendar_status.get("report_market_mode")
    news_text = ""
    if mode != "SKIP_ALL_MARKETS_CLOSED":
        logger.info("Google Docs news loading...")
        news_text = reader.prepare_news_context(reader.fetch_news_document())

    prepared_snapshots, diagnostics = _prepare_watchlist_snapshots(bundle["watchlist_bundle"]["snapshots"])
    prepared_snapshots = _attach_signal_scores(prepared_snapshots, bundle["ranking_bundle"], bundle["macro"])
    event_map = _collect_ranking_event_map(reader, bundle["ranking_bundle"]) if _should_include_kr_sections(calendar_status) else {}
    gemini_tracker = {"count": 0, "purposes": []}

    _log_report_diagnostics(bundle, prepared_snapshots, gemini_tracker)
    logger.info("signal_model_version=%s", SIGNAL_MODEL_VERSION)
    logger.warning("StockData 전달용 수정 명령어:\n%s", _build_stockdata_fix_text(bundle, diagnostics))

    if report_type == "morning":
        report_content = _build_morning_report(bundle, now_kst, analyzer, news_text, prepared_snapshots, diagnostics, event_map, gemini_tracker)
    elif report_type == "closing":
        report_content = _build_closing_report(bundle, now_kst, analyzer, news_text, prepared_snapshots, diagnostics, event_map, gemini_tracker)
    else:
        report_content = _build_regular_report(bundle, now_kst, analyzer, news_text, prepared_snapshots, diagnostics, event_map, gemini_tracker)

    _save_report(report_type, report_content, now_kst)
    logger.info("Gemini call count and purpose=%s / %s", gemini_tracker["count"], gemini_tracker["purposes"])

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
