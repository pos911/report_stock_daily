import argparse
import datetime
import logging
import re
import sys
from collections import Counter
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
    format_signed_multiple,
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

VALID_REPORT_TYPES = ("morning", "regular", "closing")
KST = ZoneInfo("Asia/Seoul")
ETF_KEYWORDS = ("KODEX", "TIGER", "ACE", "KBSTAR", "SOL", "HANARO", "ARIRANG", "KOSEF", "TIMEFOLIO", "RISE", "ETF", "ETN")
NEWS_THEME_RULES = (
    ("실적", ("실적", "영업이익", "매출", "분기", "어닝", "컨센서스")),
    ("수주/계약", ("수주", "계약", "공급", "납품", "발주", "수출")),
    ("정책/규제", ("정책", "규제", "지원", "관세", "정부", "입법")),
    ("제품/서비스", ("출시", "신제품", "서비스", "플랫폼", "브랜드", "론칭")),
    ("투자/지분", ("투자", "지분", "인수", "합병", "매각")),
    ("주주환원", ("배당", "자사주", "소각", "주주환원")),
    ("업황/가격", ("업황", "반도체", "유가", "원유", "환율", "금리", "수요", "재고", "가격")),
    ("리스크", ("소송", "리콜", "제재", "조사", "부진", "악재", "논란")),
)
NEWS_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")
NEWS_STOPWORDS = {
    "관련", "기자", "오늘", "이번", "시장", "주가", "종목", "기업", "업계", "기준", "최신", "네이버",
    "뉴스", "증권", "투자", "전망", "이슈", "대한", "에서", "으로", "대해", "이후", "통해", "정도",
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Daily Quant Report Generator")
    parser.add_argument("--type", dest="report_type", default="regular")
    args = parser.parse_args()
    report_type = (args.report_type or "regular").strip().lower()
    if report_type in ("mornig", "morining"):
        report_type = "morning"
    if report_type not in VALID_REPORT_TYPES:
        parser.error(f"invalid choice: '{args.report_type}' (choose from {', '.join(VALID_REPORT_TYPES)})")
    args.report_type = report_type
    return args


def _get_now_kst() -> datetime.datetime:
    return datetime.datetime.now(KST)


def _get_regular_slot_label(now_kst: datetime.datetime) -> str:
    hour = now_kst.hour
    if hour == 10:
        return "오전 10:30 점검"
    if hour == 12:
        return "오후 12:30 점검"
    if hour == 14:
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


def _effective_zero(value, epsilon: float = 1e-9) -> bool:
    numeric = _safe_float(value)
    return numeric is not None and abs(numeric) <= epsilon


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


def _format_zero_sensitive_percent(value) -> str:
    numeric = _safe_float(value)
    if numeric is None or abs(numeric) <= 1e-9:
        return NA_TEXT
    return format_percent(numeric)


def _format_zero_sensitive_number(value, digits: int = 2) -> str:
    numeric = _safe_float(value)
    if numeric is None or abs(numeric) <= 1e-9:
        return NA_TEXT
    return format_plain_number(numeric, digits=digits)


def _relative_position(price_value, ma_value):
    price_num = _safe_float(price_value)
    ma_num = _safe_float(ma_value)
    if price_num is None or ma_num is None or ma_num == 0:
        return None
    return (price_num / ma_num) - 1.0


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


def _truncate_text(text: str, max_len: int = 36) -> str:
    stripped = (text or "").strip()
    if len(stripped) <= max_len:
        return stripped
    return stripped[: max_len - 1].rstrip() + "…"


def _extract_news_theme(news_items: list[dict], stock_name: str) -> tuple[str | None, str | None]:
    joined_text = " ".join(
        f"{item.get('title', '')} {item.get('description', '')}".lower()
        for item in news_items
    )
    theme_scores = []
    for label, keywords in NEWS_THEME_RULES:
        score = sum(joined_text.count(keyword.lower()) for keyword in keywords)
        if score > 0:
            theme_scores.append((score, label))
    theme = max(theme_scores)[1] if theme_scores else None

    stock_tokens = {token.lower() for token in NEWS_TOKEN_RE.findall(stock_name or "") if len(token) >= 2}
    counter = Counter()
    for item in news_items:
        tokens = NEWS_TOKEN_RE.findall(f"{item.get('title', '')} {item.get('description', '')}")
        for token in tokens:
            lowered = token.lower()
            if len(lowered) < 2 or lowered in NEWS_STOPWORDS or lowered in stock_tokens:
                continue
            counter[lowered] += 1
    keyword = counter.most_common(1)[0][0] if counter else None
    return theme, keyword


def _format_news_reason_from_titles(news_items: list[dict], stock_name: str) -> str:
    if not news_items:
        return ""
    theme, keyword = _extract_news_theme(news_items, stock_name)
    title = _truncate_text(news_items[0].get("title") or "")
    if theme and title:
        return f"네이버 최근 뉴스 최대 10건 기준 `{theme}` 테마가 반복됐고, `{title}` 흐름이 단기 관심을 자극했습니다."
    if theme:
        return f"네이버 최근 뉴스 최대 10건에서 `{theme}` 관련 이슈가 반복 노출됐습니다."
    if keyword and title:
        return f"네이버 최근 뉴스 최대 10건에서 `{keyword}` 키워드가 반복됐고, `{title}` 이슈가 주목을 받았습니다."
    if title:
        return f"네이버 최근 뉴스 최대 10건에서 `{title}` 이슈가 가장 먼저 포착됐습니다."
    return "네이버 최근 뉴스 노출이 이어지며 단기 거래 관심이 확대됐습니다."


def _build_top_volume_reason(stock: dict, naver_service: NaverNewsService, analyzer: GeminiAnalyzer | None) -> str:
    stock_name = stock.get("name") or stock.get("symbol") or ""
    upper_name = stock_name.upper()
    if any(keyword in upper_name for keyword in ETF_KEYWORDS):
        return "ETF/ETN 거래대금 상위는 방향성 신호보다 단기 트레이딩 수요와 변동성 확대 신호로 해석하는 편이 적절합니다."

    news_items = naver_service.search_news(stock_name, display=10)
    if news_items:
        theme, keyword = _extract_news_theme(news_items, stock_name)
        if analyzer and not theme and not keyword:
            try:
                summarized = analyzer.summarize_news_reason(stock_name, news_items)
                if summarized:
                    return summarized
            except Exception as exc:
                logger.warning(f"Gemini reason fallback for {stock_name}: {exc}")
        return _format_news_reason_from_titles(news_items, stock_name)

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
    short_row = snapshot.get("short_selling") or {}

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
    short_ratio = _safe_float(short_row.get("short_ratio"))
    short_value = _safe_float(short_row.get("short_value"))
    short_volume = _safe_float(short_row.get("short_volume"))

    rel_ma5 = _relative_position(close_price, ma5)
    rel_ma20 = _relative_position(close_price, ma20)

    if return_5d is not None and rel_ma5 is not None and rel_ma20 is not None:
        if return_5d > 0 and rel_ma5 > 0 and rel_ma20 > 0:
            comments.append("5일 흐름과 이동평균 위치를 함께 보면 단기 추세는 우호적인 편입니다.")
        elif return_5d < 0 and rel_ma5 < 0 and rel_ma20 < 0:
            comments.append("5일 흐름과 이동평균 위치가 함께 약해 단기 추세 복원 확인이 먼저 필요합니다.")
        else:
            comments.append("수익률과 이동평균 신호가 엇갈려 추세 해석은 혼재 구간으로 보는 편이 적절합니다.")
    elif return_5d is not None:
        comments.append(f"5일 수익률은 {return_5d * 100:.1f}%입니다.")

    if volatility is not None:
        if volatility >= 0.04:
            comments.append(f"20일 변동성 {volatility * 100:.1f}%로 높은 편이라 추격보다 분할 접근이 더 적절합니다.")
        elif volatility <= 0.02:
            comments.append(f"20일 변동성 {volatility * 100:.1f}%로 낮아 추세 추종 부담은 상대적으로 제한적입니다.")

    if foreign_z is not None:
        if foreign_z >= 1:
            comments.append(f"외국인 수급 z-score {foreign_z:.2f}는 수급 확인 신호로는 우호적입니다.")
        elif foreign_z <= -1:
            comments.append(f"외국인 수급 z-score {foreign_z:.2f}로 수급 확인은 약한 편입니다.")

    if foreign_holding is not None:
        comments.append(f"외국인 보유율은 {foreign_holding:.1f}% 수준입니다.")

    if per is not None or pbr is not None:
        ratios = []
        if per is not None:
            ratios.append(f"PER {per:.1f}배")
        if pbr is not None:
            ratios.append(f"PBR {pbr:.1f}배")
        comments.append(" / ".join(ratios) + "는 절대 저평가·고평가 단정보다 업종 맥락 안에서 참고하는 편이 적절합니다.")

    if short_ratio is not None and abs(short_ratio) > 1e-9:
        comments.append(f"공매도 비중 {short_ratio:.1f}%는 단기 수급 압력 점검 포인트입니다.")
    elif (short_value is not None and short_value > 0) or (short_volume is not None and short_volume > 0):
        comments.append("공매도 금액·수량은 있으나 비중값은 0으로 적재돼 방향성 해석에는 제한이 있습니다.")

    if not comments:
        return "핵심 지표가 제한적이라 우호·중립·부담 판단을 서두르기보다 추가 신호 확인이 우선입니다."
    return " ".join(comments[:4])


def _format_short_selling_summary(short_row: dict) -> str:
    ratio = _safe_float(short_row.get("short_ratio"))
    short_value = _safe_float(short_row.get("short_value"))
    short_volume = _safe_float(short_row.get("short_volume"))

    parts = [f"비중 {NA_TEXT if _effective_zero(ratio) and ((short_value or 0) > 0 or (short_volume or 0) > 0) else format_ratio_metric(ratio)}"]
    if short_value is not None and short_value > 0:
        parts.append(f"거래금액 {format_trading_value(short_value)}")
    if short_volume is not None and short_volume > 0:
        parts.append(f"거래량 {format_volume(short_volume)}")
    return " / ".join(parts)


def _build_readiness_text(readiness: dict) -> str:
    return "데이터 점검: 정상" if readiness.get("report_guard_pass") else "데이터 점검: 커버리지 부족 또는 파이프라인 경고 확인 필요"


def _build_header_lines(title: str, now_kst: datetime.datetime, readiness: dict) -> list[str]:
    latest_price_date = format_date(readiness.get("latest_price_date"))
    latest_macro_date = format_date(readiness.get("latest_macro_date"))
    coverage = (readiness.get("price_coverage") or {}).get("covered_symbols", 0)
    static_count = readiness.get("static_enabled_count", 0)
    report_guard_pass = bool(readiness.get("report_guard_pass"))
    recent_problem_count = len(readiness.get("recent_problem_logs") or [])
    latest_full_price_processed = readiness.get("latest_full_price_records_processed", 0)
    lines = [
        f"# {title}",
        f"- 작성시각: {now_kst.strftime('%Y-%m-%d %H:%M KST')}",
        "- 수치 기준: Supabase StockData 최신 적재값",
        f"- 가격 기준일: {latest_price_date}",
        f"- 매크로 기준일: {latest_macro_date}",
        f"- report_guard_pass: {'true' if report_guard_pass else 'false'}",
        f"- {_build_readiness_text(readiness)}",
        f"- 전체시장 가격 커버리지: {coverage:,}종목",
        f"- Static 관심종목: {static_count:,}개",
        f"- daily_stock_full_price_pipeline 최신 처리건수: {latest_full_price_processed:,}",
        f"- 최근 3일 파이프라인 경고/실패: {recent_problem_count}건",
    ]
    if not report_guard_pass:
        lines.append("- 데이터 커버리지 경고: 전체시장 거래량 상위 섹션의 신뢰도를 반드시 함께 확인하세요.")
    return lines


def _rule_based_us_summary(macro: dict, news_text: str) -> str:
    headlines = _extract_news_headlines(news_text, limit=2)
    lines = [
        f"S&P500 {format_index(macro.get('sp500'))} ({format_percent(macro.get('sp500_change_rate'))}), NASDAQ {format_index(macro.get('nasdaq'))} ({format_percent(macro.get('nasdaq_change_rate'))}) 기준 미국 증시는 최신 적재값상 혼조 흐름입니다.",
        f"미국 10년물 {format_rate_percent(macro.get('us10y'))}, DXY {format_plain_number(macro.get('dxy'))}, VIX {format_plain_number(macro.get('vix'))}, WTI {format_plain_number(macro.get('wti'))}를 함께 보면 금리·달러·변동성 부담과 위험선호 신호가 공존합니다.",
    ]
    if headlines:
        lines.append(f"해외 뉴스에서는 {_truncate_text(headlines[0], 60)} 이슈가 먼저 포착됐습니다.")
    else:
        lines.append("해외 뉴스 요약은 제한적이어서 지수와 금리·원자재 수치 중심으로 해석했습니다.")
    return "\n".join(f"- {line}" for line in lines[:3])


def _generate_us_market_summary(macro: dict, news_text: str, analyzer: GeminiAnalyzer | None) -> str:
    if analyzer:
        try:
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
            generated = analyzer.generate_morning_us_summary(summary_input, news_text)
            if generated:
                lines = [line.strip().lstrip("- ").strip() for line in generated.splitlines() if line.strip()]
                return "\n".join(f"- {line}" for line in lines[:3])
        except Exception as exc:
            logger.warning(f"Gemini US summary fallback to rule-based: {exc}")
    return _rule_based_us_summary(macro, news_text)


def _build_market_impact_lists(macro: dict, derivatives: dict) -> tuple[list[str], list[str], list[str]]:
    positives, burdens, watchpoints = [], [], []
    nasdaq_chg = _safe_float(macro.get("nasdaq_change_rate"))
    sp500_chg = _safe_float(macro.get("sp500_change_rate"))
    kospi_chg = _safe_float(macro.get("kospi_change_rate"))
    kosdaq_chg = _safe_float(macro.get("kosdaq_change_rate"))
    usdkrw = _safe_float(macro.get("usdkrw"))
    dxy = _safe_float(macro.get("dxy"))
    us10y = _safe_float(macro.get("us10y"))
    kr10y = _safe_float(macro.get("kr10y"))
    vix = _safe_float(macro.get("vix"))
    commodity = _safe_float(macro.get("wti")) or _safe_float(macro.get("brent"))
    night_ret = _safe_float(derivatives.get("night_futures_return"))

    if nasdaq_chg is not None and nasdaq_chg > 0:
        positives.append(f"NASDAQ가 {format_percent(nasdaq_chg)}로 마감해 성장주 심리는 완전히 꺾이지 않았습니다.")
    if sp500_chg is not None and sp500_chg > 0:
        positives.append(f"S&P500이 {format_percent(sp500_chg)}로 마감해 미국 대형주 전반의 위험선호는 유지됐습니다.")
    if kospi_chg is not None and kospi_chg > 0:
        positives.append(f"전일 KOSPI가 {format_percent(kospi_chg)}로 마감해 국내 대형주 흐름은 우호적입니다.")

    if kosdaq_chg is not None and kosdaq_chg < 0:
        burdens.append(f"전일 KOSDAQ이 {format_percent(kosdaq_chg)}로 약세여서 중소형 성장주는 변동성 경계가 필요합니다.")
    if usdkrw is not None and usdkrw >= 1450:
        burdens.append(f"원/달러 환율이 {format_usdkrw(usdkrw)} 수준이라 외국인 위험자산 선호에는 부담입니다.")
    if dxy is not None and dxy >= 100:
        burdens.append(f"DXY가 {format_plain_number(dxy)}로 높은 편이라 달러 강세 부담이 남아 있습니다.")
    if us10y is not None and us10y >= 4.3:
        burdens.append(f"미국 10년물이 {format_rate_percent(us10y)}로 높아 밸류에이션 부담이 이어질 수 있습니다.")
    if commodity is not None and commodity >= 90:
        burdens.append(f"유가가 {format_plain_number(commodity)} 수준으로 높아 원가 부담 점검이 필요합니다.")

    if kr10y is not None and us10y is not None:
        watchpoints.append(f"한국 10년물 {format_rate_percent(kr10y)}와 미국 10년물 {format_rate_percent(us10y)}의 금리 격차 변화를 계속 봐야 합니다.")
    if not _effective_zero(derivatives.get("futures_basis")):
        watchpoints.append(f"선물 베이시스는 {format_plain_number(derivatives.get('futures_basis'))}입니다.")
    if night_ret is not None and abs(night_ret) > 1e-9:
        watchpoints.append(f"야간선물 수익률은 {format_percent(night_ret)}로 개장 체감 심리에 영향을 줄 수 있습니다.")
    watchpoints.append(
        f"KOSPI 외국인 {_format_market_flow_value(macro.get('kospi_foreign_net_buy'))}, 기관 {_format_market_flow_value(macro.get('kospi_institutional_net_buy'))} 흐름은 현물 방향 확인 신호입니다."
    )

    if not positives:
        positives.append("미국·국내 지수 신호가 혼재해 뚜렷한 상방 우위는 아직 제한적입니다.")
    if not burdens:
        burdens.append("금리·환율·원자재 신호가 즉시 위험회피를 강제하는 수준은 아니어서 과도한 비관은 경계가 필요합니다.")
    return positives[:3], burdens[:3], watchpoints[:3]


def _infer_market_judgment(macro: dict, breadth: dict) -> str:
    score = 0
    for field in ("kospi_change_rate", "sp500_change_rate", "nasdaq_change_rate"):
        value = _safe_float(macro.get(field))
        if value is not None:
            score += 1 if value > 0 else -1 if value < 0 else 0
    if (_safe_float(macro.get("vix")) or 0) >= 22:
        score -= 1
    if (_safe_float(macro.get("usdkrw")) or 0) >= 1450:
        score -= 1
    advances = _safe_float(breadth.get("advances"))
    declines = _safe_float(breadth.get("declines"))
    if advances is not None and declines is not None:
        score += 1 if advances > declines else -1 if advances < declines else 0

    if score >= 2:
        return "우호"
    if score <= -2:
        return "부담"
    return "중립"


def _format_top_volume_sections(top_volume: dict, readiness: dict, naver_service: NaverNewsService, analyzer: GeminiAnalyzer | None) -> list[str]:
    coverage = (top_volume.get("coverage") or {}).get("covered_symbols", 0)
    lines = [
        "## 거래량 상위 종목",
        f"_기준일: {format_date(top_volume.get('base_date'))} (`normalized_stock_prices_daily`, Supabase 최신 적재 기준)_",
        f"- 전체시장 가격 커버리지: {coverage:,}종목",
    ]
    if coverage <= 2000:
        lines.append("- 커버리지 부족으로 신뢰도 낮음: 전체시장 거래량 상위 해석은 참고용으로만 보세요.")

    for market_key in ("KOSPI", "KOSDAQ", "ETF"):
        lines.append(f"[{market_key} Top 5]")
        rows = top_volume.get(market_key) or []
        if market_key in ("KOSPI", "KOSDAQ") and coverage <= 2000:
            lines.append("- 커버리지 부족으로 스킵")
            lines.append("")
            continue
        if not rows:
            lines.append("- 데이터 없음")
            lines.append("")
            continue
        for idx, stock in enumerate(rows, 1):
            lines.append(
                f"{idx}) {stock.get('name', stock.get('symbol'))}({stock.get('symbol')}) | "
                f"종가 {format_price(stock.get('close_price'))} | 거래량 {format_volume(stock.get('volume'))} | 거래대금 {format_trading_value(stock.get('trading_value'))}"
            )
            lines.append(f"   - 주목 사유: {_build_top_volume_reason(stock, naver_service, analyzer)}")
        lines.append("")
    return lines


def _format_static_section(static_snapshots: list[dict], readiness: dict) -> list[str]:
    lines = [
        "## Static 관심종목 점검",
        f"- Static 관심종목: {len(static_snapshots):,}개",
        "_기준 universe: `static_stock_universe.enabled = true`_",
    ]
    if not static_snapshots:
        lines.append("- 데이터 없음")
        return lines

    for idx, snapshot in enumerate(static_snapshots, 1):
        price = snapshot.get("price") or {}
        supply = snapshot.get("supply") or {}
        fundamentals = snapshot.get("fundamentals") or {}
        short_row = snapshot.get("short_selling") or {}
        event = snapshot.get("event") or {}
        features = snapshot.get("features") or {}
        price_date = format_date(price.get("base_date"))
        supply_date = format_date(supply.get("base_date"))
        fundamentals_date = format_date(fundamentals.get("base_date"))
        short_date = format_date(short_row.get("base_date"))
        feature_date = format_date(features.get("base_date"))

        lines.extend(
            [
                f"{idx}) {snapshot.get('name')}({snapshot.get('symbol')})",
                f"- 시장: {snapshot.get('market', NA_TEXT)} / 가격 기준일: {price_date}",
                f"- 종가: {format_price(price.get('close_price'))} / 거래량: {format_volume(price.get('volume'))} / 거래대금: {format_trading_value(price.get('trading_value'))} / 시가총액: {format_market_cap(price.get('market_cap'))}",
                f"- 상장주식수: {format_outstanding_shares(price.get('outstanding_shares'))}",
                f"- 수급: 개인 {_format_supply_value(supply.get('individual_net_buy'))}, 외국인 {_format_supply_value(supply.get('foreign_net_buy'))}, 기관 {_format_supply_value(supply.get('institutional_net_buy'))}, 외국인 보유율 {format_ratio_metric(supply.get('foreign_holding_ratio'))} (수급 기준일: {supply_date})",
                f"- 밸류에이션: PER {format_multiple(fundamentals.get('per'), '배')}, PBR {format_multiple(fundamentals.get('pbr'), '배')}, ROE {format_ratio_metric(fundamentals.get('roe'))}, 부채비율 {format_ratio_metric(fundamentals.get('debt_ratio'))} (밸류 기준일: {fundamentals_date})",
                f"- 퀀트 수치: 5일 수익률 {format_percent((_safe_float(features.get('return_5d')) or 0) * 100) if _safe_float(features.get('return_5d')) is not None else NA_TEXT}, MA5 {format_index(features.get('moving_avg_5'))}, MA20 {format_index(features.get('moving_avg_20'))}, 변동성 {format_ratio_metric((_safe_float(features.get('volatility_20d')) or 0) * 100) if _safe_float(features.get('volatility_20d')) is not None else NA_TEXT}, 외국인 수급 z-score {format_signed_multiple(features.get('foreign_flow_zscore'), '')} (퀀트 기준일: {feature_date})",
                f"- 공매도: {_format_short_selling_summary(short_row)} (공매도 기준일: {short_date})",
                f"- 공시 이벤트: {event.get('event_type', NA_TEXT) or NA_TEXT} / {event.get('event_score', NA_TEXT) if event.get('event_score') is not None else NA_TEXT} / {event.get('sentiment_score', NA_TEXT) if event.get('sentiment_score') is not None else NA_TEXT}",
                f"- 퀀트 해석: {_build_quant_comment(snapshot)}",
                "",
            ]
        )

    lines.append("- 일부 수급/밸류 데이터는 최신 기준일이 가격 기준일과 다를 수 있음")
    return lines


def _build_morning_report(bundle: dict, now_kst: datetime.datetime, analyzer: GeminiAnalyzer | None, news_text: str) -> str:
    macro = bundle["macro"]
    breadth = bundle["breadth"]
    derivatives = bundle["derivatives"]
    readiness = bundle["readiness"]
    top_volume = bundle["top_volume"]
    static_snapshots = bundle["static_snapshots"]

    positives, burdens, watchpoints = _build_market_impact_lists(macro, derivatives)
    naver_service = NaverNewsService()
    lines = _build_header_lines("Morning Market Brief", now_kst, readiness)
    lines.extend(
        [
            "",
            "## 미국 시장 정리",
            f"_기준일: {format_date(macro.get('base_date'))} (`normalized_global_macro_daily`)_",
            f"- S&P500: {format_index(macro.get('sp500'))} ({format_percent(macro.get('sp500_change_rate'))})",
            f"- NASDAQ: {format_index(macro.get('nasdaq'))} ({format_percent(macro.get('nasdaq_change_rate'))})",
            f"- SOX: {format_index(macro.get('sox'))}",
            f"- VIX: {format_plain_number(macro.get('vix'))}",
            f"- 미국 10년물: {format_rate_percent(macro.get('us10y'))}",
            f"- DXY: {format_plain_number(macro.get('dxy'))}",
            f"- WTI: {format_plain_number(macro.get('wti'))}",
            f"- Brent: {format_plain_number(macro.get('brent'))}",
            f"- Gold: {format_plain_number(macro.get('gold'))}",
            f"- Copper: {format_plain_number(macro.get('copper'))}",
            "- 미국 시장 핵심 요약:",
            _generate_us_market_summary(macro, news_text, analyzer),
            "",
            "## 한국 시장 영향 전망",
            f"_매크로 기준일: {format_date(macro.get('base_date'))}, 파생 기준일: {format_date(derivatives.get('base_date'))}_",
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
            f"- 원/달러: {format_usdkrw(macro.get('usdkrw'))}",
            f"- 한국 10년물: {format_rate_percent(macro.get('kr10y'))}",
            f"- 미국 10년물: {format_rate_percent(macro.get('us10y'))}",
            f"- 전일 KOSPI/KOSDAQ: {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))}) / {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))})",
            f"- KOSPI 외국인/기관: {_format_market_flow_value(macro.get('kospi_foreign_net_buy'))} / {_format_market_flow_value(macro.get('kospi_institutional_net_buy'))}",
            f"- KOSDAQ 외국인/기관: {_format_market_flow_value(macro.get('kosdaq_foreign_net_buy'))} / {_format_market_flow_value(macro.get('kosdaq_institutional_net_buy'))}",
            f"- 파생 보조: KOSPI200 선물 {format_index(derivatives.get('kospi200_futures'))} / 베이시스 {_format_zero_sensitive_number(derivatives.get('futures_basis'))} / 미결제약정 {format_plain_number(derivatives.get('open_interest'), 0)} / 야간선물 수익률 {_format_zero_sensitive_percent(derivatives.get('night_futures_return'))}",
            "",
            "## 전일 한국 시장 요약",
            f"_기준일: {format_date(macro.get('base_date'))} (`normalized_global_macro_daily`), 시장 폭 기준일: {format_date(breadth.get('base_date'))} (`market_breadth_daily`)_",
            f"- KOSPI: {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))})",
            f"- KOSDAQ: {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))})",
            f"- KOSPI 수급: 개인 {_format_market_flow_value(macro.get('kospi_individual_net_buy'))}, 외국인 {_format_market_flow_value(macro.get('kospi_foreign_net_buy'))}, 기관 {_format_market_flow_value(macro.get('kospi_institutional_net_buy'))}",
            f"- KOSDAQ 수급: 개인 {_format_market_flow_value(macro.get('kosdaq_individual_net_buy'))}, 외국인 {_format_market_flow_value(macro.get('kosdaq_foreign_net_buy'))}, 기관 {_format_market_flow_value(macro.get('kosdaq_institutional_net_buy'))}",
            f"- 상승/하락/보합: 상승 {breadth.get('advances', NA_TEXT)}개 / 하락 {breadth.get('declines', NA_TEXT)}개 / 보합 {breadth.get('unchanged', NA_TEXT)}개",
            f"- 상승 거래량/하락 거래량: {format_volume(breadth.get('advancing_volume'))} / {format_volume(breadth.get('declining_volume'))}",
            "",
        ]
    )
    lines.extend(_format_top_volume_sections(top_volume, readiness, naver_service, analyzer))
    lines.append("")
    lines.extend(_format_static_section(static_snapshots, readiness))
    return "\n".join(lines).strip() + "\n"


def _build_regular_report(bundle: dict, now_kst: datetime.datetime, analyzer: GeminiAnalyzer | None, news_text: str) -> str:
    macro = bundle["macro"]
    breadth = bundle["breadth"]
    derivatives = bundle["derivatives"]
    readiness = bundle["readiness"]
    top_volume = bundle["top_volume"]
    static_snapshots = bundle["static_snapshots"]
    naver_service = NaverNewsService()
    market_judgment = _infer_market_judgment(macro, breadth)
    slot_label = _get_regular_slot_label(now_kst)

    lines = _build_header_lines(f"Intraday Market Brief | {slot_label}", now_kst, readiness)
    lines.extend(
        [
            "",
            "## 1. 장중 매크로 점검",
            f"- 원/달러: {format_usdkrw(macro.get('usdkrw'))}",
            f"- DXY: {format_plain_number(macro.get('dxy'))}",
            f"- 미국 10년물 / 한국 10년물: {format_rate_percent(macro.get('us10y'))} / {format_rate_percent(macro.get('kr10y'))}",
            f"- WTI / Brent / Gold / Copper: {format_plain_number(macro.get('wti'))} / {format_plain_number(macro.get('brent'))} / {format_plain_number(macro.get('gold'))} / {format_plain_number(macro.get('copper'))}",
            f"- 파생 참고: KOSPI200 선물 {format_index(derivatives.get('kospi200_futures'))}, 야간선물 수익률 {_format_zero_sensitive_percent(derivatives.get('night_futures_return'))}",
            "",
            "## 2. 지수 흐름 점검",
            f"- KOSPI: {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))})",
            f"- KOSDAQ: {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))})",
            f"- 미국 지수 참고: S&P500 {format_index(macro.get('sp500'))} ({format_percent(macro.get('sp500_change_rate'))}), NASDAQ {format_index(macro.get('nasdaq'))} ({format_percent(macro.get('nasdaq_change_rate'))})",
            f"- 시장 폭: 상승 {breadth.get('advances', NA_TEXT)}개 / 하락 {breadth.get('declines', NA_TEXT)}개 / 보합 {breadth.get('unchanged', NA_TEXT)}개",
            f"- 시장 수급: KOSPI 외국인 {_format_market_flow_value(macro.get('kospi_foreign_net_buy'))}, 기관 {_format_market_flow_value(macro.get('kospi_institutional_net_buy'))}",
            "",
            "## 3. 거래량 상위 종목 기반 시장 상황",
            f"- 요약 판단: 현재 시장은 `{market_judgment}` 구간으로 보되, 수치는 Supabase 최신 적재값 기준입니다.",
            "",
        ]
    )
    lines.extend(_format_top_volume_sections(top_volume, readiness, naver_service, analyzer))
    lines.append("")
    lines.extend(["## 4. Static 관심종목 점검"])
    lines.extend(_format_static_section(static_snapshots, readiness)[1:])
    lines.extend(
        [
            "",
            "## 5. 요약 판단",
            f"- 종합 판단: `{market_judgment}`",
            "- 행동 해석: 우호면 선별 접근, 중립이면 관망·분할, 부담이면 추격주의 관점이 적절합니다.",
            "- 주의: 장중 리포트라도 가격 기준일과 매크로 기준일은 Supabase 최신 적재 기준이며 실시간 시세가 아닐 수 있습니다.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _build_closing_report(bundle: dict, now_kst: datetime.datetime, analyzer: GeminiAnalyzer | None, news_text: str) -> str:
    macro = bundle["macro"]
    breadth = bundle["breadth"]
    derivatives = bundle["derivatives"]
    readiness = bundle["readiness"]
    top_volume = bundle["top_volume"]
    static_snapshots = bundle["static_snapshots"]
    naver_service = NaverNewsService()
    market_judgment = _infer_market_judgment(macro, breadth)

    lines = _build_header_lines("Closing Market Brief", now_kst, readiness)
    lines.extend(
        [
            "",
            "## 1. 마감 지수와 수급",
            f"- KOSPI 마감: {format_index(macro.get('kospi'))} ({format_percent(macro.get('kospi_change_rate'))})",
            f"- KOSDAQ 마감: {format_index(macro.get('kosdaq'))} ({format_percent(macro.get('kosdaq_change_rate'))})",
            f"- KOSPI 수급: 개인 {_format_market_flow_value(macro.get('kospi_individual_net_buy'))}, 외국인 {_format_market_flow_value(macro.get('kospi_foreign_net_buy'))}, 기관 {_format_market_flow_value(macro.get('kospi_institutional_net_buy'))}",
            f"- KOSDAQ 수급: 개인 {_format_market_flow_value(macro.get('kosdaq_individual_net_buy'))}, 외국인 {_format_market_flow_value(macro.get('kosdaq_foreign_net_buy'))}, 기관 {_format_market_flow_value(macro.get('kosdaq_institutional_net_buy'))}",
            f"- 시장 폭: 상승 {breadth.get('advances', NA_TEXT)}개 / 하락 {breadth.get('declines', NA_TEXT)}개 / 보합 {breadth.get('unchanged', NA_TEXT)}개",
            "",
            "## 2. 마감 매크로와 파생 체크",
            f"- 원/달러: {format_usdkrw(macro.get('usdkrw'))}",
            f"- 미국 10년물 / 한국 10년물: {format_rate_percent(macro.get('us10y'))} / {format_rate_percent(macro.get('kr10y'))}",
            f"- DXY / VIX / SOX: {format_plain_number(macro.get('dxy'))} / {format_plain_number(macro.get('vix'))} / {format_index(macro.get('sox'))}",
            f"- 파생 참고: KOSPI200 선물 {format_index(derivatives.get('kospi200_futures'))} / 베이시스 {_format_zero_sensitive_number(derivatives.get('futures_basis'))} / 야간선물 수익률 {_format_zero_sensitive_percent(derivatives.get('night_futures_return'))}",
            "",
            "## 3. 거래량 상위 종목 테마",
            f"- 마감 판단: `{market_judgment}`",
            "",
        ]
    )
    lines.extend(_format_top_volume_sections(top_volume, readiness, naver_service, analyzer))
    lines.append("")
    lines.extend(["## 4. Static 관심종목 종가/수급/퀀트/밸류"])
    lines.extend(_format_static_section(static_snapshots, readiness)[1:])
    lines.extend(
        [
            "",
            "## 5. 다음 거래일 체크포인트",
            "- 환율과 미국 장 마감 방향이 국내 위험선호를 유지시키는지 확인",
            "- 외국인 수급이 가격 흐름을 따라오는지, 혹은 가격과 엇갈리는지 확인",
            "- 거래량 상위 종목이 단기 순환매인지, 특정 테마 확산인지 구분해 추격 여부를 결정",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _build_report_bundle(reader: SupabaseReader) -> dict:
    return {
        "readiness": reader.fetch_report_readiness(),
        "macro": reader.fetch_latest_global_macro_snapshot(),
        "breadth": reader.fetch_latest_market_breadth(),
        "derivatives": reader.fetch_latest_derivatives_snapshot(),
        "static_universe": reader.fetch_static_stock_universe(),
        "static_snapshots": reader.fetch_static_universe_stock_snapshot(),
        "top_volume": reader.fetch_top_volume_stocks_by_market(limit=5),
    }


def _save_and_send_report(report_type: str, report_content: str, now_kst: datetime.datetime):
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    file_name = f"daily_quant_report_{report_type}_{now_kst.strftime('%Y%m%d_%H%M')}.md"
    file_path = reports_dir / file_name
    file_path.write_text(report_content, encoding="utf-8")
    logger.info(f"Report saved: {file_path}")

    try:
        sender = TelegramSender()
        sent = sender.send_report(report_content)
        if sent:
            logger.info("Telegram send succeeded.")
        else:
            logger.warning("Telegram send request finished but delivery failed.")
    except Exception as exc:
        logger.warning(f"Telegram send failed (non-fatal): {exc}")


def run_report(report_type: str, now_kst: datetime.datetime):
    reader = SupabaseReader()
    analyzer = _safe_get_analyzer()
    logger.info("Supabase official tables bundle loading...")
    bundle = _build_report_bundle(reader)
    logger.info("Google Docs news loading...")
    news_text = reader.prepare_news_context(reader.fetch_news_document())

    if report_type == "morning":
        report_content = _build_morning_report(bundle, now_kst, analyzer, news_text)
    elif report_type == "closing":
        report_content = _build_closing_report(bundle, now_kst, analyzer, news_text)
    else:
        report_content = _build_regular_report(bundle, now_kst, analyzer, news_text)

    _save_and_send_report(report_type, report_content, now_kst)


def main():
    args = _parse_args()
    now_kst = _get_now_kst()
    logger.info(f"=== Daily Report Pipeline start [type={args.report_type}] ===")
    run_report(args.report_type, now_kst)


if __name__ == "__main__":
    main()
