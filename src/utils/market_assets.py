from typing import Iterable


ETF_ETN_PATTERNS = (
    "ETF",
    "ETN",
    "KODEX",
    "TIGER",
    "ACE",
    "KBSTAR",
    "SOL",
    "HANARO",
    "ARIRANG",
    "KOSEF",
    "TIMEFOLIO",
    "RISE",
    "PLUS",
    "인버스",
    "레버리지",
    "선물",
    "액티브",
)
NON_COMMON_STOCK_PATTERNS = (
    "스팩",
    "SPAC",
    "리츠",
    "REIT",
    "우선주",
)
THEME_KEYWORDS = {
    "WTI원유": ["WTI", "원유", "유가"],
    "원유": ["원유", "유가", "WTI"],
    "2차전지": ["2차전지", "배터리"],
    "반도체": ["반도체", "SOX", "AI", "HBM"],
    "채권": ["채권", "금리"],
    "인버스": ["하락", "헤지", "인버스"],
    "레버리지": ["상승", "레버리지"],
}
PREFERRED_PATTERNS = ("우선주",)


def canonicalize_symbol(symbol: str | None) -> str:
    raw = (symbol or "").strip().upper()
    if not raw:
        return ""
    if raw.startswith("Q"):
        raw = raw[1:]
    if raw.isdigit():
        return raw.zfill(6)
    return raw


def display_symbol(symbol: str | None) -> str:
    canonical = canonicalize_symbol(symbol)
    return canonical or (symbol or "")


def is_q_prefixed(symbol: str | None) -> bool:
    return (symbol or "").strip().upper().startswith("Q")


def normalize_market_label(market: str | None) -> str | None:
    if not market:
        return None
    upper = market.strip().upper()
    mapping = {
        "J": "KOSPI",
        "Q": "KOSDAQ",
        "T": "ETF",
        "KOSPI": "KOSPI",
        "KOSDAQ": "KOSDAQ",
        "ETF": "ETF",
        "ETN": "ETN",
        "KONEX": "KONEX",
    }
    return mapping.get(upper, upper)


def infer_asset_type(name: str | None, market: str | None, symbol: str | None) -> str:
    upper_name = (name or "").upper()
    normalized_market = normalize_market_label(market) or ""
    symbol_text = canonicalize_symbol(symbol)

    if "ETN" in upper_name or normalized_market == "ETN":
        return "ETN"
    if normalized_market == "ETF" or any(pattern in upper_name for pattern in ETF_ETN_PATTERNS):
        return "ETF"
    if "ELW" in upper_name:
        return "ELW"
    if any(pattern in upper_name for pattern in PREFERRED_PATTERNS) or upper_name.endswith("우"):
        return "PREFERRED_STOCK"
    if any(pattern in upper_name for pattern in NON_COMMON_STOCK_PATTERNS):
        if "SPAC" in upper_name or "스팩" in upper_name:
            return "SPAC"
        return "REIT"
    if normalized_market in {"KOSPI", "KOSDAQ"} and symbol_text:
        return "COMMON_STOCK"
    return "UNKNOWN"


def is_common_stock_top_eligible(row: dict) -> bool:
    return normalize_market_label(row.get("market")) in {"KOSPI", "KOSDAQ"} and row.get("asset_type") == "COMMON_STOCK"


def is_etf_etn_top_eligible(row: dict) -> bool:
    return row.get("asset_type") in {"ETF", "ETN"}


def has_minimum_top_data(row: dict) -> bool:
    close_price = row.get("close_price")
    volume = row.get("volume")
    trading_value = row.get("trading_value")
    try:
        return (
            close_price is not None
            and volume is not None
            and trading_value is not None
            and float(close_price) > 0
            and float(volume) > 0
            and float(trading_value) > 0
        )
    except (TypeError, ValueError):
        return False


def pick_preferred_duplicate(rows: Iterable[dict]) -> dict | None:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("trading_value") or 0),
            float(row.get("volume") or 0),
            1 if row.get("close_price") is not None else 0,
            1 if not is_q_prefixed(row.get("symbol")) else 0,
        ),
        reverse=True,
    )
    return sorted_rows[0] if sorted_rows else None


def deduplicate_by_canonical_symbol(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    duplicates: list[dict] = []
    for row in rows:
        canonical = canonicalize_symbol(row.get("symbol"))
        normalized = {
            **row,
            "canonical_symbol": canonical,
            "display_symbol": canonical or row.get("symbol"),
            "join_symbol": row.get("symbol"),
            "market": normalize_market_label(row.get("market")),
        }
        grouped.setdefault(canonical, []).append(normalized)

    deduped: list[dict] = []
    for canonical, group in grouped.items():
        picked = pick_preferred_duplicate(group)
        if not picked:
            continue
        deduped.append(picked)
        if len(group) > 1:
            duplicates.append(
                {
                    "canonical_symbol": canonical,
                    "symbols": [item.get("symbol") for item in group],
                    "picked_symbol": picked.get("symbol"),
                    "name": picked.get("name"),
                    "base_date": picked.get("base_date"),
                }
            )
    return deduped, duplicates


def extract_theme_keywords(name: str | None) -> list[str]:
    source = name or ""
    themes: list[str] = []
    for label, keywords in THEME_KEYWORDS.items():
        if label in source:
            themes.extend(keywords)
    if "ETN" in source and "ETN" not in themes:
        themes.append("ETN")
    if "ETF" in source and "ETF" not in themes:
        themes.append("ETF")
    return list(dict.fromkeys(themes))


def label_for_column(column_name: str) -> str:
    mapping = {
        "kospi200_futures": "KOSPI200 선물",
        "sp500": "S&P500",
        "nasdaq": "NASDAQ",
        "kospi": "KOSPI",
        "kosdaq": "KOSDAQ",
    }
    return mapping.get(column_name, column_name)

