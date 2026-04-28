import math


NA_TEXT = "N/A"


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def format_date(value) -> str:
    return NA_TEXT if is_missing(value) else str(value)


def format_index(value) -> str:
    return NA_TEXT if is_missing(value) else f"{float(value):,.2f}"


def format_percent(value) -> str:
    if is_missing(value):
        return NA_TEXT
    numeric = float(value)
    return f"{numeric:+.2f}%"


def format_rate_percent(value) -> str:
    return NA_TEXT if is_missing(value) else f"{float(value):.2f}%"


def format_usdkrw(value) -> str:
    return NA_TEXT if is_missing(value) else f"1달러 = {float(value):,.2f}원"


def format_plain_number(value, digits: int = 2) -> str:
    return NA_TEXT if is_missing(value) else f"{float(value):,.{digits}f}"


def format_volume(value) -> str:
    return NA_TEXT if is_missing(value) else f"{int(round(float(value))):,}주"


def format_price(value) -> str:
    return NA_TEXT if is_missing(value) else f"{int(round(float(value))):,}원"


def format_trading_value(value) -> str:
    if is_missing(value):
        return NA_TEXT
    eok = float(value) / 100_000_000
    return f"{int(round(eok)):,}억원"


def format_market_cap(value) -> str:
    if is_missing(value):
        return NA_TEXT
    numeric = float(value)
    if numeric >= 1_000_000_000_000:
        return f"{numeric / 1_000_000_000_000:.1f}조원"
    return f"{numeric / 100_000_000:,.0f}억원"


def format_outstanding_shares(value) -> str:
    return NA_TEXT if is_missing(value) else f"{int(round(float(value))):,}주"


def format_flow_amount(value) -> str:
    if is_missing(value):
        return NA_TEXT
    eok = float(value) / 100_000_000
    label = "순매수" if eok >= 0 else "순매도"
    return f"{label} {eok:+,.0f}억원"


def format_flow_generic(value) -> str:
    if is_missing(value):
        return NA_TEXT
    numeric = float(value)
    label = "순매수" if numeric >= 0 else "순매도"
    return f"{label} {numeric:+,.0f}"


def format_multiple(value, suffix: str) -> str:
    return NA_TEXT if is_missing(value) else f"{float(value):.1f}{suffix}"


def format_signed_multiple(value, suffix: str) -> str:
    return NA_TEXT if is_missing(value) else f"{float(value):+.2f}{suffix}"


def format_ratio_metric(value, suffix: str = "%") -> str:
    return NA_TEXT if is_missing(value) else f"{float(value):.1f}{suffix}"

