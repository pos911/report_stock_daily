# Report Data Usage Guide

## Purpose

`report_stock_daily` consumes StockData with a readiness-first policy.

- `kr_full_market_price_ready=true`인 경우에만 국내 전체시장 거래대금 Top, 시총 Top을 노출한다.
- `kr_full_market_price_ready=false`이면 국내 전체시장 Top 섹션은 생성하지 않는다.
- `kis_volume_ranking_ready=true`이면 `normalized_market_rankings_daily source='KIS' rank_type='volume'` 기준의 거래량 상위를 참고 섹션으로 노출한다.
- `kis_universe_ready=true`이면 관심종목·랭킹 후보 기반 `watchlist_signal`을 노출한다.

## Naming Rules

- KIS universe, KIS_DETAIL, KIS ranking 데이터는 국내 전체시장 통계처럼 표현하지 않는다.
- 허용 표현:
  - `KIS 거래량 순위 기준`
  - `관심종목·랭킹 후보 Signal`
  - `관심종목·KIS 후보군 기준`
- 금지 표현:
  - `국내 전체시장 거래대금 상위`
  - `국내 전체시장 시가총액 상위`
  - `오늘 한국시장 전체 Top`

## Section Gating

- always allowed: `macro`, `us_market`
- conditional:
  - `kis_volume_top`
  - `watchlist_signal`
  - `etf_etn`
  - `kr_full_market_trading_value_top`
  - `kr_full_market_market_cap_top`

## LLM Guardrails

- Gemini는 blocked section을 임의 생성하지 않는다.
- KIS universe 기반 데이터를 전체시장으로 표현하지 않는다.
- BUY / HOLD / SELL 표현을 사용하지 않는다.
