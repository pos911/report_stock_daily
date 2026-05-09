# Report Data Usage Guide

## Purpose

`report_stock_daily` consumes StockData with a readiness-first policy.

- `kr_full_market_price_ready=true`일 때만 국내 전체시장 거래대금 Top, 시총 Top을 노출한다.
- `kr_full_market_price_ready=false`이면 국내 전체시장 Top 섹션은 생성하지 않는다.
- `kis_volume_ranking_ready=true`이면 `normalized_market_rankings_daily source='KIS' rank_type='volume'` 기준 거래량 순위를 참고 섹션으로 노출한다.
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

- Gemini는 blocked section을 임의로 생성하지 않는다.
- KIS universe 기반 데이터를 전체시장 통계처럼 해석하지 않는다.
- BUY / HOLD / SELL 표현을 사용하지 않는다.

## Workflow Operations

- `daily_report.yml`
  - 단일 `report` job 구조를 사용한다.
  - `workflow_dispatch`에서 `report_type=morning/regular/closing/auto`를 모두 지원한다.
  - `notify_on_skip=true`이면 양 시장 휴장일에도 짧은 skip 알림을 Telegram으로 보낸다.
- `daily_morning_required_data.yml`
  - `scripts/verify_data.py`를 실행해 morning contract readiness를 검증한다.
- `daily_market_close_pipeline.yml`
  - close pipeline 실행 뒤 `scripts/verify_data.py`를 실행한다.
- `weekly_cleanup.yml`
  - `scripts/cleanup_raw_tables.py`를 기본 dry-run으로 실행 가능하다.
- `weekly_master_refresh.yml`, `monthly_calendar_sync.yml`
  - 현재는 policy/audit workflow다.
  - 실제 master refresh/sync 스크립트가 추가되기 전까지는 운영 정책 점검 용도로만 사용한다.

## Secret Handling

- workflow는 `API_KEYS_JSON`을 `.runtime_env`로 분리 주입한다.
- `::add-mask::`로 `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_ACCESS_TOKEN`, `SUPABASE_KEY`, `GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`를 마스킹한다.
- secret 원문을 `echo`하지 않는다.
- `set -x`는 사용하지 않는다.
