# Report Data Usage Guide

## Purpose

`report_stock_daily` is an operational report consumer for StockData and Supabase.

- This repository does not ingest market data.
- StockData is responsible for data collection, normalization, and readiness policy.
- `report_stock_daily` reads Supabase data, applies readiness gating, and generates Telegram reports.

## Report Data Policy

- `kr_full_market_price_ready=true`:
  - Full-market trading value and market-cap sections may be shown.
- `kr_full_market_price_ready=false`:
  - Full-market trading value and market-cap sections must be omitted.
  - Domestic equity commentary must stay within KIS ranking and watchlist candidate coverage.
- `kis_volume_ranking_ready=true`:
  - `normalized_market_rankings_daily` with `source='KIS'` and `rank_type='volume'` may be shown as a limited ranking reference.
- `kis_universe_ready=true`:
  - Watchlist and ranking-candidate signal sections may be shown.

## Naming Rules

- KIS universe, KIS_DETAIL, and KIS ranking data must not be described as full-market statistics.
- Allowed phrases:
  - `KIS 거래량 순위 기준`
  - `관심종목·랭킹 후보 Signal`
  - `관심종목·KIS 후보군 기준`
- Forbidden phrases when full-market readiness is false:
  - `국내 전체시장 거래대금 상위`
  - `국내 전체시장 시가총액 상위`
  - `오늘 한국시장 전체 Top`

## Workflow Operations

### Operational workflow

- `daily_report.yml`
  - This is the only scheduled operational workflow in this repository.
  - It generates `morning`, `regular`, and `closing` reports.
  - It supports Telegram sending and market-closed skip notifications.
  - It supports `workflow_dispatch` with `report_type`, `report_date`, and `notify_on_skip`.

### Manual diagnostics

- `report_data_diagnostics.yml`
  - Manual-only workflow.
  - Runs `scripts/verify_data.py`.
  - Can optionally run `morning`, `regular`, and `closing` dry-runs after verification.
  - Intended for operations checks and troubleshooting, not scheduled production use.

### Manual cleanup

- `weekly_cleanup.yml`
  - Manual-only workflow.
  - Default input is `dry_run=true`.
  - Use it to review raw-data cleanup behavior before any destructive run.

### Not handled in this repository

- Weekly master refresh is not performed in `report_stock_daily`.
- Monthly calendar sync is not performed in `report_stock_daily`.
- Market close data ingestion is not performed in `report_stock_daily`.
- If those jobs are needed, they must run in the StockData repository.

## LLM Guardrails

- Gemini must not invent blocked sections.
- Gemini must not describe KIS universe data as full-market statistics.
- Gemini must not create trading value or market-cap full-market rankings when readiness blocks them.
- BUY / HOLD / SELL language is not allowed in report body generation.

## Secret Handling

- Workflows load secrets from `API_KEYS_JSON` into a temporary `.runtime_env` file.
- Sensitive values are masked with `::add-mask::`.
- Secrets must not be echoed directly in workflow logs.
- `set -x` must not be used in workflow shell steps.
