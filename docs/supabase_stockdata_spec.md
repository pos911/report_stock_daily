# Supabase StockData Spec

Guardrails:

- preserve existing normalized core tables
- do not create duplicate ETF price tables
- keep `market_trading_calendar` based XKRX/XNYS gating
- use `static_stock_universe.enabled = true` as the baseline watchlist source

Coverage policy:

- latest prices for report-required ETFs must be upserted into `normalized_stock_prices_daily` even when not top-ranked
- no carry-forward writes on market holidays
- stale interpretation belongs to report views, not ingestion writes
