# Pipeline Frequency Policy

A-grade daily latest:

- report-required stocks, ETFs, and macro series
- `normalized_market_rankings_daily`
- `market_breadth_daily`
- watchlist and ranking price/supply coverage

B-grade daily close or once daily:

- `normalized_stock_short_selling`
- `normalized_stock_supply_daily`
- `normalized_stock_snapshots_daily`
- sector ETF coverage check

C-grade weekly or monthly:

- `stocks_master` full refresh
- ETF and ETN master refresh
- fundamentals and ratio refresh
- `market_trading_calendar` monthly sync
- `macro_series_master` monthly validation

D-grade diagnostic only:

- raw API payload tables with retention cleanup
