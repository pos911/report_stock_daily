# Report Data Contract

`report_stock_daily` should read report-oriented views first:

- `report_morning_macro_view`
- `report_sector_etf_signal_view`
- `report_watchlist_snapshot_view`
- `report_market_ranking_view`
- `report_data_freshness_view`

Daily-latest required:

- `normalized_global_macro_daily`
- `normalized_market_rankings_daily`
- `normalized_stock_prices_daily` for watchlist, ranking names, and report-required ETFs
- `market_breadth_daily`
- `normalized_stock_supply_daily` for watchlist and ranking extensions

Stale-tolerant:

- `normalized_stock_short_selling`
- `normalized_stock_snapshots_daily`
- `normalized_stock_fundamentals_ratios`

Not part of the report contract:

- raw API payload tables
- broad text/news archives
- ad hoc exploratory rankings outside report views
