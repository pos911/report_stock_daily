# StockData Data Dictionary

Report-facing normalized tables:

- `normalized_stock_prices_daily`: latest price, volume, trading value
- `normalized_market_rankings_daily`: volume, trading value, market cap rankings
- `normalized_global_macro_daily`: macro snapshot for morning reports
- `normalized_stock_supply_daily`: investor flow and holding ratio
- `normalized_stock_short_selling`: short ratio and short value
- `market_breadth_daily`: market breadth status

Report-facing views:

- `report_morning_macro_view`
- `report_sector_etf_signal_view`
- `report_watchlist_snapshot_view`
- `report_market_ranking_view`
- `report_data_freshness_view`

Out of scope for Supabase growth:

- Naver news bulk text archives
- Google Docs text mirrors beyond lightweight report context
- raw diagnostics beyond retention windows
