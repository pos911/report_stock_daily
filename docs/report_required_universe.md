# Report Required Universe

Primary source remains `static_stock_universe.enabled = true`.

Extension layers:

- `config/report_required_stock_universe.yml`
- `config/report_required_etf_universe.yml`
- `config/report_required_macro_series.yml`

Rules:

- `report_required_stock_universe.yml` extends watchlist coverage without replacing static universe.
- `report_required_etf_universe.yml` guarantees sector ETF freshness even when ETFs are not in top rankings.
- leverage, inverse, ETN, covered-call, synthetic, and futures-style products may exist in the ETF universe, but default to `exclude_from_signal=true`.
