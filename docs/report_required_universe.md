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
- 레포의 YAML/seed SQL과 Supabase live table은 한글 `name`, `sector_group`, `theme_group` 기준으로 일치해야 합니다.
- seed SQL 재실행 시에도 `반도체`, `2차전지`, `조선`, `방산`, `금융/증권`, `바이오/헬스케어` 등 한글 섹터명이 유지되어야 합니다.
