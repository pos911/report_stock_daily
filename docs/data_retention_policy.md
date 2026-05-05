# Data Retention Policy

Normalized tables are long-retention analytical assets.

Raw retention:

- `raw_stock_prices_daily`: 60 days
- `raw_market_rankings`: 60 days
- `raw_macro`: 90 days
- `pipeline_run_logs`: at least 180 days
- raw news/disclosure tables: disabled by default, otherwise 7-14 days max

Operational notes:

- `scripts/cleanup_raw_tables.py` supports dry-run by default.
- cleanup summary should be logged into `pipeline_run_logs`.
