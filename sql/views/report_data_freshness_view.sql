create or replace view report_data_freshness_view as
with latest_dates as (
    select
        (select max(base_date) from normalized_stock_prices_daily) as latest_stock_price_date,
        (select max(base_date) from normalized_market_rankings_daily) as latest_ranking_date,
        (select max(base_date) from normalized_global_macro_daily) as latest_macro_date,
        (select max(base_date) from normalized_stock_supply_daily) as latest_supply_date,
        (select max(base_date) from normalized_stock_short_selling) as latest_short_selling_date,
        (select max(base_date) from market_breadth_daily) as latest_breadth_date
),
sector_etf as (
    select
        count(*) filter (where data_status = 'FRESH') as fresh_count,
        count(*) filter (where data_status in ('STALE', 'NO_DATA')) as problem_count,
        string_agg(symbol, ', ' order by symbol) filter (where data_status in ('STALE', 'NO_DATA')) as stale_symbols
    from report_sector_etf_signal_view
),
watchlist as (
    select
        count(*) filter (where data_status = 'FRESH') as fresh_count,
        count(*) filter (where data_status in ('STALE', 'NO_DATA', 'DATA_MISSING')) as problem_count,
        string_agg(symbol, ', ' order by symbol) filter (where data_status in ('STALE', 'NO_DATA', 'DATA_MISSING')) as stale_symbols
    from report_watchlist_snapshot_view
)
select
    l.latest_stock_price_date as target_date,
    l.latest_stock_price_date,
    l.latest_ranking_date,
    l.latest_macro_date,
    l.latest_supply_date,
    l.latest_short_selling_date,
    l.latest_breadth_date,
    case when s.problem_count = 0 then 'PASS' else 'WARN' end as sector_etf_coverage_status,
    case when w.problem_count = 0 then 'PASS' else 'WARN' end as watchlist_coverage_status,
    concat_ws(' | ',
        case when s.problem_count > 0 then 'sector_etf: ' || coalesce(s.stale_symbols, '') end,
        case when w.problem_count > 0 then 'watchlist: ' || coalesce(w.stale_symbols, '') end
    ) as stale_warnings,
    case
        when l.latest_stock_price_date is null or l.latest_ranking_date is null or l.latest_macro_date is null
            then 'core report tables missing'
        else null
    end as missing_required_data
from latest_dates l
cross join sector_etf s
cross join watchlist w;
