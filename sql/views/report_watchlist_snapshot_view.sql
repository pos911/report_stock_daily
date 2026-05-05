create or replace view report_watchlist_snapshot_view as
with price_anchor as (
    select max(base_date) as target_date
    from normalized_stock_prices_daily
),
watchlist_union as (
    select
        s.symbol,
        s.name,
        s.market,
        cast(null as text) as sector_group,
        cast(null as text) as theme_group,
        cast('static_stock_universe' as text) as source_name
    from static_stock_universe s
    where s.enabled = true
    union
    select
        r.symbol,
        r.name,
        r.market,
        r.sector_group,
        r.theme_group,
        cast('report_required_stock_universe' as text) as source_name
    from report_required_stock_universe r
    where r.is_active = true
),
watchlist as (
    select distinct on (symbol)
        u.symbol,
        coalesce(sm.name, u.name, u.symbol) as name,
        coalesce(sm.market, u.market) as market,
        u.sector_group,
        u.theme_group
    from watchlist_union u
    left join stocks_master sm on sm.symbol = u.symbol
    order by u.symbol, u.source_name desc
),
price_history as (
    select
        p.symbol,
        p.base_date,
        p.close_price,
        p.volume,
        p.trading_value,
        row_number() over (partition by p.symbol order by p.base_date desc) as rn
    from normalized_stock_prices_daily p
    join watchlist w on w.symbol = p.symbol
),
price_pivot as (
    select
        symbol,
        max(case when rn = 1 then base_date end) as base_date,
        max(case when rn = 1 then close_price end) as close_price,
        max(case when rn = 1 then trading_value end) as trading_value,
        max(case when rn = 1 then volume end) as volume,
        max(case when rn = 2 then close_price end) as prev_close_price,
        max(case when rn = 6 then close_price end) as close_price_5d_ago,
        max(case when rn = 21 then close_price end) as close_price_20d_ago,
        max(case when rn = 61 then close_price end) as close_price_60d_ago,
        avg(case when rn between 1 and 20 then trading_value end) as trading_value_20d_avg
    from price_history
    group by symbol
),
latest_supply as (
    select distinct on (s.symbol)
        s.symbol,
        s.foreign_net_buy,
        s.institutional_net_buy,
        s.individual_net_buy,
        s.foreign_holding_ratio
    from normalized_stock_supply_daily s
    join watchlist w on w.symbol = s.symbol
    order by s.symbol, s.base_date desc
),
latest_short as (
    select distinct on (s.symbol)
        s.symbol,
        s.short_ratio,
        s.short_value
    from normalized_stock_short_selling s
    join watchlist w on w.symbol = s.symbol
    order by s.symbol, s.base_date desc
),
latest_fundamental as (
    select distinct on (f.symbol)
        f.symbol,
        f.per,
        f.pbr,
        f.roe,
        f.debt_ratio
    from normalized_stock_fundamentals_ratios f
    join watchlist w on w.symbol = f.symbol
    order by f.symbol, f.base_date desc
)
select
    w.symbol,
    w.name,
    w.market,
    w.sector_group,
    p.base_date,
    p.close_price,
    case
        when p.close_price is null or p.prev_close_price is null or p.prev_close_price = 0 then null
        else (p.close_price / p.prev_close_price) - 1
    end as change_rate_1d,
    case
        when p.close_price is null or p.close_price_5d_ago is null or p.close_price_5d_ago = 0 then null
        else (p.close_price / p.close_price_5d_ago) - 1
    end as return_5d,
    case
        when p.close_price is null or p.close_price_20d_ago is null or p.close_price_20d_ago = 0 then null
        else (p.close_price / p.close_price_20d_ago) - 1
    end as return_20d,
    case
        when p.close_price is null or p.close_price_60d_ago is null or p.close_price_60d_ago = 0 then null
        else (p.close_price / p.close_price_60d_ago) - 1
    end as return_60d,
    p.trading_value,
    case
        when p.trading_value_20d_avg is null or p.trading_value_20d_avg = 0 then null
        else p.trading_value / p.trading_value_20d_avg
    end as trading_value_ratio_20d,
    s.foreign_net_buy,
    s.institutional_net_buy,
    s.individual_net_buy,
    s.foreign_holding_ratio,
    ss.short_ratio,
    ss.short_value,
    f.per,
    f.pbr,
    f.roe,
    f.debt_ratio,
    case
        when p.base_date is null then 'NO_DATA'
        when a.target_date - p.base_date >= 4 then 'STALE'
        when a.target_date - p.base_date >= 1 then 'STALE_BUT_USABLE'
        else 'FRESH'
    end as data_status
from watchlist w
cross join price_anchor a
left join price_pivot p on p.symbol = w.symbol
left join latest_supply s on s.symbol = w.symbol
left join latest_short ss on ss.symbol = w.symbol
left join latest_fundamental f on f.symbol = w.symbol;
