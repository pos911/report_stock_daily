create or replace view report_sector_etf_signal_view as
with price_anchor as (
    select max(base_date) as target_date
    from normalized_stock_prices_daily
),
required_etf as (
    select
        u.symbol,
        coalesce(sm.name, u.name, u.symbol) as name,
        coalesce(sm.market, 'ETF') as market,
        u.sector_group,
        u.theme_group,
        u.provider,
        u.role,
        u.is_active,
        u.exclude_from_signal,
        u.exclude_reason,
        u.notes
    from report_required_etf_universe u
    left join stocks_master sm on sm.symbol = u.symbol
    where u.is_active = true
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
    join required_etf r on r.symbol = p.symbol
),
price_pivot as (
    select
        symbol,
        max(case when rn = 1 then base_date end) as latest_price_date,
        max(case when rn = 1 then close_price end) as close_price,
        max(case when rn = 1 then volume end) as volume,
        max(case when rn = 1 then trading_value end) as trading_value,
        max(case when rn = 2 then close_price end) as prev_close_price,
        max(case when rn = 6 then close_price end) as close_price_5d_ago,
        max(case when rn = 21 then close_price end) as close_price_20d_ago,
        max(case when rn = 61 then close_price end) as close_price_60d_ago,
        avg(case when rn between 1 and 20 then trading_value end) as trading_value_20d_avg,
        max(case when rn between 1 and 252 then close_price end) as high_52w
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
    join required_etf r on r.symbol = s.symbol
    order by s.symbol, s.base_date desc
)
select
    r.symbol,
    r.name,
    r.sector_group,
    r.theme_group,
    r.role,
    r.exclude_from_signal,
    p.latest_price_date,
    a.target_date,
    case
        when p.latest_price_date is null or a.target_date is null then null
        else a.target_date - p.latest_price_date
    end as stale_days,
    case
        when p.latest_price_date is null then 'NO_DATA'
        when a.target_date - p.latest_price_date >= 4 then 'STALE'
        when a.target_date - p.latest_price_date >= 1 then 'STALE_BUT_USABLE'
        else 'FRESH'
    end as data_status,
    p.close_price,
    p.volume,
    p.trading_value,
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
    p.trading_value_20d_avg,
    case
        when p.trading_value_20d_avg is null or p.trading_value_20d_avg = 0 then null
        else p.trading_value / p.trading_value_20d_avg
    end as trading_value_ratio_20d,
    p.high_52w,
    case
        when p.close_price is null or p.high_52w is null or p.high_52w = 0 then null
        else (p.close_price / p.high_52w) * 100
    end as near_52w_high_pct,
    s.foreign_net_buy,
    s.institutional_net_buy,
    s.individual_net_buy,
    s.foreign_holding_ratio,
    array_remove(
        array[
            case when r.exclude_from_signal then 'EXCLUDED_FROM_SIGNAL' end,
            case
                when p.close_price is not null and p.close_price_20d_ago is not null and p.close_price_20d_ago <> 0
                     and ((p.close_price / p.close_price_20d_ago) - 1) >= 0.30
                    then 'OVERHEATED_20D'
            end
        ],
        null
    ) as warnings
from required_etf r
cross join price_anchor a
left join price_pivot p on p.symbol = r.symbol
left join latest_supply s on s.symbol = r.symbol;
