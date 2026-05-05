create or replace view report_market_ranking_view as
with ranking_anchor as (
    select max(base_date) as target_date
    from normalized_market_rankings_daily
)
select
    r.base_date,
    r.market,
    r.rank_type,
    r.rank,
    r.symbol,
    coalesce(sm.name, r.name, r.symbol) as name,
    r.volume,
    r.trading_value,
    r.market_cap,
    r.change_rate,
    r.source,
    case
        when a.target_date - r.base_date >= 2 then 'STALE'
        else 'FRESH'
    end as data_status
from normalized_market_rankings_daily r
cross join ranking_anchor a
left join stocks_master sm on sm.symbol = r.symbol
where r.market in ('KOSPI', 'KOSDAQ', 'ETF', 'ETN')
  and r.rank_type in ('volume', 'trading_value', 'market_cap')
  and r.rank <= 50;
