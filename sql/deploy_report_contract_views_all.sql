drop view if exists report_data_freshness_view;
drop view if exists report_market_ranking_view;
drop view if exists report_watchlist_snapshot_view;
drop view if exists report_sector_etf_signal_view;
drop view if exists report_morning_macro_view;

create or replace view report_morning_macro_view as
select
    base_date,
    sp500,
    sp500_change_rate,
    nasdaq,
    nasdaq_change_rate,
    sox,
    vix,
    usdkrw,
    dxy,
    us10y,
    us3y,
    (us10y - us3y) as us10y_us3y_spread,
    kr10y,
    brent,
    wti,
    kospi,
    kospi_change_rate,
    kosdaq,
    kosdaq_change_rate,
    kospi_foreign_net_buy,
    kospi_institutional_net_buy,
    kospi_individual_net_buy,
    kosdaq_foreign_net_buy,
    kosdaq_institutional_net_buy,
    kosdaq_individual_net_buy
from normalized_global_macro_daily;

create or replace view report_sector_etf_signal_view as
with report_required_etf_universe(symbol, name, sector_group, theme_group, provider, role, is_active, exclude_from_signal, exclude_reason, notes) as (
    values
        ('396500', 'TIGER Semiconductor TOP10', 'Semiconductor', 'Semiconductor', 'Mirae Asset', 'primary', true, false, null, 'Primary semiconductor sector ETF.'),
        ('091160', 'KODEX Semiconductor', 'Semiconductor', 'Semiconductor', 'Samsung', 'primary', true, false, null, null),
        ('091230', 'TIGER Semiconductor', 'Semiconductor', 'Semiconductor', 'Mirae Asset', 'secondary', true, false, null, null),
        ('395270', 'HANARO Fn K-Semiconductor', 'Semiconductor', 'Semiconductor', 'NH-Amundi', 'secondary', true, false, null, null),
        ('305720', 'KODEX Secondary Battery Industry', 'Battery', 'Battery', 'Samsung', 'primary', true, false, null, null),
        ('465330', 'RISE Secondary Battery TOP10', 'Battery', 'Battery', 'KB', 'secondary', true, false, null, null),
        ('461950', 'KODEX Secondary Battery Core Materials 10', 'Battery', 'Battery Materials', 'Samsung', 'secondary', true, false, null, null),
        ('462330', 'KODEX Secondary Battery Industry Leverage', 'Battery', 'Battery', 'Samsung', 'satellite', true, true, 'Leverage product should not anchor sector signals.', null),
        ('494670', 'TIGER Shipbuilding TOP10', 'Shipbuilding', 'Shipbuilding', 'Mirae Asset', 'primary', true, false, null, null),
        ('0115D0', 'KODEX Shipbuilding TOP10', 'Shipbuilding', 'Shipbuilding', 'Samsung', 'secondary', true, false, null, null),
        ('466920', 'SOL Shipbuilding TOP3 Plus', 'Shipbuilding', 'Shipbuilding', 'Shinhan', 'secondary', true, false, null, null),
        ('441540', 'HANARO Fn Shipbuilding Shipping', 'Shipbuilding', 'Shipping', 'NH-Amundi', 'secondary', true, false, null, null),
        ('0080G0', 'KODEX Defense TOP10', 'Defense', 'Defense', 'Samsung', 'primary', true, false, null, null),
        ('449450', 'PLUS K-Defense', 'Defense', 'Defense', 'Hanwha', 'secondary', true, false, null, null),
        ('0090B0', 'PLUS K-Defense Components', 'Defense', 'Defense Components', 'Hanwha', 'secondary', true, false, null, null),
        ('0104G0', 'PLUS K-Defense Leverage', 'Defense', 'Defense', 'Hanwha', 'satellite', true, true, 'Leverage product should not anchor sector signals.', null),
        ('487240', 'KODEX AI Power Core Equipment', 'AI Power', 'Power Infrastructure', 'Samsung', 'primary', true, false, null, null),
        ('0117V0', 'TIGER Korea AI Power Equipment TOP3 Plus', 'AI Power', 'Power Infrastructure', 'Mirae Asset', 'secondary', true, false, null, null),
        ('0101N0', 'RISE AI Power Infra', 'AI Power', 'Power Infrastructure', 'KB', 'secondary', true, false, null, null),
        ('0091P0', 'TIGER Korea Nuclear', 'Nuclear', 'Nuclear', 'Mirae Asset', 'primary', true, false, null, null),
        ('0098F0', 'KODEX Nuclear SMR', 'Nuclear', 'SMR', 'Samsung', 'secondary', true, false, null, null),
        ('117460', 'KODEX Energy Chemicals', 'Energy Chemicals', 'Chemicals', 'Samsung', 'primary', true, false, null, null),
        ('139250', 'TIGER 200 Energy Chemicals', 'Energy Chemicals', 'Chemicals', 'Mirae Asset', 'secondary', true, false, null, null),
        ('228790', 'TIGER Cosmetics', 'Consumer', 'Cosmetics', 'Mirae Asset', 'primary', true, false, null, null),
        ('266410', 'KODEX Consumer Staples', 'Consumer', 'Staples', 'Samsung', 'secondary', true, false, null, null),
        ('091170', 'KODEX Banks', 'Financials', 'Banks', 'Samsung', 'primary', true, false, null, null),
        ('102970', 'KODEX Securities', 'Financials', 'Brokerage', 'Samsung', 'primary', true, false, null, null),
        ('091180', 'KODEX Automobiles', 'Automobile', 'Autos', 'Samsung', 'primary', true, false, null, null),
        ('244580', 'KODEX Bio', 'Healthcare', 'Bio', 'Samsung', 'primary', true, false, null, null),
        ('266420', 'KODEX Healthcare', 'Healthcare', 'Healthcare', 'Samsung', 'secondary', true, false, null, null)
),
required_etf as (
    select *
    from report_required_etf_universe
    where is_active = true
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
    current_date as target_date,
    case
        when p.latest_price_date is null then null
        else current_date - p.latest_price_date
    end as stale_days,
    case
        when p.latest_price_date is null then 'NO_DATA'
        when current_date - p.latest_price_date >= 4 then 'STALE'
        when current_date - p.latest_price_date >= 1 then 'STALE_BUT_USABLE'
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
left join price_pivot p on p.symbol = r.symbol
left join latest_supply s on s.symbol = r.symbol;

create or replace view report_watchlist_snapshot_view as
with report_required_stock_universe(symbol, name, market, sector_group, theme_group, role, is_active, notes) as (
    values
        ('005930', 'Samsung Electronics', 'KOSPI', 'Semiconductor', 'AI Hardware', 'primary', true, 'Core report-required large cap.'),
        ('000660', 'SK hynix', 'KOSPI', 'Semiconductor', 'HBM', 'primary', true, 'Core semiconductor bellwether.'),
        ('277810', 'Rainbow Robotics', 'KOSDAQ', 'Robotics', 'AI Automation', 'satellite', true, 'High-beta growth watchlist.'),
        ('058470', 'Leeno Industrial', 'KOSDAQ', 'Semiconductor Equipment', 'Backend', 'primary', true, 'Representative mid-cap semiconductor name.'),
        ('042700', 'Hanmi Semiconductor', 'KOSPI', 'Semiconductor Equipment', 'HBM', 'satellite', true, 'Supplementary equipment watchlist.'),
        ('012330', 'Hyundai Mobis', 'KOSPI', 'Automobile', 'EV Components', 'primary', true, 'Existing report coverage target.'),
        ('071050', 'Korea Investment Holdings', 'KOSPI', 'Financials', 'Brokerage', 'primary', true, 'Existing report coverage target.'),
        ('247540', 'Ecopro BM', 'KOSDAQ', 'Battery', 'Cathode', 'primary', true, 'Battery complex benchmark.'),
        ('329180', 'HD Hyundai Heavy Industries', 'KOSPI', 'Shipbuilding', 'Shipyard', 'satellite', true, 'Shipbuilding cycle monitor.'),
        ('012450', 'Hanwha Aerospace', 'KOSPI', 'Defense', 'Defense', 'primary', true, 'Defense lead stock.'),
        ('017670', 'SK Telecom', 'KOSPI', 'Telecom', 'Stable Cashflow', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
        ('063080', 'Com2uS', 'KOSDAQ', 'Gaming', 'Content', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
        ('047810', 'Korea Aerospace Industries', 'KOSPI', 'Aerospace', 'Defense', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
        ('278470', 'APR', 'KOSPI', 'Consumer', 'Beauty', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
        ('069960', 'Hyundai Department Store', 'KOSPI', 'Consumer', 'Retail', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
        ('035420', 'NAVER', 'KOSPI', 'Internet', 'Platform', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
        ('015760', 'Korea Electric Power', 'KOSPI', 'Utilities', 'Power', 'legacy', true, 'Migrated from legacy target_stocks.json.')
),
watchlist_union as (
    select
        s.symbol,
        s.name,
        s.market,
        cast(null as text) as sector_group
    from static_stock_universe s
    where s.enabled = true
    union
    select
        r.symbol,
        r.name,
        r.market,
        r.sector_group
    from report_required_stock_universe r
    where r.is_active = true
),
watchlist as (
    select distinct on (symbol)
        symbol,
        name,
        market,
        sector_group
    from watchlist_union
    order by symbol
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
        when current_date - p.base_date >= 4 then 'STALE'
        else 'FRESH'
    end as data_status
from watchlist w
left join price_pivot p on p.symbol = w.symbol
left join latest_supply s on s.symbol = w.symbol
left join latest_short ss on ss.symbol = w.symbol
left join latest_fundamental f on f.symbol = w.symbol;

create or replace view report_market_ranking_view as
select
    base_date,
    market,
    rank_type,
    rank,
    symbol,
    name,
    volume,
    trading_value,
    market_cap,
    change_rate,
    source,
    case
        when current_date - base_date >= 2 then 'STALE'
        else 'FRESH'
    end as data_status
from normalized_market_rankings_daily
where market in ('KOSPI', 'KOSDAQ', 'ETF', 'ETN')
  and rank_type in ('volume', 'trading_value', 'market_cap')
  and rank <= 50;

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
    l.latest_stock_price_date,
    l.latest_ranking_date,
    l.latest_macro_date,
    l.latest_supply_date,
    l.latest_short_selling_date,
    l.latest_breadth_date,
    case when s.problem_count = 0 then 'PASS' else 'WARN' end as sector_etf_coverage_status,
    case when w.problem_count = 0 then 'PASS' else 'WARN' end as watchlist_coverage_status,
    concat_ws(
        ' | ',
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

-- Verification SQL
-- select * from report_morning_macro_view limit 1;
-- select * from report_sector_etf_signal_view limit 5;
-- select * from report_watchlist_snapshot_view limit 5;
-- select * from report_market_ranking_view limit 5;
-- select * from report_data_freshness_view limit 1;
