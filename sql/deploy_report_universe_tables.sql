-- Execution order in Supabase SQL Editor:
-- 1. Run this file first: sql/deploy_report_universe_tables.sql
-- 2. Run sql/deploy_report_contract_views.sql

create table if not exists report_required_stock_universe (
    symbol text primary key,
    name text,
    market text,
    sector_group text,
    theme_group text,
    role text,
    is_active boolean not null default true,
    notes text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists report_required_etf_universe (
    symbol text primary key,
    name text,
    sector_group text,
    theme_group text,
    provider text,
    role text,
    is_active boolean not null default true,
    exclude_from_signal boolean not null default false,
    exclude_reason text,
    notes text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists report_required_macro_series (
    symbol text primary key,
    series_id text,
    name text,
    market text,
    role text,
    is_active boolean not null default true,
    notes text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

insert into report_required_stock_universe (
    symbol, name, market, sector_group, theme_group, role, is_active, notes
) values
    ('005930', '삼성전자', 'KOSPI', 'Semiconductor', 'AI Hardware', 'primary', true, 'Core report-required large cap.'),
    ('000660', 'SK하이닉스', 'KOSPI', 'Semiconductor', 'HBM', 'primary', true, 'Core semiconductor bellwether.'),
    ('277810', '레인보우로보틱스', 'KOSDAQ', 'Robotics', 'AI Automation', 'satellite', true, 'High-beta growth watchlist.'),
    ('058470', '리노공업', 'KOSDAQ', 'Semiconductor Equipment', 'Backend', 'primary', true, 'Representative mid-cap semiconductor name.'),
    ('042700', '한미반도체', 'KOSPI', 'Semiconductor Equipment', 'HBM', 'satellite', true, 'Supplementary equipment watchlist.'),
    ('012330', '현대모비스', 'KOSPI', 'Automobile', 'EV Components', 'primary', true, 'Existing report coverage target.'),
    ('071050', '한국금융지주', 'KOSPI', 'Financials', 'Brokerage', 'primary', true, 'Existing report coverage target.'),
    ('247540', '에코프로비엠', 'KOSDAQ', 'Battery', 'Cathode', 'primary', true, 'Battery complex benchmark.'),
    ('329180', 'HD현대중공업', 'KOSPI', 'Shipbuilding', 'Shipyard', 'satellite', true, 'Shipbuilding cycle monitor.'),
    ('012450', '한화에어로스페이스', 'KOSPI', 'Defense', 'Defense', 'primary', true, 'Defense lead stock.'),
    ('017670', 'SK텔레콤', 'KOSPI', 'Telecom', 'Stable Cashflow', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
    ('063080', '컴투스홀딩스', 'KOSDAQ', 'Gaming', 'Content', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
    ('047810', '한국항공우주', 'KOSPI', 'Aerospace', 'Defense', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
    ('278470', '에이피알', 'KOSPI', 'Consumer', 'Beauty', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
    ('069960', '현대백화점', 'KOSPI', 'Consumer', 'Retail', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
    ('035420', 'NAVER', 'KOSPI', 'Internet', 'Platform', 'legacy', true, 'Migrated from legacy target_stocks.json.'),
    ('015760', '한국전력공사', 'KOSPI', 'Utilities', 'Power', 'legacy', true, 'Migrated from legacy target_stocks.json.')
on conflict (symbol) do update
set
    name = excluded.name,
    market = excluded.market,
    sector_group = excluded.sector_group,
    theme_group = excluded.theme_group,
    role = excluded.role,
    is_active = excluded.is_active,
    notes = excluded.notes,
    updated_at = now();

insert into report_required_etf_universe (
    symbol, name, sector_group, theme_group, provider, role, is_active, exclude_from_signal, exclude_reason, notes
) values
    ('396500', 'TIGER 반도체TOP10', 'Semiconductor', 'Semiconductor', 'Mirae Asset', 'primary', true, false, null, 'Primary semiconductor sector ETF.'),
    ('091160', 'KODEX 반도체', 'Semiconductor', 'Semiconductor', 'Samsung', 'primary', true, false, null, null),
    ('091230', 'TIGER 반도체', 'Semiconductor', 'Semiconductor', 'Mirae Asset', 'secondary', true, false, null, null),
    ('395270', 'HANARO Fn K-반도체', 'Semiconductor', 'Semiconductor', 'NH-Amundi', 'secondary', true, false, null, null),
    ('305720', 'KODEX 2차전지산업', 'Battery', 'Battery', 'Samsung', 'primary', true, false, null, null),
    ('465330', 'RISE 2차전지TOP10', 'Battery', 'Battery', 'KB', 'secondary', true, false, null, null),
    ('461950', 'KODEX 2차전지핵심소재10', 'Battery', 'Battery Materials', 'Samsung', 'secondary', true, false, null, null),
    ('462330', 'KODEX 2차전지산업레버리지', 'Battery', 'Battery', 'Samsung', 'satellite', true, true, 'Leverage product should not anchor sector signals.', null),
    ('494670', 'TIGER 조선TOP10', 'Shipbuilding', 'Shipbuilding', 'Mirae Asset', 'primary', true, false, null, null),
    ('0115D0', 'KODEX 조선TOP10', 'Shipbuilding', 'Shipbuilding', 'Samsung', 'secondary', true, false, null, null),
    ('466920', 'SOL 조선TOP3플러스', 'Shipbuilding', 'Shipbuilding', 'Shinhan', 'secondary', true, false, null, null),
    ('441540', 'HANARO Fn조선해운', 'Shipbuilding', 'Shipping', 'NH-Amundi', 'secondary', true, false, null, null),
    ('0080G0', 'KODEX 방산TOP10', 'Defense', 'Defense', 'Samsung', 'primary', true, false, null, null),
    ('449450', 'PLUS K방산', 'Defense', 'Defense', 'Hanwha', 'secondary', true, false, null, null),
    ('0090B0', 'PLUS K방산소부장', 'Defense', 'Defense Components', 'Hanwha', 'secondary', true, false, null, null),
    ('0104G0', 'PLUS K방산레버리지', 'Defense', 'Defense', 'Hanwha', 'satellite', true, true, 'Leverage product should not anchor sector signals.', null),
    ('487240', 'KODEX AI전력핵심설비', 'AI Power', 'Power Infrastructure', 'Samsung', 'primary', true, false, null, null),
    ('0117V0', 'TIGER 코리아AI전력기기TOP3플러스', 'AI Power', 'Power Infrastructure', 'Mirae Asset', 'secondary', true, false, null, null),
    ('0101N0', 'RISE AI전력인프라', 'AI Power', 'Power Infrastructure', 'KB', 'secondary', true, false, null, null),
    ('0091P0', 'TIGER 코리아원자력', 'Nuclear', 'Nuclear', 'Mirae Asset', 'primary', true, false, null, null),
    ('0098F0', 'KODEX 원자력SMR', 'Nuclear', 'SMR', 'Samsung', 'secondary', true, false, null, null),
    ('117460', 'KODEX 에너지화학', 'Energy Chemicals', 'Chemicals', 'Samsung', 'primary', true, false, null, null),
    ('139250', 'TIGER 200 에너지화학', 'Energy Chemicals', 'Chemicals', 'Mirae Asset', 'secondary', true, false, null, null),
    ('228790', 'TIGER 화장품', 'Consumer', 'Cosmetics', 'Mirae Asset', 'primary', true, false, null, null),
    ('266410', 'KODEX 필수소비재', 'Consumer', 'Staples', 'Samsung', 'secondary', true, false, null, null),
    ('091170', 'KODEX 은행', 'Financials', 'Banks', 'Samsung', 'primary', true, false, null, null),
    ('102970', 'KODEX 증권', 'Financials', 'Brokerage', 'Samsung', 'primary', true, false, null, null),
    ('091180', 'KODEX 자동차', 'Automobile', 'Autos', 'Samsung', 'primary', true, false, null, null),
    ('244580', 'KODEX 바이오', 'Healthcare', 'Bio', 'Samsung', 'primary', true, false, null, null),
    ('266420', 'KODEX 헬스케어', 'Healthcare', 'Healthcare', 'Samsung', 'secondary', true, false, null, null)
on conflict (symbol) do update
set
    name = excluded.name,
    sector_group = excluded.sector_group,
    theme_group = excluded.theme_group,
    provider = excluded.provider,
    role = excluded.role,
    is_active = excluded.is_active,
    exclude_from_signal = excluded.exclude_from_signal,
    exclude_reason = excluded.exclude_reason,
    notes = excluded.notes,
    updated_at = now();

insert into report_required_macro_series (
    symbol, series_id, name, market, role, is_active, notes
) values
    ('SP500', 'SP500', 'S&P500', 'US', 'primary', true, 'Morning macro contract.'),
    ('NASDAQ', 'NASDAQ', 'Nasdaq', 'US', 'primary', true, null),
    ('SOX', 'SOX', 'SOX', 'US', 'primary', true, null),
    ('VIX', 'VIX', 'VIX', 'US', 'primary', true, null),
    ('USDKRW', 'USDKRW', 'USDKRW', 'FX', 'primary', true, null),
    ('DXY', 'DXY', 'DXY', 'FX', 'primary', true, null),
    ('US10Y', 'US10Y', 'US10Y', 'Rates', 'primary', true, null),
    ('US3Y', 'US3Y', 'US3Y', 'Rates', 'primary', true, null),
    ('KR10Y', 'KR10Y', 'KR10Y', 'Rates', 'primary', true, null),
    ('BRENT', 'BRENT', 'Brent', 'Commodities', 'primary', true, null),
    ('WTI', 'WTI', 'WTI', 'Commodities', 'primary', true, null),
    ('GOLD', 'GOLD', 'Gold', 'Commodities', 'primary', true, null),
    ('COPPER', 'COPPER', 'Copper', 'Commodities', 'primary', true, null),
    ('HY_SPREAD', 'HY_SPREAD', 'HY Spread', 'Credit', 'primary', true, null),
    ('KOSPI', 'KOSPI', 'KOSPI', 'KR', 'primary', true, null),
    ('KOSDAQ', 'KOSDAQ', 'KOSDAQ', 'KR', 'primary', true, null),
    ('KOSPI_FOREIGN_NET_BUY', 'KOSPI_FOREIGN_NET_BUY', 'KOSPI Foreign Net Buy', 'KR', 'primary', true, null),
    ('KOSPI_INSTITUTIONAL_NET_BUY', 'KOSPI_INSTITUTIONAL_NET_BUY', 'KOSPI Institutional Net Buy', 'KR', 'primary', true, null),
    ('KOSPI_INDIVIDUAL_NET_BUY', 'KOSPI_INDIVIDUAL_NET_BUY', 'KOSPI Individual Net Buy', 'KR', 'primary', true, null),
    ('KOSDAQ_FOREIGN_NET_BUY', 'KOSDAQ_FOREIGN_NET_BUY', 'KOSDAQ Foreign Net Buy', 'KR', 'primary', true, null),
    ('KOSDAQ_INSTITUTIONAL_NET_BUY', 'KOSDAQ_INSTITUTIONAL_NET_BUY', 'KOSDAQ Institutional Net Buy', 'KR', 'primary', true, null),
    ('KOSDAQ_INDIVIDUAL_NET_BUY', 'KOSDAQ_INDIVIDUAL_NET_BUY', 'KOSDAQ Individual Net Buy', 'KR', 'primary', true, null)
on conflict (symbol) do update
set
    series_id = excluded.series_id,
    name = excluded.name,
    market = excluded.market,
    role = excluded.role,
    is_active = excluded.is_active,
    notes = excluded.notes,
    updated_at = now();
