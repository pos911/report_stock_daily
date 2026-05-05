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
    ('005930', '삼성전자', 'KOSPI', '반도체', 'AI 하드웨어', 'primary', true, '핵심 대형주'),
    ('000660', 'SK하이닉스', 'KOSPI', '반도체', 'HBM', 'primary', true, '핵심 반도체 벨웨더'),
    ('277810', '레인보우로보틱스', 'KOSDAQ', '로봇', 'AI 자동화', 'satellite', true, '고베타 성장주 관찰 대상'),
    ('058470', '리노공업', 'KOSDAQ', '반도체 장비', '후공정', 'primary', true, '대표 중형 반도체 종목'),
    ('042700', '한미반도체', 'KOSPI', '반도체 장비', 'HBM', 'satellite', true, '보조 장비 관찰 대상'),
    ('012330', '현대모비스', 'KOSPI', '자동차', 'EV 부품', 'primary', true, '기존 리포트 핵심 종목'),
    ('071050', '한국금융지주', 'KOSPI', '금융/증권', '증권', 'primary', true, '기존 리포트 핵심 종목'),
    ('247540', '에코프로비엠', 'KOSDAQ', '2차전지', '양극재', 'primary', true, '2차전지 대표 종목'),
    ('329180', 'HD현대중공업', 'KOSPI', '조선', '조선소', 'satellite', true, '조선 업황 점검 종목'),
    ('012450', '한화에어로스페이스', 'KOSPI', '방산', '방산', 'primary', true, '방산 대표 종목'),
    ('017670', 'SK텔레콤', 'KOSPI', '통신', '안정 현금흐름', 'legacy', true, 'legacy target_stocks.json 이관 종목'),
    ('063080', '컴투스', 'KOSDAQ', '게임', '콘텐츠', 'legacy', true, 'legacy target_stocks.json 이관 종목'),
    ('047810', '한국항공우주', 'KOSPI', '방산', '항공우주', 'legacy', true, 'legacy target_stocks.json 이관 종목'),
    ('278470', '에이피알', 'KOSPI', '화장품/소비재', '뷰티', 'legacy', true, 'legacy target_stocks.json 이관 종목'),
    ('069960', '현대백화점', 'KOSPI', '화장품/소비재', '리테일', 'legacy', true, 'legacy target_stocks.json 이관 종목'),
    ('035420', 'NAVER', 'KOSPI', '인터넷/플랫폼', '플랫폼', 'legacy', true, 'legacy target_stocks.json 이관 종목'),
    ('015760', '한국전력', 'KOSPI', '전력/유틸리티', '전력', 'legacy', true, 'legacy target_stocks.json 이관 종목')
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
    ('396500', 'TIGER 반도체TOP10', '반도체', '반도체', '미래에셋', 'primary', true, false, null, '대표 반도체 섹터 ETF'),
    ('091160', 'KODEX 반도체', '반도체', '반도체', '삼성', 'primary', true, false, null, null),
    ('091230', 'TIGER 반도체', '반도체', '반도체', '미래에셋', 'secondary', true, false, null, null),
    ('395270', 'HANARO Fn K-반도체', '반도체', '반도체', 'NH-아문디', 'secondary', true, false, null, null),
    ('305720', 'KODEX 2차전지산업', '2차전지', '2차전지', '삼성', 'primary', true, false, null, null),
    ('465330', 'RISE 2차전지TOP10', '2차전지', '2차전지', 'KB', 'secondary', true, false, null, null),
    ('461950', 'KODEX 2차전지핵심소재10', '2차전지', '2차전지 소재', '삼성', 'secondary', true, false, null, null),
    ('462330', 'KODEX 2차전지산업레버리지', '2차전지', '2차전지', '삼성', 'satellite', true, true, '레버리지 상품은 섹터 주신호에서 제외', null),
    ('494670', 'TIGER 조선TOP10', '조선', '조선', '미래에셋', 'primary', true, false, null, null),
    ('0115D0', 'KODEX 조선TOP10', '조선', '조선', '삼성', 'secondary', true, false, null, null),
    ('466920', 'SOL 조선TOP3플러스', '조선', '조선', '신한', 'secondary', true, false, null, null),
    ('441540', 'HANARO Fn조선해운', '조선', '조선/해운', 'NH-아문디', 'secondary', true, false, null, null),
    ('0080G0', 'KODEX 방산TOP10', '방산', '방산', '삼성', 'primary', true, false, null, null),
    ('449450', 'PLUS K방산', '방산', '방산', '한화', 'secondary', true, false, null, null),
    ('0090B0', 'PLUS K방산소부장', '방산', '방산 소부장', '한화', 'secondary', true, false, null, null),
    ('0104G0', 'PLUS K방산레버리지', '방산', '방산', '한화', 'satellite', true, true, '레버리지 상품은 섹터 주신호에서 제외', null),
    ('487240', 'KODEX AI전력핵심설비', 'AI전력/인프라', 'AI전력/인프라', '삼성', 'primary', true, false, null, null),
    ('0117V0', 'TIGER 코리아AI전력기기TOP3플러스', 'AI전력/인프라', 'AI전력/인프라', '미래에셋', 'secondary', true, false, null, null),
    ('0101N0', 'RISE AI전력인프라', 'AI전력/인프라', 'AI전력/인프라', 'KB', 'secondary', true, false, null, null),
    ('0091P0', 'TIGER 코리아원자력', '원자력', '원자력', '미래에셋', 'primary', true, false, null, null),
    ('0098F0', 'KODEX 원자력SMR', '원자력', 'SMR', '삼성', 'secondary', true, false, null, null),
    ('117460', 'KODEX 에너지화학', '정유화학', '정유화학', '삼성', 'primary', true, false, null, null),
    ('139250', 'TIGER 200 에너지화학', '정유화학', '정유화학', '미래에셋', 'secondary', true, false, null, null),
    ('228790', 'TIGER 화장품', '화장품/소비재', '화장품', '미래에셋', 'primary', true, false, null, null),
    ('266410', 'KODEX 필수소비재', '화장품/소비재', '필수소비재', '삼성', 'secondary', true, false, null, null),
    ('091170', 'KODEX 은행', '금융/증권', '은행', '삼성', 'primary', true, false, null, null),
    ('102970', 'KODEX 증권', '금융/증권', '증권', '삼성', 'primary', true, false, null, null),
    ('091180', 'KODEX 자동차', '자동차', '자동차', '삼성', 'primary', true, false, null, null),
    ('244580', 'KODEX 바이오', '바이오/헬스케어', '바이오', '삼성', 'primary', true, false, null, null),
    ('266420', 'KODEX 헬스케어', '바이오/헬스케어', '헬스케어', '삼성', 'secondary', true, false, null, null)
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
