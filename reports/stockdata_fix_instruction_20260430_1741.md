# StockData 데이터 적재/스키마 수정 요청

대상 저장소:
- https://github.com/pos911/StockData.git

문제 요약:
- full price coverage 미완료 또는 PARTIAL 상태
- Q prefix symbol 중복
- Top 5 후보 중 최소 데이터 조건 미충족 row 존재
- fundamentals ratios 0값 대량 또는 유효성 점검 필요
- short_value/short_volume 대비 short_ratio 0/null 불일치

증거:
| category | name | symbol | base_date | details |
|---|---|---|---|---|
| full_market_coverage | 전체 시장 커버리지 | - | 2026-04-30 | covered_symbols=428, records_processed=2658 |
| duplicate_symbol | 삼성 인버스 2X WTI원유 선물 ETN | 530036/Q530036 | 2026-04-30 | duplicate canonical symbol -> 530036 |
| duplicate_symbol | N2 인버스 레버리지 WTI원유 선물 ETN(H) | 550043/Q550043 | 2026-04-30 | duplicate canonical symbol -> 550043 |
| duplicate_symbol | 삼성 인버스 2X 코스닥150 선물 ETN | 530107/Q530107 | 2026-04-30 | duplicate canonical symbol -> 530107 |
| duplicate_symbol | 삼성 레버리지 WTI원유 선물 ETN | 530031/Q530031 | 2026-04-30 | duplicate canonical symbol -> 530031 |
| duplicate_symbol | 삼성 인버스 2X WTI원유 선물 ETN | 530036/Q530036 | 2026-04-30 | duplicate canonical symbol -> 530036 |
| duplicate_symbol | N2 인버스 레버리지 WTI원유 선물 ETN(H) | 550043/Q550043 | 2026-04-30 | duplicate canonical symbol -> 550043 |
| duplicate_symbol | 삼성 인버스 2X 코스닥150 선물 ETN | 530107/Q530107 | 2026-04-30 | duplicate canonical symbol -> 530107 |
| duplicate_symbol | 삼성 레버리지 WTI원유 선물 ETN | 530031/Q530031 | 2026-04-30 | duplicate canonical symbol -> 530031 |
| duplicate_symbol | KODEX 인버스 | 114800/114800 | 2026-04-30 | duplicate canonical symbol -> 114800 |
| duplicate_symbol | 삼성 인버스 2X WTI원유 선물 ETN | Q530036/Q530036 | 2026-04-30 | duplicate canonical symbol -> 530036 |
| duplicate_symbol | KODEX 2차전지산업레버리지 | 462330/462330 | 2026-04-30 | duplicate canonical symbol -> 462330 |
| duplicate_symbol | N2 인버스 레버리지 WTI원유 선물 ETN(H) | Q550043/Q550043 | 2026-04-30 | duplicate canonical symbol -> 550043 |
| duplicate_symbol | TIGER 반도체TOP10 | 396500/396500 | 2026-04-30 | duplicate canonical symbol -> 396500 |
| duplicate_symbol | TIGER 2차전지TOP10레버리지 | 412570/412570 | 2026-04-30 | duplicate canonical symbol -> 412570 |
| duplicate_symbol | KODEX 레버리지 | 122630/122630 | 2026-04-30 | duplicate canonical symbol -> 122630 |
| duplicate_symbol | KODEX 미국S&P500 | 379800/379800 | 2026-04-30 | duplicate canonical symbol -> 379800 |
| duplicate_symbol | 삼성 인버스 2X 코스닥150 선물 ETN | Q530107/Q530107 | 2026-04-30 | duplicate canonical symbol -> 530107 |
| duplicate_symbol | KODEX 2차전지산업레버리지 | 305720/305720 | 2026-04-30 | duplicate canonical symbol -> 305720 |
| duplicate_symbol | KODEX 삼성전자SK하이닉스채권혼합50 | 0177N0/0177N0 | 2026-04-30 | duplicate canonical symbol -> 0177N0 |
| duplicate_symbol | RISE 삼성전자SK하이닉스채권혼합50 | 0162Z0/0162Z0 | 2026-04-30 | duplicate canonical symbol -> 0162Z0 |
| duplicate_symbol | KODEX 미국우주항공 | 0167Z0/0167Z0 | 2026-04-30 | duplicate canonical symbol -> 0167Z0 |
| duplicate_symbol | TIGER 미국우주테크 | 0183J0/0183J0 | 2026-04-30 | duplicate canonical symbol -> 0183J0 |
| duplicate_symbol | 삼성 레버리지 WTI원유 선물 ETN | Q530031/Q530031 | 2026-04-30 | duplicate canonical symbol -> 530031 |
| duplicate_symbol | KODEX 머니마켓액티브 | 488770/488770 | 2026-04-30 | duplicate canonical symbol -> 488770 |
| invalid_top_volume_row | 삼성 블룸버그 인버스2X WTI원유선물 ETN B | 530134 | 2026-04-30 | close_price=2390.0, volume=16100071.0, trading_value=0.0 |
| valuation_zero_issue | 리노공업 | 058470 | 2026-04-30 | per=0.0, pbr=0.0, roe=0.0, debt_ratio=0.0 |
| short_ratio_issue | 리노공업 | 058470 | 2026-04-30 | short_value=17586522550, short_volume=147615, short_ratio=0.0 |
| short_ratio_issue | SK하이닉스 | 000660 | 2026-04-30 | short_value=13055850000, short_volume=9978, short_ratio=0.0 |
| short_ratio_issue | 삼성전자 | 005930 | 2026-04-30 | short_value=66862207500, short_volume=298627, short_ratio=0.0 |
| short_ratio_issue | 한국전력 | 015760 | 2026-04-30 | short_value=5154710500, short_volume=117686, short_ratio=0.0 |
| short_ratio_issue | NAVER | 035420 | 2026-04-30 | short_value=50234940750, short_volume=234729, short_ratio=0.0 |
| short_ratio_issue | 현대백화점 | 069960 | 2026-04-30 | short_value=483229600, short_volume=4407, short_ratio=0.0 |
| short_ratio_issue | 한국금융지주 | 071050 | 2026-04-30 | short_value=1131531500, short_volume=4628, short_ratio=0.0 |
| short_ratio_issue | 에이피알 | 278470 | 2026-04-30 | short_value=29947339000, short_volume=70114, short_ratio=0.0 |

StockData 수정 지시:
1. `stocks_master`에 `asset_type` 필드를 추가하거나 기존 분류를 정비하라.
2. `market`과 `asset_type`을 분리하라.
3. `Q530036`과 `530036`처럼 동일 상품이 중복 적재되지 않도록 `canonical_symbol`, `display_symbol`, `source_symbol` 체계를 도입하라.
4. `normalized_stock_prices_daily`에 prefix만 다른 중복 row가 올라오지 않게 upsert key를 재검토하라.
5. `static_stock_universe`, `stocks_master`, `normalized_*` 테이블 간 symbol join 기준을 명확히 하라.
6. `normalized_stock_supply_daily`의 net_buy 계열 컬럼 단위를 spec과 코드에 명확히 정의하라.
7. `normalized_stock_fundamentals_ratios`는 수집 실패 시 0이 아니라 null로 적재하라.
8. `normalized_stock_fundamentals` 원천값과 ratio 정합성 검증 로직을 추가하라.
9. `normalized_stock_short_selling.short_ratio`가 미수집/계산불가이면 0이 아니라 null로 적재하라.
10. `short_value > 0` 또는 `short_volume > 0`인데 `short_ratio = 0`인 row를 경고로 남겨라.
11. `daily_stock_full_price_pipeline.records_processed > 2000`을 full market 완료 기준으로 유지하되 부분완료 시 `WARN` 또는 `PARTIAL` 상태를 남겨라.
12. `pipeline_run_logs`에 `PARTIAL` 상태를 명확히 기록하라.
13. `supabase_stockdata_spec.md`에 단위, asset_type, symbol normalization 정책을 반영하라.

StockData 검증 SQL:
```sql
-- asset_type 미분류 종목 확인
SELECT market, COUNT(*) 
FROM stocks_master
GROUP BY market
ORDER BY market;

-- Q prefix 중복 symbol 확인
SELECT
    REGEXP_REPLACE(symbol, '^Q', '') AS canonical_symbol,
    ARRAY_AGG(symbol ORDER BY symbol) AS symbols,
    COUNT(*) AS dup_count
FROM normalized_stock_prices_daily
GROUP BY REGEXP_REPLACE(symbol, '^Q', ''), base_date
HAVING COUNT(*) > 1;

-- fundamentals ratios 0값 대량 확인
SELECT
    base_date,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (
        WHERE COALESCE(per, 0) = 0
          AND COALESCE(pbr, 0) = 0
          AND COALESCE(roe, 0) = 0
          AND COALESCE(debt_ratio, 0) = 0
    ) AS zero_ratio_rows
FROM normalized_stock_fundamentals_ratios
GROUP BY base_date
ORDER BY base_date DESC
LIMIT 10;

-- short_value > 0 인데 short_ratio 0/null 인 row
SELECT symbol, base_date, short_value, short_volume, short_ratio
FROM normalized_stock_short_selling
WHERE (COALESCE(short_value, 0) > 0 OR COALESCE(short_volume, 0) > 0)
  AND (short_ratio IS NULL OR short_ratio = 0)
ORDER BY base_date DESC
LIMIT 100;

-- full market coverage 확인
WITH latest_price AS (
    SELECT MAX(base_date) AS base_date
    FROM normalized_stock_prices_daily
)
SELECT COUNT(DISTINCT p.symbol) AS covered_symbols
FROM normalized_stock_prices_daily p
JOIN stocks_master m
  ON m.symbol = p.symbol
WHERE p.base_date = (SELECT base_date FROM latest_price)
  AND m.market IN ('KOSPI', 'KOSDAQ');
```

주의:
- 이 문서는 report_stock_daily가 생성한 수정 요청 문서이며, report 저장소가 DB를 직접 수정하지 않는다.
