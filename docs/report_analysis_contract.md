# Report Analysis Contract

## 목적

- `Morning`:
  밤사이 글로벌 변수와 한국장 예상 영향을 연결하고, 오늘 볼 섹터·관심종목·리스크 조건을 제시한다.
- `Regular`:
  오전 View와 실제 장중 데이터를 비교해 유지/일부 유지/약화/확인 제한을 판정한다. KIS 거래량 상위와 관심종목 반응을 연결한다.
- `Closing`:
  오늘 움직임이 추세인지 단기 이벤트인지 복기하고, 다음 거래일의 공격적 조건/보수적 조건/필수 확인 데이터를 제시한다.

## 분석 프레임

- 매크로: `USD/KRW`, `DXY`, `US10Y`, `US3Y`, `10Y-3Y spread`, `Nasdaq`, `SOX`, `VIX`, `Brent/WTI`
- 수급·모멘텀: `KIS 거래량 순위`, 관심종목 가격, 등락률, score, 거래대금/feature
- 섹터: 반도체, 2차전지, 증권, 소재, 조선/LNG 등 현재 입력 데이터와 연결 가능한 범위
- 리스크: 환율, 금리, 유가, 과열, `source_mixed`, `stale`, 데이터 부재

## Gemini 사용 원칙

- Python이 숫자, 가격, 등락률, rank, score, source quality, allowed/blocked sections를 확정한다.
- Gemini는 숫자·가격·등락률·rank·score를 새로 만들거나 수정하지 않는다.
- Gemini는 전체 리포트를 자유 작성하지 않고, 세션별 해석 문장만 JSON으로 반환한다.
- Gemini는 입력된 데이터 범위 안에서만 해석 문장을 보강한다.
- Gemini 실패 시 리포트 생성과 Telegram 발송은 중단되지 않고 rule-based 결과로 fallback한다.

## 금지 사항

- 입력에 없는 외국인 선물, 프로그램 매매, 실시간 뉴스, 공시를 만들지 말 것
- KIS 거래량 순위를 전체시장 Top처럼 표현하지 말 것
- 전체시장 거래대금 Top / 전체시장 시총 Top을 임의 생성하지 말 것
- `BUY / SELL / HOLD / 목표가 / 매수추천 / 매도추천` 금지
- 같은 문장을 여러 종목에 반복하지 말 것
