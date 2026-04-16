import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

from src.utils import config

class GeminiAnalyzer:
    def __init__(self, settings_config_path="config/analyzer_settings.json"):
        """
        Initialize Gemini API and load model settings.
        Priority for API Key: 1. Env Var, 2. config/api_keys.json
        Priority for Settings: 1. analyzer_settings.json, 2. Env Vars, 3. Defaults
        """
        self.api_key = config.get("api_key", section="gemini")

        if not self.api_key:
            raise ValueError("Gemini API Key must be provided via config/api_keys.json or GEMINI_API_KEY environment variable.")

        genai.configure(api_key=self.api_key)

        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        self.system_instruction = os.getenv("GEMINI_SYSTEM_INSTRUCTION")

        if os.path.exists(settings_config_path):
            with open(settings_config_path, "r", encoding="utf-8") as f:
                settings = json.load(f).get("gemini", {})
                self.model_name = settings.get("model_name", self.model_name)
                self.system_instruction = settings.get("system_instruction", self.system_instruction)

        if not self.system_instruction:
            self.system_instruction = "You are a financial analyst."

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_silent_skip_rules(self):
        """공통 Silent Skip 규칙 텍스트 반환."""
        return """
## 공통 엄격 금지 사항 (Silent Skip 원칙)
- "데이터가 없습니다", "데이터가 부족하여", "제공되지 않아", "N/A", "Not available" 등 절대 금지.
- 데이터가 없는 섹션/지표/종목은 **아무 언급 없이 조용히 생략(Silent Skip)**.
- 마크다운 최상위 헤더 `#`, `##` 금지 (### 부터 사용).
- 인사말·전체 서론·전체 결론 금지 — 곧바로 본론만 출력.
- If any metric or section has no data, do not mention it. Simply omit and analyze only what is available.
"""

    def _call_model(self, prompt: str, temperature: float = 0.7) -> str:
        """Gemini API 호출 공통 래퍼."""
        response = self.model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        return response.text

    # -------------------------------------------------------------------------
    # Step 1: Market Summary (시간대별 분기)
    # -------------------------------------------------------------------------

    def generate_market_summary(self, macro_data, market_breadth, news_text,
                                generation_time, report_type: str = "regular"):
        """
        [Step 1] 매크로 시황 분석.

        report_type:
          - 'morning'  : 미국 야간 변화율 기반 Risk-On/Off + 시초가 대응 전략
          - 'closing'  : 장 마감 후 거래량/외인 Z-Score 기반 매집주 포착
          - 'regular'  : 기본 매크로 종합 분석
        """
        base_data_block = f"""
[Report Generation Info]
- 작성 시간: {generation_time}
- 리포트 유형: {report_type.upper()}

[거시 지표 (normalized_macro_series)]
{json.dumps(macro_data, indent=2, ensure_ascii=False)}

[시장 폭 (market_breadth_daily)]
{json.dumps(market_breadth, indent=2, ensure_ascii=False)}

[모멘텀 변화율 피처 (feature_store_daily: macro_ / _1d_chg / _5d_chg)]
※ 이 섹션의 데이터가 핵심입니다. 반드시 분석하세요.
"""

        if report_type == "morning":
            type_instruction = """
## [Morning Report 07:00 KST] 핵심 분석 가이드

### 필수 항목 1 — 미국 시장 야간 변화율 분석 (최우선)
- `_1d_chg` (1일 변화율), `_5d_chg` (5일 변화율) 피처를 최우선으로 분석하라.
- 특히 미국 주요 지수(S&P 500, NASDAQ, 다우), 달러인덱스(DXY), 미국채 10년물 금리의 변화율에 집중하라.
- "어제 밤(미국 시간) 주요 자산의 움직임이 오늘 한국 시장에 어떤 영향을 줄 것인가"를 명확히 서술하라.

### 필수 항목 2 — Risk-On / Risk-Off 체제 판정
- 위 분석을 토대로 오늘 한국 시장이 **Risk-On** 인지 **Risk-Off** 인지 **반드시 단정하여** 명시하라.
- 근거를 한 문장으로 요약하라. 예: "NASDAQ +1.8%, 달러 약세 → Risk-On 개장 예상"

### 필수 항목 3 — 시초가 대응 전략
- 오늘 코스피/코스닥 시초가 방향성(갭업/갭다운/보합)을 예측하고,
  투자자가 시초가에서 취해야 할 구체적 대응 전략(매수 접근 또는 관망 등)을 제시하라.
"""
        elif report_type == "closing":
            type_instruction = """
## [Closing Report 15:30 KST] 핵심 분석 가이드

### 필수 항목 1 — 오늘 장 마감 매크로 정리
- 오늘 장 중 매크로 지표(환율, 금리, 상품가격 등)의 변화를 간결하게 요약하라.

### 필수 항목 2 — Risk-On / Risk-Off 체제 평가
- 오늘 장 마감 기준으로 **Risk-On** 인지 **Risk-Off** 인지 판정하고 근거를 명시하라.

### 필수 항목 3 — 다음 날 시장 전망
- 오늘 마감 데이터를 기반으로 내일 장 방향성과 주의해야 할 리스크 요인을 간략히 제시하라.
"""
        else:  # regular
            type_instruction = """
## [Regular Report] 종합 매크로 분석 가이드

### 필수 항목 1 — 매크로 체제(Regime) 판단
- 환율(USD/KRW)과 금리의 `_1d_chg`, `_5d_chg`를 최우선으로 분석하라.
- **Risk-On** 또는 **Risk-Off** 체제를 반드시 명시하고 근거를 서술하라.

### 필수 항목 2 — 거시경제 데이터 분석
- 시장 방향성을 단순 수치 나열이 아닌 논리적 서사(narrative)로 작성하라.
"""

        news_block = f"""
[News Text from Google Docs]
{news_text}
"""

        prompt = base_data_block + news_block + type_instruction + self._build_silent_skip_rules() + "\n마크다운 형식으로 작성해줘.\n"
        return self._call_model(prompt, temperature=0.7)

    # -------------------------------------------------------------------------
    # Step 2: Stock Analysis (시간대별 분기)
    # -------------------------------------------------------------------------

    def generate_stock_analysis(self, market_summary, top_volume_data,
                                target_stocks_data, generation_time,
                                report_type: str = "regular"):
        """
        [Step 2] 종목 심층 분석.

        report_type:
          - 'morning'  : 오늘 주목할 선매수 후보 발굴
          - 'closing'  : 거래량이 터진 종목 중 외인 Z-Score 매집주 포착
          - 'regular'  : 기본 퀀트 분석 + 3단계 개조식
        """
        base_block = f"""
[Report Generation Info]
- 작성 시간: {generation_time}
- 리포트 유형: {report_type.upper()}

[Market Summary Context]
{market_summary}

[Top Volume Stocks (KOSPI/KOSDAQ/ETF) — volume_value 키 포함]
{json.dumps(top_volume_data, indent=2, ensure_ascii=False)}

[Target Stocks Data]
데이터 포함: supply, fundamentals(market_cap/per/pbr), short selling, feature store(_1d_chg/_5d_chg)
{json.dumps(target_stocks_data, indent=2, ensure_ascii=False)}
"""

        if report_type == "morning":
            type_instruction = """
## [Morning Report 07:00 KST] 종목 분석 가이드

### 분석 1 — 오늘 시초가 주목 종목 선별
- 어제 종가 대비 `_1d_chg`가 큰 종목을 우선 선별하라.
- 시초가에서 매수를 고려할 만한 모멘텀 종목을 근거와 함께 제시하라.

### 분석 2 — 스마트 머니 선 포착 (Smart Money Detection) ⭐
거래량 상위 종목 중 아래 두 조건을 **동시에** 충족하는 종목을 `<외인/기관 강한 매집 우량주>` 레이블로 특별 강조하라:
- 조건 A: 외국인 수급 Z-Score(`foreign_net_zscore` 또는 유사 피처)가 **양수이고 높은 값**
- 조건 B: PER 또는 PBR이 **시장 평균 대비 낮음** (밸류에이션 매력)
조건 미충족 시 이 섹션 전체를 Silent Skip.

### 분석 3 — 타겟 관심 종목 아침 브리핑
각 종목별로 3단계 개조식 구조로 작성하라:
1. 🔴 공격적인 포인트 (모멘텀, 수급 강도)
2. 🔵 보수적인 포인트 (리스크, 하방 요소)
3. ⚖️ 시초가 대응: **매수 접근 / 관망 / 회피**
"""
        elif report_type == "closing":
            type_instruction = """
## [Closing Report 15:30 KST] 종목 분석 가이드

### 분석 1 — 오늘 거래량 폭발 종목 포착 (핵심)
- `volume_value`가 높은 종목 중 오늘 특별히 거래량이 터진 종목을 식별하라.
- 각 종목의 거래량 급증 원인(모멘텀, 뉴스, 수급)을 추정하여 서술하라.

### 분석 2 — 외인 수급 Z-Score 매집주 포착 ⭐ (핵심)
오늘 장 마감 기준, 거래량 상위 종목 중 아래 조건을 **동시에** 충족하는 종목을
`<외인/기관 강한 매집 우량주>` 레이블로 반드시 별도 강조 섹션으로 작성하라:
- 조건 A: `foreign_net_zscore` 또는 유사 외인 수급 Z-Score가 **양수이고 높은 값**
- 조건 B: `per` 또는 `pbr`이 **낮아 밸류에이션 매력**이 있는 종목
→ 이 종목들은 내일 추가 매수 가능성이 높은 스마트 머니 신호이다.
조건 미충족 시 Silent Skip.

### 분석 3 — 타겟 관심 종목 마감 결산
각 종목별로 3단계 개조식 구조로 작성하라:
1. 🔴 오늘 매수 근거 (수급, 모멘텀)
2. 🔵 리스크 및 보수적 관점
3. ⚖️ 내일 전략: **BUY / HOLD / SELL**
"""
        else:  # regular
            type_instruction = """
## [Regular Report] 종목 분석 가이드

### 분석 1 — 거래량 상위 종목 퀀트 평가
KOSPI / KOSDAQ / ETF 거래량 상위 종목의 퀀트 지표를 짧고 핵심만 평가하라.
종목별로 소속 시장(KOSPI, KOSDAQ, ETF)을 명시하라.

### 분석 2 — 스마트 머니 포착 ⭐
외인 수급 Z-Score 높고 PER/PBR 낮은 종목을 `<외인/기관 강한 매집 우량주>`로 특별 언급.
조건 미충족 시 Silent Skip.

### 분석 3 — 타겟 관심 종목 심층 분석
각 종목별로 3단계 개조식 구조로 작성하라:
1. 🔴 공격적인 포인트
2. 🔵 보수적인 포인트
3. ⚖️ 최종 결론: **BUY / HOLD / SELL**
"""

        prompt = base_block + type_instruction + self._build_silent_skip_rules() + "\n마크다운 형식으로 작성해줘.\n"
        return self._call_model(prompt, temperature=0.7)
