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

    def generate_top_volume_analysis(self, top_volume_data, report_type: str = "regular"):
        """
        [Step 2] 거래량 상위 종목 및 '스마트 머니' 전용 분석.
        집중력 분산을 막기 위해 별도 메서드로 분리함.
        """
        prompt = f"""
[Report Generation Info]
- 리포트 유형: {report_type.upper()}

[Top Volume Stocks (KOSPI/KOSDAQ/ETF) — enriched with zscore/per/pbr]
{json.dumps(top_volume_data, indent=2, ensure_ascii=False)}

[System Instruction]
거래량 상위 종목을 바탕으로 '스마트 머니'의 흐름을 날카롭게 포착하라. 

### 필히 포함되어야 할 분석 룰:
1. **거래량 상위 종목 퀀트 평가**: KOSPI/KOSDAQ/ETF 별로 상위 종목들의 수급 강도와 기술적 상태(`moving_avg_20`, `return_5d`)를 짧고 강렬하게 평가하라.
2. **스마트 머니 포착 (Smart Money Detection) ⭐**: 
   - `foreign_flow_zscore`가 높고(양수), `per` 또는 `pbr`이 낮은 종목이 있다면 이를 `<외인/기관 강한 매집 우량주>`로 강력 추천 섹션으로 분류하라.
   - 조건에 부합하는 종목이 전혀 없다면 이 분석 섹션 자체를 조용히 생략(Silent Skip)하라.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
        return self._call_model(prompt, temperature=0.7)

    def generate_stock_analysis(self, market_summary, target_stocks_data, macro_data, generation_time,
                                report_type: str = "regular"):
        """
        [Step 3] 타겟 관심 종목 심층 분석.
        오직 개별 타겟 종목에 대한 3단계 개조식 전략만 생성함.
        """
        base_block = f"""
[Report Generation Info]
- 작성 시간: {generation_time}
- 리포트 유형: {report_type.upper()}

[Market Summary Context]
{market_summary}

[Macro Data Context (normalized_macro_series)]
{json.dumps(macro_data, indent=2, ensure_ascii=False)}

[Target Stocks Data]
{json.dumps(target_stocks_data, indent=2, ensure_ascii=False)}
"""

        intelligent_instruction = """
## [중요] 타겟 종목 지능형 분석 지침
- **매크로 연계 전략**: 현재의 `Macro Data Context`(환율, 금리 trend)를 바탕으로 각 종목의 대응 전략을 수립하라.
  - 예: 고금리 유지 전망 시 부채비율이 높은 성장주는 보수적(SELL/HOLD)으로, 수출 비중이 높고 환율 수혜가 예상되는 종목은 공격적(BUY)으로 평가.
- **3단계 개조식(Bullet points) 구조 강제**:
  1. 🔴 공격적인 포인트 (매수 근거, 모멘텀, 수급 강도)
  2. 🔵 보수적인 포인트 (리스크, 밸류에이션 부담, 하방 요소)
  3. ⚖️ 최종 결론: **BUY / HOLD / SELL** (명확한 판단)
"""

        prompt = base_block + intelligent_instruction + self._build_silent_skip_rules() + "\n마크다운 형식으로 작성해줘.\n"
        return self._call_model(prompt, temperature=0.7)

        prompt = base_block + type_instruction + self._build_silent_skip_rules() + "\n마크다운 형식으로 작성해줘.\n"
        return self._call_model(prompt, temperature=0.7)
