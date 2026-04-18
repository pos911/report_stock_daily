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

    def generate_market_summary(self, macro_data, market_breadth, momentum_data, data_guardrails, news_text,
                                generation_time, report_type: str = "regular"):
        """
        [Step 1] 매크로 시황 분석.
        'momentum_data'를 추가 수신하여 글로벌 피처(GLOBAL)의 모멘텀 변화율을 분석함.
        """
        base_data_block = f"""
[Report Generation Info]
- 작성 시간: {generation_time}
- 리포트 유형: {report_type.upper()}

[거시 지표 (normalized_macro_series)]
{json.dumps(macro_data, indent=2, ensure_ascii=False)}

[시장 폭 (market_breadth_daily)]
{json.dumps(market_breadth, indent=2, ensure_ascii=False)}

[모멘텀 변화율 피처 (feature_store_daily: symbol=GLOBAL)]
{json.dumps(momentum_data, indent=2, ensure_ascii=False)}

[Data Quality Guardrails]
{json.dumps(data_guardrails, indent=2, ensure_ascii=False)}
"""

        type_instruction = f"""
## [필수] 분석 가이드 (Global-to-Local Impact)
1. **해외 주요 경제 뉴스 분석**: 아래 `[News Text from Google Docs]` 섹션의 내용은 **해외/미국 주요 경제 뉴스 및 글로벌 시황**이다. 이 정보가 가장 중요한 외부 변수임을 명심하라.
2. **글로벌 모멘텀 결합**: 위 뉴스 텍스트와 `[모멘텀 변화율 피처]` 데이터(환율, 금리 등의 _1d_chg, _5d_chg)를 결합하여, 현재 글로벌 매크로 환경이 오늘 한국 시장에 미칠 구체적인 영향력을 서술하라.
3. **Risk-On/Off 판정**: 해외 뉴스와 지표 변화율을 종합하여 현재 시장의 심리를 Risk-On 또는 Risk-Off로 명확히 판정하라.
"""

        news_block = f"""
[News Text from Google Docs] - 해외 주요 경제 뉴스 및 글로벌 시황
{news_text}
"""

        prompt = base_data_block + news_block + type_instruction + self._build_silent_skip_rules() + "\n마크다운 형식으로 작성해줘.\n"
        return self._call_model(prompt, temperature=0.7)

    # -------------------------------------------------------------------------
    # Step 2: Top Volume Analysis (기존 유지)
    # -------------------------------------------------------------------------

    def generate_top_volume_analysis(self, top_volume_data, report_type: str = "regular"):
        """
        [Step 2] 거래량 상위 종목 및 '스마트 머니' 전용 분석.
        """
        prompt = f"""
[Report Generation Info]
- 리포트 유형: {report_type.upper()}

[Top Volume Stocks (KOSPI/KOSDAQ/ETF) — enriched with zscore/per/pbr]
{json.dumps(top_volume_data, indent=2, ensure_ascii=False)}

[System Instruction]
거래량 상위 종목을 바탕으로 '스마트 머니'의 흐름을 포착하라.

### 분석 룰:
1. **스마트 머니 포착 ⭐**: 
   - `foreign_flow_zscore`가 높고, `per` 또는 `pbr`이 낮은 종목을 강조.
   - 종목별로 소속 섹터가 현재 글로벌 매크로 트렌드(AI, 미국 국채 금리 민감도 등)와 부합하는지 짧게 언급하라.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
        return self._call_model(prompt, temperature=0.7)

    def generate_stock_analysis(self, market_summary, target_stocks_data, macro_market_data, generation_time,
                                data_guardrails=None, report_type: str = "regular"):
        """
        [Step 3] 타겟 관심 종목 심층 분석.
        'macro_market_data'를 수신하여 매크로/글로벌 시황과 연계 분석함.
        """
        macro_data = macro_market_data.get("normalized_macro_series")
        global_macro_data = macro_market_data.get("normalized_global_macro_daily")
        momentum_data = macro_market_data.get("momentum")

        base_block = f"""
[Report Generation Info]
- 작성 시간: {generation_time}
- 리포트 유형: {report_type.upper()}

[Market Summary Context]
{market_summary}

[Macro/Global Context Information]
- Macro Series: {json.dumps(macro_data, indent=1, ensure_ascii=False)}
- Global Macro Daily: {json.dumps(global_macro_data, indent=1, ensure_ascii=False)}
- Global Momentum: {json.dumps(momentum_data, indent=1, ensure_ascii=False)}
- Data Guardrails: {json.dumps(data_guardrails or {}, indent=1, ensure_ascii=False)}

[Target Stocks Data - Supply/Fundamentals/Features]
{json.dumps(target_stocks_data, indent=2, ensure_ascii=False)}
"""

        intelligent_instruction = """
## [중요] 타겟 종목 지능형 분석 지침
1. **연기금 수급 해석**: `normalized_stock_supply_daily` 테이블에서 `pension_net_buy`가 유의미한 양수를 기록할 경우, 이를 **스마트 머니의 추세적 매집**으로 규정하고 긍정적 요인으로 분석하라.
2. **글로벌 트렌드 연결**: 거래량 상위 종목이나 타겟 종목의 섹터(예: 반도체, 2차전지, 자동차)가 현재 해외 뉴스와 매크로 지표에서 나타나는 유망 섹터와 연결되는지 분석하라.
3. **3단계 개조식 구조**:
  1. 🔴 공격적인 포인트 (매수 근거, 모멘텀, 연기금/외인 수입)
  2. 🔵 보수적인 포인트 (리스크, 글로벌 매크로 압박)
  3. ⚖️ 최종 결론: **BUY / HOLD / SELL**
"""

        prompt = base_block + intelligent_instruction + self._build_silent_skip_rules() + "\n마크다운 형식으로 작성해줘.\n"
        return self._call_model(prompt, temperature=0.7)
