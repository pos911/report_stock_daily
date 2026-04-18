import json
import os
import time

import google.generativeai as genai
from dotenv import load_dotenv

from src.utils import config


load_dotenv()


class GeminiAnalyzer:
    MAX_RETRIES = 4
    BASE_RETRY_SECONDS = 5

    def __init__(self, settings_config_path="config/analyzer_settings.json"):
        """
        Initialize Gemini API and load model settings.
        Priority for API Key: 1. Env Var, 2. config/api_keys.json
        Priority for Settings: 1. analyzer_settings.json, 2. Env Vars, 3. Defaults
        """
        self.api_key = config.get("api_key", section="gemini")

        if not self.api_key:
            raise ValueError(
                "Gemini API Key must be provided via config/api_keys.json "
                "or GEMINI_API_KEY environment variable."
            )

        genai.configure(api_key=self.api_key)

        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        self.system_instruction = os.getenv("GEMINI_SYSTEM_INSTRUCTION")

        if os.path.exists(settings_config_path):
            with open(settings_config_path, "r", encoding="utf-8") as f:
                settings = json.load(f).get("gemini", {})
                self.model_name = settings.get("model_name", self.model_name)
                self.system_instruction = settings.get(
                    "system_instruction", self.system_instruction
                )

        if not self.system_instruction:
            self.system_instruction = "You are a financial analyst."

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction,
        )

    def _build_silent_skip_rules(self):
        """Common silent-skip rules shared across prompts."""
        return """
## 공통 엄격 금지 사항 (Silent Skip 원칙)
- "데이터가 없습니다", "데이터가 부족하여", "제공되지 않아", "N/A", "Not available" 등은 절대 금지.
- 데이터가 없는 섹션/지표/종목은 아무 언급 없이 조용히 생략(Silent Skip).
- 마크다운 최상위 헤더 `#`, `##` 금지 (### 부터 사용).
- 인사말, 장황한 서론, 전체 결론 금지. 바로 본론만 작성.
- If any metric or section has no data, do not mention it. Simply omit and analyze only what is available.
"""

    def _call_model(self, prompt: str, temperature: float = 0.7) -> str:
        """Shared Gemini API wrapper."""
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.model.generate_content(
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature
                    ),
                )
                return response.text
            except Exception as exc:
                last_error = exc
                error_text = str(exc)
                is_retryable = "429" in error_text or "ResourceExhausted" in error_text
                if not is_retryable or attempt == self.MAX_RETRIES:
                    raise

                sleep_seconds = self.BASE_RETRY_SECONDS * (2 ** (attempt - 1))
                print(
                    f"Warning: Gemini call retry {attempt}/{self.MAX_RETRIES} "
                    f"after {sleep_seconds}s due to: {exc}"
                )
                time.sleep(sleep_seconds)

        raise last_error

    def generate_market_summary(
        self,
        macro_data,
        market_breadth,
        momentum_data,
        data_guardrails,
        news_text,
        generation_time,
        report_type: str = "regular",
    ):
        """
        [Step 1] Market regime and macro summary.
        """
        prompt = f"""
[Report Generation Info]
- 작성 시각: {generation_time}
- 리포트 유형: {report_type.upper()}

[거시 지표]
{json.dumps(macro_data, indent=2, ensure_ascii=False)}

[시장 폭]
{json.dumps(market_breadth, indent=2, ensure_ascii=False)}

[모멘텀 변화율]
{json.dumps(momentum_data, indent=2, ensure_ascii=False)}

[Data Quality Guardrails]
{json.dumps(data_guardrails, indent=2, ensure_ascii=False)}

[News Context]
{news_text}

[System Instruction]
오늘 시장의 큰 방향만 간결하게 요약하라.

### 출력 구조
- `### 시장 한줄 요약`
- `### 핵심 포인트`
- `### 오늘의 시장 판단`

### 작성 규칙
1. 긴 숫자 나열보다 해석 중심으로 써라.
2. 핵심 포인트는 3개 안팎의 bullet로 제한하라.
3. 뉴스는 시장 방향 설명에 꼭 필요한 부분만 짧게 반영하라.
4. `### 오늘의 시장 판단`에는 반드시 `Risk-On`, `Risk-Off`, `중립` 중 하나를 명시하라.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
        return self._call_model(prompt, temperature=0.6)

    def generate_news_summary(self, news_text, report_type: str = "regular"):
        """
        [Step 2] News summary and investment implications.
        """
        prompt = f"""
[Report Generation Info]
- 리포트 유형: {report_type.upper()}

[News Text from Google Docs]
{news_text}

[System Instruction]
해외 주요 뉴스만 짧고 명확하게 정리하라.

### 출력 구조
- `### 뉴스 요약`
- `### 섹터 영향`
- `### 투자 시사점`

### 작성 규칙
1. 뉴스 요약은 3개 안팎의 핵심 이슈만 남겨라.
2. 섹터 영향은 국내 투자자가 바로 이해할 수 있게 업종 중심으로 정리하라.
3. 투자 시사점은 실행 가능한 관점으로 2~3개만 제시하라.
4. 문장은 짧게 쓰고, 사건의 의미를 해석 위주로 전달하라.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
        return self._call_model(prompt, temperature=0.5)

    def generate_top_volume_analysis(self, top_volume_data, report_type: str = "regular"):
        """
        [Step 3] Top-volume names and smart-money angle.
        """
        prompt = f"""
[Report Generation Info]
- 리포트 유형: {report_type.upper()}

[Top Volume Stocks]
{json.dumps(top_volume_data, indent=2, ensure_ascii=False)}

[System Instruction]
거래대금 상위 종목에서 오늘 눈에 띄는 이름만 골라 간결하게 정리하라.

### 출력 구조
- `### 오늘 눈에 띄는 종목`
- 종목별 bullet: `- 종목명(코드): 한 줄 요약`
- `### 한줄 시사점`

### 작성 규칙
1. 긴 서론은 금지한다.
2. `foreign_flow_zscore`, 수익률, 밸류에이션 중 의미 있는 근거만 짧게 사용하라.
3. 업종 흐름이나 테마 연결은 한 문장 이내로 제한하라.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
        return self._call_model(prompt, temperature=0.6)

    def generate_stock_analysis(
        self,
        market_summary,
        target_stocks_data,
        macro_market_data,
        generation_time,
        data_guardrails=None,
        report_type: str = "regular",
    ):
        """
        [Step 4] Focus-stock analysis with bullish/bearish/final verdict structure.
        """
        macro_data = macro_market_data.get("normalized_macro_series")
        global_macro_data = macro_market_data.get("normalized_global_macro_daily")
        momentum_data = macro_market_data.get("momentum")

        prompt = f"""
[Report Generation Info]
- 작성 시각: {generation_time}
- 리포트 유형: {report_type.upper()}

[Market Summary Context]
{market_summary}

[Macro/Global Context Information]
- Macro Series: {json.dumps(macro_data, indent=1, ensure_ascii=False)}
- Global Macro Daily: {json.dumps(global_macro_data, indent=1, ensure_ascii=False)}
- Global Momentum: {json.dumps(momentum_data, indent=1, ensure_ascii=False)}
- Data Guardrails: {json.dumps(data_guardrails or {}, indent=1, ensure_ascii=False)}

[Target Stocks Data]
{json.dumps(target_stocks_data, indent=2, ensure_ascii=False)}

[System Instruction]
관심 종목을 종목별로 매우 실용적으로 정리하라.

### 종목별 고정 형식
1. `1) 공격적인 포인트`
2. `2) 최대한 보수적인 포인트`
3. `3) 최종 결론 (BUY/HOLD/SELL)`

### 작성 규칙
1. 공격적인 포인트에는 모멘텀, 수급, 밸류에이션, 업황 중 강점만 압축해 적어라.
2. 보수적인 포인트에는 리스크, 데이터 지연 가능성, 업황 역풍을 적어라.
3. 결론은 종목마다 하나만 명시하라.
4. 종목별 장문 서론은 금지하고 바로 핵심만 적어라.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
        return self._call_model(prompt, temperature=0.6)
