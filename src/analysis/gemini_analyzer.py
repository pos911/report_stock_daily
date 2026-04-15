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
        # 1. Load API Key using unified config loader
        self.api_key = config.get("api_key", section="gemini")
        
        if not self.api_key:
            raise ValueError("Gemini API Key must be provided via config/api_keys.json or GEMINI_API_KEY environment variable.")

        genai.configure(api_key=self.api_key)

        # 2. Load Model Settings
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        self.system_instruction = os.getenv("GEMINI_SYSTEM_INSTRUCTION")

        if os.path.exists(settings_config_path):
            with open(settings_config_path, "r", encoding="utf-8") as f:
                settings = json.load(f).get("gemini", {})
                self.model_name = settings.get("model_name", self.model_name)
                self.system_instruction = settings.get("system_instruction", self.system_instruction)

        if not self.system_instruction:
            # Minimal fallback if no instruction provided
            self.system_instruction = "You are a financial analyst."

        # Initialize the model with system instruction
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction
        )

    def generate_market_summary(self, macro_data, market_breadth, news_text, generation_time):
        """
        [Step 1] Generates a market summary analyzing macro factors, market breadth, and news.
        """
        prompt = f"""
[Report Generation Info]
- 작성 시간: {generation_time}

[거시 지표 / Market Breadth Data]
{json.dumps({"macro_data": macro_data, "market_breadth": market_breadth}, indent=2, ensure_ascii=False)}

[News Text from Google Docs]
{news_text}

[System Instruction]
거시 지표와 뉴스를 바탕으로 코스피/코스닥/선물옵션 시장의 전체적인 매크로 시황만 상세히 분석하라. 
종목별 지표(PER, PBR, 수급, 공매도 등)나 개별 주식에 대한 언급은 이 단계에서 절대 하지 마라. 오직 거시경제(환율, 금리, 유가 등)와 뉴스 요약에만 집중하라.
단순 값 나열이 아닌, 데이터가 의미하는 향후 시장 방향성을 논리적으로 서술하라.
오직 '거시경제/뉴스 요약' 내용만 단일 섹션으로 생성하며, 인사말/전체 서론/전체 결론은 절대 금지한다.
# 이나 ## 같은 최상위/대분류 마크다운 헤더는 파이썬에서 직접 조립할 예정이므로 이곳에서는 생성하지 마라 (### 부터 사용할 것).
"데이터가 없다"는 식의 불필요한 안내 문구는 절대 출력하지 말고 결측치는 조용히 누락(Skip)하라.
If any requested data is missing, DO NOT mention that it is missing. Do not say 'Data is not provided' or 'Cannot analyze due to lack of data'. Simply skip it and analyze only what is available.
마크다운 형식으로 작성해줘.
"""
        response = self.model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
            ),
        )
        return response.text

    def generate_stock_analysis(self, market_summary, top_volume_data, target_stocks_data, generation_time):
        """
        [Step 2] Generates stock-specific deep dives using the market summary as context.
        """
        prompt = f"""
[Report Generation Info]
- 작성 시간: {generation_time}

[Market Summary Context]
{market_summary}

[Top Volume Stocks (KOSPI/KOSDAQ/ETF)]
{json.dumps(top_volume_data, indent=2, ensure_ascii=False)}

[Target Stocks Data]
데이터 포함: supply, fundamentals, short selling, feature store
{json.dumps(target_stocks_data, indent=2, ensure_ascii=False)}

[System Instruction]
앞서 분석된 시황(Market Summary Context)을 바탕으로 아래 내용을 수행하라.
인사말이나 전체 서론, 결론은 절대 생성하지 말고 곧바로 본론(분석 내용)만 출력하라.
파이썬 코드 단에서 타이틀을 조립하므로, # 이나 ## 같은 최상위 마크다운 헤더는 생성하지 마라 (필요 시 ### 부터 사용).

첫째, 코스피/코스닥/ETF 거래량 상위 최대 20종목의 퀀트 지표를 짧고 핵심만 평가하라. 종목별로 소속 시장(KOSPI, KOSDAQ, ETF)을 명시하라.
둘째, 타겟 관심 종목(Target Stocks) 분석 시 나열식 설명을 완전히 폐기하고, 반드시 종목별로 다음 '3단계 개조식(Bullet points)' 뷰(View) 구조로만 답변을 강제하라:
  - 1) 공격적인 포인트
  - 2) 최대한 보수적인 포인트
  - 3) 최종 결론 (BUY / HOLD / SELL)

어떤 경우에도 "현재 데이터가 제공되지 않아..." 혹은 "데이터가 부족하여..." 등의 문구는 절대 출력하지 마라.
If a specific metric (e.g., PER, PBR, Z-score, supply data) is missing for a stock, DO NOT write 'N/A' or 'Not available'. Completely omit any mention of that metric and base your analysis only on the provided data.
분석 데이터가 없는 부분이나 누락된 종목은 설명 없이 완전히 무시(Silent Skip)하고, 해당 항목에 대한 마크다운 제목조차 출력하지 마라.
마크다운 형식으로 작성해줘.
"""
        response = self.model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
            ),
        )
        return response.text
