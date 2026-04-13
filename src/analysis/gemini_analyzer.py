import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiAnalyzer:
    def __init__(self, key_config_path="config/api_keys.json", settings_config_path="config/analyzer_settings.json"):
        """
        Initialize Gemini API and load model settings.
        Priority for API Key: 1. Env Var, 2. api_keys.json
        Priority for Settings: 1. analyzer_settings.json, 2. Env Vars, 3. Defaults
        """
        load_dotenv()
        
        # 1. Load API Key (Priority: api_keys.json > Env Var)
        self.api_key = None
        if os.path.exists(key_config_path):
            with open(key_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                # Check nested 'gemini' or root 'gemini_api_key'
                self.api_key = config.get("gemini", {}).get("api_key") or config.get("gemini_api_key")
        
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")

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
거시 지표와 뉴스를 바탕으로 코스피/코스닥/선물옵션 시장의 시황을 상세히 분석하라. 
단순 값 나열이 아닌, 데이터가 의미하는 향후 시장 방향성을 논리적으로 서술하라.
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
앞서 분석된 시황(Market Summary Context)을 배경으로 다음을 수행하라. 
첫째, 코스피/코스닥/ETF 거래량 상위 5종목의 퀀트 지표를 짧게 평가하라. 
둘째, 타겟 관심 종목(Target Stocks)에 대해서는 수급(연기금/외국인), 밸류에이션(PER/PBR), 공매도 데이터를 모두 활용하여 '심층적인 투자 뷰(Z-score 및 시그널 포함)'를 개별적으로 작성하라. 
결측치는 언급하지 말고 있는 데이터를 중심으로 분석하라.
마크다운 형식으로 작성해줘.
"""
        response = self.model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
            ),
        )
        return response.text
