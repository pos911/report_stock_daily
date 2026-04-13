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

    def generate_report(self, quant_data_json, news_text):
        """
        Generates a markdown report using Gemini based on quant data and news.
        """
        prompt = f"""
[Quant Data from Supabase]
제공 데이터에는 다음 테이블 정보가 포함되어 있습니다:
- normalized_global_macro_daily / market_breadth_daily (매크로/마켓 지표)
- normalized_stock_short_selling (공매도 지표: short_ratio 등)
- normalized_stock_fundamentals_ratios (밸류에이션: PER, PBR 등)
- normalized_stock_supply_daily (수급: pension_net_buy, corporate_net_buy 등)
- feature_store_daily (상세 퀀트 팩터)
* 모든 종목 데이터에는 'stock_name' 필드가 포함되어 있으니 가독성을 위해 적극 활용하세요.

데이터 상세:
{json.dumps(quant_data_json, indent=2, ensure_ascii=False)}

[News Text from Google Docs]
{news_text}

위 데이터를 바탕으로 리포트를 작성해줘.
"""
        
        response = self.model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
            ),
        )
        
        return response.text
