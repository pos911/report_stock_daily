import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiAnalyzer:
    def __init__(self, config_path="config/api_keys.json"):
        """
        Initialize Gemini API by reading from environment variables or a JSON file.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.api_key = config.get("gemini_api_key")

        if not self.api_key:
            raise ValueError("Gemini API Key must be provided via Env or JSON.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def generate_report(self, quant_data_json, news_text):
        """
        Generates a markdown report using Gemini based on quant data and news.
        """
        system_instruction = (
            "너는 최고 수준의 퀀트 애널리스트야. 제공된 정량적 데이터(Supabase)와 정성적 뉴스(Google Docs)를 결합하여 마크다운 리포트를 작성해. "
            "반드시 다음의 금융 공학 이론을 적용해서 뷰를 작성할 것:\n"
            "1. Fama-French 요인 모델 및 한국 주식 시장의 반전(Reversal) 전략 관점.\n"
            "2. 외국인/기관 순매수 Z-Score와 이동평균선(MA)을 활용한 기술적 매수/매도 시그널 판단.\n"
            "3. VIX, 하이일드 스프레드, 구리/금 비율, 그리고 코스피 시장 전체의 등락 종목 수(Market Breadth)를 바탕으로 한 마켓 타이밍 및 현금 비중(Risk) 조절 전략.\n"
            "결과는 서론-매크로 분석-퀀트 시그널-최종 투자 전략(포트폴리오 비중) 순서로 상세히 작성해."
        )

        prompt = f"""
[Quant Data from Supabase]
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
            # Optional: system_instruction can be passed during model initialization 
            # but for simplicity in this script we include it in the prompt or as a separate instruction if supported.
        )
        
        # If the model was initialized with system_instruction, we don't need to repeat it.
        # However, to be safe and follow user instructions exactly:
        model_with_sys = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            system_instruction=system_instruction
        )
        response = model_with_sys.generate_content(prompt)

        return response.text
