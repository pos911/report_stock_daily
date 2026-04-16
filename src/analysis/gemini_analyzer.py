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
        [Step 1] 매크로 체제(Regime) 분석 + 시장 시황 요약.
        - 환율·금리의 1d/5d 변화율로 Risk-On / Risk-Off 체제를 명시.
        - Silent Skip: 데이터 누락 시 언급 없이 해당 섹션 생략.
        """
        prompt = f"""
[Report Generation Info]
- 작성 시간: {generation_time}

[거시 지표 / Market Breadth Data]
{json.dumps({"macro_data": macro_data, "market_breadth": market_breadth}, indent=2, ensure_ascii=False)}

[News Text from Google Docs]
{news_text}

[System Instruction]
아래 규칙을 엄격하게 준수하여 거시경제 시황 섹션만 작성하라.

## 필수 분석 규칙

1. **매크로 체제(Regime) 판단 [최우선]**
   - 환율(USD/KRW)과 금리(FRED: 미국채 10년물 등)의 `_1d_chg`(1일 변화율), `_5d_chg`(5일 변화율)을 최우선으로 분석하라.
   - 이를 바탕으로 오늘 시장이 **Risk-On** 인지 **Risk-Off** 인지 반드시 명시하고, 그 근거를 간결하게 서술하라.
   - 예: "환율 _1d_chg +0.8%, 금리 _5d_chg +12bp → Risk-Off 국면 진입"

2. **거시경제 데이터 분석**
   - 코스피/코스닥/선물옵션 시장의 전체적인 매크로 시황 분석
   - 오직 거시경제(환율, 금리, 유가 등)와 뉴스 요약에 집중하라.
   - 단순 값 나열이 아닌, 데이터가 의미하는 **향후 시장 방향성**을 논리적으로 서술하라.

## 엄격 금지 사항 (Silent Skip 원칙)
- 종목별 지표(PER, PBR, 수급, 공매도 등)나 개별 주식 언급 금지
- "데이터가 없습니다", "데이터가 부족하여", "제공되지 않아" 등의 문구 절대 금지
- 데이터가 없는 섹션이나 None/NaN 값은 **아무 언급 없이 조용히 생략(Silent Skip)**
- 인사말/전체 서론/전체 결론 금지
- `#`, `##` 최상위 마크다운 헤더 금지 (### 부터 사용)
- If any requested data is missing, DO NOT mention it. Simply skip and analyze only what is available.

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
        [Step 2] 종목 심층 분석.
        - 스마트 머니 포착: 외인 수급 Z-Score 높고 PER/PBR 낮은 종목 특별 언급
        - Silent Skip: 데이터 없는 섹션/지표 완전 생략
        - 3단계 개조식 구조 강제
        """
        prompt = f"""
[Report Generation Info]
- 작성 시간: {generation_time}

[Market Summary Context]
{market_summary}

[Top Volume Stocks (KOSPI/KOSDAQ/ETF)]
{json.dumps(top_volume_data, indent=2, ensure_ascii=False)}

[Target Stocks Data]
데이터 포함: supply, fundamentals(market_cap/per/pbr), short selling, feature store(_1d_chg/_5d_chg)
{json.dumps(target_stocks_data, indent=2, ensure_ascii=False)}

[System Instruction]
앞서 분석된 시황(Market Summary Context)을 바탕으로 아래 내용을 수행하라.
인사말이나 전체 서론, 결론은 절대 생성하지 말고 곧바로 본론(분석 내용)만 출력하라.
파이썬 코드 단에서 타이틀을 조립하므로, `#`, `##` 최상위 마크다운 헤더는 생성하지 마라 (필요 시 ### 부터 사용).

---

### [분석 1] 거래량 상위 종목 퀀트 평가
KOSPI / KOSDAQ / ETF 거래량 상위 종목의 퀀트 지표를 짧고 핵심만 평가하라.
종목별로 소속 시장(KOSPI, KOSDAQ, ETF)을 명시하라.

---

### [분석 2] 스마트 머니 포착 (Smart Money Detection) ⭐
거래량(Volume) 상위 종목 중 아래 두 조건을 **동시에** 충족하는 종목이 있다면 반드시 **특별 섹션**으로 강조 언급하라:
- 조건 A: 외국인 수급 Z-Score(`foreign_net_zscore` 또는 유사 피처)가 **양수이고 높은 종목**
- 조건 B: PER 또는 PBR이 **시장 평균 대비 낮은 종목** (밸류에이션 매력)

해당 종목은 `<외인/기관 강한 매집 우량주>` 레이블로 별도 강조하라.
조건을 충족하는 종목이 없으면 이 섹션 전체를 Silent Skip하라.

---

### [분석 3] 타겟 관심 종목 심층 분석
타겟 관심 종목(Target Stocks) 분석 시 나열식 설명을 완전히 폐기하고,
반드시 종목별로 다음 **3단계 개조식(Bullet points)** 구조로만 답변하라:
1. 🔴 공격적인 포인트 (매수 근거, 모멘텀, 수급 강도)
2. 🔵 보수적인 포인트 (리스크, 밸류에이션 부담, 하방 요소)
3. ⚖️ 최종 결론: **BUY / HOLD / SELL** (명확하게 단정 지을 것)

---

## 엄격 금지 사항 (Silent Skip 원칙)
- "현재 데이터가 제공되지 않아...", "데이터가 부족하여...", "N/A", "Not available" 등 절대 금지
- 특정 지표(PER, PBR, Z-score, 수급 데이터 등)가 없으면 **그 지표 자체를 언급하지 말고** 있는 데이터만으로 분석
- 분석 데이터가 없는 종목이나 섹션은 설명 없이 완전히 생략 (마크다운 제목조차 출력 금지)
- If a specific metric is missing for a stock, completely omit any mention of that metric. Base analysis only on provided data.

마크다운 형식으로 작성해줘.
"""
        response = self.model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
            ),
        )
        return response.text
