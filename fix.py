import codecs
with open('src/analysis/gemini_analyzer.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. API Migration
text = text.replace('import google.generativeai as genai', 'from google import genai')
text = text.replace('genai.configure(api_key=self.api_key)', 'self.client = genai.Client(api_key=self.api_key)')
search_model_init = '''        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction,
        )
        self.fallback_models = [
            (
                model_name,
                genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=self.system_instruction,
                ),
            )
            for model_name in self.fallback_model_names
        ]'''
replace_model_init_empty = ''
text = text.replace(search_model_init, replace_model_init_empty)

search_call_model = '''        model_candidates = [(self.model_name, self.model, self.MAX_RETRIES)] + [
            (model_name, model, self.FALLBACK_RETRIES)
            for model_name, model in self.fallback_models
        ]

        for model_index, (model_name, model, max_retries) in enumerate(model_candidates):'''
replace_call_model = '''        model_candidates = [(self.model_name, self.MAX_RETRIES)] + [
            (model_name, self.FALLBACK_RETRIES)
            for model_name in self.fallback_model_names
        ]

        for model_index, (model_name, max_retries) in enumerate(model_candidates):'''
text = text.replace(search_call_model, replace_call_model)

search_generate = '''                    response = model.generate_content(
                        contents=[{"role": "user", "parts": [{"text": prompt}]}],
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature
                        ),
                    )'''
replace_generate = '''                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=self.system_instruction,
                            temperature=temperature
                        ),
                    )'''
text = text.replace(search_generate, replace_generate)

# 2A. Update _generate_market_snapshot_summary rules
search_snapshot_rules = '''### 작성 규칙
1. ### 시장 한줄 요약의 첫 문장은 반드시 한 문장으로 오늘 시장 분위기를 요약하라.
2. 그 아래에 KOSPI, KOSDAQ 종가와 등락(전일 대비), 개인/외인/기관 동향(순매수), 나스닥 지수 등락폭을 bullet로 명확하게 정리하라. (제공된 값만 사용)
3. 지수와 수급 숫자 바로 아래에 평가: 한 줄을 붙여 이 수치들이 종합적으로 어떤 의미인지 설명하라.
4. ### 오늘의 시장 판단에는 현재 시장이 Risk-On 인지 Risk-Off 인지 혹은 중립 인지 명시하고, 왜 그렇게 판단했는지(예: "Risk-Off 라는데 이게 맞는 판단인지") 데이터를 근거로 짧은 의견을 덧붙여라.'''
replace_snapshot_rules = '''### 작성 규칙
1. ### 시장 한줄 요약의 첫 문장은 반드시 한 문장으로 오늘 시장 분위기를 요약하라.
2. 주요 증권사 시황 리포트 양식을 참고하여, 지수 및 지표명은 굵게 표시하고, 현재 수치와 전일 대비 변화율(%)을 시각적으로 돋보이게 작성하라. 
   (작성 예시: - **KOSPI**: 2,750.12 (+1.23%) / - **개인 순매수**: +1,500억)
3. 수치 나열 직후 **[시장 평가]** 항목을 추가하여, 현재 변화율이 주는 종합적인 의미를 한 줄로 요약하라.
4. ### 오늘의 시장 판단에는 현재 시장이 Risk-On 인지 Risk-Off 인지 혹은 중립 인지 명시하고, 왜 그렇게 판단했는지(예: "Risk-Off 라는데 이게 맞는 판단인지") 데이터를 근거로 짧은 의견을 덧붙여라.'''
text = text.replace(search_snapshot_rules, replace_snapshot_rules)

# 2B. Update generate_stock_analysis rules
search_stock_rules = '''### 작성 규칙
1. 공격적인 포인트에는 모멘텀, 수급, 밸류에이션, 업황 중 강점만 압축해 적어라.
2. 보수적인 포인트에는 리스크, 데이터 지연 가능성, 업황 역풍을 적어라.
3. 결론은 종목마다 하나만 명시하라.
4. 종목별 장문 서론은 금지하고 바로 핵심만 적어라.
5. Report 작성 시간, ### 시장 한줄 요약, ### 매크로 분석, ### 최종 투자 전략 같은 추가 섹션은 절대 쓰지 마라.'''
replace_stock_rules = '''### 작성 규칙
1. 종목명(코드) 옆에 반드시 1일 수익률(return_1d) 등 변화율(%)을 괄호와 부호(+, -)를 포함해 명확히 표기하라. (예: **삼성전자(005930)** (+1.50%))
2. 1) 공격적인 포인트: 상승을 기대할 수 있는 모멘텀, 수급(외인/기관), 밸류에이션 매력 등 긍정적 뷰를 강력하게 서술하라.
3. 2) 최대한 보수적인 포인트: 하방 리스크, 매크로 역풍, 수급 이탈 등 리스크 요소를 최대한 보수적인 관점에서 서술하라.
4. 3) 최종 결론: 위 두 가지 뷰를 검토한 후, (BUY / HOLD / SELL) 중 하나의 최종 결론을 도출하고 이유를 짧게 적어라.
5. 장문 서론은 금지하고 바로 종목 핵심만 적어라.'''
text = text.replace(search_stock_rules, replace_stock_rules)

# 2C. Dead code removal in generate_market_summary
search_split = '''        return "\\n\\n".join(
            part for part in (market_snapshot_md, news_implications_md) if part
        ).strip()

        prompt = f"""'''
index = text.find(search_split)
if index != -1:
    end_of_func = index + len('''        return "\\n\\n".join(
            part for part in (market_snapshot_md, news_implications_md) if part
        ).strip()

''')
    next_func = text.find('    def summarize_news_context', end_of_func)
    if next_func != -1:
        text = text[:end_of_func] + text[next_func:]

with open('src/analysis/gemini_analyzer.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done!')
