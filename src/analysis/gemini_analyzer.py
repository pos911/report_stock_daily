import json
import os
import time
import re

from google import genai
from dotenv import load_dotenv

from src.utils import config


load_dotenv()


class GeminiAnalyzer:
    MAX_RETRIES = 2
    FALLBACK_RETRIES = 2
    BASE_RETRY_SECONDS = 5
    NEWS_BATCH_CHARS = 4000
    STOCK_BATCH_SIZE = 4
    TOP_VOLUME_PER_MARKET = 3
    FEATURE_PRIORITY = (
        "return_1d",
        "return_5d",
        "return_20d",
        "moving_avg_5",
        "moving_avg_20",
        "moving_avg_60",
        "foreign_flow_zscore",
        "institutional_flow_zscore",
        "individual_flow_zscore",
        "pension_flow_zscore",
        "volume",
        "volume_ratio",
        "trading_value",
        "trading_value_zscore",
        "rsi_14",
        "rsi",
    )

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

        self.client = genai.Client(api_key=self.api_key)

        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        self.system_instruction = os.getenv("GEMINI_SYSTEM_INSTRUCTION")
        configured_fallbacks = []

        if os.path.exists(settings_config_path):
            with open(settings_config_path, "r", encoding="utf-8") as f:
                settings = json.load(f).get("gemini", {})
                self.model_name = settings.get("model_name", self.model_name)
                self.system_instruction = settings.get(
                    "system_instruction", self.system_instruction
                )
                configured_fallbacks = settings.get("fallback_model_names", [])

        env_fallbacks = os.getenv("GEMINI_FALLBACK_MODEL_NAMES", "")
        fallback_model_names = [
            name.strip()
            for name in env_fallbacks.split(",")
            if name.strip()
        ] or configured_fallbacks
        self.fallback_model_names = [
            name for name in fallback_model_names if name and name != self.model_name
        ]

        if not self.system_instruction:
            self.system_instruction = "You are a financial analyst."



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

    def _build_silent_skip_rules(self):
        """Common silent-skip rules shared across prompts."""
        return """
## 공통 엄격 금지 사항 (Silent Skip 원칙)
- "데이터가 없습니다", "데이터가 부족합니다", "제공되지 않아", "N/A", "Not available" 같은 표현 금지.
- 데이터가 없는 섹션/지표/종목은 아무 언급 없이 조용히 생략.
- 마크다운 최상위 헤더 `#`, `##` 금지. Python 조립 단계가 구조 헤더를 담당하므로 `###` 이하만 사용.
- 인사말, 상황 설명식 서론, 전체 결론 금지. 바로 본문만 작성.
- If any metric or section has no data, do not mention it. Simply omit and analyze only what is available.
"""

    @staticmethod
    def _sanitize_llm_text(text: str) -> str:
        if not text:
            return ""

        blocked_fragments = (
            "N/A",
            "데이터가 없어",
            "데이터가 없",
            "데이터가 부족",
            "제공된 데이터",
            "제공되지 않아",
            "미제공",
            "한국 지수",
            "판단이 어렵",
            "구체적인 투자 판단은 어렵",
        )
        cleaned_lines = []
        for line in text.splitlines():
            if any(fragment in line for fragment in blocked_fragments):
                continue
            cleaned_lines.append(line.rstrip())

        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _call_model(self, prompt: str, temperature: float = 0.7) -> str:
        """Shared Gemini API wrapper."""
        last_error = None
        prompt_chars = len(prompt)
        model_candidates = [(self.model_name, self.MAX_RETRIES)] + [
            (model_name, self.FALLBACK_RETRIES)
            for model_name in self.fallback_model_names
        ]

        for model_index, (model_name, max_retries) in enumerate(model_candidates):
            is_last_model = model_index == len(model_candidates) - 1
            for attempt in range(1, max_retries + 1):
                try:
                    print(
                        f"Gemini request: model={model_name}, attempt {attempt}/{max_retries}, "
                        f"prompt_chars={prompt_chars}, temperature={temperature}"
                    )
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=self.system_instruction,
                            temperature=temperature
                        ),
                    )
                    return self._sanitize_llm_text(response.text)
                except Exception as exc:
                    last_error = exc
                    error_text = str(exc)
                    
                    # 429(Rate Limit)와 404(Not Found) 에러 구분
                    is_rate_limit = "429" in error_text or "ResourceExhausted" in error_text
                    is_not_found = "404" in error_text or "not found" in error_text.lower()

                    # 둘 다 아니면 즉시 에러 발생 (Fatal Error)
                    if not (is_rate_limit or is_not_found):
                        raise

                    # 404 에러이거나 재시도 횟수를 초과한 경우 -> 다음 Fallback 모델로 즉시 이동
                    if is_not_found or attempt == max_retries:
                        if is_last_model:
                            raise
                        next_model = model_candidates[model_index + 1][0]
                        print(
                            f"Warning: Gemini model={model_name} failed with {exc}. "
                            f"Immediately trying fallback model={next_model}."
                        )
                        break # 내부 재시도 루프를 빠져나가 다음 모델 시도로 이동

                    # 429 에러인 경우 -> 백오프 대기 후 동일 모델 재시도
                    sleep_seconds = self.BASE_RETRY_SECONDS * (2 ** (attempt - 1))
                    print(
                        f"Warning: Gemini call retry {attempt}/{max_retries} "
                        f"after {sleep_seconds}s due to: {exc}"
                    )
                    time.sleep(sleep_seconds)

        raise last_error

    @staticmethod
    def _compact_global_macro_data(macro_data):
        if not isinstance(macro_data, dict):
            return macro_data

        preferred_keys = (
            "base_date",
            "kospi",
            "kospi_change_rate",
            "kosdaq",
            "kosdaq_change_rate",
            "kospi_individual_net_buy",
            "kospi_foreign_net_buy",
            "kospi_institutional_net_buy",
            "kosdaq_individual_net_buy",
            "kosdaq_foreign_net_buy",
            "kosdaq_institutional_net_buy",
            "usdkrw",
            "dxy",
            "us10y",
            "kr10y",
            "wti",
            "gold",
            "copper",
            "vix",
            "sp500",
            "sp500_change_rate",
            "nasdaq",
            "nasdaq_change_rate",
        )
        compact = {key: macro_data.get(key) for key in preferred_keys if macro_data.get(key) is not None}
        return compact or macro_data

    @staticmethod
    def _compact_market_breadth(market_breadth):
        if isinstance(market_breadth, list):
            market_breadth = market_breadth[0] if market_breadth else {}
        if not isinstance(market_breadth, dict):
            return market_breadth

        preferred_keys = (
            "base_date",
            "advances",
            "declines",
            "advancing_volume",
            "declining_volume",
            "advance_decline_ratio",
            "advance_decline_volume_ratio",
        )
        compact = {key: market_breadth.get(key) for key in preferred_keys if market_breadth.get(key) is not None}
        return compact or market_breadth

    @staticmethod
    def _compact_momentum_data(momentum_data):
        if not isinstance(momentum_data, list):
            return momentum_data

        compact = []
        for item in momentum_data:
            if not isinstance(item, dict):
                continue
            compact.append(
                {
                    "feature_name": item.get("feature_name"),
                    "feature_value": item.get("feature_value"),
                    "base_date": item.get("base_date"),
                }
            )
        return compact

    @staticmethod
    def _compact_guardrails(data_guardrails):
        if not isinstance(data_guardrails, dict):
            return {}

        zero = data_guardrails.get("zero_volume_guardrail") or {}
        lag_map = data_guardrails.get("lag_days_by_table") or {}
        return {
            "lag_days_by_table": lag_map,
            "zero_volume_guardrail": {
                "latest_zero_volume_pct": zero.get("latest_zero_volume_pct"),
                "delta_pct": zero.get("delta_pct"),
                "latest_base_date": zero.get("latest_base_date"),
            },
            "pipeline_alert_count": len(data_guardrails.get("pipeline_alert_logs") or []),
        }

    @staticmethod
    def _compact_news_context(news_text: str) -> str:
        return news_text.strip() if news_text else ""

    def _compact_top_volume_data(self, top_volume_data):
        compact = {}
        if not isinstance(top_volume_data, dict):
            return compact

        for market_name, items in top_volume_data.items():
            compact_items = []
            for item in (items or [])[: self.TOP_VOLUME_PER_MARKET]:
                if not isinstance(item, dict):
                    continue
                compact_items.append(
                    {
                        "symbol": item.get("symbol"),
                        "stock_name": item.get("stock_name"),
                        "market": item.get("market"),
                        "volume_value": item.get("volume_value"),
                        "return_5d": item.get("return_5d"),
                        "foreign_flow_zscore": item.get("foreign_flow_zscore"),
                        "moving_avg_20": item.get("moving_avg_20"),
                        "per": item.get("per"),
                        "pbr": item.get("pbr"),
                    }
                )
            if compact_items:
                compact[market_name] = compact_items
        return compact

    def _compact_target_stocks_data(self, target_stocks_data):
        if not isinstance(target_stocks_data, dict):
            return {}

        compact = {}

        for row in target_stocks_data.get("normalized_stock_fundamentals_ratios", []) or []:
            if not isinstance(row, dict):
                continue
            symbol = row.get("symbol")
            if not symbol:
                continue
            compact[symbol] = {
                "symbol": symbol,
                "stock_name": row.get("stock_name"),
                "base_date": row.get("base_date"),
                "per": row.get("per"),
                "pbr": row.get("pbr"),
                "market_cap": row.get("market_cap"),
            }

        for row in target_stocks_data.get("normalized_stock_supply_daily", []) or []:
            if not isinstance(row, dict):
                continue
            symbol = row.get("symbol")
            if not symbol:
                continue
            bucket = compact.setdefault(symbol, {"symbol": symbol, "stock_name": row.get("stock_name")})
            bucket["supply"] = {
                "individual_net_buy": row.get("individual_net_buy"),
                "foreign_net_buy": row.get("foreign_net_buy"),
                "institutional_net_buy": row.get("institutional_net_buy"),
                "pension_net_buy": row.get("pension_net_buy"),
                "corporate_net_buy": row.get("corporate_net_buy"),
            }

        for row in target_stocks_data.get("normalized_stock_short_selling", []) or []:
            if not isinstance(row, dict):
                continue
            symbol = row.get("symbol")
            if not symbol:
                continue
            bucket = compact.setdefault(symbol, {"symbol": symbol, "stock_name": row.get("stock_name")})
            bucket["short_selling"] = {
                "short_ratio": row.get("short_ratio"),
                "short_volume": row.get("short_volume"),
                "short_balance": row.get("short_balance"),
            }

        feature_groups = {}
        for row in target_stocks_data.get("feature_store_daily", []) or []:
            if not isinstance(row, dict):
                continue
            symbol = row.get("symbol")
            feature_name = row.get("feature_name")
            if not symbol or not feature_name:
                continue
            feature_groups.setdefault(symbol, {})
            if feature_name in self.FEATURE_PRIORITY:
                feature_groups[symbol][feature_name] = row.get("feature_value")

        for symbol, feature_map in feature_groups.items():
            bucket = compact.setdefault(symbol, {"symbol": symbol})
            ordered_features = {}
            for feature_name in self.FEATURE_PRIORITY:
                if feature_name in feature_map:
                    ordered_features[feature_name] = feature_map[feature_name]
            if ordered_features:
                bucket["features"] = ordered_features

        return compact

    @staticmethod
    def _chunk_list(items, chunk_size):
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    @staticmethod
    def _chunk_text_by_lines(text: str, max_chars: int):
        chunks = []
        current = []
        current_len = 0

        for line in (text or "").splitlines():
            line_len = len(line) + 1
            if current and current_len + line_len > max_chars:
                chunks.append("\n".join(current).strip())
                current = []
                current_len = 0
            current.append(line)
            current_len += line_len

        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _extract_section(text: str, heading: str) -> str:
        if not text:
            return ""

        lines = text.splitlines()
        target = f"### {heading}"
        collecting = False
        collected = []

        for line in lines:
            stripped = line.strip()
            if stripped == target:
                collecting = True
                collected.append(target)
                continue
            if collecting and stripped.startswith("### "):
                break
            if collecting:
                collected.append(line.rstrip())

        return "\n".join(collected).strip()

    @staticmethod
    def _strip_stock_noise(text: str) -> str:
        if not text:
            return ""

        cleaned = re.sub(r"(?im)^Report 작성 시간:.*\n?", "", text)
        cleaned = re.sub(r"(?im)^### 시장 한줄 요약.*?(?=^###|\Z)", "", cleaned, flags=re.S)
        cleaned = re.sub(r"(?im)^### 매크로 분석.*?(?=^###|\Z)", "", cleaned, flags=re.S)
        cleaned = re.sub(r"(?im)^### 최종 투자 전략.*?(?=^###|\Z)", "", cleaned, flags=re.S)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _generate_market_snapshot_summary(
        self,
        compact_macro_data,
        compact_market_breadth,
        compact_momentum,
        compact_guardrails,
        korean_market_snapshot,
        generation_time,
        report_type: str,
    ) -> str:
        prompt = f"""
[Report Generation Info]
- 작성 시각: {generation_time}
- 리포트 유형: {report_type.upper()}

[거시 지표]
{json.dumps(compact_macro_data, ensure_ascii=False)}

[시장 폭]
{json.dumps(compact_market_breadth, ensure_ascii=False)}

[모멘텀 변화율]
{json.dumps(compact_momentum, ensure_ascii=False)}

[한국 시장 스냅샷]
{json.dumps(korean_market_snapshot or {}, ensure_ascii=False)}

[Data Quality Guardrails]
{json.dumps(compact_guardrails, ensure_ascii=False)}

[System Instruction]
국내 지수, 투자자별 수급, 미국 지수, 매크로 데이터를 바탕으로 시장 상태를 판단하라. 뉴스 해석은 다음 단계에서 별도로 한다.

### 출력 구조
- `### 시장 한줄 요약`
- `### 오늘의 시장 판단`

### 작성 규칙
1. `### 시장 한줄 요약`의 첫 문장은 반드시 한 문장으로 오늘 시장 분위기를 요약하라.
2. 주요 증권사 시황 리포트 양식을 참고하여, 지수 및 지표명은 굵게 표시하고, 현재 수치와 전일 대비 변화율(%)을 시각적으로 돋보이게 작성하라. 
   (작성 예시: `- **KOSPI**: 2,750.12 (+1.23%)` / `- **개인 순매수**: +1,500억`)
3. 수치 나열 직후 `**[시장 평가]**` 항목을 추가하여, 현재 변화율이 주는 종합적인 의미를 한 줄로 요약하라.
4. `### 오늘의 시장 판단`에는 현재 시장이 `Risk-On` 인지 `Risk-Off` 인지 혹은 `중립` 인지 명시하고, 왜 그렇게 판단했는지(예: "Risk-Off 라는데 이게 맞는 판단인지") 데이터를 근거로 짧은 의견을 덧붙여라.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
        return self._call_model(prompt, temperature=0.5).strip()

    def _generate_news_implications_summary(
        self,
        compact_news,
        market_snapshot_md,
        compact_momentum,
        report_type: str,
    ) -> str:
        prompt = f"""
[Report Generation Info]
- 리포트 유형: {report_type.upper()}

[시장/수급 판단 요약]
{market_snapshot_md}

[StockData 요약]
{json.dumps(compact_momentum, ensure_ascii=False)}

[Google Docs News Context]
{compact_news}

[System Instruction]
Google Docs 뉴스와 앞 단계의 시장/수급 판단을 종합해 투자자가 읽을 수 있는 핵심 해석을 작성하라.

### 출력 구조
- `### 뉴스 요약`
- `### 핵심 포인트`
- `### 투자 시사점`

### 작성 규칙
1. `### 뉴스 요약`은 핵심 뉴스가 누락되지 않게 서로 다른 이슈를 3~5개로 묶어라.
2. `### 핵심 포인트`는 뉴스, 시장 지수, 수급 등의 평가를 전체적으로 종합해서 3개 안팎의 bullet로 정리하라.
3. `### 투자 시사점`은 제공된 StockData의 모멘텀, 수급, 밸류에이션, 거래대금 등의 다양한 데이터를 투자공학 관점에서 분석하고, 위 핵심 포인트에서 뉴스와 종합한 내용을 기초로 2~3개만 제시하라.
4. 데이터 품질 문제는 별도 본문 섹션으로 만들지 말고, 꼭 필요한 경우 마지막 bullet에 짧게만 반영하라.
5. 새 사실을 지어내지 말고, 입력에 없는 종목/수치는 추가하지 마라.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
        return self._call_model(prompt, temperature=0.5).strip()

    def generate_market_summary(
        self,
        macro_data,
        market_breadth,
        momentum_data,
        data_guardrails,
        news_text,
        korean_market_snapshot,
        generation_time,
        report_type: str = "regular",
    ):
        """
        [Step 1] Market regime, macro summary, and news summary.
        """
        compact_macro_data = self._compact_global_macro_data(macro_data)
        compact_market_breadth = self._compact_market_breadth(market_breadth)
        compact_momentum = self._compact_momentum_data(momentum_data)
        compact_guardrails = self._compact_guardrails(data_guardrails)
        compact_news = self._compact_news_context(news_text)

        market_snapshot_md = self._generate_market_snapshot_summary(
            compact_macro_data=compact_macro_data,
            compact_market_breadth=compact_market_breadth,
            compact_momentum=compact_momentum,
            compact_guardrails=compact_guardrails,
            korean_market_snapshot=korean_market_snapshot,
            generation_time=generation_time,
            report_type=report_type,
        )
        news_implications_md = self._generate_news_implications_summary(
            compact_news=compact_news,
            market_snapshot_md=market_snapshot_md,
            compact_momentum=compact_momentum,
            report_type=report_type,
        )
        return "\n\n".join(
            part for part in (market_snapshot_md, news_implications_md) if part
        ).strip()

    def summarize_news_context(self, news_text: str, report_type: str = "regular") -> str:
        """
        Summarize normalized news in batches before the main market prompt.
        This keeps all news items represented while reducing the heavy first Gemini call.
        """
        if not news_text:
            return ""

        chunks = self._chunk_text_by_lines(news_text, self.NEWS_BATCH_CHARS)
        if not chunks:
            return ""

        summaries = []
        for idx, chunk in enumerate(chunks, 1):
            prompt = f"""
[Report Generation Info]
- 리포트 유형: {report_type.upper()}
- 뉴스 배치: {idx}/{len(chunks)}

[Normalized Google Docs News Items]
{chunk}

[System Instruction]
아래 뉴스 항목들을 투자자가 읽을 수 있게 압축하라.

### 출력 구조
- `### 뉴스 배치 요약`

### 작성 규칙
1. 원문 항목을 빠뜨리지 말고, 같은 이슈는 묶어서 정리하라.
2. 각 bullet은 `- 이슈: 시장 영향` 형식으로 쓴다.
3. 새 사실을 지어내지 말고, 원문에 없는 종목/수치는 추가하지 마라.
4. 긴 서론, 인사말, 전체 결론은 금지한다.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
            summaries.append(self._call_model(prompt, temperature=0.4).strip())

        return "\n\n".join(summaries).strip()

    def generate_top_volume_analysis(
        self,
        top_volume_data,
        market_summary: str = "",
        report_type: str = "regular",
    ):
        """
        [Step 3] Top-volume names and smart-money angle.
        """
        compact_top_volume = self._compact_top_volume_data(top_volume_data)
        prompt = f"""
[Report Generation Info]
- 리포트 유형: {report_type.upper()}

[Top Volume Stocks]
{json.dumps(compact_top_volume, indent=2, ensure_ascii=False)}

[Market/News Context]
{market_summary}

[System Instruction]
거래대금 상위 종목에서 오늘 눈에 띄는 이름을 고르고, 시장/뉴스 맥락상 왜 거래대금이 상위로 몰렸는지 해석하라.

### 출력 구조
- `### 거래대금 상위 종목`
- 종목별 bullet: `- 종목명(코드): 왜 거래대금이 상위인지 설명`

### 작성 규칙
1. 긴 서론은 금지한다.
2. 뉴스/시장 맥락과 연결하여 특정 종목에 거래대금이 왜 쏠렸는지 그 이유와 의미를 뉴스/모멘텀을 통해 알려줘라.
3. `foreign_flow_zscore`, 수익률, 밸류에이션 중 의미 있는 근거만 짧게 포함하라. 단, 근거 없는 추정은 금지한다.
4. `Report 작성 시간`, `### 매크로 분석`, `### 최종 투자 전략` 같은 추가 섹션은 절대 쓰지 마라.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
        response_text = self._call_model(prompt, temperature=0.6)
        extracted = self._extract_section(response_text, "거래대금 상위 종목")
        return extracted or response_text.strip()

    def generate_stock_analysis(
        self,
        market_summary,
        top_volume_data,
        target_stocks_data,
        macro_market_data,
        generation_time,
        data_guardrails=None,
        report_type: str = "regular",
    ):
        """
        [Step 2] Focus-stock analysis.
        """
        compact_macro_data = self._compact_global_macro_data(
            macro_market_data.get("normalized_global_macro_daily")
        )
        compact_momentum = self._compact_momentum_data(macro_market_data.get("momentum"))
        compact_guardrails = self._compact_guardrails(data_guardrails)
        compact_target_stocks = self._compact_target_stocks_data(target_stocks_data)

        prompt = f"""
[Report Generation Info]
- 작성 시각: {generation_time}
- 리포트 유형: {report_type.upper()}

[Market Summary Context]
{market_summary}

[Macro/Global Context Information]
- Global Macro Daily: {json.dumps(compact_macro_data, indent=1, ensure_ascii=False)}
- Global Momentum: {json.dumps(compact_momentum, indent=1, ensure_ascii=False)}
- Data Guardrails: {json.dumps(compact_guardrails, indent=1, ensure_ascii=False)}

[Target Stocks Data]
{json.dumps(compact_target_stocks, indent=2, ensure_ascii=False)}

[System Instruction]
관심 종목만 매우 실용적으로 정리하라.

### 출력 구조
- `### 관심 종목 분석`
- 종목별 고정 형식:
1. `1) 공격적인 포인트`
2. `2) 최대한 보수적인 포인트`
3. `3) 최종 결론 (BUY/HOLD/SELL)`

### 작성 규칙
1. 종목명(코드) 옆에 반드시 1일 수익률(return_1d) 등 변화율(%)을 괄호와 부호(+, -)를 포함해 명확히 표기하라. (예: `**삼성전자(005930)** (+1.50%)`)
2. `1) 공격적인 포인트`: 상승을 기대할 수 있는 모멘텀, 수급(외인/기관), 밸류에이션 매력 등 긍정적 뷰를 강력하게 서술하라.
3. `2) 최대한 보수적인 포인트`: 하방 리스크, 매크로 역풍, 수급 이탈 등 리스크 요소를 최대한 보수적인 관점에서 서술하라.
4. `3) 최종 결론`: 위 두 가지 뷰를 검토한 후, (BUY / HOLD / SELL) 중 하나의 최종 결론을 도출하고 이유를 짧게 적어라.
5. 장문 서론은 금지하고 바로 종목 핵심만 적어라.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
        response_text = self._call_model(prompt, temperature=0.6)
        extracted = self._extract_section(response_text, "관심 종목 분석")
        return self._strip_stock_noise(extracted or response_text.strip())

    def generate_batched_stock_analysis(
        self,
        market_summary,
        target_stocks_data,
        macro_market_data,
        generation_time,
        data_guardrails=None,
        report_type: str = "regular",
    ):
        compact_target_stocks = self._compact_target_stocks_data(target_stocks_data)
        symbols = list(compact_target_stocks.keys())
        if not symbols:
            return ""

        sections = []
        for batch_symbols in self._chunk_list(symbols, self.STOCK_BATCH_SIZE):
            batch_payload = {
                table_name: [
                    row for row in (rows or []) if isinstance(row, dict) and row.get("symbol") in batch_symbols
                ]
                for table_name, rows in (target_stocks_data or {}).items()
            }
            sections.append(
                self.generate_stock_analysis(
                    market_summary=market_summary,
                    top_volume_data=None,
                    target_stocks_data=batch_payload,
                    macro_market_data=macro_market_data,
                    generation_time=generation_time,
                    data_guardrails=data_guardrails,
                    report_type=report_type,
                ).strip()
            )

        merged_sections = []
        seen_heading = False
        for section in sections:
            if not section:
                continue
            if seen_heading:
                section = section.replace("### 관심 종목 분석", "", 1).strip()
            elif "### 관심 종목 분석" in section:
                seen_heading = True
            merged_sections.append(section)

        return "\n\n".join(part for part in merged_sections if part).strip()
