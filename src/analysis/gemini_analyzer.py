import json
import os
import time
import re

import google.generativeai as genai
from dotenv import load_dotenv

from src.utils import config


load_dotenv()


class GeminiAnalyzer:
    MAX_RETRIES = 4
    BASE_RETRY_SECONDS = 5
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
        prompt_chars = len(prompt)
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                print(
                    f"Gemini request: attempt {attempt}/{self.MAX_RETRIES}, "
                    f"prompt_chars={prompt_chars}, temperature={temperature}"
                )
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

    @staticmethod
    def _compact_global_macro_data(macro_data):
        if not isinstance(macro_data, dict):
            return macro_data

        preferred_keys = (
            "base_date",
            "usdkrw",
            "dxy",
            "us10y",
            "kr10y",
            "wti",
            "gold",
            "copper",
            "vix",
            "sp500",
            "nasdaq",
            "kospi",
            "kosdaq",
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
        [Step 1] Market regime, macro summary, and news summary.
        """
        compact_macro_data = self._compact_global_macro_data(macro_data)
        compact_market_breadth = self._compact_market_breadth(market_breadth)
        compact_momentum = self._compact_momentum_data(momentum_data)
        compact_guardrails = self._compact_guardrails(data_guardrails)
        compact_news = self._compact_news_context(news_text)

        prompt = f"""
[Report Generation Info]
- 작성 시각: {generation_time}
- 리포트 유형: {report_type.upper()}

[거시 지표]
{json.dumps(compact_macro_data, indent=2, ensure_ascii=False)}

[시장 폭]
{json.dumps(compact_market_breadth, indent=2, ensure_ascii=False)}

[모멘텀 변화율]
{json.dumps(compact_momentum, indent=2, ensure_ascii=False)}

[Data Quality Guardrails]
{json.dumps(compact_guardrails, indent=2, ensure_ascii=False)}

[News Context]
{compact_news}

[System Instruction]
오늘 시장의 큰 방향과 핵심 뉴스를 한 번에 간결하게 요약하라.

### 출력 구조
- `### 시장 한줄 요약`
- `### 핵심 포인트`
- `### 뉴스 요약`
- `### 투자 시사점`
- `### 오늘의 시장 판단`

### 작성 규칙
1. 긴 숫자 나열보다 해석 중심으로 써라.
2. 핵심 포인트는 3개 안팎의 bullet로 제한하라.
3. 뉴스 요약은 3개 안팎의 핵심 이슈만 남겨라.
4. 투자 시사점은 국내 투자자가 바로 이해할 수 있게 2~3개만 제시하라.
5. `### 오늘의 시장 판단`에는 반드시 `Risk-On`, `Risk-Off`, `중립` 중 하나를 명시하라.

{self._build_silent_skip_rules()}
마크다운 형식으로 작성해줘.
"""
        return self._call_model(prompt, temperature=0.6)

    def generate_top_volume_analysis(self, top_volume_data, report_type: str = "regular"):
        """
        [Step 3] Top-volume names and smart-money angle.
        """
        compact_top_volume = self._compact_top_volume_data(top_volume_data)
        prompt = f"""
[Report Generation Info]
- 리포트 유형: {report_type.upper()}

[Top Volume Stocks]
{json.dumps(compact_top_volume, indent=2, ensure_ascii=False)}

[System Instruction]
거래대금 상위 종목에서 오늘 눈에 띄는 이름만 골라 간결하게 정리하라.

### 출력 구조
- `### 거래대금 상위 종목`
- 종목별 bullet: `- 종목명(코드): 한 줄 요약`

### 작성 규칙
1. 긴 서론은 금지한다.
2. `foreign_flow_zscore`, 수익률, 밸류에이션 중 의미 있는 근거만 짧게 사용하라.
3. 업종 흐름이나 테마 연결은 한 문장 이내로 제한하라.
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
1. 공격적인 포인트에는 모멘텀, 수급, 밸류에이션, 업황 중 강점만 압축해 적어라.
2. 보수적인 포인트에는 리스크, 데이터 지연 가능성, 업황 역풍을 적어라.
3. 결론은 종목마다 하나만 명시하라.
4. 종목별 장문 서론은 금지하고 바로 핵심만 적어라.
5. `Report 작성 시간`, `### 시장 한줄 요약`, `### 매크로 분석`, `### 최종 투자 전략` 같은 추가 섹션은 절대 쓰지 마라.

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
