# report_stock_daily Project

Stock daily report generation system using Supabase data and Gemini AI analysis.

## Development Commands

### Execution
- **Run Daily Report Generation**: `python src/jobs/generate_report.py`
- **Run Morning Market Brief**: `python src/jobs/generate_report.py --type morning`
- **Run Formatter Tests**: `python -m unittest tests.test_formatters`
- **Install Dependencies**: `pip install -r requirements.txt`

### Configuration Files
- `config/api_keys.json`: **Main storage for API Keys** (KIS, KRX, Gemini, Supabase).
- `config/analyzer_settings.json`: AI model parameters (model name, system instructions).

### Environment Variables
Environment-specific settings or those required for automated pipelines:
- `GOOGLE_DOCS_NEWS_URL`: Export URL (txt format) for the news source Google Doc.
- `DATABASE_URL`: Postgres connection string (for MCP/Database access).
- `SUPABASE_URL` / `SUPABASE_KEY`: (Optional overrides, priority given to `api_keys.json`).
- `GEMINI_API_KEY`: (Optional override, priority given to environment variables).
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`: (Optional overrides).

## Project Structure
- `src/data/`: Modules for fetching data from external sources (Supabase, Google Docs).
- `src/analysis/`: Modules for data processing and AI-driven analysis using Gemini.
- `src/jobs/`: Entry points for pipeline execution and orchestration.
- `src/utils/`: Shared utilities including the unified `config_loader`.
- `config/`: JSON configuration files for API keys and analysis settings.
- `reports/`: Generated markdown reports.

## Code Conventions
- **Credential Management**: Use `src.utils.config` for all credential lookups.
- **Priority**: 1. Environment Variables, 2. `config/api_keys.json`.
- **No Hardcoding**: Avoid hardcoding environment-specific values like URLs and API keys.
- **Modularity**: Keep data fetching, analysis, and job orchestration logic separate.

## Report Generation Principles
- **Two-Step Strict Separation & Assembly**: 
    - The LLM prompts must exclusively generate their assigned section's deep content (Step 1: Macro/News, Step 2: Stock Analysis) without overarching headers (`#`, `##`), greetings, or conclusions.
    - Python (`generate_report.py`) handles the structural assembly and layout headers.
- **Aggressive vs Conservative View**: Stock analysis must avoid plain narrative lists. It strictly enforces a 3-step bullet point structure: `1) 공격적인 포인트`, `2) 최대한 보수적인 포인트`, `3) 최종 결론 (BUY/HOLD/SELL)`.
- **Zero-Waste Prompting**: The LLM must NEVER output excuses about missing data. If data is absent, the model must silently skip it. Do not say "Data is not provided" or "결측치입니다".
- **Morning Report Data Source Rule**: Morning Market Brief sections must use Supabase StockData official tables as the source of truth. Market, macro, price, supply, valuation, short-selling, event, and feature numbers must come only from the official Supabase tables, not from external market-price crawlers.
