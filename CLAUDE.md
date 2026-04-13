# report_stock_daily Project

Stock daily report generation system using Supabase data and Gemini AI analysis.

## Development Commands

### Execution
- **Run Daily Report Generation**: `python src/jobs/generate_report.py`
- **Install Dependencies**: `pip install -r requirements.txt`

### Configuration Files
- `config/api_keys.json`: **Main storage for API Keys** (KIS, KRX, Gemini, Supabase).
- `config/analyzer_settings.json`: AI model parameters (model name, system instructions).

### Environment Variables
Environment-specific settings or those required for automated pipelines:
- `GOOGLE_DOCS_NEWS_URL`: Export URL (txt format) for the news source Google Doc.
- `DATABASE_URL`: Postgres connection string (for MCP/Database access).
- `SUPABASE_URL` / `SUPABASE_KEY`: (Optional overrides, priority given to `api_keys.json`).
- `GEMINI_API_KEY`: (Optional override, priority given to `api_keys.json`).

## Project Structure
- `src/data/`: Modules for fetching data from external sources (Supabase, Google Docs).
- `src/analysis/`: Modules for data processing and AI-driven analysis using Gemini.
- `src/jobs/`: Entry points for pipeline execution and orchestration.
- `config/`: JSON configuration files for API keys and analysis settings.
- `reports/`: Generated markdown reports.

## Code Conventions
- **Credential Management**: Use `config/api_keys.json` for local development. Use environment variables as fallback/overrides (useful for GitHub Actions).
- **No Hardcoding**: Avoid hardcoding environment-specific values like URLs and API keys.
- **Environment Variables**: Use `os.getenv` for URLs that change per environment.
- **Modularity**: Keep data fetching, analysis, and job orchestration logic separate.
