import os
import sys
import datetime
from pathlib import Path

# Add src to python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.data.supabase_reader import SupabaseReader
from src.data.news_reader import fetch_news_document
from src.analysis.gemini_analyzer import GeminiAnalyzer

def main():
    print("Starting daily report generation pipeline...")
    
    # 1. Initialize modules
    try:
        reader = SupabaseReader()
        analyzer = GeminiAnalyzer()
    except Exception as e:
        print(f"Error initializing modules: {e}")
        return

    # 2. Fetch data
    print("Fetching data from Supabase...")
    quant_data = reader.fetch_latest_data()
    
    print("Fetching news from Google Docs...")
    news_text = fetch_news_document()

    # 3. Generate report
    print("Generating report via Gemini...")
    report_content = analyzer.generate_report(quant_data, news_text)

    # 4. Save report
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    now = datetime.datetime.now()
    file_name = f"daily_quant_report_{now.strftime('%Y%m%d_%H%M')}.md"
    file_path = reports_dir / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Report successfully saved to: {file_path}")

if __name__ == "__main__":
    main()
