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
from src.notification.telegram_sender import TelegramSender

import json

def main():
    print("Starting daily report generation pipeline (Two-Step)...")
    
    # 1. Load target stocks
    target_stocks_path = project_root / "config" / "target_stocks.json"
    target_symbols = []
    if target_stocks_path.exists():
        with open(target_stocks_path, "r", encoding="utf-8") as f:
            targets = json.load(f)
            target_symbols = [t["symbol"] for t in targets if t.get("enabled", True)]
    
    # 2. Initialize modules
    try:
        reader = SupabaseReader()
        analyzer = GeminiAnalyzer()
    except Exception as e:
        print(f"Error initializing modules: {e}")
        return

    # 3. Fetch data (Separated concerns)
    print("Fetching data from Supabase...")
    macro_market_data = reader.fetch_macro_and_market_data()
    
    print("Fetching Top Volume Stocks...")
    top_volume_data = reader.fetch_top_volume_stocks(limit=5)
    
    print(f"Fetching Target Stocks Data ({len(target_symbols)} symbols)...")
    target_stocks_data = reader.fetch_target_stocks_data(target_symbols)
    
    print("Fetching news from Google Docs...")
    news_text = fetch_news_document()

    # Calculate KST time
    kst_tz = datetime.timezone(datetime.timedelta(hours=9))
    now_kst = datetime.datetime.now(kst_tz)
    generation_time_str = now_kst.strftime('%Y-%m-%d %H:%M (KST)')

    # 4. Generate Two-Step Report
    print("STEP 1: Generating Market Summary...")
    market_summary_md = analyzer.generate_market_summary(
        macro_data=macro_market_data.get("normalized_global_macro_daily"),
        market_breadth=macro_market_data.get("market_breadth_daily"),
        news_text=news_text,
        generation_time=generation_time_str
    )

    print("STEP 2: Generating Stock Analysis...")
    stock_analysis_md = analyzer.generate_stock_analysis(
        market_summary=market_summary_md,
        top_volume_data=top_volume_data,
        target_stocks_data=target_stocks_data,
        generation_time=generation_time_str
    )

    # Combine report
    final_report = f"{market_summary_md}\n\n---\n\n{stock_analysis_md}"

    # 5. Save report
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    file_name = f"daily_quant_report_{now_kst.strftime('%Y%m%d_%H%M')}.md"
    file_path = reports_dir / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"Report successfully saved to: {file_path}")

    # 6. Send report via Telegram
    try:
        sender = TelegramSender()
        sent = sender.send_report(final_report)
        if sent:
            print("Telegram notification sent successfully.")
        else:
            print("Telegram notification request completed, but delivery failed.")
    except Exception as e:
        print(f"Telegram notification failed (non-fatal): {e}")

if __name__ == "__main__":
    main()
