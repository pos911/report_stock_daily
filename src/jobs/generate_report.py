import os
import sys
import datetime
import logging
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

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _validate_top_volume(top_volume_data: dict) -> bool:
    """
    top_volume_data가 유효한지 검사.
    KOSPI, KOSDAQ, ETF 모두 비어 있으면 False 반환.
    """
    if not top_volume_data:
        return False
    has_any = any(isinstance(v, list) and len(v) > 0 for v in top_volume_data.values())
    return has_any


def _validate_target_stocks(target_stocks_data: dict) -> bool:
    """
    target_stocks_data가 유효한지 검사.
    딕셔너리 자체가 비어 있거나, 모든 테이블에 데이터가 없으면 False 반환.
    """
    if not target_stocks_data:
        return False
    has_any = any(isinstance(v, list) and len(v) > 0 for v in target_stocks_data.values())
    return has_any


def main():
    logger.info("Starting daily report generation pipeline (Two-Step)...")
    
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
        logger.error(f"Error initializing modules: {e}")
        return

    # 3. Fetch data (Separated concerns)
    logger.info("Fetching macro & market breadth data from Supabase...")
    macro_market_data = reader.fetch_macro_and_market_data()
    
    logger.info("Fetching Top Volume Stocks (KOSPI/KOSDAQ/ETF, 각 10개)...")
    top_volume_data = reader.fetch_top_volume_stocks(limit=10)

    # [방어 로직] top_volume_data 유효성 검사
    if not _validate_top_volume(top_volume_data):
        logger.warning(
            "[SKIP] top_volume_data가 비어 있습니다. "
            "feature_store_daily에 데이터가 없거나 날짜 조회에 실패했을 수 있습니다. "
            "Stock Analysis 섹션을 스킵합니다."
        )
        top_volume_data = None  # 하단에서 스킵 여부 판단

    logger.info(f"Fetching Target Stocks Data ({len(target_symbols)} symbols)...")
    if target_symbols:
        target_stocks_data = reader.fetch_target_stocks_data(target_symbols)
        # [방어 로직] target_stocks_data 유효성 검사
        if not _validate_target_stocks(target_stocks_data):
            logger.warning(
                "[SKIP] target_stocks_data가 비어 있습니다. "
                "해당 심볼에 대한 데이터가 Supabase에 없을 수 있습니다."
            )
            target_stocks_data = {}
    else:
        logger.warning("target_symbols가 비어 있습니다. target_stocks.json을 확인하세요.")
        target_stocks_data = {}

    logger.info("Fetching news from Google Docs...")
    news_text = fetch_news_document()

    # Calculate KST time
    kst_tz = datetime.timezone(datetime.timedelta(hours=9))
    now_kst = datetime.datetime.now(kst_tz)
    generation_time_str = now_kst.strftime('%Y-%m-%d %H:%M (KST)')

    # 4. Generate Two-Step Report
    logger.info("STEP 1: Generating Market Summary...")
    market_summary_md = analyzer.generate_market_summary(
        macro_data=macro_market_data.get("normalized_global_macro_daily"),
        market_breadth=macro_market_data.get("market_breadth_daily"),
        news_text=news_text,
        generation_time=generation_time_str
    )

    # [방어 로직] top_volume_data 또는 target_stocks_data가 없으면 Stock Analysis 스킵
    if top_volume_data is None and not target_stocks_data:
        logger.warning(
            "[SKIP] top_volume_data와 target_stocks_data 모두 비어 있어 "
            "Stock Analysis 섹션 생성을 건너뜁니다."
        )
        stock_analysis_md = "_거래량 및 종목 데이터를 조회하지 못해 종목 분석이 생략되었습니다._"
    else:
        logger.info("STEP 2: Generating Stock Analysis...")
        stock_analysis_md = analyzer.generate_stock_analysis(
            market_summary=market_summary_md,
            top_volume_data=top_volume_data or {},
            target_stocks_data=target_stocks_data,
            generation_time=generation_time_str
        )

    # Combine report with well-structured headers (assembly logic)
    final_report = (
        f"# Daily Quant Report\n"
        f"> **Generated at**: {generation_time_str}\n\n"
        f"## 1. Market Summary\n\n"
        f"{market_summary_md.strip()}\n\n"
        f"---\n\n"
        f"## 2. Stock Analysis & Strategy\n\n"
        f"{stock_analysis_md.strip()}"
    )

    # 5. Save report
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    file_name = f"daily_quant_report_{now_kst.strftime('%Y%m%d_%H%M')}.md"
    file_path = reports_dir / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    logger.info(f"Report successfully saved to: {file_path}")

    # 6. Send report via Telegram
    try:
        sender = TelegramSender()
        sent = sender.send_report(final_report)
        if sent:
            logger.info("Telegram notification sent successfully.")
        else:
            logger.warning("Telegram notification request completed, but delivery failed.")
    except Exception as e:
        logger.warning(f"Telegram notification failed (non-fatal): {e}")

if __name__ == "__main__":
    main()
