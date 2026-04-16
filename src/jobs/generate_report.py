import os
import sys
import argparse
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

VALID_REPORT_TYPES = ("morning", "closing", "regular")


def _parse_args():
    """CLI 인자 파싱. --type morning|closing|regular"""
    parser = argparse.ArgumentParser(description="Daily Quant Report Generator")
    parser.add_argument(
        "--type",
        dest="report_type",
        choices=VALID_REPORT_TYPES,
        default="regular",
        help="리포트 유형: morning(07:00), closing(15:30), regular(기타). 기본값: regular",
    )
    return parser.parse_args()


def _validate_top_volume(top_volume_data: dict) -> bool:
    """KOSPI/KOSDAQ/ETF 중 하나라도 데이터가 있으면 True."""
    if not top_volume_data:
        return False
    return any(isinstance(v, list) and len(v) > 0 for v in top_volume_data.values())


def _validate_target_stocks(target_stocks_data: dict) -> bool:
    """하나 이상의 테이블에 데이터가 있으면 True."""
    if not target_stocks_data:
        return False
    return any(isinstance(v, list) and len(v) > 0 for v in target_stocks_data.values())


def main():
    args = _parse_args()
    report_type = args.report_type

    logger.info(f"=== Daily Report Pipeline 시작 [type={report_type}] ===")

    # 1. 타겟 종목 로드
    target_stocks_path = project_root / "config" / "target_stocks.json"
    target_symbols = []
    if target_stocks_path.exists():
        with open(target_stocks_path, "r", encoding="utf-8") as f:
            targets = json.load(f)
            target_symbols = [t["symbol"] for t in targets if t.get("enabled", True)]

    # 2. 모듈 초기화
    try:
        reader = SupabaseReader()
        analyzer = GeminiAnalyzer()
    except Exception as e:
        logger.error(f"모듈 초기화 실패: {e}")
        return

    # 3. 데이터 수집 (정해진 순서: 매크로 -> 상위종목 농축 -> 타겟종목)
    logger.info("1. 매크로/시장 폭 데이터 수집 중...")
    macro_market_data = reader.fetch_macro_and_market_data()
    macro_data = macro_market_data.get("normalized_macro_series")
    
    if not macro_data:
        logger.error("매크로 데이터를 수집하지 못했습니다. 시황 분석 품질이 저하될 수 있습니다.")

    logger.info("2. 거래량 상위 종목 농축 수집 중 (KOSPI/KOSDAQ/ETF 각 10개)...")
    top_volume_data = reader.fetch_top_volume_stocks(limit=10)
    if not _validate_top_volume(top_volume_data):
        logger.error("거래량 상위 종목 데이터를 수집하지 못했습니다. (Clean Skip 대상)")
        top_volume_data = None

    logger.info(f"3. 타겟 종목 데이터 수집 중 ({len(target_symbols)}개)...")
    if target_symbols:
        target_stocks_data = reader.fetch_target_stocks_data(target_symbols)
        if not _validate_target_stocks(target_stocks_data):
            logger.error("타겟 종목 데이터를 수집하지 못했습니다. (Clean Skip 대상)")
            target_stocks_data = {}
    else:
        target_stocks_data = {}

    logger.info("4. 뉴스 문서 수집 중...")
    news_text = fetch_news_document()

    # KST 시간 계산
    kst_tz = datetime.timezone(datetime.timedelta(hours=9))
    now_kst = datetime.datetime.now(kst_tz)
    generation_time_str = now_kst.strftime("%Y-%m-%d %H:%M (KST)")

    # 리포트 유형별 타이틀 설정
    type_label_map = {
        "morning": "🌅 Morning Briefing (07:00 KST)",
        "closing": "📊 Closing Analysis (15:30 KST)",
        "regular": "📋 Regular Report",
    }
    report_label = type_label_map.get(report_type, "Daily Report")

    # 4. 3단계 리포트 생성 (Assembly)
    report_content = (
        f"# Daily Quant Report — {report_label}\n"
        f"> **Generated at**: {generation_time_str}\n\n"
    )

    # 4-1. STEP 1: Market Summary
    logger.info(f"STEP 1: Market Summary 생성 중 [{report_type}]...")
    market_summary_md = analyzer.generate_market_summary(
        macro_data=macro_data,
        market_breadth=macro_market_data.get("market_breadth_daily"),
        momentum_data=macro_market_data.get("momentum"),
        news_text=news_text,
        generation_time=generation_time_str,
        report_type=report_type,
    )
    report_content += f"## 1. Market Summary\n\n{market_summary_md.strip()}\n\n---\n\n"

    # 4-2. STEP 2: Top Volume Analysis (신규 분리 메서드)
    if top_volume_data:
        logger.info(f"STEP 2: Top Volume & Smart Money 분석 중...")
        top_volume_md = analyzer.generate_top_volume_analysis(
            top_volume_data=top_volume_data,
            report_type=report_type
        )
        report_content += f"## 2. Top Volume & Smart Money\n\n{top_volume_md.strip()}\n\n---\n\n"
    else:
        logger.warning("거래량 데이터 부재로 STEP 2를 건너뜁니다.")

    # 4-3. STEP 3: Target Stock Analysis
    if target_stocks_data:
        logger.info(f"STEP 3: Target Stock 상세 분석 중...")
        stock_analysis_md = analyzer.generate_stock_analysis(
            market_summary=market_summary_md,
            target_stocks_data=target_stocks_data,
            macro_market_data=macro_market_data,
            generation_time=generation_time_str,
            report_type=report_type,
        )
        report_content += f"## 3. Stock Analysis & Strategy\n\n{stock_analysis_md.strip()}"
    else:
        logger.warning("타겟 종목 데이터 부재로 STEP 3을 건너뜁니다.")

    # 5. 리포트 저장
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    file_name = f"daily_quant_report_{report_type}_{now_kst.strftime('%Y%m%d_%H%M')}.md"
    file_path = reports_dir / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"리포트 저장 완료: {file_path}")

    # 6. 텔레그램 발송
    try:
        sender = TelegramSender()
        sent = sender.send_report(report_content)
        if sent:
            logger.info("텔레그램 발송 성공.")
        else:
            logger.warning("텔레그램 발송 요청 완료, 전달 실패.")
    except Exception as e:
        logger.warning(f"텔레그램 발송 실패 (non-fatal): {e}")


if __name__ == "__main__":
    main()
