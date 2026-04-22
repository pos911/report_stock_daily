"""
run_daily_macro_pipeline.py
============================
KOSPI / KOSDAQ 지수 데이터를 Yahoo Finance(yfinance)로 수집하여
Supabase normalized_macro_series 테이블에 upsert 하는 파이프라인.

series_id 규칙:
  KOSPI  → "KOSPI"
  KOSDAQ → "KOSDAQ"

Usage:
    $env:PYTHONPATH="."; python src/jobs/run_daily_macro_pipeline.py
    $env:PYTHONPATH="."; python src/jobs/run_daily_macro_pipeline.py --start-date 2020-01-02
"""

import argparse
import datetime
import sys
import time
import logging
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

try:
    import yfinance as yf
except ImportError:
    print("[ERROR] yfinance 가 설치되어 있지 않습니다. pip install yfinance 를 실행하세요.")
    sys.exit(1)

import pandas as pd
from supabase import create_client, Client
from src.utils import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BATCH_UPSERT_SIZE = 500

# KOSPI: ^KS11, KOSDAQ: ^KQ11 (Yahoo Finance 공식 티커)
INDEX_MAP = {
    "KOSPI":  "^KS11",
    "KOSDAQ": "^KQ11",
}


# ══════════════════════════════════════════════════════════════════════════════
# Supabase 헬퍼
# ══════════════════════════════════════════════════════════════════════════════
class SupabaseUpserter:
    def __init__(self):
        url = config.get("url", section="supabase")
        key = config.get("service_role_key", section="supabase")
        self.client: Client = create_client(url, key)

    def get_latest_date(self, series_id: str) -> str | None:
        try:
            resp = (
                self.client.table("normalized_macro_series")
                .select("base_date")
                .eq("series_id", series_id)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data:
                return resp.data[0]["base_date"]
        except Exception:
            pass
        return None

    def upsert(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        inserted = 0
        for i in range(0, len(rows), BATCH_UPSERT_SIZE):
            batch = rows[i : i + BATCH_UPSERT_SIZE]
            try:
                self.client.table("normalized_macro_series").upsert(
                    batch, on_conflict="series_id,base_date"
                ).execute()
                inserted += len(batch)
            except Exception as e:
                logger.error(f"upsert 실패 (batch {i}~): {e}")
        return inserted


# ══════════════════════════════════════════════════════════════════════════════
# 지수 수집
# ══════════════════════════════════════════════════════════════════════════════
def fetch_index(series_id: str, ticker: str, start: str, end: str) -> list[dict]:
    """yfinance로 지수 일별 종가를 수집하여 normalized_macro_series 형식으로 반환."""
    logger.info(f"  [{series_id}] {ticker} 수집 중 ({start} ~ {end}) ...")
    try:
        df = yf.download(
            ticker,
            start=start,
            end=(datetime.date.fromisoformat(end) + datetime.timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        logger.warning(f"  [{series_id}] yfinance 다운로드 실패: {e}")
        return []

    if df is None or df.empty:
        logger.warning(f"  [{series_id}] 데이터 없음 (ticker={ticker})")
        return []

    # MultiIndex 컬럼 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rows = []
    for date_idx, row in df.iterrows():
        date_str = pd.Timestamp(date_idx).strftime("%Y-%m-%d")
        close_val = row.get("Close")
        if pd.isna(close_val):
            continue
        rows.append({
            "series_id": series_id,
            "base_date": date_str,
            "value":     float(close_val),
            "unit":      "points",
            "source":    "yfinance",
        })

    logger.info(f"  [{series_id}] {len(rows)}건 수집 완료")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args():
    parser = argparse.ArgumentParser(description="Daily Macro Pipeline (KOSPI/KOSDAQ)")
    parser.add_argument("--start-date", default=None, help="수집 시작일 YYYY-MM-DD")
    parser.add_argument("--end-date",   default=None, help="수집 종료일 YYYY-MM-DD")
    return parser.parse_args()


def main():
    args = _parse_args()
    kst_tz  = datetime.timezone(datetime.timedelta(hours=9))
    today   = datetime.datetime.now(kst_tz).strftime("%Y-%m-%d")
    end_date = args.end_date or today

    supa = SupabaseUpserter()

    all_rows = []
    for series_id, ticker in INDEX_MAP.items():
        # 시작일: 인수 > 테이블 최신일 +1 > 기본값
        if args.start_date:
            start_date = args.start_date
        else:
            latest = supa.get_latest_date(series_id)
            if latest:
                start_date = (
                    datetime.date.fromisoformat(latest) + datetime.timedelta(days=1)
                ).isoformat()
                logger.info(f"[{series_id}] 최신={latest} → {start_date}부터 수집")
            else:
                start_date = "2020-01-02"
                logger.info(f"[{series_id}] 데이터 없음 → {start_date}부터 전체 수집")

        if start_date > end_date:
            logger.info(f"[{series_id}] 이미 최신 상태. 건너뜀.")
            continue

        rows = fetch_index(series_id, ticker, start_date, end_date)
        all_rows.extend(rows)

    n = supa.upsert(all_rows)
    logger.info(f"=== 매크로 파이프라인 완료: {n}건 적재 ===")


if __name__ == "__main__":
    main()
