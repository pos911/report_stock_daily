"""
run_daily_stock_pipeline.py
============================
KIS Open API를 통해 주식 [가격, 수급, 공매도, 펀더멘털] 데이터를 수집하고
Supabase normalized 테이블에 upsert 하는 파이프라인.

Usage:
    $env:PYTHONPATH="."; python src/jobs/run_daily_stock_pipeline.py
    $env:PYTHONPATH="."; python src/jobs/run_daily_stock_pipeline.py --symbols 277470,071050,012330
    $env:PYTHONPATH="."; python src/jobs/run_daily_stock_pipeline.py --start-date 2020-01-02
"""

import argparse
import datetime
import json
import sys
import time
import requests
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from supabase import create_client, Client
from src.utils import config

# ── 로깅 ──────────────────────────────────────────────────────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────────
KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"
REQUEST_DELAY = 0.4          # KIS API rate limit 대응 (초)
BATCH_UPSERT_SIZE = 500      # Supabase upsert 배치 크기


# ══════════════════════════════════════════════════════════════════════════════
# KIS API 클라이언트
# ══════════════════════════════════════════════════════════════════════════════
class KISClient:
    """한국투자증권 Open API 클라이언트 (Bearer 토큰 캐시 포함)."""

    def __init__(self):
        self.app_key    = config.get("app_key",    section="kis")
        self.app_secret = config.get("app_secret", section="kis")
        self._token: str | None = None
        self._token_expires: datetime.datetime | None = None

    # ── 인증 ──────────────────────────────────────────────────────────────────
    def _get_token(self) -> str:
        now = datetime.datetime.now()
        if self._token and self._token_expires and now < self._token_expires:
            return self._token

        url  = f"{KIS_BASE_URL}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey":     self.app_key,
            "appsecret":  self.app_secret,
        }
        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        self._token         = data["access_token"]
        expires_in          = int(data.get("expires_in", 86400))
        self._token_expires = now + datetime.timedelta(seconds=expires_in - 60)
        logger.info("KIS 액세스 토큰 발급 완료.")
        return self._token

    def _headers(self, tr_id: str, extra: dict | None = None) -> dict:
        h = {
            "content-type":  "application/json",
            "authorization": f"Bearer {self._get_token()}",
            "appkey":        self.app_key,
            "appsecret":     self.app_secret,
            "tr_id":         tr_id,
        }
        if extra:
            h.update(extra)
        return h

    # ── 주가 (일별 OHLCV) ────────────────────────────────────────────────────
    def fetch_price_history(
        self,
        symbol: str,
        start_date: str,   # "YYYYMMDD"
        end_date: str,     # "YYYYMMDD"
        market: str = "J", # J=KOSPI/KOSDAQ, NQ=NASDAQ
    ) -> list[dict]:
        """일별 OHLCV 조회 (최대 100건/요청 → 페이지네이션)."""
        url    = f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        tr_id  = "FHKST03010100"
        params = {
            "FID_COND_MRKT_DIV_CODE": market,
            "FID_INPUT_ISCD":         symbol,
            "FID_INPUT_DATE_1":       start_date,
            "FID_INPUT_DATE_2":       end_date,
            "FID_PERIOD_DIV_CODE":    "D",
            "FID_ORG_ADJ_PRC":        "0",
        }
        rows = []
        try:
            resp = requests.get(
                url,
                headers=self._headers(tr_id),
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") == "0":
                rows = data.get("output2", [])
        except Exception as e:
            logger.warning(f"[{symbol}] 가격 조회 실패: {e}")
        return rows

    # ── 수급 (일별 투자자별 순매수) ───────────────────────────────────────────
    def fetch_supply_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """일별 투자자별 순매수 조회."""
        url   = f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-investor"
        tr_id = "FHKST01010900"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD":         symbol,
            "FID_INPUT_DATE_1":       start_date,
            "FID_INPUT_DATE_2":       end_date,
            "FID_PERIOD_DIV_CODE":    "D",
        }
        rows = []
        try:
            resp = requests.get(
                url,
                headers=self._headers(tr_id),
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") == "0":
                rows = data.get("output", [])
        except Exception as e:
            logger.warning(f"[{symbol}] 수급 조회 실패: {e}")
        return rows

    # ── 공매도 ────────────────────────────────────────────────────────────────
    def fetch_short_selling(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """공매도 일별 조회."""
        url   = f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-short-sale"
        tr_id = "FHPST04560000"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD":         symbol,
            "FID_INPUT_DATE_1":       start_date,
            "FID_INPUT_DATE_2":       end_date,
        }
        rows = []
        try:
            resp = requests.get(
                url,
                headers=self._headers(tr_id),
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") == "0":
                rows = data.get("output", [])
        except Exception as e:
            logger.warning(f"[{symbol}] 공매도 조회 실패: {e}")
        return rows

    # ── 기본 정보 (stocks_master 용) ─────────────────────────────────────────
    def fetch_stock_info(self, symbol: str) -> dict | None:
        """종목 기본 정보 조회 (상장일, 시장구분 등)."""
        url   = f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/search-stock-info"
        tr_id = "CTPF1002R"
        params = {
            "PRDT_TYPE_CD": "300",
            "PDNO":         symbol,
        }
        try:
            resp = requests.get(
                url,
                headers=self._headers(tr_id),
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") == "0":
                return data.get("output")
        except Exception as e:
            logger.warning(f"[{symbol}] 종목정보 조회 실패: {e}")
        return None

    # ── PER / PBR / 시가총액 (펀더멘털) ─────────────────────────────────────
    def fetch_fundamentals(self, symbol: str) -> dict | None:
        """현재 PER/PBR/시가총액 조회."""
        url   = f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
        tr_id = "FHKST01010100"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD":         symbol,
        }
        try:
            resp = requests.get(
                url,
                headers=self._headers(tr_id),
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") == "0":
                return data.get("output")
        except Exception as e:
            logger.warning(f"[{symbol}] 펀더멘털 조회 실패: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Supabase 업서터
# ══════════════════════════════════════════════════════════════════════════════
class SupabaseUpserter:
    def __init__(self):
        url = config.get("url", section="supabase")
        key = config.get("service_role_key", section="supabase")
        self.client: Client = create_client(url, key)

    def upsert(self, table: str, rows: list[dict], conflict_cols: list[str]) -> int:
        """배치 upsert. 성공 건수 반환."""
        if not rows:
            return 0
        inserted = 0
        for i in range(0, len(rows), BATCH_UPSERT_SIZE):
            batch = rows[i : i + BATCH_UPSERT_SIZE]
            try:
                self.client.table(table).upsert(
                    batch,
                    on_conflict=",".join(conflict_cols),
                ).execute()
                inserted += len(batch)
            except Exception as e:
                logger.error(f"[{table}] upsert 실패 (batch {i}~{i+len(batch)}): {e}")
        return inserted

    def get_latest_date(self, table: str, symbol: str) -> str | None:
        """해당 종목의 테이블 내 최신 base_date 조회."""
        try:
            resp = (
                self.client.table(table)
                .select("base_date")
                .eq("symbol", symbol)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data:
                return resp.data[0]["base_date"]
        except Exception:
            pass
        return None

    def upsert_master(self, symbol: str, name: str, market: str, listed_date: str | None):
        """stocks_master upsert."""
        row = {
            "symbol":      symbol,
            "name":        name,
            "market":      market,
            "is_active":   True,
        }
        if listed_date:
            row["listed_date"] = listed_date
        try:
            self.client.table("stocks_master").upsert(
                row, on_conflict="symbol"
            ).execute()
        except Exception as e:
            logger.warning(f"[{symbol}] stocks_master upsert 실패: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 변환 헬퍼
# ══════════════════════════════════════════════════════════════════════════════
def _safe_float(val) -> float | None:
    try:
        v = float(str(val).replace(",", "").strip())
        return None if v == 0.0 and str(val).strip() == "" else v
    except Exception:
        return None

def _safe_int(val) -> int | None:
    try:
        return int(str(val).replace(",", "").strip())
    except Exception:
        return None

def _fmt_date(raw: str) -> str | None:
    """YYYYMMDD → YYYY-MM-DD"""
    raw = str(raw).strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 종목별 파이프라인
# ══════════════════════════════════════════════════════════════════════════════
def _date_range_chunks(start: str, end: str, chunk_days: int = 90) -> list[tuple[str, str]]:
    """start~end 범위를 chunk_days 단위로 분할 (YYYY-MM-DD)."""
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    chunks = []
    cur = s
    while cur <= e:
        chunk_end = min(cur + datetime.timedelta(days=chunk_days - 1), e)
        chunks.append((cur.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cur = chunk_end + datetime.timedelta(days=1)
    return chunks


def run_symbol(
    symbol: str,
    name: str,
    market: str,
    start_date: str,          # YYYY-MM-DD
    end_date: str,            # YYYY-MM-DD
    kis: KISClient,
    supa: SupabaseUpserter,
    skip_master: bool = False,
):
    logger.info(f"━━ [{symbol}] {name} 처리 시작 ({start_date} ~ {end_date}) ━━")

    # 1) stocks_master
    if not skip_master:
        info = kis.fetch_stock_info(symbol)
        listed_date = None
        if info:
            raw_ld = info.get("lstg_stcn_dt") or info.get("scts_dvsn_cd")
            listed_date = _fmt_date(raw_ld) if raw_ld else None
        supa.upsert_master(symbol, name, market, listed_date)
        logger.info(f"  [1] stocks_master upsert 완료 (listed={listed_date})")
        time.sleep(REQUEST_DELAY)

    # 2) normalized_stock_prices_daily
    price_rows = []
    for s_chunk, e_chunk in _date_range_chunks(start_date, end_date):
        rows = kis.fetch_price_history(symbol, s_chunk, e_chunk)
        for r in rows:
            d = _fmt_date(r.get("stck_bsop_date", ""))
            if not d:
                continue
            price_rows.append({
                "symbol":    symbol,
                "base_date": d,
                "open":      _safe_float(r.get("stck_oprc")),
                "high":      _safe_float(r.get("stck_hgpr")),
                "low":       _safe_float(r.get("stck_lwpr")),
                "close":     _safe_float(r.get("stck_clpr")),
                "volume":    _safe_float(r.get("acml_vol")),
                "amount":    _safe_float(r.get("acml_tr_pbmn")),
            })
        time.sleep(REQUEST_DELAY)
    n = supa.upsert(
        "normalized_stock_prices_daily",
        price_rows,
        ["symbol", "base_date"],
    )
    logger.info(f"  [2] 가격 데이터 {n}건 upsert")

    # 3) normalized_stock_supply_daily
    supply_rows = []
    for s_chunk, e_chunk in _date_range_chunks(start_date, end_date):
        rows = kis.fetch_supply_history(symbol, s_chunk, e_chunk)
        for r in rows:
            d = _fmt_date(r.get("stck_bsop_date", ""))
            if not d:
                continue
            supply_rows.append({
                "symbol":                 symbol,
                "base_date":              d,
                "individual_net_buy":     _safe_float(r.get("prsn_ntby_qty")),
                "foreign_net_buy":        _safe_float(r.get("frgn_ntby_qty")),
                "institutional_net_buy":  _safe_float(r.get("orgn_ntby_qty")),
                "pension_net_buy":        _safe_float(r.get("pnsn_ntby_qty")),
                "corporate_net_buy":      _safe_float(r.get("corp_ntby_qty")),
            })
        time.sleep(REQUEST_DELAY)
    n = supa.upsert(
        "normalized_stock_supply_daily",
        supply_rows,
        ["symbol", "base_date"],
    )
    logger.info(f"  [3] 수급 데이터 {n}건 upsert")

    # 4) normalized_stock_short_selling
    short_rows = []
    for s_chunk, e_chunk in _date_range_chunks(start_date, end_date):
        rows = kis.fetch_short_selling(symbol, s_chunk, e_chunk)
        for r in rows:
            d = _fmt_date(r.get("stck_bsop_date", ""))
            if not d:
                continue
            short_rows.append({
                "symbol":            symbol,
                "base_date":         d,
                "short_sell_volume": _safe_float(r.get("smtn_slby_qty")),
                "short_sell_amount": _safe_float(r.get("smtn_slby_tr_pbmn")),
                "short_sell_ratio":  _safe_float(r.get("slby_tr_pbmn_smtn_rate")),
            })
        time.sleep(REQUEST_DELAY)
    n = supa.upsert(
        "normalized_stock_short_selling",
        short_rows,
        ["symbol", "base_date"],
    )
    logger.info(f"  [4] 공매도 데이터 {n}건 upsert")

    # 5) normalized_stock_fundamentals_ratios (오늘 기준)
    fund = kis.fetch_fundamentals(symbol)
    if fund:
        today = end_date  # YYYY-MM-DD
        fund_row = {
            "symbol":     symbol,
            "base_date":  today,
            "per":        _safe_float(fund.get("per")),
            "pbr":        _safe_float(fund.get("pbr")),
            "market_cap": _safe_float(fund.get("hts_avls")),  # 억 단위
        }
        n = supa.upsert(
            "normalized_stock_fundamentals_ratios",
            [fund_row],
            ["symbol", "base_date"],
        )
        logger.info(f"  [5] 펀더멘털 {n}건 upsert (PER={fund_row['per']}, PBR={fund_row['pbr']})")
    time.sleep(REQUEST_DELAY)

    logger.info(f"━━ [{symbol}] 완료 ━━")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args():
    parser = argparse.ArgumentParser(description="Daily Stock Pipeline")
    parser.add_argument(
        "--symbols",
        default=None,
        help="쉼표 구분 종목 코드. 미지정 시 config/target_stocks.json 전체 사용.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="수집 시작일 YYYY-MM-DD. 미지정 시 테이블 최신일 다음 날 자동 계산.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="수집 종료일 YYYY-MM-DD. 미지정 시 오늘.",
    )
    return parser.parse_args()


def _load_universe(symbols_arg: str | None) -> list[dict]:
    path = project_root / "config" / "target_stocks.json"
    with open(path, encoding="utf-8") as f:
        all_stocks = json.load(f)
    enabled = [s for s in all_stocks if s.get("enabled", True)]

    if symbols_arg:
        codes = {s.strip() for s in symbols_arg.split(",")}
        # 지정 코드가 유니버스에 없으면 기본 KOSPI로 추가
        found = {s["symbol"]: s for s in enabled}
        result = []
        for code in codes:
            if code in found:
                result.append(found[code])
            else:
                result.append({"symbol": code, "name": code, "market": "KOSPI"})
        return result

    return enabled


def main():
    args = _parse_args()

    kst_tz  = datetime.timezone(datetime.timedelta(hours=9))
    today   = datetime.datetime.now(kst_tz).strftime("%Y-%m-%d")
    end_date = args.end_date or today

    universe = _load_universe(args.symbols)
    logger.info(f"대상 종목: {[s['symbol'] for s in universe]}")

    kis  = KISClient()
    supa = SupabaseUpserter()

    for stock in universe:
        symbol = stock["symbol"]
        name   = stock.get("name", symbol)
        market = stock.get("market", "KOSPI")

        # 시작일 결정: 인수 > 테이블 최신일 +1 > 기본 과거
        if args.start_date:
            start_date = args.start_date
        else:
            latest = supa.get_latest_date("normalized_stock_prices_daily", symbol)
            if latest:
                next_day = (
                    datetime.date.fromisoformat(latest) + datetime.timedelta(days=1)
                ).isoformat()
                start_date = next_day
                logger.info(f"[{symbol}] 최신 데이터: {latest} → {start_date}부터 수집")
            else:
                start_date = "2020-01-02"   # 상장 이후 기본 시작일
                logger.info(f"[{symbol}] 데이터 없음 → {start_date}부터 전체 수집")

        if start_date > end_date:
            logger.info(f"[{symbol}] 이미 최신 상태 (start={start_date} > end={end_date}). 건너뜀.")
            continue

        try:
            run_symbol(
                symbol=symbol,
                name=name,
                market=market,
                start_date=start_date,
                end_date=end_date,
                kis=kis,
                supa=supa,
            )
        except Exception as e:
            logger.error(f"[{symbol}] 처리 중 오류: {e}", exc_info=True)

    logger.info("=== 주식 파이프라인 완료 ===")


if __name__ == "__main__":
    main()
