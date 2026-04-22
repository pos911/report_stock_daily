"""
scripts/verify_data.py
=======================
데이터 적재 정합성 검증 스크립트.

- normalized_stock_prices_daily 기준 최신 base_date 확인
- 3개 신규 종목(277470, 071050, 012330) 포함 여부 확인
- normalized_macro_series 내 KOSPI/KOSDAQ 최신 적재 확인

Usage:
    $env:PYTHONPATH="."; python scripts/verify_data.py
    $env:PYTHONPATH="."; python scripts/verify_data.py --date 2026-04-23
"""

import argparse
import datetime
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from supabase import create_client
from src.utils import config

CHECK_DATE_DEFAULT = "2026-04-23"
TARGET_SYMBOLS = ["277470", "071050", "012330"]
INDEX_SERIES   = ["^KS", "^KQ"]

PRICE_TABLE    = "normalized_stock_prices_daily"
SUPPLY_TABLE   = "normalized_stock_supply_daily"
SHORT_TABLE    = "normalized_stock_short_selling"
FUND_TABLE     = "normalized_stock_fundamentals_ratios"
MACRO_TABLE    = "normalized_macro_series"
MASTER_TABLE   = "stocks_master"


def _client():
    url = config.get("url", section="supabase")
    key = config.get("service_role_key", section="supabase")
    return create_client(url, key)


def check_price_latest(client, check_date: str) -> dict:
    """normalized_stock_prices_daily 최신일 및 대상 종목 포함 여부 확인."""
    result = {"status": "OK", "issues": []}

    # 1) 전체 최신 base_date
    resp = (
        client.table(PRICE_TABLE)
        .select("base_date")
        .order("base_date", desc=True)
        .limit(1)
        .execute()
    )
    latest = resp.data[0]["base_date"] if resp.data else None
    result["latest_base_date"] = latest

    if not latest:
        result["status"] = "FAIL"
        result["issues"].append("테이블이 비어있습니다.")
        return result

    # 2) 기준일 데이터 존재 여부
    if latest < check_date:
        result["status"] = "WARN"
        result["issues"].append(
            f"최신 base_date({latest})가 체크 기준일({check_date})보다 오래됨."
        )

    # 3) 대상 종목별 확인
    symbol_status = {}
    for sym in TARGET_SYMBOLS:
        sym_resp = (
            client.table(PRICE_TABLE)
            .select("base_date, close_price, volume")
            .eq("symbol", sym)
            .order("base_date", desc=True)
            .limit(1)
            .execute()
        )
        if sym_resp.data:
            row = sym_resp.data[0]
            symbol_status[sym] = {
                "latest_date": row["base_date"],
                "close":       row.get("close_price"),
                "volume":      row.get("volume"),
                "up_to_date":  row["base_date"] >= check_date,
            }
        else:
            symbol_status[sym] = {"latest_date": None, "up_to_date": False}
            result["status"] = "FAIL"
            result["issues"].append(f"{sym}: 가격 데이터 없음")

    result["symbol_status"] = symbol_status
    return result


def check_supply(client, check_date: str) -> dict:
    result = {"status": "OK", "issues": []}
    for sym in TARGET_SYMBOLS:
        try:
            resp = (
                client.table(SUPPLY_TABLE)
                .select("base_date, investor_type, net_buy_vol")
                .eq("symbol", sym)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
        except Exception:
            # Fallback to wide format
            resp = (
                client.table(SUPPLY_TABLE)
                .select("base_date, individual_net_buy, foreign_net_buy")
                .eq("symbol", sym)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
        if not resp.data:
            result["status"] = "WARN"
            result["issues"].append(f"{sym}: 수급 데이터 없음")
    return result


def check_short_selling(client, check_date: str) -> dict:
    result = {"status": "OK", "issues": []}
    for sym in TARGET_SYMBOLS:
        try:
            resp = (
                client.table(SHORT_TABLE)
                .select("base_date, short_volume")
                .eq("symbol", sym)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
        except Exception:
            resp = (
                client.table(SHORT_TABLE)
                .select("base_date, short_sell_volume")
                .eq("symbol", sym)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
        if not resp.data:
            result["status"] = "WARN"
            result["issues"].append(f"{sym}: 공매도 데이터 없음")
    return result


def check_fundamentals(client, check_date: str) -> dict:
    result = {"status": "OK", "issues": []}
    for sym in TARGET_SYMBOLS:
        resp = (
            client.table(FUND_TABLE)
            .select("base_date, per, pbr")
            .eq("symbol", sym)
            .order("base_date", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            result["status"] = "WARN"
            result["issues"].append(f"{sym}: 펀더멘털 데이터 없음")
        else:
            row = resp.data[0]
            result[sym] = {"date": row["base_date"], "per": row.get("per"), "pbr": row.get("pbr")}
    return result


def check_macro_index(client, check_date: str) -> dict:
    result = {"status": "OK", "issues": []}
    for series_id in INDEX_SERIES:
        try:
            resp = (
                client.table(MACRO_TABLE)
                .select("base_date, close_val")
                .eq("series_id", series_id)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            col = "close_val"
        except Exception:
            resp = (
                client.table(MACRO_TABLE)
                .select("base_date, value")
                .eq("series_id", series_id)
                .order("base_date", desc=True)
                .limit(1)
                .execute()
            )
            col = "value"
        
        if resp.data:
            row = resp.data[0]
            result[series_id] = {"latest_date": row["base_date"], "value": row.get(col)}
            if row["base_date"] < check_date:
                result["status"] = "WARN"
                result["issues"].append(
                    f"{series_id}: 최신일({row['base_date']}) < 체크기준일({check_date})"
                )
        else:
            result["status"] = "FAIL"
            result["issues"].append(f"{series_id}: 데이터 없음")
    return result


def check_master(client) -> dict:
    result = {"status": "OK", "issues": []}
    resp = (
        client.table(MASTER_TABLE)
        .select("symbol, name, market, is_active")
        .in_("symbol", TARGET_SYMBOLS)
        .execute()
    )
    found = {r["symbol"]: r for r in (resp.data or [])}
    for sym in TARGET_SYMBOLS:
        if sym not in found:
            result["status"] = "WARN"
            result["issues"].append(f"{sym}: stocks_master 에 없음")
        else:
            result[sym] = found[sym]
    return result


def _status_icon(status: str) -> str:
    return {"OK": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(status, "❓")


def main():
    parser = argparse.ArgumentParser(description="Data Integrity Verifier")
    parser.add_argument("--date", default=CHECK_DATE_DEFAULT, help="검증 기준일 YYYY-MM-DD")
    args = parser.parse_args()
    check_date = args.date

    print(f"\n{'='*60}")
    print(f"  데이터 정합성 검증 리포트")
    print(f"  기준일: {check_date}  /  대상 종목: {TARGET_SYMBOLS}")
    print(f"{'='*60}\n")

    client = _client()
    overall_ok = True

    checks = [
        ("stocks_master",              check_master(client)),
        ("가격 (prices_daily)",         check_price_latest(client, check_date)),
        ("수급 (supply_daily)",         check_supply(client, check_date)),
        ("공매도 (short_selling)",      check_short_selling(client, check_date)),
        ("펀더멘털 (fundamentals)",     check_fundamentals(client, check_date)),
        ("지수 (macro_series KOSPI/Q)", check_macro_index(client, check_date)),
    ]

    for label, res in checks:
        icon = _status_icon(res["status"])
        print(f"{icon}  [{res['status']}] {label}")
        issues = res.get("issues", [])
        for issue in issues:
            print(f"      → {issue}")
        if res["status"] != "OK":
            overall_ok = False
        # 상세 정보 출력
        for sym in TARGET_SYMBOLS:
            if sym in res:
                print(f"      {sym}: {res[sym]}")
        for idx in INDEX_SERIES:
            if idx in res:
                print(f"      {idx}: {res[idx]}")
        print()

    print("─" * 60)
    if overall_ok:
        print("✅  전체 검증 통과 — 데이터 정합성 OK\n")
    else:
        print("⚠️  일부 항목 검증 실패 — 위 내용을 확인하세요\n")


if __name__ == "__main__":
    main()
