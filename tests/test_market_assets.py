import unittest

from src.utils.market_assets import (
    canonicalize_symbol,
    deduplicate_by_canonical_symbol,
    infer_asset_type,
    is_common_stock_top_eligible,
)


class MarketAssetTests(unittest.TestCase):
    def test_canonicalize_symbol(self):
        self.assertEqual(canonicalize_symbol("Q530036"), "530036")
        self.assertEqual(canonicalize_symbol("71050"), "071050")

    def test_infer_asset_type(self):
        self.assertEqual(infer_asset_type("삼성 인버스 2X WTI원유 선물 ETN", "KOSPI", "530036"), "ETN")
        self.assertEqual(infer_asset_type("KODEX 2차전지산업레버리지", "KOSPI", "305720"), "ETF")
        self.assertEqual(infer_asset_type("삼성전자", "KOSPI", "005930"), "COMMON_STOCK")

    def test_deduplicate_prefers_non_q_row(self):
        rows, duplicates = deduplicate_by_canonical_symbol(
            [
                {"symbol": "Q530036", "trading_value": 100, "volume": 10, "close_price": 1000},
                {"symbol": "530036", "trading_value": 100, "volume": 10, "close_price": 1000},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "530036")
        self.assertEqual(len(duplicates), 1)

    def test_common_stock_eligibility(self):
        row = {"market": "KOSPI", "asset_type": "COMMON_STOCK"}
        self.assertTrue(is_common_stock_top_eligible(row))


if __name__ == "__main__":
    unittest.main()

