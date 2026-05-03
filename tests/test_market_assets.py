import unittest

from src.utils.market_assets import (
    canonicalize_symbol,
    deduplicate_by_canonical_symbol,
    infer_asset_type,
    is_allowed_ranking_source,
    is_common_stock_top_eligible,
    ranking_market_matches_master,
)


class MarketAssetTests(unittest.TestCase):
    def test_canonicalize_symbol(self):
        self.assertEqual(canonicalize_symbol("Q530036"), "530036")
        self.assertEqual(canonicalize_symbol("5930"), "005930")
        self.assertEqual(canonicalize_symbol(" 071050 "), "071050")

    def test_infer_asset_type(self):
        self.assertEqual(infer_asset_type("삼성 인버스 2X WTI원유 선물 ETN", "ETN", "530036"), "ETN")
        self.assertEqual(infer_asset_type("KODEX 2차전지산업레버리지", "ETF", "305720"), "ETF")
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
        self.assertTrue(is_common_stock_top_eligible({"market": "KOSPI", "asset_type": "COMMON_STOCK"}))
        self.assertFalse(is_common_stock_top_eligible({"market": "ETF", "asset_type": "ETF"}))

    def test_ranking_market_matches_master(self):
        self.assertTrue(ranking_market_matches_master("KOSPI", "KOSPI"))
        self.assertTrue(ranking_market_matches_master("KOSPI200", "KOSPI"))
        self.assertFalse(ranking_market_matches_master("KOSDAQ", "KOSPI"))

    def test_allowed_ranking_source(self):
        self.assertTrue(is_allowed_ranking_source("volume", "KIS"))
        self.assertTrue(is_allowed_ranking_source("trading_value", "KRX"))
        self.assertTrue(is_allowed_ranking_source("market_cap", "VALID_PRICE_FALLBACK"))
        self.assertFalse(is_allowed_ranking_source("trading_value", "KIS"))
        self.assertFalse(is_allowed_ranking_source("market_cap", "KIS"))


if __name__ == "__main__":
    unittest.main()
