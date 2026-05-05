import unittest
from unittest.mock import patch

from scripts.cleanup_raw_tables import build_cleanup_plan, execute_cleanup


class _FakeResponse:
    def __init__(self, data=None):
        self.data = data or []


class _FakeQuery:
    def __init__(self, table_name, store):
        self.table_name = table_name
        self.store = store
        self.cutoff = None

    def select(self, *_args, **_kwargs):
        return self

    def lt(self, _column, cutoff):
        self.cutoff = cutoff
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def delete(self):
        self.store["deleted"].append(self.table_name)
        return self

    def insert(self, payload):
        self.store["logs"].append(payload)
        return self

    def execute(self):
        count = self.store["counts"].get(self.table_name, 0)
        return _FakeResponse([{"row": i} for i in range(count)])


class _FakeClient:
    def __init__(self, counts):
        self.store = {"counts": counts, "deleted": [], "logs": []}

    def table(self, table_name):
        return _FakeQuery(table_name, self.store)


class CleanupRawTablesTests(unittest.TestCase):
    def test_build_cleanup_plan_uses_retention_cutoffs(self):
        plan = build_cleanup_plan("2026-05-05")
        first = next(item for item in plan if item["table"] == "raw_stock_prices_daily")
        self.assertEqual(first["cutoff_date"], "2026-03-06")

    def test_execute_cleanup_dry_run_does_not_delete(self):
        fake = _FakeClient({"raw_stock_prices_daily": 3, "raw_market_rankings": 1})
        with patch("scripts.cleanup_raw_tables._client", return_value=fake):
            summary = execute_cleanup("2026-05-05", apply=False)
        self.assertEqual(sum(item["candidate_rows"] for item in summary["tables"]), 4)
        self.assertEqual(fake.store["deleted"], [])
        self.assertTrue(fake.store["logs"])


if __name__ == "__main__":
    unittest.main()
