from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WorkflowContractTests(unittest.TestCase):
    def test_daily_report_is_single_job(self):
        content = (PROJECT_ROOT / ".github/workflows/daily_report.yml").read_text(encoding="utf-8")
        self.assertIn("jobs:\n  report:", content)
        self.assertNotIn("morning-report:", content)
        self.assertNotIn("regular-report:", content)
        self.assertNotIn("closing-report:", content)

    def test_daily_report_dispatch_supports_all_types(self):
        content = (PROJECT_ROOT / ".github/workflows/daily_report.yml").read_text(encoding="utf-8")
        self.assertIn("report_type", content)
        self.assertIn("morning / regular / closing / auto", content)
        self.assertIn("notify_on_skip", content)

    def test_workflows_do_not_echo_sensitive_names(self):
        for path in (PROJECT_ROOT / ".github/workflows").glob("*.yml"):
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("echo $KIS_APP_SECRET", content)
            self.assertNotIn("set -x", content)


if __name__ == "__main__":
    unittest.main()
