from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"


class WorkflowContractTests(unittest.TestCase):
    def _read(self, name: str) -> str:
        return (WORKFLOWS / name).read_text(encoding="utf-8")

    def test_daily_report_exists(self):
        self.assertTrue((WORKFLOWS / "daily_report.yml").exists())

    def test_daily_report_is_single_job(self):
        content = self._read("daily_report.yml")
        self.assertIn("jobs:\n  report:", content)
        self.assertNotIn("morning-report:", content)
        self.assertNotIn("regular-report:", content)
        self.assertNotIn("closing-report:", content)

    def test_daily_report_inputs_exist(self):
        content = self._read("daily_report.yml")
        self.assertIn("report_type", content)
        self.assertIn("report_date", content)
        self.assertIn("notify_on_skip", content)
        self.assertIn("use_gemini", content)
        self.assertIn("no_send", content)
        self.assertIn("dry_run", content)

    def test_daily_report_has_expected_schedules(self):
        content = self._read("daily_report.yml")
        self.assertIn('cron: "0 22 * * 0-4"', content)
        self.assertIn('cron: "30 1,3,5 * * 1-5"', content)
        self.assertIn('cron: "40 6 * * 1-5"', content)

    def test_removed_workflows_do_not_exist(self):
        for name in (
            "daily_morning_required_data.yml",
            "daily_market_close_pipeline.yml",
            "weekly_master_refresh.yml",
            "monthly_calendar_sync.yml",
        ):
            self.assertFalse((WORKFLOWS / name).exists(), msg=name)

    def test_report_data_diagnostics_exists(self):
        self.assertTrue((WORKFLOWS / "report_data_diagnostics.yml").exists())

    def test_report_data_diagnostics_is_dispatch_only(self):
        content = self._read("report_data_diagnostics.yml")
        self.assertIn("workflow_dispatch:", content)
        self.assertNotIn("schedule:", content)

    def test_weekly_cleanup_is_dispatch_only(self):
        content = self._read("weekly_cleanup.yml")
        self.assertIn("workflow_dispatch:", content)
        self.assertNotIn("schedule:", content)

    def test_workflows_do_not_echo_sensitive_names(self):
        for path in WORKFLOWS.glob("*.yml"):
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("echo $KIS_APP_SECRET", content)
            self.assertNotIn("set -x", content)

    def test_daily_report_passes_runtime_flags(self):
        content = self._read("daily_report.yml")
        self.assertIn("--use-gemini", content)
        self.assertIn("--no-gemini", content)
        self.assertIn("--no-send", content)
        self.assertIn("--dry-run", content)

    def test_daily_report_defaults_schedule_to_no_gemini(self):
        content = self._read("daily_report.yml")
        self.assertIn('USE_GEMINI="false"', content)

    def test_generate_report_supports_gemini_flags(self):
        report_script = (PROJECT_ROOT / "src" / "jobs" / "generate_report.py").read_text(encoding="utf-8")
        self.assertIn("--use-gemini", report_script)
        self.assertIn("--no-gemini", report_script)


if __name__ == "__main__":
    unittest.main()
