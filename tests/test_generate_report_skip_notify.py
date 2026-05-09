import datetime
import unittest
from unittest.mock import MagicMock, patch

from src.jobs.generate_report import run_report


class GenerateReportSkipNotifyTests(unittest.TestCase):
    @patch("src.jobs.generate_report.TelegramSender")
    @patch("src.jobs.generate_report._save_report")
    @patch("src.jobs.generate_report.SupabaseStockDataReader")
    @patch("src.jobs.generate_report.SupabaseReader")
    def test_skip_notifies_when_enabled(self, mock_base_reader_cls, mock_reader_cls, mock_save, mock_sender_cls):
        mock_base_reader = MagicMock()
        mock_base_reader.telegram_bot_token = "token"
        mock_base_reader.telegram_chat_id = "1234"
        mock_base_reader.fetch_market_calendar_status.return_value = {
            "report_date": "2026-05-09",
            "xkrx_is_open": False,
            "xnys_is_open": False,
            "xkrx_reason": "weekend",
            "xnys_reason": "weekend",
            "xkrx_next_trading_day": "2026-05-11",
            "xnys_next_trading_day": "2026-05-11",
            "report_market_mode": "SKIP_ALL_MARKETS_CLOSED",
        }
        mock_base_reader_cls.return_value = mock_base_reader
        mock_reader_cls.return_value = MagicMock()
        sender = MagicMock()
        mock_sender_cls.return_value = sender

        run_report("regular", datetime.datetime(2026, 5, 9, 9, 0), report_date="2026-05-09", send_enabled=True, notify_on_skip=True)
        sender.send_report.assert_called_once()
        mock_save.assert_called_once()

    @patch("src.jobs.generate_report.TelegramSender")
    @patch("src.jobs.generate_report._save_report")
    @patch("src.jobs.generate_report.SupabaseStockDataReader")
    @patch("src.jobs.generate_report.SupabaseReader")
    def test_skip_does_not_notify_when_disabled(self, mock_base_reader_cls, mock_reader_cls, mock_save, mock_sender_cls):
        mock_base_reader = MagicMock()
        mock_base_reader.telegram_bot_token = "token"
        mock_base_reader.telegram_chat_id = "1234"
        mock_base_reader.fetch_market_calendar_status.return_value = {
            "report_date": "2026-05-09",
            "xkrx_is_open": False,
            "xnys_is_open": False,
            "xkrx_reason": "weekend",
            "xnys_reason": "weekend",
            "xkrx_next_trading_day": "2026-05-11",
            "xnys_next_trading_day": "2026-05-11",
            "report_market_mode": "SKIP_ALL_MARKETS_CLOSED",
        }
        mock_base_reader_cls.return_value = mock_base_reader
        mock_reader_cls.return_value = MagicMock()

        run_report("regular", datetime.datetime(2026, 5, 9, 9, 0), report_date="2026-05-09", send_enabled=True, notify_on_skip=False)
        mock_sender_cls.assert_not_called()
        mock_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
