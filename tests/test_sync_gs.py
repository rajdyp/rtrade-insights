import pandas as pd
import pytest

from stock_calculator.calculations import POSITION_SOURCE_COLUMNS
from stock_calculator.robinhood import PLANNED_STOP_COLUMNS, TRANSACTION_COLUMNS
from stock_calculator.storage import POSITION_ARCHIVE_COLUMNS
from tools import sync_gs


def test_load_sync_config_reads_helper_only_config(tmp_path):
    config_path = _write_config(tmp_path)

    config = sync_gs.load_sync_config(config_path)

    assert config.spreadsheet_id == "sheet-id"
    assert config.service_account["client_email"] == "bot@example.com"


def test_load_sync_config_rejects_partial_config(tmp_path):
    config_path = tmp_path / "google_sheets.toml"
    config_path.write_text(
        """
        [google_sheets]
        spreadsheet_id = "sheet-id"
        """,
        encoding="utf-8",
    )

    with pytest.raises(sync_gs.SyncError, match="requires both"):
        sync_gs.load_sync_config(config_path)


def test_pull_google_sheets_writes_normalized_local_csv_files(tmp_path):
    config_path = _write_config(tmp_path)
    data_dir = tmp_path / "data"
    spreadsheet = FakeSpreadsheet(
        {
            "positions": FakeWorksheet(
                [
                    POSITION_SOURCE_COLUMNS,
                    ["pos_1", "aapl", "2026-04-01", "100", "95", "2.5", "20000", "0.5"],
                ]
            ),
            "positions_archive": FakeWorksheet(
                [
                    POSITION_ARCHIVE_COLUMNS,
                    ["pos_1", "aapl", "2026-04-01", "100", "95", "2.5", "", "EP", "", "20", "", "", "", "", "", ""],
                ]
            ),
            "planned_stops": FakeWorksheet(
                [
                    PLANNED_STOP_COLUMNS,
                    ["aapl", "2026-04-01", "20", "95", "ep", "2.5", "selective go"],
                ]
            ),
            "robinhood_transactions": FakeWorksheet(
                [
                    TRANSACTION_COLUMNS,
                    ["2026-04-01", "2026-04-01", "2026-04-02", "aapl", "Buy AAPL", "Buy", "20", "100", "-2000"],
                ]
            ),
        }
    )

    results = sync_gs.pull_google_sheets(
        config_path=config_path,
        data_dir=data_dir,
        gspread_module=FakeGspreadModule(spreadsheet),
    )

    assert [(result.worksheet, result.rows, result.wrote) for result in results] == [
        ("positions", 1, True),
        ("positions_archive", 1, True),
        ("planned_stops", 1, True),
        ("robinhood_transactions", 1, True),
    ]
    assert pd.read_csv(data_dir / "positions.csv")["symbol"].tolist() == ["AAPL"]
    assert pd.read_csv(data_dir / "positions_archive.csv")["strategy"].tolist() == ["EP"]
    assert pd.read_csv(data_dir / "planned_stops.csv")["market_regime"].tolist() == ["SELECTIVE GO"]
    assert pd.read_csv(data_dir / "robinhood_transactions.csv")["symbol"].tolist() == ["AAPL"]


def test_pull_google_sheets_dry_run_does_not_write_files(tmp_path):
    config_path = _write_config(tmp_path)
    data_dir = tmp_path / "data"
    spreadsheet = FakeSpreadsheet(
        {
            "positions": FakeWorksheet([POSITION_SOURCE_COLUMNS, ["", "msft", "", "", "", "", "", ""]]),
            "positions_archive": FakeWorksheet([POSITION_ARCHIVE_COLUMNS]),
            "planned_stops": FakeWorksheet([PLANNED_STOP_COLUMNS]),
            "robinhood_transactions": FakeWorksheet([TRANSACTION_COLUMNS]),
        }
    )

    results = sync_gs.pull_google_sheets(
        config_path=config_path,
        data_dir=data_dir,
        dry_run=True,
        gspread_module=FakeGspreadModule(spreadsheet),
    )

    assert [(result.worksheet, result.rows, result.wrote) for result in results] == [
        ("positions", 1, False),
        ("positions_archive", 0, False),
        ("planned_stops", 0, False),
        ("robinhood_transactions", 0, False),
    ]
    assert not data_dir.exists()


def test_main_reports_dry_run_output(tmp_path, capsys, monkeypatch):
    config_path = _write_config(tmp_path)
    expected = [sync_gs.PullResult("positions", tmp_path / "data/positions.csv", rows=2, wrote=False)]
    captured_call = {}

    def fake_pull_google_sheets(**kwargs):
        captured_call.update(kwargs)
        return expected

    monkeypatch.setattr(sync_gs, "pull_google_sheets", fake_pull_google_sheets)

    exit_code = sync_gs.main(["pull", "--config", config_path.as_posix(), "--data-dir", (tmp_path / "data").as_posix(), "--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured_call["config_path"] == config_path
    assert captured_call["dry_run"] is True
    assert "Would pull 2 rows from 'positions'" in output


def _write_config(tmp_path):
    config_path = tmp_path / "google_sheets.toml"
    config_path.write_text(
        """
        [google_sheets]
        spreadsheet_id = "sheet-id"

        [gcp_service_account]
        client_email = "bot@example.com"
        """,
        encoding="utf-8",
    )
    return config_path


class FakeWorksheet:
    def __init__(self, values):
        self.values = values

    def get_all_values(self):
        return self.values


class FakeSpreadsheet:
    def __init__(self, worksheets):
        self.worksheets = worksheets

    def worksheet(self, title):
        return self.worksheets[title]


class FakeGspreadClient:
    def __init__(self, spreadsheet):
        self.spreadsheet = spreadsheet

    def open_by_key(self, spreadsheet_id):
        assert spreadsheet_id == "sheet-id"
        return self.spreadsheet


class FakeGspreadModule:
    def __init__(self, spreadsheet):
        self.spreadsheet = spreadsheet

    def service_account_from_dict(self, service_account):
        assert service_account["client_email"] == "bot@example.com"
        return FakeGspreadClient(self.spreadsheet)
