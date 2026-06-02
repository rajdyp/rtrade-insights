import pandas as pd
import pytest

from stock_calculator.calculations import CAMPAIGN_OVERRIDE_COLUMNS, POSITION_SOURCE_COLUMNS
from stock_calculator.robinhood import PLANNED_STOP_COLUMNS, TRANSACTION_COLUMNS
from tools import update_stops


def test_load_stop_updates_normalizes_symbols_and_stops(tmp_path):
    path = tmp_path / "campaign_stops.csv"
    path.write_text("symbol,campaign_stop\n akam ,147.25\nNTAP,174.50\n", encoding="utf-8")

    updates = update_stops.load_stop_updates(path)

    assert updates == [
        update_stops.StopUpdate("AKAM", 147.25),
        update_stops.StopUpdate("NTAP", 174.5),
    ]


def test_load_stop_updates_rejects_duplicate_symbols(tmp_path):
    path = tmp_path / "campaign_stops.csv"
    path.write_text("symbol,campaign_stop\nAKAM,147.25\nakam,148.00\n", encoding="utf-8")

    with pytest.raises(update_stops.StopUpdateError, match="duplicate symbol"):
        update_stops.load_stop_updates(path)


def test_load_stop_updates_rejects_non_positive_stops(tmp_path):
    path = tmp_path / "campaign_stops.csv"
    path.write_text("symbol,campaign_stop\nAKAM,0\nNTAP,-1\n", encoding="utf-8")

    with pytest.raises(update_stops.StopUpdateError, match="greater than 0"):
        update_stops.load_stop_updates(path)


def test_load_stop_updates_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "campaign_stops.csv"
    path.write_text("symbol,stop\nAKAM,147.25\n", encoding="utf-8")

    with pytest.raises(update_stops.StopUpdateError, match="campaign_stop"):
        update_stops.load_stop_updates(path)


def test_upsert_campaign_stop_override_preserves_current_shares():
    overrides = pd.DataFrame([{"symbol": "AKAM", "current_shares": 3, "campaign_stop": 146.00}])

    updated, action = update_stops.upsert_campaign_stop_override(
        overrides,
        update_stops.StopUpdate("AKAM", 147.25),
    )

    assert action == "updated"
    assert updated.to_dict("records") == [{"symbol": "AKAM", "current_shares": 3, "campaign_stop": 147.25}]


def test_apply_stop_updates_writes_google_sheets_campaign_overrides():
    storage, spreadsheet = _storage_with_position(
        overrides=[CAMPAIGN_OVERRIDE_COLUMNS, ["akam", "3", "146.00"]],
    )

    results = update_stops.apply_stop_updates(
        storage=storage,
        updates=[update_stops.StopUpdate("AKAM", 147.25)],
    )

    assert results == [update_stops.StopUpdateResult("AKAM", 146.0, 147.25, "updated")]
    assert spreadsheet.worksheets["campaign_overrides"].values == [
        CAMPAIGN_OVERRIDE_COLUMNS,
        ["AKAM", 3, 147.25],
    ]


def test_apply_stop_updates_dry_run_does_not_write_google_sheets():
    storage, spreadsheet = _storage_with_position(overrides=[CAMPAIGN_OVERRIDE_COLUMNS])

    results = update_stops.apply_stop_updates(
        storage=storage,
        updates=[update_stops.StopUpdate("AKAM", 147.25)],
        dry_run=True,
    )

    assert results == [update_stops.StopUpdateResult("AKAM", 146.0, 147.25, "inserted")]
    assert spreadsheet.worksheets["campaign_overrides"].values == [CAMPAIGN_OVERRIDE_COLUMNS]


def test_apply_stop_updates_does_not_save_unchanged_stop_overrides_when_other_symbols_change():
    storage, spreadsheet = _storage_with_position(
        overrides=[CAMPAIGN_OVERRIDE_COLUMNS],
        position_rows=[
            ["pos_1", "akam", "2026-06-01", "151.78", "146.00", "6.4", "19250", "0.12"],
            ["pos_2", "ntap", "2026-06-01", "187.32", "172.00", "3.97", "19250", "0.12"],
        ],
    )

    results = update_stops.apply_stop_updates(
        storage=storage,
        updates=[
            update_stops.StopUpdate("AKAM", 147.25),
            update_stops.StopUpdate("NTAP", 172.00),
        ],
    )

    assert results == [
        update_stops.StopUpdateResult("AKAM", 146.0, 147.25, "inserted"),
        update_stops.StopUpdateResult("NTAP", 172.0, 172.0, "unchanged"),
    ]
    assert spreadsheet.worksheets["campaign_overrides"].values == [
        CAMPAIGN_OVERRIDE_COLUMNS,
        ["AKAM", "", 147.25],
    ]


def test_apply_stop_updates_rejects_inactive_symbols():
    storage, _ = _storage_with_position(overrides=[CAMPAIGN_OVERRIDE_COLUMNS])

    with pytest.raises(update_stops.StopUpdateError, match="not active"):
        update_stops.apply_stop_updates(
            storage=storage,
            updates=[update_stops.StopUpdate("MSFT", 300.00)],
        )


def test_main_returns_exit_code_2_for_update_errors(capsys, monkeypatch, tmp_path):
    def fake_update_google_sheets_stops(**kwargs):
        raise update_stops.StopUpdateError("bad input")

    monkeypatch.setattr(update_stops, "update_google_sheets_stops", fake_update_google_sheets_stops)

    exit_code = update_stops.main(["apply", "--file", (tmp_path / "campaign_stops.csv").as_posix()])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Error: bad input" in captured.err


def _storage_with_position(overrides, position_rows=None):
    position_rows = position_rows or [
        ["pos_1", "akam", "2026-06-01", "151.78", "146.00", "6.4", "19250", "0.12"],
    ]
    spreadsheet = FakeSpreadsheet(
        {
            "positions": FakeWorksheet([POSITION_SOURCE_COLUMNS, *position_rows]),
            "planned_stops": FakeWorksheet([PLANNED_STOP_COLUMNS]),
            "robinhood_transactions": FakeWorksheet([TRANSACTION_COLUMNS]),
            "campaign_overrides": FakeWorksheet(overrides),
        }
    )
    return update_stops.GoogleSheetsStorage(spreadsheet), spreadsheet


class WorksheetNotFound(Exception):
    pass


class FakeWorksheet:
    def __init__(self, values):
        self.values = values

    def get_all_values(self):
        return self.values

    def clear(self):
        self.values = []

    def update(self, values, value_input_option=None):
        self.values = values


class FakeSpreadsheet:
    def __init__(self, worksheets):
        self.worksheets = worksheets

    def worksheet(self, title):
        if title not in self.worksheets:
            raise WorksheetNotFound(title)
        return self.worksheets[title]

    def add_worksheet(self, title, rows, cols):
        worksheet = FakeWorksheet([])
        self.worksheets[title] = worksheet
        return worksheet
