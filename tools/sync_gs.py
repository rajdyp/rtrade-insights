from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_calculator.calculations import POSITION_SOURCE_COLUMNS, committed_positions
from stock_calculator.robinhood import PLANNED_STOP_COLUMNS, TRANSACTION_COLUMNS
from stock_calculator.storage import (
    GOOGLE_SHEETS_SECRET,
    GCP_SERVICE_ACCOUNT_SECRET,
    PLANNED_STOPS_WORKSHEET,
    POSITIONS_ARCHIVE_WORKSHEET,
    POSITIONS_WORKSHEET,
    POSITION_ARCHIVE_COLUMNS,
    ROBINHOOD_TRANSACTIONS_WORKSHEET,
    SPREADSHEET_ID_SECRET,
    LocalCsvStorage,
    _normalize_planned_stops,
    _normalize_positions_archive,
    _normalize_robinhood_transactions,
)


DEFAULT_CONFIG_PATH = Path(".sync/google_sheets.toml")
DEFAULT_DATA_DIR = Path("data")


class SyncError(RuntimeError):
    """Raised when Google Sheets sync cannot continue."""


@dataclass(frozen=True)
class SyncConfig:
    spreadsheet_id: str
    service_account: Mapping[str, Any]


@dataclass(frozen=True)
class SyncTable:
    worksheet: str
    filename: str
    columns: list[str]
    normalize: Callable[[pd.DataFrame], pd.DataFrame]
    save: Callable[[LocalCsvStorage, pd.DataFrame], None]


@dataclass(frozen=True)
class PullResult:
    worksheet: str
    path: Path
    rows: int
    wrote: bool


SYNC_TABLES = (
    SyncTable(
        worksheet=POSITIONS_WORKSHEET,
        filename="positions.csv",
        columns=POSITION_SOURCE_COLUMNS,
        normalize=committed_positions,
        save=lambda storage, df: storage.save_positions(df),
    ),
    SyncTable(
        worksheet=POSITIONS_ARCHIVE_WORKSHEET,
        filename="positions_archive.csv",
        columns=POSITION_ARCHIVE_COLUMNS,
        normalize=_normalize_positions_archive,
        save=lambda storage, df: storage.save_positions_archive(df),
    ),
    SyncTable(
        worksheet=PLANNED_STOPS_WORKSHEET,
        filename="planned_stops.csv",
        columns=PLANNED_STOP_COLUMNS,
        normalize=_normalize_planned_stops,
        save=lambda storage, df: storage.save_planned_stops(df),
    ),
    SyncTable(
        worksheet=ROBINHOOD_TRANSACTIONS_WORKSHEET,
        filename="robinhood_transactions.csv",
        columns=TRANSACTION_COLUMNS,
        normalize=_normalize_robinhood_transactions,
        save=lambda storage, df: storage.save_robinhood_transactions(df),
    ),
)


def load_sync_config(path: Path = DEFAULT_CONFIG_PATH) -> SyncConfig:
    if not path.exists():
        raise SyncError(f"Sync config does not exist: {path}")

    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise SyncError(f"Sync config is not valid TOML: {path}") from exc
    except OSError as exc:
        raise SyncError(f"Could not read sync config {path}: {exc}") from exc

    sheet_config = _mapping_value(parsed, GOOGLE_SHEETS_SECRET)
    service_account = _mapping_value(parsed, GCP_SERVICE_ACCOUNT_SECRET)
    if not sheet_config or not service_account:
        raise SyncError("Sync config requires both [google_sheets] and [gcp_service_account] sections.")
    if not isinstance(service_account, Mapping):
        raise SyncError("Sync config [gcp_service_account] must contain service account fields.")

    spreadsheet_id = _mapping_value(sheet_config, SPREADSHEET_ID_SECRET)
    if not spreadsheet_id:
        raise SyncError("Sync config requires google_sheets.spreadsheet_id.")

    return SyncConfig(spreadsheet_id=str(spreadsheet_id), service_account=service_account)


def pull_google_sheets(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    dry_run: bool = False,
    gspread_module: Any | None = None,
) -> list[PullResult]:
    config = load_sync_config(config_path)
    spreadsheet = _open_spreadsheet(config, gspread_module=gspread_module)
    storage = _local_storage(data_dir)
    results = []

    for table in SYNC_TABLES:
        frame = table.normalize(_read_worksheet(spreadsheet, table))
        destination = data_dir / table.filename
        if not dry_run:
            table.save(storage, frame)
        results.append(PullResult(table.worksheet, destination, len(frame), wrote=not dry_run))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        results = pull_google_sheets(config_path=args.config, data_dir=args.data_dir, dry_run=args.dry_run)
    except SyncError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    action = "Would pull" if args.dry_run else "Pulled"
    for result in results:
        write_note = "dry run" if not result.wrote else "written"
        print(f"{action} {result.rows} rows from '{result.worksheet}' to {result.path} ({write_note})")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pull Google Sheets tabs into local rTrade CSV files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pull = subparsers.add_parser("pull", help="Download Google Sheets tabs into local data/*.csv files.")
    pull.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Sync config TOML path.")
    pull.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Local data directory.")
    pull.add_argument("--dry-run", action="store_true", help="Report row counts without writing CSV files.")

    return parser


def _open_spreadsheet(config: SyncConfig, *, gspread_module: Any | None = None) -> Any:
    if gspread_module is None:
        try:
            import gspread as gspread_module
        except ImportError as exc:
            raise SyncError("Google Sheets sync requires gspread to be installed.") from exc

    try:
        client = gspread_module.service_account_from_dict(dict(config.service_account))
        return client.open_by_key(config.spreadsheet_id)
    except Exception as exc:
        raise SyncError(f"Could not connect to Google Sheets: {exc}") from exc


def _read_worksheet(spreadsheet: Any, table: SyncTable) -> pd.DataFrame:
    try:
        worksheet = spreadsheet.worksheet(table.worksheet)
        values = worksheet.get_all_values()
    except Exception as exc:
        raise SyncError(f"Could not read Google Sheets tab '{table.worksheet}': {exc}") from exc

    if not values or not values[0]:
        return pd.DataFrame(columns=table.columns)

    header = values[0]
    records = []
    for raw_row in values[1:]:
        padded = [*raw_row, *[""] * max(0, len(header) - len(raw_row))]
        records.append(dict(zip(header, padded, strict=False)))
    return pd.DataFrame(records)


def _local_storage(data_dir: Path) -> LocalCsvStorage:
    return LocalCsvStorage(
        positions_path=data_dir / "positions.csv",
        positions_archive_path=data_dir / "positions_archive.csv",
        planned_stops_path=data_dir / "planned_stops.csv",
        transactions_path=data_dir / "robinhood_transactions.csv",
    )


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except (KeyError, TypeError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
