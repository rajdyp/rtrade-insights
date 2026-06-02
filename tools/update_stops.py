from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_calculator.calculations import (
    CAMPAIGN_OVERRIDE_COLUMNS,
    apply_campaign_overrides,
    calculate_positions,
    campaign_view_positions,
    normalize_campaign_overrides,
)
from stock_calculator.robinhood import derive_fifo_trades
from stock_calculator.storage import GoogleSheetsStorage, StorageError
from tools.sync_gs import DEFAULT_CONFIG_PATH, SyncError, _open_spreadsheet, load_sync_config


class StopUpdateError(RuntimeError):
    """Raised when stop updates cannot be applied."""


@dataclass(frozen=True)
class StopUpdate:
    symbol: str
    campaign_stop: float


@dataclass(frozen=True)
class StopUpdateResult:
    symbol: str
    old_stop: float | None
    new_stop: float
    action: str


def load_stop_updates(path: Path) -> list[StopUpdate]:
    if not path.exists():
        raise StopUpdateError(f"Stop update file does not exist: {path}")

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise StopUpdateError(f"Could not read stop update file {path}: {exc}") from exc

    required = {"symbol", "campaign_stop"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise StopUpdateError(f"Stop update file is missing required column(s): {', '.join(missing)}")

    rows: list[StopUpdate] = []
    seen: set[str] = set()
    duplicate_symbols: list[str] = []
    invalid_rows: list[str] = []

    for row_number, row in frame.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        stop = pd.to_numeric(row.get("campaign_stop"), errors="coerce")
        display_row = row_number + 2

        if not symbol:
            invalid_rows.append(f"row {display_row}: symbol is required")
            continue
        if symbol in seen:
            duplicate_symbols.append(symbol)
            continue
        seen.add(symbol)
        if pd.isna(stop) or float(stop) <= 0:
            invalid_rows.append(f"row {display_row}: campaign_stop must be greater than 0")
            continue

        rows.append(StopUpdate(symbol=symbol, campaign_stop=float(stop)))

    if duplicate_symbols:
        unique_duplicates = sorted(set(duplicate_symbols))
        raise StopUpdateError(f"Stop update file contains duplicate symbol(s): {', '.join(unique_duplicates)}")
    if invalid_rows:
        raise StopUpdateError("; ".join(invalid_rows))
    if not rows:
        raise StopUpdateError("Stop update file does not contain any updates.")

    return rows


def apply_stop_updates(
    *,
    storage: GoogleSheetsStorage,
    updates: list[StopUpdate],
    dry_run: bool = False,
) -> list[StopUpdateResult]:
    planned_stops = storage.load_planned_stops()
    transactions = storage.load_robinhood_transactions()
    positions = storage.load_positions()
    existing_overrides = storage.load_campaign_overrides()

    derived = derive_fifo_trades(transactions, planned_stops)
    campaign_view = campaign_view_positions(derived.open_lots, calculate_positions(positions))
    active_symbols = set(campaign_view["symbol"].fillna("").astype(str).str.upper().str.strip())
    missing_symbols = sorted(update.symbol for update in updates if update.symbol not in active_symbols)
    if missing_symbols:
        raise StopUpdateError(f"Symbol(s) are not active in Campaign View: {', '.join(missing_symbols)}")

    displayed_view = apply_campaign_overrides(campaign_view, existing_overrides)
    displayed_stops = {
        str(row["symbol"]): _optional_float(row.get("campaign_stop"))
        for _, row in displayed_view.iterrows()
    }

    updated_overrides = normalize_campaign_overrides(existing_overrides)
    results: list[StopUpdateResult] = []
    for update in updates:
        old_stop = displayed_stops.get(update.symbol)
        if old_stop is not None and _same_number(old_stop, update.campaign_stop):
            action = "unchanged"
        else:
            updated_overrides, action = upsert_campaign_stop_override(updated_overrides, update)
        results.append(
            StopUpdateResult(
                symbol=update.symbol,
                old_stop=old_stop,
                new_stop=update.campaign_stop,
                action=action,
            )
        )

    if not dry_run and any(result.action != "unchanged" for result in results):
        storage.save_campaign_overrides(updated_overrides)

    return results


def upsert_campaign_stop_override(overrides: pd.DataFrame, update: StopUpdate) -> tuple[pd.DataFrame, str]:
    normalized = normalize_campaign_overrides(overrides)
    matches = normalized.index[normalized["symbol"] == update.symbol].tolist()
    if matches:
        index = matches[0]
        existing_stop = _optional_float(normalized.loc[index, "campaign_stop"])
        if existing_stop is not None and _same_number(existing_stop, update.campaign_stop):
            return normalized, "unchanged"
        normalized.loc[index, "campaign_stop"] = update.campaign_stop
        return normalize_campaign_overrides(normalized), "updated"

    appended = pd.concat(
        [
            normalized,
            pd.DataFrame(
                [
                    {
                        "symbol": update.symbol,
                        "current_shares": None,
                        "campaign_stop": update.campaign_stop,
                    }
                ],
                columns=CAMPAIGN_OVERRIDE_COLUMNS,
            ),
        ],
        ignore_index=True,
    )
    return normalize_campaign_overrides(appended), "inserted"


def update_google_sheets_stops(
    *,
    file_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
    dry_run: bool = False,
    gspread_module: Any | None = None,
) -> list[StopUpdateResult]:
    updates = load_stop_updates(file_path)
    try:
        config = load_sync_config(config_path)
        spreadsheet = _open_spreadsheet(config, gspread_module=gspread_module)
        storage = GoogleSheetsStorage(spreadsheet)
        return apply_stop_updates(storage=storage, updates=updates, dry_run=dry_run)
    except SyncError as exc:
        raise StopUpdateError(str(exc)) from exc
    except StorageError as exc:
        raise StopUpdateError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        results = update_google_sheets_stops(
            file_path=args.file,
            config_path=args.config,
            dry_run=args.dry_run,
        )
    except StopUpdateError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    prefix = "Would update" if args.dry_run else "Updated"
    for result in results:
        print(
            f"{prefix} {result.symbol}: "
            f"{_format_stop(result.old_stop)} -> {_format_stop(result.new_stop)} ({result.action})"
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update Campaign View stop overrides in Google Sheets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    apply = subparsers.add_parser("apply", help="Apply stop updates from a CSV file.")
    apply.add_argument("--file", type=Path, required=True, help="CSV file with symbol,campaign_stop columns.")
    apply.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Google Sheets config TOML path.")
    apply.add_argument("--dry-run", action="store_true", help="Print planned changes without writing Google Sheets.")

    return parser


def _optional_float(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return float(number)


def _same_number(left: float, right: float) -> bool:
    return abs(left - right) < 1e-9


def _format_stop(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
