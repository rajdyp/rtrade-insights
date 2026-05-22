from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import pandas as pd

from stock_calculator.calculations import (
    POSITION_ID_COLUMN,
    POSITION_SOURCE_COLUMNS,
    committed_positions,
    ensure_position_ids,
    normalize_input_frame,
)
from stock_calculator.risk import normalize_market_regime
from stock_calculator.robinhood import PLANNED_STOP_COLUMNS, TRANSACTION_COLUMNS, normalize_strategy


DATA_PATH = Path("data/positions.csv")
POSITIONS_ARCHIVE_PATH = Path("data/positions_archive.csv")
ROBINHOOD_TRANSACTIONS_PATH = Path("data/robinhood_transactions.csv")
PLANNED_STOPS_PATH = Path("data/planned_stops.csv")

POSITIONS_WORKSHEET = "positions"
POSITIONS_ARCHIVE_WORKSHEET = "positions_archive"
PLANNED_STOPS_WORKSHEET = "planned_stops"
ROBINHOOD_TRANSACTIONS_WORKSHEET = "robinhood_transactions"

GOOGLE_SHEETS_SECRET = "google_sheets"
GCP_SERVICE_ACCOUNT_SECRET = "gcp_service_account"
SPREADSHEET_ID_SECRET = "spreadsheet_id"

POSITION_ARCHIVE_COLUMNS = [
    POSITION_ID_COLUMN,
    "symbol",
    "buy_date",
    "share_price",
    "stop_price",
    "atr",
    "risk_in_atr",
    "strategy",
    "stop_loss_percent",
    "number_of_shares",
    "sell_lot",
    "hold_count",
    "position_size",
    "risk_percent",
    "risk_amount",
    "portfolio_amount",
]

_DEFAULT_BACKEND: StorageBackend | None = None


class StorageError(RuntimeError):
    """Raised when configured storage cannot be read or written."""


class StorageConfigurationError(StorageError):
    """Raised when hosted storage is partially or incorrectly configured."""


class StorageBackend(Protocol):
    @property
    def label(self) -> str: ...

    def load_positions(self) -> pd.DataFrame: ...

    def save_positions(self, df: pd.DataFrame) -> None: ...

    def load_positions_archive(self) -> pd.DataFrame: ...

    def save_positions_archive(self, df: pd.DataFrame) -> None: ...

    def load_planned_stops(self) -> pd.DataFrame: ...

    def save_planned_stops(self, df: pd.DataFrame) -> None: ...

    def load_robinhood_transactions(self) -> pd.DataFrame: ...

    def save_robinhood_transactions(self, df: pd.DataFrame) -> None: ...

    def transactions_label(self) -> str: ...

    def planned_stops_label(self) -> str: ...


@dataclass(frozen=True)
class LocalCsvStorage:
    positions_path: Path = DATA_PATH
    positions_archive_path: Path = POSITIONS_ARCHIVE_PATH
    planned_stops_path: Path = PLANNED_STOPS_PATH
    transactions_path: Path = ROBINHOOD_TRANSACTIONS_PATH

    @property
    def label(self) -> str:
        return "local CSV"

    def load_positions(self) -> pd.DataFrame:
        return _load_positions_csv(self.positions_path)

    def save_positions(self, df: pd.DataFrame) -> None:
        _save_positions_csv(df, self.positions_path)

    def load_positions_archive(self) -> pd.DataFrame:
        return _load_positions_archive_csv(self.positions_archive_path)

    def save_positions_archive(self, df: pd.DataFrame) -> None:
        _save_positions_archive_csv(df, self.positions_archive_path)

    def load_planned_stops(self) -> pd.DataFrame:
        return _load_planned_stops_csv(self.planned_stops_path)

    def save_planned_stops(self, df: pd.DataFrame) -> None:
        _save_planned_stops_csv(df, self.planned_stops_path)

    def load_robinhood_transactions(self) -> pd.DataFrame:
        return _load_robinhood_transactions_csv(self.transactions_path)

    def save_robinhood_transactions(self, df: pd.DataFrame) -> None:
        _save_robinhood_transactions_csv(df, self.transactions_path)

    def transactions_label(self) -> str:
        return str(self.transactions_path)

    def planned_stops_label(self) -> str:
        return str(self.planned_stops_path)


@dataclass(frozen=True)
class GoogleSheetsStorage:
    spreadsheet: Any

    @property
    def label(self) -> str:
        return "Google Sheets"

    def load_positions(self) -> pd.DataFrame:
        return committed_positions(self._read_table(POSITIONS_WORKSHEET, POSITION_SOURCE_COLUMNS))

    def save_positions(self, df: pd.DataFrame) -> None:
        self._write_table(POSITIONS_WORKSHEET, committed_positions(df), POSITION_SOURCE_COLUMNS)

    def load_positions_archive(self) -> pd.DataFrame:
        return _normalize_positions_archive(self._read_table(POSITIONS_ARCHIVE_WORKSHEET, POSITION_ARCHIVE_COLUMNS))

    def save_positions_archive(self, df: pd.DataFrame) -> None:
        self._write_table(
            POSITIONS_ARCHIVE_WORKSHEET,
            _normalize_positions_archive(df),
            POSITION_ARCHIVE_COLUMNS,
        )

    def load_planned_stops(self) -> pd.DataFrame:
        return _normalize_planned_stops(self._read_table(PLANNED_STOPS_WORKSHEET, PLANNED_STOP_COLUMNS))

    def save_planned_stops(self, df: pd.DataFrame) -> None:
        self._write_table(PLANNED_STOPS_WORKSHEET, _normalize_planned_stops(df), PLANNED_STOP_COLUMNS)

    def load_robinhood_transactions(self) -> pd.DataFrame:
        return _normalize_robinhood_transactions(
            self._read_table(ROBINHOOD_TRANSACTIONS_WORKSHEET, TRANSACTION_COLUMNS)
        )

    def save_robinhood_transactions(self, df: pd.DataFrame) -> None:
        self._write_table(
            ROBINHOOD_TRANSACTIONS_WORKSHEET,
            _normalize_robinhood_transactions(df),
            TRANSACTION_COLUMNS,
        )

    def transactions_label(self) -> str:
        return f"Google Sheets tab '{ROBINHOOD_TRANSACTIONS_WORKSHEET}'"

    def planned_stops_label(self) -> str:
        return f"Google Sheets tab '{PLANNED_STOPS_WORKSHEET}'"

    def _read_table(self, title: str, columns: list[str]) -> pd.DataFrame:
        worksheet = self._worksheet(title, columns)
        values = worksheet.get_all_values()
        if not values:
            self._write_headers(worksheet, columns)
            return pd.DataFrame(columns=columns)

        header = values[0]
        rows = values[1:]
        if not header:
            self._write_headers(worksheet, columns)
            return pd.DataFrame(columns=columns)

        records = []
        for raw_row in rows:
            padded = [*raw_row, *[""] * max(0, len(header) - len(raw_row))]
            records.append(dict(zip(header, padded, strict=False)))
        return pd.DataFrame(records)

    def _write_table(self, title: str, df: pd.DataFrame, columns: list[str]) -> None:
        worksheet = self._worksheet(title, columns)
        worksheet.clear()
        worksheet.update(_worksheet_values(df, columns), value_input_option="USER_ENTERED")

    def _worksheet(self, title: str, columns: list[str]) -> Any:
        try:
            return self.spreadsheet.worksheet(title)
        except Exception as exc:
            if exc.__class__.__name__ != "WorksheetNotFound":
                raise StorageError(f"Could not open Google Sheets tab '{title}': {exc}") from exc

        try:
            worksheet = self.spreadsheet.add_worksheet(title=title, rows=1, cols=len(columns))
            self._write_headers(worksheet, columns)
            return worksheet
        except Exception as exc:
            raise StorageError(f"Could not create Google Sheets tab '{title}': {exc}") from exc

    @staticmethod
    def _write_headers(worksheet: Any, columns: list[str]) -> None:
        worksheet.clear()
        worksheet.update([columns], value_input_option="USER_ENTERED")


def get_storage_backend() -> StorageBackend:
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = build_storage_backend()
    return _DEFAULT_BACKEND


def reset_storage_backend() -> None:
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = None


def build_storage_backend(
    secrets: Mapping[str, Any] | None = None,
    *,
    gspread_module: Any | None = None,
) -> StorageBackend:
    secrets = _streamlit_secrets() if secrets is None else secrets
    sheet_config = _mapping_value(secrets, GOOGLE_SHEETS_SECRET)
    service_account = _mapping_value(secrets, GCP_SERVICE_ACCOUNT_SECRET)

    has_sheet_config = bool(sheet_config)
    has_service_account = bool(service_account)
    if not has_sheet_config and not has_service_account:
        return LocalCsvStorage()
    if not has_sheet_config or not has_service_account:
        raise StorageConfigurationError(
            "Google Sheets storage requires both [google_sheets] and [gcp_service_account] secrets."
        )

    spreadsheet_id = _mapping_value(sheet_config, SPREADSHEET_ID_SECRET)
    if not spreadsheet_id:
        raise StorageConfigurationError("Google Sheets storage requires google_sheets.spreadsheet_id.")
    if not isinstance(service_account, Mapping):
        raise StorageConfigurationError("gcp_service_account must contain the Google service account JSON fields.")

    if gspread_module is None:
        try:
            import gspread as gspread_module
        except ImportError as exc:
            raise StorageConfigurationError("Google Sheets storage requires gspread to be installed.") from exc

    try:
        client = gspread_module.service_account_from_dict(dict(service_account))
        spreadsheet = client.open_by_key(str(spreadsheet_id))
    except Exception as exc:
        raise StorageConfigurationError(f"Could not connect to Google Sheets: {exc}") from exc

    return GoogleSheetsStorage(spreadsheet)


def storage_label() -> str:
    return get_storage_backend().label


def robinhood_transactions_label() -> str:
    return get_storage_backend().transactions_label()


def planned_stops_label() -> str:
    return get_storage_backend().planned_stops_label()


def load_positions(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        return get_storage_backend().load_positions()
    return _load_positions_csv(path)


def save_positions(df: pd.DataFrame, path: Path | None = None) -> None:
    if path is None:
        get_storage_backend().save_positions(df)
        return
    _save_positions_csv(df, path)


def load_positions_archive(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        return get_storage_backend().load_positions_archive()
    return _load_positions_archive_csv(path)


def save_positions_archive(df: pd.DataFrame, path: Path | None = None) -> None:
    if path is None:
        get_storage_backend().save_positions_archive(df)
        return
    _save_positions_archive_csv(df, path)


def load_planned_stops(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        return get_storage_backend().load_planned_stops()
    return _load_planned_stops_csv(path)


def save_planned_stops(df: pd.DataFrame, path: Path | None = None) -> None:
    if path is None:
        get_storage_backend().save_planned_stops(df)
        return
    _save_planned_stops_csv(df, path)


def load_robinhood_transactions(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        return get_storage_backend().load_robinhood_transactions()
    return _load_robinhood_transactions_csv(path)


def save_robinhood_transactions(df: pd.DataFrame, path: Path | None = None) -> None:
    if path is None:
        get_storage_backend().save_robinhood_transactions(df)
        return
    _save_robinhood_transactions_csv(df, path)


def _load_positions_csv(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return normalize_input_frame(pd.DataFrame(columns=POSITION_SOURCE_COLUMNS))
    return committed_positions(pd.read_csv(path))


def _save_positions_csv(df: pd.DataFrame, path: Path = DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    committed_positions(df).to_csv(path, index=False, columns=POSITION_SOURCE_COLUMNS)


def _load_positions_archive_csv(path: Path = POSITIONS_ARCHIVE_PATH) -> pd.DataFrame:
    if not path.exists():
        return _normalize_positions_archive(pd.DataFrame(columns=POSITION_ARCHIVE_COLUMNS))
    return _normalize_positions_archive(pd.read_csv(path))


def _save_positions_archive_csv(df: pd.DataFrame, path: Path = POSITIONS_ARCHIVE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_positions_archive(df).to_csv(path, index=False, columns=POSITION_ARCHIVE_COLUMNS)


def upsert_positions_archive(archive: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    current = _normalize_positions_archive(archive)
    incoming = _normalize_positions_archive(positions)
    if incoming.empty:
        return current

    incoming_ids = set(incoming[POSITION_ID_COLUMN].fillna("").astype(str))
    kept = current[~current[POSITION_ID_COLUMN].isin(incoming_ids)]
    return _normalize_positions_archive(pd.concat([incoming, kept], ignore_index=True))


def _load_planned_stops_csv(path: Path = PLANNED_STOPS_PATH) -> pd.DataFrame:
    if not path.exists():
        return _normalize_planned_stops(pd.DataFrame(columns=PLANNED_STOP_COLUMNS))
    return _normalize_planned_stops(pd.read_csv(path))


def _save_planned_stops_csv(df: pd.DataFrame, path: Path = PLANNED_STOPS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_planned_stops(df).to_csv(path, index=False, columns=PLANNED_STOP_COLUMNS)


def upsert_planned_stop(planned_stops: pd.DataFrame, calculated_position: pd.Series) -> pd.DataFrame:
    row = _planned_stop_from_calculated_position(calculated_position)
    if row is None:
        return _normalize_planned_stops(planned_stops)

    normalized = _normalize_planned_stops(planned_stops)
    key = _planned_stop_key(row)
    if not row.get("market_regime"):
        row["market_regime"] = _planned_market_regime_for_key(normalized, key)
    keep_rows = [_planned_stop_key(existing) != key for _, existing in normalized.iterrows()]
    updated = pd.concat([pd.DataFrame([row]), normalized.loc[keep_rows]], ignore_index=True)
    return _normalize_planned_stops(updated)


def generate_planned_stops_from_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_robinhood_transactions(transactions)
    buys = normalized[normalized["trans_code"] == "Buy"]
    rows = [
        {
            "symbol": row["symbol"],
            "buy_date": row["activity_date"],
            "quantity": row["quantity"],
            "planned_stop": None,
            "strategy": "",
            "atr": None,
            "market_regime": "",
        }
        for _, row in buys.iterrows()
    ]
    return _normalize_planned_stops(pd.DataFrame(rows, columns=PLANNED_STOP_COLUMNS))


def _load_robinhood_transactions_csv(path: Path = ROBINHOOD_TRANSACTIONS_PATH) -> pd.DataFrame:
    if not path.exists():
        return _normalize_robinhood_transactions(pd.DataFrame(columns=TRANSACTION_COLUMNS))
    return _normalize_robinhood_transactions(pd.read_csv(path))


def _save_robinhood_transactions_csv(df: pd.DataFrame, path: Path = ROBINHOOD_TRANSACTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_robinhood_transactions(df).to_csv(path, index=False, columns=TRANSACTION_COLUMNS)


def append_robinhood_transactions(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    current = _normalize_robinhood_transactions(existing)
    additions = _normalize_robinhood_transactions(incoming)
    if additions.empty:
        return current, 0

    seen = set(_transaction_keys(current))
    new_rows = []
    for _, row in additions.iterrows():
        key = _transaction_key(row)
        if key in seen:
            continue
        seen.add(key)
        new_rows.append(row)

    if not new_rows:
        return current, 0

    appended = pd.DataFrame(new_rows, columns=TRANSACTION_COLUMNS)
    return _normalize_robinhood_transactions(pd.concat([current, appended], ignore_index=True)), len(appended)


def _normalize_robinhood_transactions(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in TRANSACTION_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    normalized = normalized[TRANSACTION_COLUMNS]
    for column in ["activity_date", "process_date", "settle_date", "symbol", "description", "trans_code"]:
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()

    normalized["symbol"] = normalized["symbol"].str.upper()
    for column in ["quantity", "price", "amount"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized[normalized["symbol"] != ""].reset_index(drop=True)


def _normalize_positions_archive(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in POSITION_ARCHIVE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    normalized = normalized[POSITION_ARCHIVE_COLUMNS]
    normalized[POSITION_ID_COLUMN] = normalized[POSITION_ID_COLUMN].fillna("").astype(str).str.strip()
    normalized["symbol"] = normalized["symbol"].fillna("").astype(str).str.upper().str.strip()
    normalized["buy_date"] = normalized["buy_date"].fillna("").astype(str).str.strip()
    normalized["strategy"] = normalized["strategy"].apply(normalize_strategy)

    for column in [
        "share_price",
        "stop_price",
        "atr",
        "risk_in_atr",
        "stop_loss_percent",
        "number_of_shares",
        "sell_lot",
        "hold_count",
        "position_size",
        "risk_percent",
        "risk_amount",
        "portfolio_amount",
    ]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized[normalized["symbol"] != ""].reset_index(drop=True)
    normalized = ensure_position_ids(normalized)
    return normalized[POSITION_ARCHIVE_COLUMNS]


def _normalize_planned_stops(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in PLANNED_STOP_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    normalized = normalized[PLANNED_STOP_COLUMNS]
    normalized["symbol"] = normalized["symbol"].fillna("").astype(str).str.upper().str.strip()
    normalized["buy_date"] = normalized["buy_date"].fillna("").astype(str).str.strip()
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce")
    normalized["planned_stop"] = pd.to_numeric(normalized["planned_stop"], errors="coerce")
    normalized["strategy"] = normalized["strategy"].apply(normalize_strategy)
    normalized["atr"] = pd.to_numeric(normalized["atr"], errors="coerce")
    normalized["market_regime"] = normalized["market_regime"].apply(_normalize_market_regime_or_blank)

    return normalized[normalized["symbol"] != ""].reset_index(drop=True)


def _planned_stop_from_calculated_position(row: pd.Series) -> dict[str, object] | None:
    quantity = pd.to_numeric(row.get("number_of_shares"), errors="coerce")
    planned_stop = pd.to_numeric(row.get("stop_price"), errors="coerce")
    if pd.isna(quantity) or pd.isna(planned_stop) or quantity <= 0:
        return None

    symbol = str(row.get("symbol") or "").upper().strip()
    buy_date = str(row.get("buy_date") or "").strip()
    if not symbol or not buy_date:
        return None

    return {
        "symbol": symbol,
        "buy_date": buy_date,
        "quantity": quantity,
        "planned_stop": planned_stop,
        "strategy": normalize_strategy(row.get("strategy")),
        "atr": pd.to_numeric(row.get("atr"), errors="coerce"),
        "market_regime": _normalize_market_regime_or_blank(row.get("market_regime")),
    }


def _normalize_market_regime_or_blank(value: object) -> str:
    regime = str(value or "").strip()
    if not regime:
        return ""
    normalized = normalize_market_regime(regime)
    return normalized if normalized == regime.upper() else ""


def _planned_stop_key(row: pd.Series | dict[str, object]) -> tuple[str, str, float | None]:
    quantity = _key_number(row.get("quantity"))
    return (
        str(row.get("symbol") or "").upper().strip(),
        str(row.get("buy_date") or "").strip(),
        quantity,
    )


def _planned_market_regime_for_key(
    planned_stops: pd.DataFrame,
    key: tuple[str, str, float | None],
) -> str:
    regimes = {
        str(row.get("market_regime") or "").strip()
        for _, row in planned_stops.iterrows()
        if _planned_stop_key(row) == key and str(row.get("market_regime") or "").strip()
    }
    return next(iter(regimes)) if len(regimes) == 1 else ""


def _transaction_keys(df: pd.DataFrame) -> list[tuple[str, str, str, str, str, str, float | None, float | None, float | None]]:
    return [_transaction_key(row) for _, row in df.iterrows()]


def _transaction_key(row: pd.Series) -> tuple[str, str, str, str, str, str, float | None, float | None, float | None]:
    return (
        str(row.get("activity_date") or ""),
        str(row.get("process_date") or ""),
        str(row.get("settle_date") or ""),
        str(row.get("symbol") or "").upper().strip(),
        str(row.get("description") or "").strip(),
        str(row.get("trans_code") or "").strip(),
        _key_number(row.get("quantity")),
        _key_number(row.get("price")),
        _key_number(row.get("amount")),
    )


def _key_number(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return round(float(number), 8)


def _worksheet_values(df: pd.DataFrame, columns: list[str]) -> list[list[Any]]:
    normalized = df.copy()
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = None
    normalized = normalized[columns].astype(object)
    normalized = normalized.where(pd.notna(normalized), "")
    return [columns, *normalized.values.tolist()]


def _streamlit_secrets() -> Mapping[str, Any]:
    try:
        import streamlit as st

        return st.secrets
    except Exception:
        return {}


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except (KeyError, TypeError):
        return None
