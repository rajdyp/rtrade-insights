from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_calculator.calculations import INPUT_COLUMNS, committed_positions, normalize_input_frame
from stock_calculator.robinhood import PLANNED_STOP_COLUMNS, TRANSACTION_COLUMNS


DATA_PATH = Path("data/positions.csv")
ROBINHOOD_TRANSACTIONS_PATH = Path("data/robinhood_transactions.csv")
PLANNED_STOPS_PATH = Path("data/planned_stops.csv")


def load_positions(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return normalize_input_frame(pd.DataFrame(columns=INPUT_COLUMNS))
    return committed_positions(pd.read_csv(path))


def save_positions(df: pd.DataFrame, path: Path = DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    committed_positions(df).to_csv(path, index=False, columns=INPUT_COLUMNS)


def load_planned_stops(path: Path = PLANNED_STOPS_PATH) -> pd.DataFrame:
    if not path.exists():
        return _normalize_planned_stops(pd.DataFrame(columns=PLANNED_STOP_COLUMNS))
    return _normalize_planned_stops(pd.read_csv(path))


def save_planned_stops(df: pd.DataFrame, path: Path = PLANNED_STOPS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_planned_stops(df).to_csv(path, index=False, columns=PLANNED_STOP_COLUMNS)


def upsert_planned_stop(planned_stops: pd.DataFrame, calculated_position: pd.Series) -> pd.DataFrame:
    row = _planned_stop_from_calculated_position(calculated_position)
    if row is None:
        return _normalize_planned_stops(planned_stops)

    normalized = _normalize_planned_stops(planned_stops)
    key = _planned_stop_key(row)
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
        }
        for _, row in buys.iterrows()
    ]
    return _normalize_planned_stops(pd.DataFrame(rows, columns=PLANNED_STOP_COLUMNS))


def load_robinhood_transactions(path: Path = ROBINHOOD_TRANSACTIONS_PATH) -> pd.DataFrame:
    if not path.exists():
        return _normalize_robinhood_transactions(pd.DataFrame(columns=TRANSACTION_COLUMNS))
    return _normalize_robinhood_transactions(pd.read_csv(path))


def save_robinhood_transactions(df: pd.DataFrame, path: Path = ROBINHOOD_TRANSACTIONS_PATH) -> None:
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
    }


def _planned_stop_key(row: pd.Series | dict[str, object]) -> tuple[str, str, float | None]:
    quantity = _key_number(row.get("quantity"))
    return (
        str(row.get("symbol") or "").upper().strip(),
        str(row.get("buy_date") or "").strip(),
        quantity,
    )


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
