from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


INPUT_COLUMNS = [
    "symbol",
    "buy_date",
    "share_price",
    "stop_price",
    "atr",
    "portfolio_amount",
    "risk_percent",
]

OUTPUT_COLUMNS = [
    *INPUT_COLUMNS,
    "stop_loss_percent",
    "risk_in_atr",
    "risk_amount",
    "number_of_shares",
    "hold_count",
    "sell_lot",
    "position_size",
    "validation_error",
]

PUBLIC_OUTPUT_COLUMNS = [
    "symbol",
    "buy_date",
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
]


@dataclass(frozen=True)
class PositionCalculation:
    stop_loss_percent: float | None
    risk_in_atr: float | None
    risk_amount: float | None
    number_of_shares: int | None
    hold_count: int | None
    sell_lot: int | None
    position_size: float | None
    validation_error: str


def empty_positions(rows: int = 8) -> pd.DataFrame:
    today = date.today().isoformat()
    return pd.DataFrame(
        [
            {
                "symbol": "",
                "buy_date": today,
                "share_price": None,
                "stop_price": None,
                "atr": None,
                "portfolio_amount": 20_000.0,
                "risk_percent": 0.50,
            }
            for _ in range(rows)
        ],
        columns=INPUT_COLUMNS,
    )


def normalize_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in INPUT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    normalized = normalized[INPUT_COLUMNS]
    normalized["symbol"] = normalized["symbol"].fillna("").astype(str).str.upper().str.strip()
    normalized["buy_date"] = normalized["buy_date"].fillna("").astype(str).str.strip()

    for column in ["share_price", "stop_price", "atr", "portfolio_amount", "risk_percent"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized


def committed_positions(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_input_frame(df)
    return normalized[normalized["symbol"] != ""].reset_index(drop=True)


def delete_positions_by_index(df: pd.DataFrame, row_indexes: list[int]) -> pd.DataFrame:
    positions = committed_positions(df)
    if not row_indexes:
        return positions

    delete_indexes = {int(row_index) for row_index in row_indexes}
    keep_mask = ~positions.index.isin(delete_indexes)
    return positions.loc[keep_mask].reset_index(drop=True)


def draft_position(
    *,
    symbol: str,
    buy_date: Any,
    share_price: float | None,
    stop_price: float | None,
    portfolio_amount: float | None,
    risk_percent: float | None,
    atr: float | None = None,
) -> pd.DataFrame:
    return normalize_input_frame(
        pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "buy_date": buy_date,
                    "share_price": share_price,
                    "stop_price": stop_price,
                    "atr": atr,
                    "portfolio_amount": portfolio_amount,
                    "risk_percent": risk_percent,
                }
            ]
        )
    )


def append_committed_position(positions: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    draft_row = committed_positions(draft)
    if draft_row.empty:
        return committed_positions(positions)

    calculated = calculate_positions(draft_row)
    validation_error = str(calculated.iloc[0]["validation_error"] or "")
    if validation_error:
        return committed_positions(positions)

    return committed_positions(pd.concat([draft_row, committed_positions(positions)], ignore_index=True))


def calculate_position(row: pd.Series, *, as_of: date | None = None) -> PositionCalculation:
    symbol = str(row.get("symbol") or "").strip()
    share_price = _to_float(row.get("share_price"))
    stop_price = _to_float(row.get("stop_price"))
    atr = _to_float(row.get("atr"))
    portfolio_amount = _to_float(row.get("portfolio_amount"))
    risk_percent = _to_float(row.get("risk_percent"))
    hold_count = weekday_hold_count(row.get("buy_date"), as_of=as_of)

    if not symbol:
        return PositionCalculation(None, None, None, None, None, None, None, "")
    if share_price is None or share_price <= 0:
        return PositionCalculation(None, None, None, None, hold_count, None, None, "Share price must be greater than 0.")
    if stop_price is None or stop_price <= 0:
        return PositionCalculation(None, None, None, None, hold_count, None, None, "Stop price must be greater than 0.")
    if stop_price >= share_price:
        return PositionCalculation(None, None, None, None, hold_count, None, None, "Stop price must be below share price.")
    if portfolio_amount is None or portfolio_amount <= 0:
        return PositionCalculation(None, None, None, None, hold_count, None, None, "Portfolio amount must be greater than 0.")
    if risk_percent is None or risk_percent <= 0:
        return PositionCalculation(None, None, None, None, hold_count, None, None, "Risk percent must be greater than 0.")

    risk_per_share = share_price - stop_price
    stop_loss_percent = round((risk_per_share / share_price) * 100, 2)
    risk_in_atr = round(stop_loss_percent / atr, 2) if atr is not None and atr > 0 else None
    risk_amount = portfolio_amount * (risk_percent / 100)
    number_of_shares = int(risk_amount // risk_per_share)

    if number_of_shares <= 0:
        return PositionCalculation(
            stop_loss_percent,
            risk_in_atr,
            round(risk_amount, 2),
            0,
            hold_count,
            0,
            0.0,
            "Risk amount is too small to buy one share at this stop distance.",
        )

    position_size = number_of_shares * share_price

    return PositionCalculation(
        stop_loss_percent=stop_loss_percent,
        risk_in_atr=risk_in_atr,
        risk_amount=round(risk_amount, 2),
        number_of_shares=number_of_shares,
        hold_count=hold_count,
        sell_lot=max(1, number_of_shares // 3),
        position_size=round(position_size, 2),
        validation_error="",
    )


def calculate_positions(df: pd.DataFrame, *, as_of: date | None = None) -> pd.DataFrame:
    normalized = normalize_input_frame(df)
    calculations = normalized.apply(lambda row: calculate_position(row, as_of=as_of), axis=1)

    result = normalized.copy()
    for field in PositionCalculation.__dataclass_fields__:
        result[field] = [getattr(calculation, field) for calculation in calculations]

    return result[OUTPUT_COLUMNS]


def percent_of_portfolio(amount: Any, portfolio_amount: Any) -> float | None:
    amount_value = _to_float(amount)
    portfolio_value = _to_float(portfolio_amount)
    if amount_value is None or portfolio_value is None or portfolio_value <= 0:
        return None
    return round((amount_value / portfolio_value) * 100, 2)


def symbol_exposure_breaches(
    positions: pd.DataFrame,
    *,
    portfolio_amount: Any,
    max_symbol_exposure_percent: Any,
) -> pd.DataFrame:
    columns = ["symbol", "position_size", "exposure_percent"]
    portfolio_value = _to_float(portfolio_amount)
    limit = _to_float(max_symbol_exposure_percent)
    if positions.empty or portfolio_value is None or portfolio_value <= 0 or limit is None or limit <= 0:
        return pd.DataFrame(columns=columns)
    if "symbol" not in positions.columns or "position_size" not in positions.columns:
        return pd.DataFrame(columns=columns)

    exposure = positions[["symbol", "position_size"]].copy()
    exposure["symbol"] = exposure["symbol"].fillna("").astype(str).str.upper().str.strip()
    exposure["position_size"] = pd.to_numeric(exposure["position_size"], errors="coerce")
    exposure = exposure[(exposure["symbol"] != "") & (exposure["position_size"] > 0)]
    if exposure.empty:
        return pd.DataFrame(columns=columns)

    grouped = exposure.groupby("symbol", as_index=False, sort=True)["position_size"].sum()
    grouped["exposure_percent"] = ((grouped["position_size"] / portfolio_value) * 100).round(2)
    breached = grouped[grouped["exposure_percent"] > limit].copy()
    if breached.empty:
        return pd.DataFrame(columns=columns)
    breached = breached.sort_values(["exposure_percent", "symbol"], ascending=[False, True]).reset_index(drop=True)
    breached["position_size"] = breached["position_size"].round(2)
    return breached[columns]


def prospective_symbol_exposure_breach(
    positions: pd.DataFrame,
    draft: pd.DataFrame | pd.Series,
    *,
    portfolio_amount: Any,
    max_symbol_exposure_percent: Any,
) -> pd.Series | None:
    draft_frame = draft.to_frame().T if isinstance(draft, pd.Series) else draft.copy()
    if draft_frame.empty or "symbol" not in draft_frame.columns:
        return None

    symbol = str(draft_frame.iloc[0].get("symbol") or "").upper().strip()
    position_size = _to_float(draft_frame.iloc[0].get("position_size"))
    if not symbol or position_size is None or position_size <= 0:
        return None

    combined = pd.concat([positions, draft_frame], ignore_index=True)
    breaches = symbol_exposure_breaches(
        combined,
        portfolio_amount=portfolio_amount,
        max_symbol_exposure_percent=max_symbol_exposure_percent,
    )
    if breaches.empty:
        return None

    matches = breaches[breaches["symbol"] == symbol]
    if matches.empty:
        return None
    return matches.iloc[0]


def weekday_hold_count(buy_date: Any, *, as_of: date | None = None) -> int | None:
    parsed_buy_date = _to_date(buy_date)
    if parsed_buy_date is None:
        return None

    end_date = as_of or date.today()
    if parsed_buy_date >= end_date:
        return 0

    days = pd.date_range(parsed_buy_date, end_date, inclusive="right")
    return int(sum(day.weekday() < 5 for day in days))


def format_currency(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return ""
    return f"${number:,.2f}"


def format_percent(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return ""
    return f"{number:.2f}%"


def _to_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_date(value: Any) -> date | None:
    if pd.isna(value):
        return None
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()
