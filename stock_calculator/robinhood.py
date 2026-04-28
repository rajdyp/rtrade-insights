from __future__ import annotations

import csv
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from stock_calculator.calculations import weekday_hold_count


ROBINHOOD_COLUMNS = [
    "Activity Date",
    "Process Date",
    "Settle Date",
    "Instrument",
    "Description",
    "Trans Code",
    "Quantity",
    "Price",
    "Amount",
]

TRADE_CODES = {"Buy", "Sell"}
STRATEGY_OPTIONS = ["EP", "5% BO", "BO"]
UNCLASSIFIED_STRATEGY = "Unclassified"

TRANSACTION_COLUMNS = [
    "activity_date",
    "process_date",
    "settle_date",
    "symbol",
    "description",
    "trans_code",
    "quantity",
    "price",
    "amount",
]

PLANNED_STOP_COLUMNS = [
    "symbol",
    "buy_date",
    "quantity",
    "planned_stop",
    "strategy",
]

CLOSED_TRADE_COLUMNS = [
    "symbol",
    "buy_date",
    "sell_date",
    "quantity",
    "planned_stop",
    "strategy",
    "buy_price",
    "buy_amount",
    "sell_price",
    "sell_amount",
    "realized_pnl",
    "realized_pnl_percent",
    "hold_days",
]

OPEN_POSITION_COLUMNS = [
    "symbol",
    "buy_date",
    "quantity",
    "planned_stop",
    "strategy",
    "buy_price",
    "cost_basis",
    "hold_days",
]

UNMATCHED_SELL_COLUMNS = ["symbol", "sell_date", "quantity", "sell_price"]
IMPORT_ISSUE_COLUMNS = ["row_number", "issue", "raw_row"]
STRATEGY_METRIC_COLUMNS = [
    "strategy",
    "trade_count",
    "total_realized_pnl",
    "win_rate",
    "expectancy",
    "profit_factor",
    "average_win",
    "average_loss",
    "average_win_r",
    "average_loss_r",
    "r_ratio",
    "average_win_hold",
    "average_loss_hold",
    "rolling_10r_exp",
    "mode",
    "action",
]


@dataclass(frozen=True)
class ImportResult:
    transactions: pd.DataFrame
    ignored_rows: pd.DataFrame
    malformed_rows: pd.DataFrame


@dataclass(frozen=True)
class TradeDerivation:
    closed_trades: pd.DataFrame
    open_positions: pd.DataFrame
    unmatched_sells: pd.DataFrame
    missing_planned_stops: int


def calculate_trade_metrics(closed_trades: pd.DataFrame) -> dict[str, float | int | None]:
    if closed_trades.empty:
        return _empty_trade_metrics()

    trades = closed_trades.copy()
    trades["realized_pnl"] = pd.to_numeric(trades["realized_pnl"], errors="coerce")
    trades["realized_pnl_percent"] = pd.to_numeric(trades["realized_pnl_percent"], errors="coerce")
    trades["hold_days"] = pd.to_numeric(trades["hold_days"], errors="coerce")
    trades = trades.dropna(subset=["realized_pnl"]).reset_index(drop=True)
    if trades.empty:
        return _empty_trade_metrics()

    r_trades = trades.copy()
    for column in ["planned_stop", "buy_price", "quantity"]:
        r_trades[column] = pd.to_numeric(r_trades.get(column), errors="coerce")
    r_trades["initial_risk"] = (r_trades["buy_price"] - r_trades["planned_stop"]) * r_trades["quantity"]
    r_trades = r_trades[r_trades["initial_risk"] > 0].copy()
    r_trades["r_multiple"] = r_trades["realized_pnl"] / r_trades["initial_risk"]
    r_wins = r_trades[r_trades["realized_pnl"] > 0]
    r_losses = r_trades[r_trades["realized_pnl"] < 0]
    average_win_r = _mean(r_wins["r_multiple"])
    average_loss_r = _mean(r_losses["r_multiple"])
    expectancy_r = _mean(r_trades["r_multiple"])

    wins = trades[trades["realized_pnl"] > 0]
    losses = trades[trades["realized_pnl"] < 0]
    breakevens = trades[trades["realized_pnl"] == 0]
    gross_win = float(wins["realized_pnl"].sum())
    gross_loss = abs(float(losses["realized_pnl"].sum()))
    average_win = _mean(wins["realized_pnl"])
    average_loss = _mean(losses["realized_pnl"])

    return {
        "trade_count": len(trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "breakeven_count": len(breakevens),
        "win_rate": round((len(wins) / len(trades)) * 100, 1),
        "win_loss_ratio": _safe_round(average_win / abs(average_loss)) if average_win is not None and average_loss else None,
        "average_win_r": _safe_round(average_win_r),
        "average_loss_r": _safe_round(average_loss_r),
        "r_ratio": _safe_round(average_win_r / abs(average_loss_r))
        if average_win_r is not None and average_loss_r
        else None,
        "expectancy_r": _safe_round(expectancy_r),
        "expectancy": _safe_round(_mean(trades["realized_pnl"])),
        "profit_factor": _safe_round(gross_win / gross_loss) if gross_loss else None,
        "average_win": _safe_round(average_win),
        "average_win_percent": _safe_round(_mean(wins["realized_pnl_percent"])),
        "average_win_hold": _safe_round(_mean(wins["hold_days"]), digits=1),
        "win_streak": _longest_streak(trades, winning=True),
        "top_win": _safe_round(wins["realized_pnl"].max()) if not wins.empty else None,
        "average_loss": _safe_round(average_loss),
        "average_loss_percent": _safe_round(_mean(losses["realized_pnl_percent"])),
        "average_loss_hold": _safe_round(_mean(losses["hold_days"]), digits=1),
        "loss_streak": _longest_streak(trades, winning=False),
        "top_loss": _safe_round(losses["realized_pnl"].min()) if not losses.empty else None,
    }


def calculate_total_realized_pnl(closed_trades: pd.DataFrame) -> float:
    if closed_trades.empty or "realized_pnl" not in closed_trades.columns:
        return 0.0

    realized_pnl = pd.to_numeric(closed_trades["realized_pnl"], errors="coerce")
    return round(float(realized_pnl.dropna().sum()), 2)


def calculate_rolling_10r_mode(closed_trades: pd.DataFrame) -> dict[str, str]:
    rolling_10r = _calculate_rolling_10r_exp(closed_trades)
    if rolling_10r is None:
        return {"rolling_10r_exp": "N/A", "mode": "Unknown", "action": "Tiny size only"}

    if rolling_10r > 0.30:
        mode = "Working"
        action = "Normal size"
    elif rolling_10r >= 0:
        mode = "Caution"
        action = "Half size"
    elif rolling_10r >= -0.25:
        mode = "Weak"
        action = "Quarter size"
    else:
        mode = "Failing"
        action = "Probe only / pause"

    return {"rolling_10r_exp": f"{rolling_10r:+.2f}R", "mode": mode, "action": action}


def calculate_strategy_metrics(closed_trades: pd.DataFrame) -> pd.DataFrame:
    if closed_trades.empty:
        return pd.DataFrame(columns=STRATEGY_METRIC_COLUMNS)

    trades = closed_trades.copy()
    strategy_values = trades["strategy"] if "strategy" in trades.columns else pd.Series([""] * len(trades), index=trades.index)
    trades["strategy_group"] = strategy_values.apply(display_strategy)
    rows = []
    grouped_by_strategy = dict(tuple(trades.groupby("strategy_group", sort=False, dropna=False)))
    for strategy in [*STRATEGY_OPTIONS, UNCLASSIFIED_STRATEGY]:
        grouped_trades = grouped_by_strategy.get(strategy)
        if grouped_trades is None:
            continue
        metrics = calculate_trade_metrics(grouped_trades)
        if metrics["trade_count"] == 0:
            continue
        rolling_10r_mode = calculate_rolling_10r_mode(grouped_trades)
        rows.append(
            {
                "strategy": strategy,
                "trade_count": metrics["trade_count"],
                "total_realized_pnl": calculate_total_realized_pnl(grouped_trades),
                "win_rate": metrics["win_rate"],
                "expectancy": metrics["expectancy"],
                "profit_factor": metrics["profit_factor"],
                "average_win": metrics["average_win"],
                "average_loss": metrics["average_loss"],
                "average_win_r": metrics["average_win_r"],
                "average_loss_r": metrics["average_loss_r"],
                "r_ratio": metrics["r_ratio"],
                "average_win_hold": metrics["average_win_hold"],
                "average_loss_hold": metrics["average_loss_hold"],
                "rolling_10r_exp": rolling_10r_mode["rolling_10r_exp"],
                "mode": rolling_10r_mode["mode"],
                "action": rolling_10r_mode["action"],
            }
        )

    if not rows:
        return pd.DataFrame(columns=STRATEGY_METRIC_COLUMNS)
    return pd.DataFrame(rows, columns=STRATEGY_METRIC_COLUMNS)


def _calculate_rolling_10r_exp(closed_trades: pd.DataFrame) -> float | None:
    if len(closed_trades) < 10:
        return None

    trades = closed_trades.copy()
    trades["sell_date"] = pd.to_datetime(trades.get("sell_date"), errors="coerce")
    trades = trades.sort_values("sell_date", kind="mergesort").tail(10).copy()
    if len(trades) < 10:
        return None

    for column in ["realized_pnl", "buy_price", "planned_stop", "quantity"]:
        trades[column] = pd.to_numeric(trades.get(column), errors="coerce")

    trades["initial_risk"] = (trades["buy_price"] - trades["planned_stop"]) * trades["quantity"]
    trades["r_multiple"] = trades["realized_pnl"] / trades["initial_risk"]
    valid_r_multiples = trades.loc[trades["initial_risk"] > 0, "r_multiple"].dropna()
    if len(valid_r_multiples) != 10:
        return None

    return float(valid_r_multiples.sum()) / 10


def parse_robinhood_csv(source: str | Path | TextIO | Any) -> ImportResult:
    rows = _read_csv_rows(source)
    if not rows:
        return ImportResult(_empty_transactions(), _empty_issues(), _empty_issues())

    header = rows[0]
    if header != ROBINHOOD_COLUMNS:
        return ImportResult(
            _empty_transactions(),
            _empty_issues(),
            pd.DataFrame(
                [{"row_number": 1, "issue": "Unexpected Robinhood CSV header.", "raw_row": repr(header)}],
                columns=IMPORT_ISSUE_COLUMNS,
            ),
        )

    transactions: list[dict[str, Any]] = []
    ignored_rows: list[dict[str, Any]] = []
    malformed_rows: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows[1:], start=2):
        if _is_robinhood_footer_row(row):
            continue

        if len(row) != len(ROBINHOOD_COLUMNS):
            malformed_rows.append(
                {
                    "row_number": row_number,
                    "issue": f"Expected {len(ROBINHOOD_COLUMNS)} columns, found {len(row)}.",
                    "raw_row": repr(row),
                }
            )
            continue

        record = dict(zip(ROBINHOOD_COLUMNS, row, strict=True))
        trans_code = str(record["Trans Code"]).strip()
        if trans_code not in TRADE_CODES:
            ignored_rows.append(
                {
                    "row_number": row_number,
                    "issue": f"Ignored transaction code: {trans_code or 'blank'}.",
                    "raw_row": repr(row),
                }
            )
            continue

        activity_date = pd.to_datetime(record["Activity Date"], errors="coerce")
        process_date = pd.to_datetime(record["Process Date"], errors="coerce")
        settle_date = pd.to_datetime(record["Settle Date"], errors="coerce")
        quantity = _parse_number(record["Quantity"])
        price = _parse_money(record["Price"])
        amount = _parse_money(record["Amount"])
        symbol = str(record["Instrument"]).upper().strip()

        if pd.isna(activity_date) or not symbol or quantity is None or price is None or amount is None:
            malformed_rows.append(
                {
                    "row_number": row_number,
                    "issue": "Missing or invalid trade field.",
                    "raw_row": repr(row),
                }
            )
            continue

        transactions.append(
            {
                "activity_date": activity_date.date().isoformat(),
                "process_date": "" if pd.isna(process_date) else process_date.date().isoformat(),
                "settle_date": "" if pd.isna(settle_date) else settle_date.date().isoformat(),
                "symbol": symbol,
                "description": record["Description"],
                "trans_code": trans_code,
                "quantity": quantity,
                "price": price,
                "amount": amount,
            }
        )

    return ImportResult(
        pd.DataFrame(transactions, columns=TRANSACTION_COLUMNS),
        pd.DataFrame(ignored_rows, columns=IMPORT_ISSUE_COLUMNS),
        pd.DataFrame(malformed_rows, columns=IMPORT_ISSUE_COLUMNS),
    )


def derive_fifo_trades(
    transactions: pd.DataFrame,
    planned_stops: pd.DataFrame | None = None,
    *,
    as_of: date | None = None,
) -> TradeDerivation:
    if transactions.empty:
        return TradeDerivation(_empty_closed_trades(), _empty_open_positions(), _empty_unmatched_sells(), 0)

    planned_stop_lookup = _planned_stop_lookup(planned_stops)
    normalized = _normalize_transactions(transactions)
    lots: defaultdict[str, deque[dict[str, Any]]] = defaultdict(deque)
    closed_trades: list[dict[str, Any]] = []
    unmatched_sells: list[dict[str, Any]] = []

    for _, row in normalized.iterrows():
        symbol = row["symbol"]
        quantity = float(row["quantity"])
        if row["trans_code"] == "Buy":
            buy_date = row["activity_date"]
            lots[symbol].append(
                {
                    "symbol": symbol,
                    "buy_date": buy_date,
                    "quantity": quantity,
                    "buy_price": float(row["price"]),
                    **planned_stop_lookup.get(
                        (symbol, buy_date, _quantity_key(quantity)),
                        {"planned_stop": None, "strategy": ""},
                    ),
                }
            )
            continue

        remaining_quantity = quantity
        sell_price = float(row["price"])
        sell_date = row["activity_date"]
        while remaining_quantity > 1e-9 and lots[symbol]:
            lot = lots[symbol][0]
            matched_quantity = min(remaining_quantity, lot["quantity"])
            buy_price = float(lot["buy_price"])
            buy_amount = round(matched_quantity * buy_price, 2)
            sell_amount = round(matched_quantity * sell_price, 2)
            realized_pnl = round(sell_amount - buy_amount, 2)
            realized_pnl_percent = round((realized_pnl / buy_amount) * 100, 2) if buy_amount else None

            closed_trades.append(
                {
                    "symbol": symbol,
                    "buy_date": lot["buy_date"],
                    "sell_date": sell_date,
                    "quantity": _clean_quantity(matched_quantity),
                    "planned_stop": lot["planned_stop"],
                    "strategy": lot["strategy"],
                    "buy_price": buy_price,
                    "buy_amount": buy_amount,
                    "sell_price": sell_price,
                    "sell_amount": sell_amount,
                    "realized_pnl": realized_pnl,
                    "realized_pnl_percent": realized_pnl_percent,
                    "hold_days": weekday_hold_count(lot["buy_date"], as_of=_to_date(sell_date)),
                }
            )

            lot["quantity"] -= matched_quantity
            remaining_quantity -= matched_quantity
            if lot["quantity"] <= 1e-9:
                lots[symbol].popleft()

        if remaining_quantity > 1e-9:
            unmatched_sells.append(
                {
                    "symbol": symbol,
                    "sell_date": sell_date,
                    "quantity": _clean_quantity(remaining_quantity),
                    "sell_price": sell_price,
                }
            )

    open_positions = []
    for symbol in sorted(lots):
        for lot in lots[symbol]:
            if lot["quantity"] <= 1e-9:
                continue
            quantity = _clean_quantity(lot["quantity"])
            open_positions.append(
                {
                    "symbol": symbol,
                    "buy_date": lot["buy_date"],
                    "quantity": quantity,
                    "planned_stop": lot["planned_stop"],
                    "strategy": lot["strategy"],
                    "buy_price": float(lot["buy_price"]),
                    "cost_basis": round(float(lot["quantity"]) * float(lot["buy_price"]), 2),
                    "hold_days": weekday_hold_count(lot["buy_date"], as_of=as_of),
                }
            )

    closed_frame = pd.DataFrame(closed_trades, columns=CLOSED_TRADE_COLUMNS)
    open_frame = pd.DataFrame(open_positions, columns=OPEN_POSITION_COLUMNS)
    unmatched_frame = pd.DataFrame(unmatched_sells, columns=UNMATCHED_SELL_COLUMNS)
    missing_planned_stops = _missing_stop_count(closed_frame) + _missing_stop_count(open_frame)

    return TradeDerivation(closed_frame, open_frame, unmatched_frame, missing_planned_stops)


def _read_csv_rows(source: str | Path | TextIO | Any) -> list[list[str]]:
    if isinstance(source, str | Path):
        with Path(source).open(newline="", encoding="utf-8-sig") as file:
            return list(csv.reader(file))

    if hasattr(source, "getvalue"):
        value = source.getvalue()
        if isinstance(value, bytes):
            value = value.decode("utf-8-sig")
        return list(csv.reader(str(value).splitlines()))

    return list(csv.reader(source))


def _is_robinhood_footer_row(row: list[str]) -> bool:
    values = [str(value).strip() for value in row]
    if not any(values):
        return True

    text = " ".join(value for value in values if value).lower()
    return "the data provided is for informational purposes only" in text


def _normalize_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    normalized = transactions.copy()
    normalized["activity_date"] = pd.to_datetime(normalized["activity_date"], errors="coerce").dt.date.astype(str)
    normalized["symbol"] = normalized["symbol"].fillna("").astype(str).str.upper().str.strip()
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce")
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    normalized["side_rank"] = normalized["trans_code"].map({"Buy": 0, "Sell": 1}).fillna(9)
    return normalized.sort_values(["activity_date", "symbol", "side_rank"]).reset_index(drop=True)


def normalize_strategy(value: Any) -> str:
    strategy = str(value or "").strip()
    return strategy if strategy in STRATEGY_OPTIONS else ""


def display_strategy(value: Any) -> str:
    strategy = normalize_strategy(value)
    return strategy or UNCLASSIFIED_STRATEGY


def _planned_stop_lookup(planned_stops: pd.DataFrame | None) -> dict[tuple[str, str, float], dict[str, float | str | None]]:
    if planned_stops is None or planned_stops.empty:
        return {}

    stops_by_key: defaultdict[tuple[str, str, float], set[float]] = defaultdict(set)
    strategies_by_key: defaultdict[tuple[str, str, float], set[str]] = defaultdict(set)
    for _, row in planned_stops.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        buy_date = pd.to_datetime(row.get("buy_date"), errors="coerce")
        if pd.isna(buy_date):
            continue
        quantity = pd.to_numeric(row.get("quantity"), errors="coerce")
        if pd.isna(quantity):
            continue
        key = (symbol, buy_date.date().isoformat(), _quantity_key(quantity))

        planned_stop = pd.to_numeric(row.get("planned_stop"), errors="coerce")
        if not pd.isna(planned_stop):
            stops_by_key[key].add(float(planned_stop))

        strategy = normalize_strategy(row.get("strategy"))
        if strategy:
            strategies_by_key[key].add(strategy)

    lookup = {}
    for key in set(stops_by_key) | set(strategies_by_key):
        stops = stops_by_key[key]
        strategies = strategies_by_key[key]
        lookup[key] = {
            "planned_stop": next(iter(stops)) if len(stops) == 1 else None,
            "strategy": next(iter(strategies)) if len(strategies) == 1 else "",
        }
    return lookup


def _parse_money(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    text = re.sub(r"[^0-9.-]", "", text)
    if not text or text in {"-", ".", "-."}:
        return None
    return float(text)


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    return float(text)


def _to_date(value: Any) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _clean_quantity(value: float) -> float | int:
    if float(value).is_integer():
        return int(value)
    return round(float(value), 6)


def _quantity_key(value: Any) -> float:
    return round(float(value), 8)


def _missing_stop_count(frame: pd.DataFrame) -> int:
    if frame.empty or "planned_stop" not in frame.columns:
        return 0
    return int(frame["planned_stop"].isna().sum())


def _empty_trade_metrics() -> dict[str, float | int | None]:
    return {
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "breakeven_count": 0,
        "win_rate": None,
        "win_loss_ratio": None,
        "average_win_r": None,
        "average_loss_r": None,
        "r_ratio": None,
        "expectancy_r": None,
        "expectancy": None,
        "profit_factor": None,
        "average_win": None,
        "average_win_percent": None,
        "average_win_hold": None,
        "win_streak": 0,
        "top_win": None,
        "average_loss": None,
        "average_loss_percent": None,
        "average_loss_hold": None,
        "loss_streak": 0,
        "top_loss": None,
    }


def _mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    value = values.mean()
    if pd.isna(value):
        return None
    return float(value)


def _safe_round(value: Any, *, digits: int = 2) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _longest_streak(trades: pd.DataFrame, *, winning: bool) -> int:
    if trades.empty:
        return 0

    ordered = trades.copy()
    ordered["_original_order"] = range(len(ordered))
    ordered = ordered.sort_values(["sell_date", "_original_order"])
    best_streak = 0
    current_streak = 0

    for pnl in ordered["realized_pnl"]:
        is_match = pnl > 0 if winning else pnl < 0
        if is_match:
            current_streak += 1
        else:
            current_streak = 0
        best_streak = max(best_streak, current_streak)

    return best_streak


def _empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSACTION_COLUMNS)


def _empty_closed_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=CLOSED_TRADE_COLUMNS)


def _empty_open_positions() -> pd.DataFrame:
    return pd.DataFrame(columns=OPEN_POSITION_COLUMNS)


def _empty_unmatched_sells() -> pd.DataFrame:
    return pd.DataFrame(columns=UNMATCHED_SELL_COLUMNS)


def _empty_issues() -> pd.DataFrame:
    return pd.DataFrame(columns=IMPORT_ISSUE_COLUMNS)
