from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


POSITION_ID_COLUMN = "position_id"

INPUT_COLUMNS = [
    "symbol",
    "buy_date",
    "share_price",
    "stop_price",
    "atr",
    "portfolio_amount",
    "risk_percent",
]

POSITION_SOURCE_COLUMNS = [POSITION_ID_COLUMN, *INPUT_COLUMNS]

OUTPUT_COLUMNS = [
    *POSITION_SOURCE_COLUMNS,
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

POSITION_CAMPAIGN_COLUMNS = [
    "symbol",
    "lots",
    "current_shares",
    "avg_entry",
    "campaign_stop",
    "sell_lot",
    "position_size",
    "risk_at_campaign_stop",
    "planned_lot_risk",
    "strategy",
    "source",
]

CAMPAIGN_OVERRIDE_COLUMNS = [
    "symbol",
    "current_shares",
    "campaign_stop",
]

CAMPAIGN_VIEW_COLUMNS = [
    "symbol",
    "lots",
    "current_shares",
    "avg_entry",
    "campaign_stop",
    "sell_lot",
    "position_size",
    "risk_at_campaign_stop",
    "planned_lot_risk",
    "strategy",
    "source",
]

CAMPAIGN_TRIM_COLUMNS = [
    "stop_itm",
    "live_price",
    "trim_count",
    "free_roll",
    "max_add",
    "add_risk",
    "profit_at_stop",
]

CAMPAIGN_TRIM_VIEW_COLUMNS = [
    "symbol",
    "lots",
    "current_shares",
    "avg_entry",
    "campaign_stop",
    "sell_lot",
    "position_size",
    "risk_at_campaign_stop",
    "planned_lot_risk",
    *CAMPAIGN_TRIM_COLUMNS,
]

DISPLAY_PLACEHOLDER = "-"


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


@dataclass(frozen=True)
class RiskNeutralAddOn:
    symbol: str
    source: str
    current_shares: int | float
    avg_entry: float
    draft_shares: int
    combined_shares: int | float
    combined_avg_entry: float
    combined_risk_at_stop: float
    max_risk_neutral_shares: int
    max_risk_neutral_risk_percent: float | None
    risk_neutral: bool


@dataclass(frozen=True)
class ProfitProtectedAddOn:
    applicable: bool
    max_add_shares: int
    add_risk: float
    profit_at_stop: float | None


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
    for column in POSITION_SOURCE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    normalized = normalized[POSITION_SOURCE_COLUMNS]
    normalized[POSITION_ID_COLUMN] = normalized[POSITION_ID_COLUMN].fillna("").astype(str).str.strip()
    normalized["symbol"] = normalized["symbol"].fillna("").astype(str).str.upper().str.strip()
    normalized["buy_date"] = normalized["buy_date"].fillna("").astype(str).str.strip()

    for column in ["share_price", "stop_price", "atr", "portfolio_amount", "risk_percent"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized


def ensure_position_ids(
    df: pd.DataFrame,
    *,
    existing_ids: set[str] | None = None,
    generate: str = "deterministic",
) -> pd.DataFrame:
    normalized = df.copy()
    if POSITION_ID_COLUMN not in normalized.columns:
        normalized[POSITION_ID_COLUMN] = ""

    normalized[POSITION_ID_COLUMN] = normalized[POSITION_ID_COLUMN].fillna("").astype(str).str.strip()
    used_ids = set(existing_ids or set())
    generated_counts: dict[str, int] = {}
    ids: list[str] = []
    for _, row in normalized.iterrows():
        symbol = str(row.get("symbol") or "").strip()
        position_id = str(row.get(POSITION_ID_COLUMN) or "").strip()
        if not symbol:
            ids.append("")
            continue
        if position_id and position_id not in used_ids:
            used_ids.add(position_id)
            ids.append(position_id)
            continue

        position_id = _new_position_id(row, used_ids, generated_counts, generate=generate)
        used_ids.add(position_id)
        ids.append(position_id)

    normalized[POSITION_ID_COLUMN] = ids
    return normalized


def committed_positions(df: pd.DataFrame, *, assign_missing_ids: bool = True) -> pd.DataFrame:
    normalized = normalize_input_frame(df)
    committed = normalized[normalized["symbol"] != ""].reset_index(drop=True)
    if assign_missing_ids:
        committed = ensure_position_ids(committed)
    return committed[POSITION_SOURCE_COLUMNS]


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
    current_positions = committed_positions(positions)
    draft_row = committed_positions(draft, assign_missing_ids=False)
    if draft_row.empty:
        return current_positions

    calculated = calculate_positions(draft_row)
    validation_error = str(calculated.iloc[0]["validation_error"] or "")
    if validation_error:
        return current_positions

    existing_ids = set(current_positions[POSITION_ID_COLUMN].fillna("").astype(str))
    draft_row = ensure_position_ids(draft_row, existing_ids=existing_ids, generate="uuid")
    return committed_positions(pd.concat([draft_row, current_positions], ignore_index=True))


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


def risk_neutral_add_on(
    *,
    symbol: str,
    draft: pd.Series,
    open_lots: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
) -> RiskNeutralAddOn | None:
    normalized_symbol = str(symbol or "").upper().strip()
    if not normalized_symbol:
        return None

    draft_price = _to_float(draft.get("share_price"))
    draft_stop = _to_float(draft.get("stop_price"))
    draft_shares_value = _to_float(draft.get("number_of_shares"))
    portfolio_amount = _to_float(draft.get("portfolio_amount"))
    if (
        draft_price is None
        or draft_stop is None
        or draft_shares_value is None
        or draft_shares_value <= 0
        or draft_stop >= draft_price
    ):
        return None

    existing_lots, source = _risk_neutral_existing_lots(normalized_symbol, open_lots, positions)
    if existing_lots.empty:
        return None

    current_shares = float(existing_lots["quantity"].sum())
    avg_entry = _weighted_average(existing_lots, "quantity", "buy_price")
    if current_shares <= 0 or avg_entry is None:
        return None

    draft_shares = int(draft_shares_value)
    risk_per_new_share = draft_price - draft_stop
    existing_risk_at_stop = (avg_entry - draft_stop) * current_shares
    max_safe_shares = int(max(0, (-existing_risk_at_stop) // risk_per_new_share))
    combined_shares = current_shares + draft_shares
    combined_avg_entry = ((avg_entry * current_shares) + (draft_price * draft_shares)) / combined_shares
    combined_risk_at_stop = round((combined_avg_entry - draft_stop) * combined_shares, 2)
    max_safe_risk_percent = (
        round(((max_safe_shares * risk_per_new_share) / portfolio_amount) * 100, 2)
        if portfolio_amount is not None and portfolio_amount > 0
        else None
    )

    return RiskNeutralAddOn(
        symbol=normalized_symbol,
        source=source,
        current_shares=_whole_number_if_possible(current_shares),
        avg_entry=round(avg_entry, 2),
        draft_shares=draft_shares,
        combined_shares=_whole_number_if_possible(combined_shares),
        combined_avg_entry=round(combined_avg_entry, 2),
        combined_risk_at_stop=combined_risk_at_stop,
        max_risk_neutral_shares=max_safe_shares,
        max_risk_neutral_risk_percent=max_safe_risk_percent,
        risk_neutral=combined_risk_at_stop <= 0,
    )


def risk_neutral_add_on_message(add_on: RiskNeutralAddOn | None) -> tuple[str, str] | None:
    if add_on is None:
        return None

    if add_on.risk_neutral:
        return (
            f"Risk-neutral: Yes. Max size {add_on.max_risk_neutral_shares}; draft {add_on.draft_shares}.",
            "ready",
        )

    cap_text = (
        f" (cap {format_percent(add_on.max_risk_neutral_risk_percent)})"
        if add_on.max_risk_neutral_risk_percent is not None
        else ""
    )
    return (
        f"Risk-neutral: No. Max size {add_on.max_risk_neutral_shares}{cap_text}; draft {add_on.draft_shares}.",
        "idle",
    )


def _risk_neutral_existing_lots(
    symbol: str,
    open_lots: pd.DataFrame | None,
    positions: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str]:
    robinhood_lots = _campaign_lots_from_robinhood(open_lots)
    robinhood_matches = robinhood_lots[robinhood_lots["symbol"] == symbol]
    if not robinhood_matches.empty:
        return robinhood_matches, "Robinhood"

    position_lots = _campaign_lots_from_positions(positions)
    position_matches = position_lots[position_lots["symbol"] == symbol]
    if not position_matches.empty:
        return position_matches, "Positions"

    return pd.DataFrame(), ""


def normalize_position_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in POSITION_CAMPAIGN_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    normalized = normalized[POSITION_CAMPAIGN_COLUMNS]
    normalized["symbol"] = normalized["symbol"].fillna("").astype(str).str.upper().str.strip()
    for column in [
        "lots",
        "current_shares",
        "avg_entry",
        "campaign_stop",
        "sell_lot",
        "position_size",
        "risk_at_campaign_stop",
        "planned_lot_risk",
    ]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["strategy"] = normalized["strategy"].fillna("").astype(str).str.strip()
    normalized["source"] = normalized["source"].fillna("").astype(str).str.strip()
    return normalized[normalized["symbol"] != ""].reset_index(drop=True)


def normalize_campaign_overrides(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in CAMPAIGN_OVERRIDE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    normalized = normalized[CAMPAIGN_OVERRIDE_COLUMNS]
    normalized["symbol"] = normalized["symbol"].fillna("").astype(str).str.upper().str.strip()
    for column in ["current_shares", "campaign_stop"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized[
        (normalized["symbol"] != "")
        & (normalized[["current_shares", "campaign_stop"]].notna().any(axis=1))
    ]
    normalized = normalized.drop_duplicates(subset=["symbol"], keep="first")
    return normalized.reset_index(drop=True)


def campaign_view_positions(open_lots: pd.DataFrame, positions: pd.DataFrame | None = None) -> pd.DataFrame:
    robinhood_lots = _campaign_lots_from_robinhood(open_lots)
    position_lots = _campaign_lots_from_positions(positions)
    symbols = sorted(set(robinhood_lots["symbol"]) | set(position_lots["symbol"]))
    rows: list[dict[str, object]] = []

    for symbol in symbols:
        robinhood_group = robinhood_lots[robinhood_lots["symbol"] == symbol]
        position_group = position_lots[position_lots["symbol"] == symbol]
        if not robinhood_group.empty:
            rows.append(_campaign_snapshot_row(symbol, robinhood_group, position_group, source="Robinhood"))
            continue

        rows.append(_campaign_snapshot_row(symbol, position_group, pd.DataFrame(), source="Positions"))

    return normalize_position_campaigns(pd.DataFrame(rows, columns=CAMPAIGN_VIEW_COLUMNS))


def _campaign_lots_from_robinhood(open_lots: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["symbol", "buy_date", "quantity", "buy_price", "planned_stop", "strategy", "risk_amount", "row_order"]
    if open_lots is None or open_lots.empty:
        return pd.DataFrame(columns=columns)

    source = open_lots.copy()
    for column in ["symbol", "buy_date", "quantity", "buy_price", "planned_stop", "strategy"]:
        if column not in source.columns:
            source[column] = None

    source["symbol"] = source["symbol"].fillna("").astype(str).str.upper().str.strip()
    source["quantity"] = pd.to_numeric(source["quantity"], errors="coerce")
    source["buy_price"] = pd.to_numeric(source["buy_price"], errors="coerce")
    source["planned_stop"] = pd.to_numeric(source["planned_stop"], errors="coerce")
    source["risk_amount"] = None
    source["row_order"] = range(len(source))
    source = source[(source["symbol"] != "") & (source["quantity"] > 0) & (source["buy_price"] > 0)]
    return source[columns].reset_index(drop=True)


def _campaign_lots_from_positions(positions: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["symbol", "buy_date", "quantity", "buy_price", "planned_stop", "strategy", "risk_amount", "row_order"]
    if positions is None or positions.empty:
        return pd.DataFrame(columns=columns)

    source = positions.copy()
    for column in ["symbol", "buy_date", "number_of_shares", "share_price", "stop_price", "strategy", "risk_amount"]:
        if column not in source.columns:
            source[column] = None

    lots = pd.DataFrame(
        {
            "symbol": source["symbol"].fillna("").astype(str).str.upper().str.strip(),
            "buy_date": source["buy_date"],
            "quantity": pd.to_numeric(source["number_of_shares"], errors="coerce"),
            "buy_price": pd.to_numeric(source["share_price"], errors="coerce"),
            "planned_stop": pd.to_numeric(source["stop_price"], errors="coerce"),
            "strategy": source["strategy"].fillna("").astype(str).str.strip(),
            "risk_amount": pd.to_numeric(source["risk_amount"], errors="coerce"),
            "row_order": range(len(source)),
        }
    )
    lots = lots[(lots["symbol"] != "") & (lots["quantity"] > 0) & (lots["buy_price"] > 0)]
    return lots[columns].reset_index(drop=True)


def _campaign_snapshot_row(
    symbol: str,
    lots: pd.DataFrame,
    fallback_lots: pd.DataFrame,
    *,
    source: str,
) -> dict[str, object]:
    current_shares = float(lots["quantity"].sum())
    avg_entry = _weighted_average(lots, "quantity", "buy_price")
    fallback_used = False

    campaign_stop = _newest_lot_value(lots, "planned_stop")
    if campaign_stop is None and not fallback_lots.empty:
        campaign_stop = _newest_lot_value(fallback_lots, "planned_stop")
        fallback_used = campaign_stop is not None

    strategy = _campaign_strategy_label(lots.get("strategy"))
    if not strategy and not fallback_lots.empty:
        strategy = _campaign_strategy_label(fallback_lots.get("strategy"))
        fallback_used = bool(strategy)

    planned_lot_risk = _planned_position_risk(lots) if source == "Positions" else _remaining_lot_risk(lots)
    if planned_lot_risk is None and source == "Robinhood":
        planned_lot_risk = _planned_position_risk(fallback_lots)
        fallback_used = planned_lot_risk is not None and source == "Robinhood"

    display_source = "Hybrid" if source == "Robinhood" and fallback_used else source
    risk_at_campaign_stop = _risk_at_campaign_stop(avg_entry, campaign_stop, current_shares)

    return {
        "symbol": symbol,
        "lots": int(len(lots)),
        "current_shares": _whole_number_if_possible(current_shares),
        "avg_entry": round(avg_entry, 2) if avg_entry is not None else None,
        "campaign_stop": campaign_stop,
        "sell_lot": max(1, int(current_shares // 3)) if current_shares > 0 else 0,
        "position_size": round(current_shares * avg_entry, 2) if avg_entry is not None else None,
        "risk_at_campaign_stop": risk_at_campaign_stop,
        "planned_lot_risk": planned_lot_risk,
        "strategy": strategy,
        "source": display_source,
    }


def _weighted_average(lots: pd.DataFrame, weight_column: str, value_column: str) -> float | None:
    weights = pd.to_numeric(lots[weight_column], errors="coerce")
    values = pd.to_numeric(lots[value_column], errors="coerce")
    valid = weights.notna() & values.notna() & (weights > 0)
    if not valid.any():
        return None
    total_weight = float(weights[valid].sum())
    if total_weight <= 0:
        return None
    return float((weights[valid] * values[valid]).sum()) / total_weight


def _newest_lot_value(lots: pd.DataFrame, column: str) -> float | None:
    if lots.empty or column not in lots.columns:
        return None
    ordered = lots.copy()
    ordered["buy_date_sort"] = pd.to_datetime(ordered["buy_date"], errors="coerce")
    ordered = ordered.sort_values(["buy_date_sort", "row_order"], ascending=[False, True], na_position="last")
    for _, lot in ordered.iterrows():
        value = _to_float(lot.get(column))
        if value is not None:
            return value
    return None


def _remaining_lot_risk(lots: pd.DataFrame) -> float | None:
    if lots.empty:
        return None
    required = ["quantity", "buy_price", "planned_stop"]
    if any(column not in lots.columns for column in required):
        return None
    values = lots[required].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        return None
    return round(float(((values["buy_price"] - values["planned_stop"]) * values["quantity"]).sum()), 2)


def _risk_at_campaign_stop(avg_entry: float | None, campaign_stop: float | None, current_shares: float | None) -> float | None:
    if avg_entry is None or campaign_stop is None or current_shares is None:
        return None
    return max(0.0, round((avg_entry - campaign_stop) * current_shares, 2))


def _planned_position_risk(lots: pd.DataFrame) -> float | None:
    if lots.empty or "risk_amount" not in lots.columns:
        return None
    risk = pd.to_numeric(lots["risk_amount"], errors="coerce")
    if risk.notna().any():
        return round(float(risk.fillna(0).sum()), 2)
    return None


def campaign_view_from_saved(campaigns: pd.DataFrame) -> pd.DataFrame:
    columns = CAMPAIGN_VIEW_COLUMNS
    if campaigns.empty:
        return pd.DataFrame(columns=columns)
    return normalize_position_campaigns(campaigns)[columns]


def recalculate_position_campaigns(campaigns: pd.DataFrame) -> pd.DataFrame:
    recalculated = normalize_position_campaigns(campaigns)
    if recalculated.empty:
        return recalculated

    for index, row in recalculated.iterrows():
        current_shares = _to_float(row.get("current_shares"))
        avg_entry = _to_float(row.get("avg_entry"))
        campaign_stop = _to_float(row.get("campaign_stop"))
        if current_shares is None or current_shares < 0:
            continue

        recalculated.loc[index, "sell_lot"] = max(1, int(current_shares // 3)) if current_shares > 0 else 0
        if avg_entry is not None:
            recalculated.loc[index, "position_size"] = round(current_shares * avg_entry, 2)
        recalculated.loc[index, "risk_at_campaign_stop"] = _risk_at_campaign_stop(
            avg_entry,
            campaign_stop,
            current_shares,
        )

    return normalize_position_campaigns(recalculated)


def apply_campaign_overrides(campaigns: pd.DataFrame, overrides: pd.DataFrame | None) -> pd.DataFrame:
    current = normalize_position_campaigns(campaigns)
    if current.empty or overrides is None or overrides.empty:
        return current

    normalized_overrides = normalize_campaign_overrides(overrides)
    overrides_by_symbol = {
        str(row["symbol"]): row
        for _, row in normalized_overrides.iterrows()
    }
    if not overrides_by_symbol:
        return current

    updated = current.copy()
    for index, row in updated.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        override = overrides_by_symbol.get(symbol)
        if override is None:
            continue
        if pd.notna(override.get("current_shares")):
            updated.loc[index, "current_shares"] = override.get("current_shares")
        if pd.notna(override.get("campaign_stop")):
            updated.loc[index, "campaign_stop"] = override.get("campaign_stop")
    return recalculate_position_campaigns(updated)


def campaign_trim_view(
    campaigns: pd.DataFrame,
    live_prices: dict[str, float] | None = None,
    campaign_trim_credits: pd.DataFrame | None = None,
    *,
    add_on_unrealized_profit_preserve_percent: float = 50.0,
) -> pd.DataFrame:
    frame = campaign_view_from_saved(campaigns)
    for column in CAMPAIGN_TRIM_COLUMNS:
        frame[column] = None

    prices = {
        str(symbol or "").upper().strip(): _to_float(price)
        for symbol, price in (live_prices or {}).items()
    }
    trim_credits = _campaign_trim_credit_by_symbol(campaign_trim_credits)
    for index, row in frame.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        live_price = prices.get(symbol)
        realized_trim_credit = trim_credits.get(symbol, 0.0)
        trim_count, free_roll = calculate_trim_to_free_roll(row, live_price, realized_trim_credit)
        add_on = calculate_profit_protected_add_on(
            row,
            live_price,
            realized_trim_credit,
            preserve_percent=add_on_unrealized_profit_preserve_percent,
        )
        frame.loc[index, "stop_itm"] = calculate_stop_itm(row, realized_trim_credit)
        frame.loc[index, "live_price"] = format_currency(live_price) or DISPLAY_PLACEHOLDER
        frame.loc[index, "trim_count"] = _format_display_count(trim_count)
        frame.loc[index, "free_roll"] = "Yes" if free_roll else "No"
        frame.loc[index, "max_add"] = _format_add_on_count(add_on)
        frame.loc[index, "add_risk"] = _format_add_on_currency(add_on, "add_risk")
        frame.loc[index, "profit_at_stop"] = _format_add_on_currency(add_on, "profit_at_stop")

    return frame[CAMPAIGN_TRIM_VIEW_COLUMNS]


def campaign_free_roll_summary(frame: pd.DataFrame) -> str:
    total = len(frame)
    if total == 0:
        return "Free Roll: 0 / 0 (0%)"

    if "free_roll" not in frame.columns:
        free_roll_count = 0
    else:
        free_roll_count = int((frame["free_roll"].fillna("").astype(str).str.strip() == "Yes").sum())
    percent = round((free_roll_count / total) * 100)
    return f"Free Roll: {free_roll_count} / {total} ({percent}%)"


def campaign_overrides_from_editor(base_campaigns: pd.DataFrame, edited_campaigns: pd.DataFrame) -> pd.DataFrame:
    if base_campaigns.empty or edited_campaigns.empty:
        return normalize_campaign_overrides(pd.DataFrame(columns=CAMPAIGN_OVERRIDE_COLUMNS))

    base = normalize_position_campaigns(base_campaigns)
    edited = normalize_campaign_overrides(edited_campaigns)
    base_shares = {
        str(row["symbol"]): _to_float(row["current_shares"])
        for _, row in base.iterrows()
    }
    base_stops = {
        str(row["symbol"]): _to_float(row["campaign_stop"])
        for _, row in base.iterrows()
    }

    rows: list[dict[str, object]] = []
    for _, row in edited.iterrows():
        symbol = str(row["symbol"])
        edited_shares = _to_float(row["current_shares"])
        original_shares = base_shares.get(symbol)
        edited_stop = _to_float(row.get("campaign_stop"))
        original_stop = base_stops.get(symbol)
        override: dict[str, object] = {"symbol": symbol}

        if edited_shares is not None and original_shares is not None and not math.isclose(
            edited_shares, original_shares, rel_tol=0, abs_tol=1e-9
        ):
            override["current_shares"] = edited_shares
        if edited_stop is not None and original_stop is not None and not math.isclose(
            edited_stop, original_stop, rel_tol=0, abs_tol=1e-9
        ):
            override["campaign_stop"] = edited_stop

        if len(override) > 1:
            rows.append(override)

    return normalize_campaign_overrides(pd.DataFrame(rows, columns=CAMPAIGN_OVERRIDE_COLUMNS))


def calculate_trim_to_free_roll(row: pd.Series, live_price: Any, realized_trim_credit: Any = 0) -> tuple[int | None, bool]:
    current_shares = _to_float(row.get("current_shares"))
    avg_entry = _to_float(row.get("avg_entry"))
    campaign_stop = _to_float(row.get("campaign_stop"))
    live_price = _to_float(live_price)
    realized_trim_credit = _to_float(realized_trim_credit) or 0.0

    if current_shares is None or avg_entry is None or campaign_stop is None or current_shares <= 0:
        return None, False

    loss_if_stopped = current_shares * (avg_entry - campaign_stop)
    effective_loss_if_stopped = max(0.0, loss_if_stopped - realized_trim_credit)
    if effective_loss_if_stopped <= 0:
        return 0, True

    if live_price is None or live_price <= campaign_stop:
        return None, False

    trim_count = math.ceil(effective_loss_if_stopped / (live_price - campaign_stop))
    trim_count = max(0, min(trim_count, math.ceil(current_shares)))
    return trim_count, False


def calculate_stop_itm(row: pd.Series, realized_trim_credit: Any = 0) -> float:
    current_shares = _to_float(row.get("current_shares"))
    avg_entry = _to_float(row.get("avg_entry"))
    campaign_stop = _to_float(row.get("campaign_stop"))
    realized_trim_credit = _to_float(realized_trim_credit) or 0.0

    if current_shares is None or avg_entry is None or campaign_stop is None or current_shares <= 0:
        return 0.0

    stop_pnl = ((campaign_stop - avg_entry) * current_shares) + realized_trim_credit
    return max(0.0, round(stop_pnl, 2))


def _campaign_trim_credit_by_symbol(campaign_trim_credits: pd.DataFrame | None) -> dict[str, float]:
    if campaign_trim_credits is None or campaign_trim_credits.empty:
        return {}

    credits = campaign_trim_credits.copy()
    if "symbol" not in credits.columns or "realized_trim_credit" not in credits.columns:
        return {}

    credits["symbol"] = credits["symbol"].fillna("").astype(str).str.upper().str.strip()
    credits["realized_trim_credit"] = pd.to_numeric(credits["realized_trim_credit"], errors="coerce")
    credits = credits[(credits["symbol"] != "") & credits["realized_trim_credit"].notna()]
    return {
        str(row["symbol"]): float(row["realized_trim_credit"])
        for _, row in credits.iterrows()
    }


def calculate_profit_protected_add_on(
    row: pd.Series,
    live_price: Any,
    realized_trim_credit: Any = 0,
    *,
    preserve_percent: float = 50.0,
) -> ProfitProtectedAddOn | None:
    current_shares = _to_float(row.get("current_shares"))
    avg_entry = _to_float(row.get("avg_entry"))
    campaign_stop = _to_float(row.get("campaign_stop"))
    live_price = _to_float(live_price)
    realized_trim_credit = _to_float(realized_trim_credit) or 0.0
    preserve_percent = _to_float(preserve_percent)

    if (
        current_shares is None
        or avg_entry is None
        or campaign_stop is None
        or current_shares <= 0
        or preserve_percent is None
        or preserve_percent < 0
        or preserve_percent > 100
    ):
        return None
    if live_price is None:
        return None
    if live_price <= avg_entry or live_price <= campaign_stop:
        return ProfitProtectedAddOn(False, 0, 0.0, None)

    unrealized_profit = max(0.0, (live_price - avg_entry) * current_shares)
    required_floor = realized_trim_credit + ((preserve_percent / 100) * unrealized_profit)
    current_floor = realized_trim_credit + ((campaign_stop - avg_entry) * current_shares)
    available_add_risk = max(0.0, current_floor - required_floor)
    risk_per_add_share = live_price - campaign_stop
    max_add_shares = math.floor(available_add_risk / risk_per_add_share)
    add_risk = max_add_shares * risk_per_add_share
    profit_at_stop = current_floor - add_risk
    return ProfitProtectedAddOn(
        applicable=True,
        max_add_shares=max_add_shares,
        add_risk=round(add_risk, 2),
        profit_at_stop=round(profit_at_stop, 2),
    )


def _format_add_on_count(add_on: ProfitProtectedAddOn | None) -> int | str:
    if add_on is None:
        return DISPLAY_PLACEHOLDER
    if not add_on.applicable:
        return "N/A"
    return add_on.max_add_shares


def _format_add_on_currency(add_on: ProfitProtectedAddOn | None, field: str) -> str:
    if add_on is None:
        return DISPLAY_PLACEHOLDER
    if not add_on.applicable or add_on.max_add_shares == 0:
        return "N/A"
    value = getattr(add_on, field)
    return format_currency(value) or DISPLAY_PLACEHOLDER


def _format_display_count(value: int | None) -> int | str:
    return DISPLAY_PLACEHOLDER if value is None else value


def _campaign_strategy_label(strategies: pd.Series | None) -> str:
    if strategies is None:
        return ""
    unique = [str(strategy or "").strip() for strategy in strategies if str(strategy or "").strip()]
    distinct = sorted(set(unique))
    if len(distinct) == 1:
        return distinct[0]
    if len(distinct) > 1:
        return "Mixed"
    return ""


def _whole_number_if_possible(value: float) -> int | float:
    return int(value) if float(value).is_integer() else round(value, 8)


def _new_position_id(
    row: pd.Series,
    used_ids: set[str],
    generated_counts: dict[str, int],
    *,
    generate: str,
) -> str:
    if generate == "uuid":
        while True:
            candidate = f"pos_{uuid.uuid4().hex[:12]}"
            if candidate not in used_ids:
                return candidate

    base = _deterministic_position_id(row)
    generated_counts[base] = generated_counts.get(base, 0) + 1
    suffix = generated_counts[base]
    candidate = base if suffix == 1 else f"{base}_{suffix}"
    while candidate in used_ids:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def _deterministic_position_id(row: pd.Series) -> str:
    values = [str(row.get(column) or "").strip() for column in INPUT_COLUMNS]
    digest = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:12]
    return f"pos_{digest}"


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
