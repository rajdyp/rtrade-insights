from __future__ import annotations

from datetime import date
from html import escape

import pandas as pd
import streamlit as st

from stock_calculator.calculations import (
    INPUT_COLUMNS,
    PUBLIC_OUTPUT_COLUMNS,
    append_committed_position,
    calculate_positions,
    committed_positions,
    delete_positions_by_index,
    draft_position,
    format_currency,
    format_percent,
)
from stock_calculator.config import load_config
from stock_calculator.robinhood import (
    STRATEGY_OPTIONS,
    UNCLASSIFIED_STRATEGY,
    calculate_strategy_attribution,
    calculate_strategy_metrics,
    calculate_total_realized_pnl,
    calculate_trade_metrics,
    derive_fifo_trades,
    display_strategy,
    display_trade_context_frame,
    normalize_strategy,
    parse_robinhood_csv,
)
from stock_calculator.risk import (
    MARKET_REGIME_OPTIONS,
    default_strategy,
    normalize_market_regime,
    strategy_mode_for_selection,
    suggested_risk_percent,
)
from stock_calculator.storage import (
    StorageError,
    append_robinhood_transactions,
    get_storage_backend,
    load_planned_stops,
    load_positions,
    load_robinhood_transactions,
    planned_stops_label,
    robinhood_transactions_label,
    save_planned_stops,
    save_positions,
    save_robinhood_transactions,
    storage_label,
    upsert_planned_stop,
)


st.set_page_config(page_title="Stock Calculator", layout="wide")

DELETE_COLUMN = "delete_selected"
POSITION_EDITOR_COLUMNS = [
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
EDITABLE_POSITION_COLUMNS = [*INPUT_COLUMNS, "strategy"]
POSITIONS_TABLE_MIN_VISIBLE_ROWS = 5
ROBINHOOD_TABLE_HEADER_HEIGHT = 36
ROBINHOOD_TABLE_ROW_HEIGHT = 35
ROBINHOOD_TABLE_BORDER_ALLOWANCE = 3
ROBINHOOD_TABLE_MAX_VISIBLE_ROWS = 10


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Barlow+Condensed:wght@400;500;600;700&display=swap');

        :root {
            --app-bg: var(--st-background-color);
            --app-surface: var(--st-secondary-background-color);
            --app-surface-alt: var(--st-gray-background-color);
            --app-border: var(--st-border-color);
            --app-border-mid: var(--st-widget-border-color, var(--st-border-color));
            --app-border-bright: color-mix(in srgb, var(--st-border-color), var(--st-text-color) 10%);
            --app-text: var(--st-text-color);
            --app-muted: var(--st-gray-text-color);
            --app-dim: color-mix(in srgb, var(--st-gray-text-color), transparent 40%);
            --app-accent: var(--st-primary-color);
            --app-accent-hover: color-mix(in srgb, var(--st-primary-color), var(--st-text-color) 15%);
            --app-accent-soft: color-mix(in srgb, var(--st-primary-color), transparent 88%);
            --app-glow: 0 0 10px color-mix(in srgb, var(--st-primary-color), transparent 62%);
            --app-grid: color-mix(in srgb, var(--st-border-color), transparent 50%);
            --app-success: var(--st-green-color);
            --app-success-soft: var(--st-green-background-color);
            --app-error: var(--st-red-color);
            --app-error-soft: var(--st-red-background-color);
            --app-mono: 'IBM Plex Mono', 'Cascadia Code', 'Fira Code', monospace;
            --app-sans: 'Barlow Condensed', system-ui, sans-serif;
        }

        /* ── Base ────────────────────────────────────────────── */
        .stApp {
            background-color: var(--app-bg) !important;
            background-image:
                linear-gradient(var(--app-grid) 1px, transparent 1px),
                linear-gradient(90deg, var(--app-grid) 1px, transparent 1px) !important;
            background-size: 40px 40px !important;
            font-family: var(--app-sans);
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .stMainBlockContainer {
            background-color: transparent !important;
            font-family: var(--app-sans);
        }

        header[data-testid="stHeader"] {
            background-color: var(--app-bg) !important;
            border-bottom: 1px solid var(--app-border-bright) !important;
        }

        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 2rem;
            max-width: 1580px;
        }

        h1, h2, h3 { color: var(--app-text); font-family: var(--app-sans) !important; }

        div[data-testid="stVerticalBlock"] > div:has(> .section-bar),
        div[data-testid="stVerticalBlock"] > div:has(> .entry-strip) {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0;
            box-shadow: none;
        }

        /* ── Header strip ────────────────────────────────────── */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            background: var(--app-surface);
            border: 1px solid var(--app-border-bright);
            border-top: 2px solid var(--app-accent);
            border-radius: 2px;
            padding: 0.5rem 0.8rem;
            margin-bottom: 1rem;
        }

        .app-title {
            font-family: var(--app-sans);
            font-size: 1.08rem;
            font-weight: 700;
            letter-spacing: 0.3em;
            color: var(--app-accent);
        }

        .app-title::after {
            content: '_';
            animation: cursor-blink 1.2s step-end infinite;
        }

        @keyframes cursor-blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }

        .app-subtitle {
            font-family: var(--app-mono);
            font-size: 0.74rem;
            color: var(--app-muted);
            margin-top: 0.15rem;
            letter-spacing: 0.04em;
        }

        .header-meta {
            display: flex;
            align-items: stretch;
            gap: 0;
            border: 1px solid var(--app-border-bright);
            border-radius: 2px;
            overflow: hidden;
        }

        .meta-pill {
            font-family: var(--app-mono);
            font-size: 0.76rem;
            color: var(--app-muted);
            white-space: nowrap;
            padding: 0.3rem 0.8rem;
            border-right: 1px solid var(--app-border-bright);
            letter-spacing: 0.03em;
        }

        .meta-pill:last-child { border-right: none; }

        .meta-pill strong {
            color: var(--app-accent);
            font-weight: 500;
            margin-right: 0.45em;
            letter-spacing: 0.08em;
        }

        /* ── Section dividers ────────────────────────────────── */
        .section-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.28rem 0;
            border-bottom: 1px solid var(--app-border-bright);
            margin-bottom: 0.65rem;
            position: relative;
        }

        .section-bar::after {
            content: '';
            position: absolute;
            bottom: -1px; left: 0;
            width: 36px; height: 2px;
            background: var(--app-accent);
        }

        .section-title {
            font-family: var(--app-sans);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--app-muted);
        }

        .section-title::before {
            content: '// ';
            color: var(--app-accent);
            opacity: 0.7;
            font-weight: 400;
        }

        .section-note {
            font-family: var(--app-mono);
            font-size: 0.69rem;
            color: var(--app-dim);
            letter-spacing: 0.02em;
        }

        /* ── Metric tiles ────────────────────────────────────── */
        div[data-testid="stMetric"] {
            background: var(--app-surface);
            border: 1px solid var(--app-border);
            border-radius: 2px;
            padding: 0.5rem 0.75rem 0.5rem 0.9rem;
            min-height: 64px;
            position: relative;
            overflow: hidden;
        }

        div[data-testid="stMetric"]::before {
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 3px; height: 100%;
            background: var(--app-accent);
            opacity: 0.5;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--app-muted) !important;
            font-family: var(--app-sans) !important;
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.18em !important;
            text-transform: uppercase !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--app-text) !important;
            font-family: var(--app-mono) !important;
            font-size: 1.28rem !important;
            font-weight: 400 !important;
            line-height: 1.2 !important;
            letter-spacing: -0.01em !important;
        }

        /* ── Inputs ──────────────────────────────────────────── */
        div[data-testid="stTextInput"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stDateInput"] label {
            font-family: var(--app-sans) !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.16em !important;
            text-transform: uppercase !important;
            color: var(--app-muted) !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stDateInput"] input {
            background-color: var(--app-surface) !important;
            border: 1px solid var(--app-border-mid) !important;
            border-radius: 2px !important;
            color: var(--app-text) !important;
            font-family: var(--app-mono) !important;
            font-size: 0.96rem !important;
        }

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus {
            border-color: var(--app-accent) !important;
            box-shadow: 0 0 0 1px color-mix(in srgb, var(--app-accent), transparent 65%), var(--app-glow) !important;
        }

        div[data-testid="stFileUploader"] section {
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 0 !important;
        }

        div[data-testid="stFileUploader"] section > div {
            padding: 0 !important;
        }

        /* ── Feedback bar ────────────────────────────────────── */
        .compact-feedback {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            border-radius: 2px;
            border: 1px solid var(--app-border);
            padding: 0.38rem 0.7rem;
            font-family: var(--app-mono);
            font-size: 0.81rem;
            margin: 0.18rem 0 0.45rem;
            letter-spacing: 0.02em;
        }

        .compact-feedback::before {
            content: '';
            display: inline-block;
            width: 7px; height: 7px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        .compact-feedback.ready {
            border-color: color-mix(in srgb, var(--app-success), transparent 55%);
            background: var(--app-success-soft);
            color: var(--app-success);
        }

        .compact-feedback.ready::before {
            background: var(--app-success);
            box-shadow: 0 0 5px var(--app-success), 0 0 10px color-mix(in srgb, var(--app-success), transparent 60%);
        }

        .compact-feedback.idle {
            background: var(--app-surface);
            color: var(--app-muted);
        }

        .compact-feedback.idle::before { background: var(--app-dim); }

        .compact-feedback.error {
            border-color: color-mix(in srgb, var(--app-error), transparent 55%);
            background: var(--app-error-soft);
            color: var(--app-error);
        }

        .compact-feedback.error::before {
            background: var(--app-error);
            box-shadow: 0 0 5px var(--app-error), 0 0 10px color-mix(in srgb, var(--app-error), transparent 60%);
        }

        /* ── Primary button ──────────────────────────────────── */
        .stButton > button {
            border-radius: 2px !important;
            background: var(--app-text) !important;
            border: none !important;
            color: var(--app-bg) !important;
            font-family: var(--app-sans) !important;
            font-weight: 700 !important;
            font-size: 0.82rem !important;
            letter-spacing: 0.2em !important;
            text-transform: uppercase !important;
            min-height: 2.2rem !important;
            transition: box-shadow 0.15s ease, background 0.15s ease !important;
        }

        .stButton > button:hover {
            background: color-mix(in srgb, var(--app-text), var(--app-accent) 22%) !important;
            color: var(--app-bg) !important;
            box-shadow: var(--app-glow) !important;
        }

        .stButton > button:disabled {
            background: var(--app-surface-alt) !important;
            color: var(--app-dim) !important;
            border: 1px solid var(--app-border) !important;
            box-shadow: none !important;
        }

        .stDownloadButton > button {
            border-radius: 2px !important;
            background: transparent !important;
            border: 1px solid var(--app-border-mid) !important;
            color: var(--app-muted) !important;
            font-family: var(--app-sans) !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            letter-spacing: 0.14em !important;
            text-transform: uppercase !important;
            min-height: 2.2rem !important;
            transition: all 0.15s ease !important;
        }

        .stDownloadButton > button:hover {
            border-color: var(--app-accent) !important;
            color: var(--app-accent) !important;
            background: var(--app-accent-soft) !important;
            box-shadow: var(--app-glow) !important;
        }

        /* ── Data editor / frame ─────────────────────────────── */
        div[data-testid="stDataFrame"],
        div[data-testid="stDataEditor"] {
            background: color-mix(in srgb, var(--app-bg), var(--app-border) 22%) !important;
            border: 1px solid var(--app-border-bright) !important;
            border-radius: 2px !important;
            overflow: hidden !important;
            font-size: 0.9rem !important;
        }

        div[data-testid="stDataFrame"] *,
        div[data-testid="stDataEditor"] * {
            font-size: 0.9rem !important;
        }

        div[data-testid="stDataFrame"] * {
            font-size: 0.84rem !important;
        }

        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"] {
            padding-left: 0.32rem !important;
            padding-right: 0.32rem !important;
        }

        div[data-testid="stDataFrameResizable"] {
            background: color-mix(in srgb, var(--app-bg), var(--app-border) 22%) !important;
        }

        /* ── Alerts ──────────────────────────────────────────── */
        div[data-testid="stAlert"] {
            border-radius: 2px !important;
            font-family: var(--app-mono) !important;
            font-size: 0.84rem !important;
        }

        /* ── Expander ────────────────────────────────────────── */
        div[data-testid="stExpander"] details {
            border: 1px solid var(--app-border-bright) !important;
            border-radius: 2px !important;
            background: var(--app-surface) !important;
        }

        div[data-testid="stExpander"] summary {
            font-family: var(--app-sans) !important;
            font-size: 0.77rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.18em !important;
            text-transform: uppercase !important;
            color: var(--app-muted) !important;
            padding: 0.5rem 0.75rem !important;
        }

        /* ── Scrollbars ──────────────────────────────────────── */
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: var(--app-bg); }
        ::-webkit-scrollbar-thumb { background: var(--app-border-bright); border-radius: 2px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--app-muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def positions_column_config() -> dict:
    return {
        "symbol": st.column_config.TextColumn("Symbol", width=72),
        "buy_date": st.column_config.DateColumn("Buy Date", format="MM/DD/YYYY", width=96),
        "share_price": st.column_config.NumberColumn("Share Price", format="$%.2f", width=94),
        "stop_price": st.column_config.NumberColumn("Stop Price", format="$%.2f", width=88),
        "atr": st.column_config.NumberColumn("ATR %", format="%.2f", width=58),
        "risk_in_atr": st.column_config.NumberColumn("Risk (ATR)", format="%.2f", width=82),
        "strategy": st.column_config.SelectboxColumn(
            "Strategy",
            options=[*STRATEGY_OPTIONS, UNCLASSIFIED_STRATEGY],
            width=82,
        ),
        "stop_loss_percent": st.column_config.NumberColumn("Stop Loss", format="%.2f%%", width=78),
        "number_of_shares": st.column_config.NumberColumn("Shares", width=68),
        "sell_lot": st.column_config.NumberColumn("Sell Lot", width=68),
        "hold_count": st.column_config.NumberColumn("Hold Days", width=74),
        "position_size": st.column_config.NumberColumn("Position Size", format="$%.2f", width=104),
        "risk_percent": st.column_config.NumberColumn("Risk %", format="%.2f%%", width=72),
        "risk_amount": st.column_config.NumberColumn("Total Risk", format="$%.2f", width=96),
        "portfolio_amount": st.column_config.NumberColumn("Portfolio", format="$%.2f", width=104),
        DELETE_COLUMN: st.column_config.CheckboxColumn("x", width=38, default=False),
    }


def closed_trades_column_config() -> dict:
    return {
        "symbol": st.column_config.TextColumn("Symbol", width=68),
        "buy_date": st.column_config.DateColumn("Buy Date", format="MM/DD/YYYY", width=94),
        "sell_date": st.column_config.DateColumn("Sell Date", format="MM/DD/YYYY", width=94),
        "quantity": st.column_config.NumberColumn("Quantity", width=58),
        "planned_stop": st.column_config.NumberColumn("Stop Price", format="$%.2f", width=82),
        "strategy": st.column_config.TextColumn("Strategy", width=72),
        "atr": st.column_config.TextColumn("ATR", width=52),
        "market_regime": st.column_config.TextColumn("Regime", width=102),
        "buy_price": st.column_config.NumberColumn("Buy Price", format="$%.2f", width=78),
        "buy_amount": st.column_config.NumberColumn("Buy Amount", format="$%.2f", width=92),
        "sell_price": st.column_config.NumberColumn("Sell Price", format="$%.2f", width=78),
        "sell_amount": st.column_config.NumberColumn("Sell Amount", format="$%.2f", width=92),
        "realized_pnl": st.column_config.NumberColumn("Realized P/L", format="$%.2f", width=92),
        "realized_pnl_percent": st.column_config.NumberColumn("P/L %", format="%.2f%%", width=62),
        "hold_days": st.column_config.NumberColumn("Hold Days", width=46),
    }


def open_lots_column_config() -> dict:
    return {
        "symbol": st.column_config.TextColumn("Symbol", width=68),
        "buy_date": st.column_config.DateColumn("Buy Date", format="MM/DD/YYYY", width=94),
        "quantity": st.column_config.NumberColumn("Quantity", width=58),
        "planned_stop": st.column_config.NumberColumn("Stop Price", format="$%.2f", width=82),
        "strategy": st.column_config.TextColumn("Strategy", width=72),
        "atr": st.column_config.TextColumn("ATR", width=52),
        "market_regime": st.column_config.TextColumn("Regime", width=102),
        "buy_price": st.column_config.NumberColumn("Buy Price", format="$%.2f", width=78),
        "cost_basis": st.column_config.NumberColumn("Cost Basis", format="$%.2f", width=92),
        "hold_days": st.column_config.NumberColumn("Hold Days", width=46),
    }


def editor_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "buy_date" in frame.columns:
        frame["buy_date"] = pd.to_datetime(frame["buy_date"], errors="coerce").dt.date
    frame[DELETE_COLUMN] = False
    return frame


def positions_editor_frame(positions: pd.DataFrame, planned_stops: pd.DataFrame) -> pd.DataFrame:
    frame = positions.copy()
    frame["strategy"] = [planned_strategy(row, planned_stops) for _, row in frame.iterrows()]
    return editor_frame(frame[POSITION_EDITOR_COLUMNS])


def planned_strategy(row: pd.Series, planned_stops: pd.DataFrame) -> str:
    symbol = str(row.get("symbol") or "").upper().strip()
    buy_date = str(row.get("buy_date") or "").strip()
    quantity = pd.to_numeric(row.get("number_of_shares"), errors="coerce")
    if not symbol or not buy_date or pd.isna(quantity):
        return UNCLASSIFIED_STRATEGY

    normalized_stops = planned_stops.copy()
    if normalized_stops.empty:
        return UNCLASSIFIED_STRATEGY

    matches = normalized_stops[
        (normalized_stops["symbol"].fillna("").astype(str).str.upper().str.strip() == symbol)
        & (normalized_stops["buy_date"].fillna("").astype(str).str.strip() == buy_date)
        & (pd.to_numeric(normalized_stops["quantity"], errors="coerce") == quantity)
    ]
    strategies = {normalize_strategy(strategy) for strategy in matches.get("strategy", [])}
    strategies.discard("")
    return next(iter(strategies)) if len(strategies) == 1 else UNCLASSIFIED_STRATEGY


def edited_position_strategies(edited: pd.DataFrame) -> list[str]:
    edited_positions = edited[edited["symbol"].fillna("").astype(str).str.strip() != ""]
    return [normalize_strategy(strategy) for strategy in edited_positions.get("strategy", [])]


def upsert_position_strategies(
    planned_stops: pd.DataFrame,
    calculated_positions: pd.DataFrame,
    strategies: list[str],
) -> pd.DataFrame:
    updated = planned_stops
    for index, (_, row) in enumerate(calculated_positions.reset_index(drop=True).iterrows()):
        row_with_strategy = row.copy()
        row_with_strategy["strategy"] = strategies[index] if index < len(strategies) else ""
        row_with_strategy["market_regime"] = existing_planned_market_regime(row_with_strategy, updated)
        updated = upsert_planned_stop(updated, row_with_strategy)
    return updated


def existing_planned_market_regime(row: pd.Series, planned_stops: pd.DataFrame) -> str:
    symbol = str(row.get("symbol") or "").upper().strip()
    buy_date = str(row.get("buy_date") or "").strip()
    quantity = pd.to_numeric(row.get("number_of_shares"), errors="coerce")
    if not symbol or not buy_date or pd.isna(quantity) or planned_stops.empty:
        return ""

    matches = planned_stops[
        (planned_stops["symbol"].fillna("").astype(str).str.upper().str.strip() == symbol)
        & (planned_stops["buy_date"].fillna("").astype(str).str.strip() == buy_date)
        & (pd.to_numeric(planned_stops["quantity"], errors="coerce") == quantity)
    ]
    regimes = {_normalize_market_regime_or_blank(value) for value in matches.get("market_regime", [])}
    regimes.discard("")
    return next(iter(regimes)) if len(regimes) == 1 else ""


def _normalize_market_regime_or_blank(value: object) -> str:
    regime = str(value or "").strip()
    if not regime:
        return ""
    normalized = normalize_market_regime(regime)
    return normalized if normalized == regime.upper() else ""


def display_date_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for column in ["buy_date", "sell_date", "activity_date", "process_date", "settle_date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    if "strategy" in frame.columns:
        frame["strategy"] = frame["strategy"].apply(display_strategy)
    return display_trade_context_frame(frame)


def filtered_output(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["symbol"].astype(str).str.strip() != ""].reset_index(drop=True)


def positions_editor_height(row_count: int) -> int:
    visible_rows = max(int(row_count), POSITIONS_TABLE_MIN_VISIBLE_ROWS)
    return (
        ROBINHOOD_TABLE_HEADER_HEIGHT
        + (visible_rows * ROBINHOOD_TABLE_ROW_HEIGHT)
        + ROBINHOOD_TABLE_BORDER_ALLOWANCE
    )


def robinhood_dataframe_height(row_count: int) -> int:
    visible_rows = max(1, min(int(row_count), ROBINHOOD_TABLE_MAX_VISIBLE_ROWS))
    return (
        ROBINHOOD_TABLE_HEADER_HEIGHT
        + (visible_rows * ROBINHOOD_TABLE_ROW_HEIGHT)
        + ROBINHOOD_TABLE_BORDER_ALLOWANCE
    )


def first_value(df: pd.DataFrame, column: str):
    return df.iloc[0][column]


def marked_delete_rows(edited: pd.DataFrame) -> list[int]:
    if DELETE_COLUMN not in edited.columns:
        return []
    return [int(index) for index, selected in edited[DELETE_COLUMN].items() if not pd.isna(selected) and bool(selected)]


def render_header(total_pnl: float, metrics: dict | None, target=st) -> None:
    total_pnl_percent = (total_pnl / config.portfolio_amount) * 100 if config.portfolio_amount else 0.0
    total_pnl_display = f"{format_currency(total_pnl)} ({format_percent(total_pnl_percent)})"
    metrics = metrics or {}
    outcome_counts_display = (
        f"WINS {int(metrics.get('win_count') or 0)}"
        f"&nbsp;&nbsp;LOSSES {int(metrics.get('loss_count') or 0)}"
        f"&nbsp;&nbsp;BREAKEVENS {int(metrics.get('breakeven_count') or 0)}"
    )
    target.markdown(
        f"""
        <div class="app-header">
          <div>
            <div class="app-title">rTRADE INSIGHTS</div>
            <div class="app-subtitle">Position sizing &amp; portfolio performance tracking</div>
          </div>
          <div class="header-meta">
            <span class="meta-pill"><strong>PORTFOLIO</strong>&nbsp;&nbsp;{escape(format_currency(config.portfolio_amount))}</span>
            <span class="meta-pill"><strong>TOTAL P&amp;L</strong>&nbsp;&nbsp;{escape(total_pnl_display)}</span>
            <span class="meta-pill">{outcome_counts_display}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-bar">
          <div class="section-title">{escape(title)}</div>
          <div class="section-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feedback(message: str, status: str) -> None:
    st.markdown(
        f'<div class="compact-feedback {escape(status)}">{escape(message)}</div>',
        unsafe_allow_html=True,
    )


def render_mode_legend() -> None:
    render_feedback(
        "Mode uses latest 15 valid R trades within the latest 20 closed trades: "
        "> +0.30R Working | 0 to +0.30R Caution | "
        "-0.10R to 0 Weak | < -0.10R Failing. Adj Score is a variance-adjusted reference.",
        "idle",
    )


def format_optional_currency(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return format_currency(value)


def format_optional_percent(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return format_percent(value)


def format_optional_number(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}"


def format_profit_factor(metrics: dict | None) -> str:
    if not metrics:
        return "N/A"
    value = metrics.get("profit_factor")
    if value is not None and not pd.isna(value):
        return f"{float(value):.2f}"
    if metrics.get("win_count", 0) > 0 and metrics.get("loss_count", 0) == 0:
        return "∞"
    return "N/A"


def format_blank_optional_number(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def strategy_metrics_display_frame(strategy_metrics: pd.DataFrame) -> pd.DataFrame:
    display = strategy_metrics.copy()
    if "profit_factor" not in display:
        return display

    def format_row(row: pd.Series) -> str:
        value = row.get("profit_factor")
        if value is not None and not pd.isna(value):
            return f"{float(value):.2f}"
        if row.get("average_win") is not None and not pd.isna(row.get("average_win")):
            return "∞"
        return "N/A"

    display["profit_factor"] = display.apply(format_row, axis=1)
    return display


def format_optional_days(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    days = round(float(value))
    return f"{days} day" if days == 1 else f"{days} days"


def format_currency_percent_pair(currency_value, percent_value) -> str:
    if currency_value is None or pd.isna(currency_value):
        return "N/A"
    if percent_value is None or pd.isna(percent_value):
        return format_currency(currency_value)
    return f"{format_currency(currency_value)} ({format_percent(percent_value)})"


def strategy_metrics_column_config() -> dict:
    return {
        "strategy": st.column_config.TextColumn("Strategy", width=74),
        "trade_count": st.column_config.NumberColumn("Trades", width=58),
        "total_realized_pnl": st.column_config.NumberColumn("P/L", format="$%.2f", width=88),
        "win_rate": st.column_config.NumberColumn("Win %", format="%.1f%%", width=72),
        "expectancy": st.column_config.NumberColumn("Exp $", format="$%.2f", width=84),
        "profit_factor": st.column_config.TextColumn("PF", width=62),
        "average_win": st.column_config.NumberColumn("Avg Win", format="$%.2f", width=84),
        "average_loss": st.column_config.NumberColumn("Avg Loss", format="$%.2f", width=84),
        "average_win_r": st.column_config.NumberColumn("Avg R W", format="%.2f", width=76),
        "average_loss_r": st.column_config.NumberColumn("Avg R L", format="%.2f", width=76),
        "r_ratio": st.column_config.NumberColumn("R Ratio", format="%.2f", width=72),
        "average_win_hold": st.column_config.NumberColumn("Win Hold", format="%.1f", width=78),
        "average_loss_hold": st.column_config.NumberColumn("Loss Hold", format="%.1f", width=82),
        "rolling_mode_exp": st.column_config.TextColumn("15R Exp", width=82),
        "mode_adjusted_score": st.column_config.TextColumn("Adj Score", width=86),
        "mode": st.column_config.TextColumn("Mode", width=82),
        "action": st.column_config.TextColumn("Action", width=122),
    }


STRATEGY_ATTRIBUTION_DISPLAY_ROWS = [
    ("Mode", "mode"),
    ("Mode Basis", "mode_basis"),
    ("Trend", "trend"),
    ("Trend Driver", "trend_driver"),
    ("Evidence", "evidence"),
    ("Playbook", "playbook"),
]


def strategy_attribution_display_strategies(strategy_attribution: pd.DataFrame) -> list[str]:
    strategies = []
    existing = [str(strategy) for strategy in strategy_attribution.get("strategy", [])]
    for strategy in STRATEGY_OPTIONS:
        strategies.append(strategy)
    for strategy in existing:
        if strategy and strategy not in strategies:
            strategies.append(strategy)
    return strategies


def strategy_attribution_display_value(value: object, *, key: str) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if key == "trend_driver":
        return text.replace("Profit factor", "PF").replace("profit factor", "PF")
    return text


def strategy_attribution_display_frame(strategy_attribution: pd.DataFrame) -> pd.DataFrame:
    strategies = strategy_attribution_display_strategies(strategy_attribution)
    rows_by_strategy = {
        str(row.get("strategy") or ""): row
        for _, row in strategy_attribution.iterrows()
    }
    display_rows = []
    for label, key in STRATEGY_ATTRIBUTION_DISPLAY_ROWS:
        display_row = {"Metric": label}
        for strategy in strategies:
            source_row = rows_by_strategy.get(strategy, {})
            display_row[strategy] = strategy_attribution_display_value(source_row.get(key), key=key)
        display_rows.append(display_row)
    return pd.DataFrame(display_rows, columns=["Metric", *strategies])


def strategy_attribution_display_column_config(columns: list[str]) -> dict:
    column_config = {"Metric": st.column_config.TextColumn("Metric", width=40)}
    for column in columns:
        if column == "Metric":
            continue
        column_config[column] = st.column_config.TextColumn(column, width=360)
    return column_config


def render_strategy_attribution(strategy_attribution: pd.DataFrame | None) -> None:
    with st.expander("Strategy Attribution", expanded=False):
        render_feedback(
            "This panel answers: What is actually making or costing money? "
            "Position sizing is driven by Market Regime and Strategy Mode.",
            "idle",
        )
        if strategy_attribution is None or strategy_attribution.empty:
            render_feedback("No strategy attribution is available yet.", "idle")
            return

        display_frame = strategy_attribution_display_frame(strategy_attribution)
        st.dataframe(
            display_frame,
            column_config=strategy_attribution_display_column_config(list(display_frame.columns)),
            hide_index=True,
            width="stretch",
        )


def render_trade_metrics(
    metrics: dict | None,
    strategy_metrics: pd.DataFrame | None,
    strategy_attribution: pd.DataFrame | None,
) -> None:
    render_section("Trade Metrics", "Derived from stored Robinhood transactions.")
    if metrics is None:
        render_feedback("Upload a Robinhood CSV to populate trade metrics.", "idle")
        return

    if metrics.get("trade_count", 0) == 0:
        render_feedback("No closed trades are available for metrics yet.", "idle")

    metric_rows = [
        [
            ("Win Rate", format_optional_percent(metrics.get("win_rate"))),
            ("Expectancy", format_optional_currency(metrics.get("expectancy"))),
            ("Profit Factor", format_profit_factor(metrics)),
            ("Win/Loss Ratio", format_optional_number(metrics.get("win_loss_ratio"))),
        ],
        [
            ("Avg R (Wins)", format_optional_number(metrics.get("average_win_r"))),
            ("Avg R (Losses)", format_optional_number(metrics.get("average_loss_r"))),
            ("R Ratio (Win/Loss)", format_optional_number(metrics.get("r_ratio"))),
            ("Expectancy in R", format_optional_number(metrics.get("expectancy_r"))),
        ],
        [
            ("Average Win", format_currency_percent_pair(metrics.get("average_win"), metrics.get("average_win_percent"))),
            ("Average Win Hold", format_optional_days(metrics.get("average_win_hold"))),
            ("Win Streak", str(metrics.get("win_streak", 0))),
            ("Top Win", format_optional_currency(metrics.get("top_win"))),
        ],
        [
            ("Average Loss", format_currency_percent_pair(metrics.get("average_loss"), metrics.get("average_loss_percent"))),
            ("Average Loss Hold", format_optional_days(metrics.get("average_loss_hold"))),
            ("Loss Streak", str(metrics.get("loss_streak", 0))),
            ("Top Loss", format_optional_currency(metrics.get("top_loss"))),
        ],
    ]

    for row_values in metric_rows:
        columns = st.columns(4)
        for column, metric_value in zip(columns, row_values, strict=True):
            if metric_value is None:
                continue
            label, value = metric_value
            column.metric(label, value)

    if strategy_metrics is None or strategy_metrics.empty:
        render_feedback("No strategy breakdown is available yet.", "idle")
        return

    st.dataframe(
        strategy_metrics_display_frame(strategy_metrics),
        column_config=strategy_metrics_column_config(),
        hide_index=True,
        width="stretch",
    )
    render_mode_legend()
    render_strategy_attribution(strategy_attribution)


config = load_config()

try:
    get_storage_backend()
except StorageError as exc:
    st.error(f"Storage is not configured correctly: {exc}")
    st.stop()


def add_current_draft() -> None:
    draft_from_state = draft_position(
        symbol=str(st.session_state.get("draft_symbol", "")).upper().strip(),
        buy_date=st.session_state.get("draft_buy_date"),
        share_price=st.session_state.get("draft_share_price"),
        stop_price=st.session_state.get("draft_stop_price"),
        atr=st.session_state.get("draft_atr"),
        portfolio_amount=st.session_state.get("draft_portfolio_amount"),
        risk_percent=st.session_state.get("draft_risk_percent"),
    )
    calculated_draft = calculate_positions(draft_from_state)
    validation_error = str(calculated_draft.iloc[0]["validation_error"] or "")
    if validation_error:
        return

    st.session_state.positions = append_committed_position(st.session_state.positions, draft_from_state)
    calculated_draft.loc[calculated_draft.index[0], "strategy"] = st.session_state.get("draft_strategy", STRATEGY_OPTIONS[0])
    calculated_draft.loc[calculated_draft.index[0], "market_regime"] = normalize_market_regime(
        st.session_state.get("draft_market_regime")
    )
    st.session_state.planned_stops = upsert_planned_stop(st.session_state.planned_stops, calculated_draft.iloc[0])
    save_positions(st.session_state.positions)
    save_planned_stops(st.session_state.planned_stops)
    st.session_state.draft_symbol = ""
    st.session_state.draft_buy_date = date.today()
    st.session_state.draft_share_price = 0.0
    st.session_state.draft_stop_price = 0.0
    st.session_state.draft_atr = 0.0
    st.session_state.draft_portfolio_amount = config.sizing_portfolio_amount
    st.session_state.draft_strategy = default_strategy()
    st.session_state.draft_risk_context = None


def delete_selected_positions(selected_rows: list[int]) -> None:
    if not selected_rows:
        return

    st.session_state.positions = delete_positions_by_index(st.session_state.positions, selected_rows)
    save_positions(st.session_state.positions)
    st.session_state.position_editor_revision += 1
    st.rerun()


if "positions" not in st.session_state:
    st.session_state.positions = load_positions()
if "planned_stops" not in st.session_state:
    st.session_state.planned_stops = load_planned_stops()
if "robinhood_transactions" not in st.session_state:
    st.session_state.robinhood_transactions = load_robinhood_transactions()
if "draft_symbol" not in st.session_state:
    st.session_state.draft_symbol = ""
if "draft_buy_date" not in st.session_state:
    st.session_state.draft_buy_date = date.today()
if "draft_share_price" not in st.session_state:
    st.session_state.draft_share_price = 0.0
if "draft_stop_price" not in st.session_state:
    st.session_state.draft_stop_price = 0.0
if "draft_atr" not in st.session_state:
    st.session_state.draft_atr = 0.0
if "draft_portfolio_amount" not in st.session_state:
    st.session_state.draft_portfolio_amount = config.sizing_portfolio_amount
if "draft_risk_percent" not in st.session_state:
    st.session_state.draft_risk_percent = config.risk_percent
if "draft_strategy" not in st.session_state:
    st.session_state.draft_strategy = default_strategy()
if "draft_market_regime" not in st.session_state:
    st.session_state.draft_market_regime = normalize_market_regime(config.market_regime)
if "draft_risk_context" not in st.session_state:
    st.session_state.draft_risk_context = None
if "position_editor_revision" not in st.session_state:
    st.session_state.position_editor_revision = 0

robinhood_derivation = derive_fifo_trades(st.session_state.robinhood_transactions, st.session_state.planned_stops)
trade_metrics = calculate_trade_metrics(robinhood_derivation.closed_trades)
strategy_metrics = calculate_strategy_metrics(robinhood_derivation.closed_trades)
strategy_attribution = calculate_strategy_attribution(robinhood_derivation.closed_trades)
total_pnl = calculate_total_realized_pnl(robinhood_derivation.closed_trades)

apply_styles()
header_container = st.empty()
render_header(total_pnl, trade_metrics, header_container)

render_section("New Position", "Enter a candidate position and review the live sizing before adding it.")

selected_strategy_mode = strategy_mode_for_selection(strategy_metrics, st.session_state.draft_strategy)
risk_context = (
    normalize_market_regime(st.session_state.draft_market_regime),
    st.session_state.draft_strategy,
    selected_strategy_mode,
)
if st.session_state.draft_risk_context != risk_context:
    st.session_state.draft_risk_percent = suggested_risk_percent(
        risk_context[0],
        risk_context[2],
        config.risk_percent,
    )
    st.session_state.draft_risk_context = risk_context

top_cols = st.columns([1.2, 0.95, 1.15, 0.85, 0.7, 1.35, 1.0, 1.0, 0.75])
symbol = top_cols[0].text_input("Symbol", key="draft_symbol").upper().strip()
buy_date = top_cols[1].date_input("Buy Date", key="draft_buy_date")
top_cols[2].selectbox("Market Regime", MARKET_REGIME_OPTIONS, key="draft_market_regime")
top_cols[3].selectbox("Strategy", STRATEGY_OPTIONS, key="draft_strategy")
risk_percent = top_cols[4].number_input(
    "Risk %",
    min_value=0.0,
    step=0.01,
    format="%.2f",
    key="draft_risk_percent",
)
portfolio_amount = top_cols[5].number_input(
    "Portfolio",
    min_value=0.0,
    step=100.0,
    format="%.2f",
    key="draft_portfolio_amount",
)
share_price = top_cols[6].number_input("Share Price", min_value=0.0, step=0.01, format="%.2f", key="draft_share_price")
stop_price = top_cols[7].number_input("Stop Price", min_value=0.0, step=0.01, format="%.2f", key="draft_stop_price")
atr = top_cols[8].number_input("ATR %", min_value=0.0, step=0.01, format="%.2f", key="draft_atr")

draft = draft_position(
    symbol=symbol,
    buy_date=buy_date,
    share_price=share_price,
    stop_price=stop_price,
    atr=atr,
    portfolio_amount=portfolio_amount,
    risk_percent=risk_percent,
)
draft_result = calculate_positions(draft)
draft_row = draft_result.iloc[0]
draft_error = str(draft_row["validation_error"] or "")
draft_is_valid = bool(symbol) and not draft_error

preview_cols = st.columns(5)
preview_cols[0].metric("Stop Loss", format_percent(first_value(draft_result, "stop_loss_percent")))
preview_cols[1].metric("Risk in ATR", format_blank_optional_number(first_value(draft_result, "risk_in_atr")))
preview_cols[2].metric("Shares", "" if pd.isna(draft_row["number_of_shares"]) else int(draft_row["number_of_shares"]))
preview_cols[3].metric("Position Size", format_currency(first_value(draft_result, "position_size")))
preview_cols[4].metric("Total Risk", format_currency(first_value(draft_result, "risk_amount")))

if draft_error:
    feedback_message = draft_error
    feedback_status = "error"
elif symbol:
    feedback_message = "Position is ready to add."
    feedback_status = "ready"
else:
    feedback_message = "Enter a symbol and prices to preview the position."
    feedback_status = "idle"

action_cols = st.columns([1.5, 1, 1.5])
with action_cols[0]:
    render_feedback(feedback_message, feedback_status)
with action_cols[1]:
    st.button("Add Position", disabled=not draft_is_valid, on_click=add_current_draft, width="stretch")

calculated = calculate_positions(st.session_state.positions)
visible_calculated = filtered_output(calculated)

total_risk = visible_calculated["risk_amount"].fillna(0).sum()
total_position_size = visible_calculated["position_size"].fillna(0).sum()
active_positions = len(visible_calculated)
invalid_positions = int((visible_calculated["validation_error"].fillna("") != "").sum())
position_editor_key = f"position_editor_{st.session_state.position_editor_revision}"

render_section("Positions", "Saved rows remain editable; calculated columns are read-only.")

summary_cols = st.columns([1, 1.15, 1, 1])
summary_cols[0].metric("Active Positions", active_positions)
summary_cols[1].metric("Total Position Size", format_currency(total_position_size))
summary_cols[2].metric("Total Risk", format_currency(total_risk))

csv_bytes = visible_calculated[PUBLIC_OUTPUT_COLUMNS].to_csv(index=False).encode("utf-8")
summary_cols[3].download_button(
    "Export CSV",
    csv_bytes,
    file_name="stock_calculator_export.csv",
    mime="text/csv",
    width="stretch",
    disabled=visible_calculated.empty,
)

if visible_calculated.empty:
    render_feedback("No positions yet. Add a valid position from the calculator above to start the list.", "idle")
else:
    edited = st.data_editor(
        positions_editor_frame(visible_calculated, st.session_state.planned_stops),
        column_config=positions_column_config(),
        disabled=[column for column in POSITION_EDITOR_COLUMNS if column not in EDITABLE_POSITION_COLUMNS],
        height=positions_editor_height(len(visible_calculated)),
        num_rows="fixed",
        width="stretch",
        hide_index=True,
        key=position_editor_key,
    )

    strategy_values = edited_position_strategies(edited)
    st.session_state.positions = committed_positions(edited[INPUT_COLUMNS])
    delete_rows = marked_delete_rows(edited)
    if delete_rows:
        delete_selected_positions(delete_rows)

    save_positions(st.session_state.positions)
    visible_calculated = filtered_output(calculate_positions(st.session_state.positions))
    st.session_state.planned_stops = upsert_position_strategies(
        st.session_state.planned_stops,
        visible_calculated,
        strategy_values,
    )
    save_planned_stops(st.session_state.planned_stops)
    invalid_positions = int((visible_calculated["validation_error"].fillna("") != "").sum())

if invalid_positions:
    invalid_rows = visible_calculated[visible_calculated["validation_error"].fillna("") != ""]
    messages = [
        f"{row['symbol'] or f'Row {index + 1}'}: {row['validation_error']}"
        for index, row in invalid_rows.iterrows()
    ]
    render_feedback("Some rows need fixes before their calculated values can be used: " + "; ".join(messages), "error")

trade_metrics_container = st.container()

with st.expander("Robinhood Import", expanded=False):
    robinhood_file = st.file_uploader(
        "Robinhood CSV report",
        type=["csv"],
        accept_multiple_files=False,
        key="robinhood_csv_upload",
    )

    if robinhood_file is None:
        st.session_state.pop("robinhood_import_result", None)
    else:
        import_result = parse_robinhood_csv(robinhood_file)
        persisted_transactions, added_count = append_robinhood_transactions(
            st.session_state.robinhood_transactions,
            import_result.transactions,
        )
        skipped_count = len(import_result.transactions) - added_count
        if added_count:
            save_robinhood_transactions(persisted_transactions)
        st.session_state.robinhood_transactions = persisted_transactions
        st.session_state.robinhood_last_added_count = added_count
        st.session_state.robinhood_last_skipped_count = skipped_count
        robinhood_derivation = derive_fifo_trades(st.session_state.robinhood_transactions, st.session_state.planned_stops)
        trade_metrics = calculate_trade_metrics(robinhood_derivation.closed_trades)
        strategy_metrics = calculate_strategy_metrics(robinhood_derivation.closed_trades)
        strategy_attribution = calculate_strategy_attribution(robinhood_derivation.closed_trades)
        total_pnl = calculate_total_realized_pnl(robinhood_derivation.closed_trades)
        render_header(total_pnl, trade_metrics, header_container)
        st.session_state.robinhood_import_result = import_result

        if import_result.transactions.empty:
            render_feedback("No buy or sell rows were found in the uploaded CSV.", "error")
        elif added_count:
            render_feedback(f"Added {added_count} new Robinhood transactions to {storage_label()}.", "ready")
        else:
            render_feedback("No new Robinhood transactions were added; all clean rows were already imported.", "idle")

    imported_result = st.session_state.get("robinhood_import_result")
    current_trade_rows = len(imported_result.transactions) if imported_result is not None else 0
    current_ignored_rows = len(imported_result.ignored_rows) if imported_result is not None else 0
    current_malformed_rows = len(imported_result.malformed_rows) if imported_result is not None else 0
    latest_added_rows = st.session_state.get("robinhood_last_added_count", 0) if imported_result is not None else 0
    latest_skipped_rows = st.session_state.get("robinhood_last_skipped_count", 0) if imported_result is not None else 0

    import_summary_cols = st.columns(10)
    import_summary_cols[0].metric("Upload Rows", current_trade_rows)
    import_summary_cols[1].metric("New Rows", latest_added_rows)
    import_summary_cols[2].metric("Skipped Rows", latest_skipped_rows)
    import_summary_cols[3].metric("Ignored Rows", current_ignored_rows)
    import_summary_cols[4].metric("Malformed Rows", current_malformed_rows)
    import_summary_cols[5].metric("Stored Rows", len(st.session_state.robinhood_transactions))
    import_summary_cols[6].metric("Closed Trades", len(robinhood_derivation.closed_trades))
    import_summary_cols[7].metric("Exit Matches", len(robinhood_derivation.exit_matches))
    import_summary_cols[8].metric("Open Lots", len(robinhood_derivation.open_lots))
    import_summary_cols[9].metric("Missing Stops", robinhood_derivation.missing_planned_stops)

    if robinhood_derivation.missing_planned_stops:
        render_feedback(
            f"{robinhood_derivation.missing_planned_stops} matched lot rows do not have a usable planned stop. "
            "Check missing or ambiguous planned stops in the Import Issues tab.",
            "idle",
        )

    if (
        imported_result is not None
        or not robinhood_derivation.unmatched_sells.empty
        or not robinhood_derivation.planned_stop_issues.empty
    ):
        issue_parts = []
        if imported_result is not None and not imported_result.malformed_rows.empty:
            issue_parts.append(f"{len(imported_result.malformed_rows)} malformed rows in the latest upload")
        if not robinhood_derivation.unmatched_sells.empty:
            issue_parts.append(f"{len(robinhood_derivation.unmatched_sells)} unmatched sells in stored transactions")
        if not robinhood_derivation.planned_stop_issues.empty:
            issue_parts.append(f"{len(robinhood_derivation.planned_stop_issues)} ambiguous planned stop keys")
        if issue_parts:
            render_feedback("Import issues found: " + ", ".join(issue_parts) + ".", "error")

    closed_tab, exit_match_tab, open_tab, issues_tab, raw_tab = st.tabs(
        ["Closed Trades", "Exit Matches", "Open Lots", "Import Issues", "Clean Transactions"]
    )

    with closed_tab:
        if robinhood_derivation.closed_trades.empty:
            render_feedback("No closed trades were derived from stored Robinhood transactions.", "idle")
        else:
            st.dataframe(
                display_date_frame(robinhood_derivation.closed_trades),
                column_config=closed_trades_column_config(),
                height=robinhood_dataframe_height(len(robinhood_derivation.closed_trades)),
                hide_index=True,
                width="stretch",
            )

    with exit_match_tab:
        if robinhood_derivation.exit_matches.empty:
            render_feedback("No FIFO exit matches were derived from stored Robinhood transactions.", "idle")
        else:
            st.dataframe(
                display_date_frame(robinhood_derivation.exit_matches),
                column_config=closed_trades_column_config(),
                height=robinhood_dataframe_height(len(robinhood_derivation.exit_matches)),
                hide_index=True,
                width="stretch",
            )

    with open_tab:
        if robinhood_derivation.open_lots.empty:
            render_feedback("No open imported lots remain after FIFO matching.", "idle")
        else:
            st.dataframe(
                display_date_frame(robinhood_derivation.open_lots),
                column_config=open_lots_column_config(),
                hide_index=True,
                width="stretch",
            )

    with issues_tab:
        shown_any_issue = False
        if imported_result is not None and not imported_result.malformed_rows.empty:
            shown_any_issue = True
            st.dataframe(imported_result.malformed_rows, hide_index=True, width="stretch")
        if not robinhood_derivation.unmatched_sells.empty:
            shown_any_issue = True
            st.dataframe(
                display_date_frame(robinhood_derivation.unmatched_sells),
                hide_index=True,
                width="stretch",
            )
        if not robinhood_derivation.planned_stop_issues.empty:
            shown_any_issue = True
            st.dataframe(robinhood_derivation.planned_stop_issues, hide_index=True, width="stretch")
        if not shown_any_issue:
            render_feedback("No malformed rows, unmatched sells, or ambiguous planned stops were found.", "idle")

    with raw_tab:
        if st.session_state.robinhood_transactions.empty:
            render_feedback("No clean Robinhood transactions are stored yet.", "idle")
        else:
            st.dataframe(
                display_date_frame(st.session_state.robinhood_transactions),
                height=robinhood_dataframe_height(len(st.session_state.robinhood_transactions)),
                hide_index=True,
                width="stretch",
            )

with trade_metrics_container:
    render_trade_metrics(trade_metrics, strategy_metrics, strategy_attribution)
