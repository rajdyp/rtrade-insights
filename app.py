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
    calculate_total_realized_pnl,
    calculate_trade_metrics,
    derive_fifo_trades,
    parse_robinhood_csv,
)
from stock_calculator.storage import (
    DATA_PATH,
    PLANNED_STOPS_PATH,
    ROBINHOOD_TRANSACTIONS_PATH,
    append_robinhood_transactions,
    load_planned_stops,
    load_positions,
    load_robinhood_transactions,
    save_planned_stops,
    save_positions,
    save_robinhood_transactions,
    upsert_planned_stop,
)


st.set_page_config(page_title="Stock Calculator", layout="wide")

DELETE_COLUMN = "delete_selected"
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
            text-transform: uppercase;
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
        "symbol": st.column_config.TextColumn("Symbol", width=84),
        "buy_date": st.column_config.DateColumn("Buy Date", format="MM/DD/YYYY", width=108),
        "share_price": st.column_config.NumberColumn("Share Price", format="$%.2f", width=110),
        "stop_price": st.column_config.NumberColumn("Stop Price", format="$%.2f", width=104),
        "stop_loss_percent": st.column_config.NumberColumn("Stop Loss", format="%.2f%%", width=86),
        "number_of_shares": st.column_config.NumberColumn("Shares", width=78),
        "sell_lot": st.column_config.NumberColumn("Sell Lot", width=78),
        "hold_count": st.column_config.NumberColumn("Hold Days", width=84),
        "position_size": st.column_config.NumberColumn("Position Size", format="$%.2f", width=126),
        "risk_percent": st.column_config.NumberColumn("Risk %", format="%.2f%%", width=82),
        "risk_amount": st.column_config.NumberColumn("Total Risk", format="$%.2f", width=116),
        "portfolio_amount": st.column_config.NumberColumn("Portfolio", format="$%.2f", width=126),
        DELETE_COLUMN: st.column_config.CheckboxColumn("x", width=44, default=False),
    }


def closed_trades_column_config() -> dict:
    return {
        "symbol": st.column_config.TextColumn("Symbol", width=84),
        "buy_date": st.column_config.DateColumn("Buy Date", format="MM/DD/YYYY", width=108),
        "sell_date": st.column_config.DateColumn("Sell Date", format="MM/DD/YYYY", width=108),
        "quantity": st.column_config.NumberColumn("Quantity", width=86),
        "planned_stop": st.column_config.NumberColumn("Stop Price", format="$%.2f", width=104),
        "buy_price": st.column_config.NumberColumn("Buy Price", format="$%.2f", width=104),
        "buy_amount": st.column_config.NumberColumn("Buy Amount", format="$%.2f", width=116),
        "sell_price": st.column_config.NumberColumn("Sell Price", format="$%.2f", width=104),
        "sell_amount": st.column_config.NumberColumn("Sell Amount", format="$%.2f", width=116),
        "realized_pnl": st.column_config.NumberColumn("Realized P/L", format="$%.2f", width=116),
        "realized_pnl_percent": st.column_config.NumberColumn("P/L %", format="%.2f%%", width=86),
        "hold_days": st.column_config.NumberColumn("Hold Days", width=88),
    }


def open_positions_column_config() -> dict:
    return {
        "symbol": st.column_config.TextColumn("Symbol", width=84),
        "buy_date": st.column_config.DateColumn("Buy Date", format="MM/DD/YYYY", width=108),
        "quantity": st.column_config.NumberColumn("Quantity", width=86),
        "planned_stop": st.column_config.NumberColumn("Stop Price", format="$%.2f", width=104),
        "buy_price": st.column_config.NumberColumn("Buy Price", format="$%.2f", width=104),
        "cost_basis": st.column_config.NumberColumn("Cost Basis", format="$%.2f", width=116),
        "hold_days": st.column_config.NumberColumn("Hold Days", width=88),
    }


def editor_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "buy_date" in frame.columns:
        frame["buy_date"] = pd.to_datetime(frame["buy_date"], errors="coerce").dt.date
    frame[DELETE_COLUMN] = False
    return frame


def display_date_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for column in ["buy_date", "sell_date", "activity_date", "process_date", "settle_date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    return frame


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


def render_header(total_pnl: float, target=st) -> None:
    total_pnl_percent = (total_pnl / config.portfolio_amount) * 100 if config.portfolio_amount else 0.0
    total_pnl_display = f"{format_currency(total_pnl)} ({format_percent(total_pnl_percent)})"
    target.markdown(
        f"""
        <div class="app-header">
          <div>
            <div class="app-title">Stock Calculator</div>
            <div class="app-subtitle">Position sizing &amp; portfolio risk tracking</div>
          </div>
          <div class="header-meta">
            <span class="meta-pill"><strong>PORTFOLIO</strong>&nbsp;&nbsp;{escape(format_currency(config.portfolio_amount))}</span>
            <span class="meta-pill"><strong>TOTAL P&amp;L</strong>&nbsp;&nbsp;{escape(total_pnl_display)}</span>
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


def render_trade_metrics(metrics: dict | None) -> None:
    render_section("Trade Metrics", "Derived from the local trade dataset.")
    if metrics is None:
        st.info("Upload a Robinhood CSV to populate trade metrics.")
        return

    if metrics.get("trade_count", 0) == 0:
        st.info("No closed trades are available for metrics yet.")

    metric_rows = [
        [
            ("Win Rate", format_optional_percent(metrics.get("win_rate"))),
            ("Expectancy", format_optional_currency(metrics.get("expectancy"))),
            ("Profit Factor", format_optional_number(metrics.get("profit_factor"))),
            ("Win/Loss Ratio", format_optional_number(metrics.get("win_loss_ratio"))),
        ],
        [
            ("Avg R (Wins)", format_optional_number(metrics.get("average_win_r"))),
            ("Avg R (Losses)", format_optional_number(metrics.get("average_loss_r"))),
            ("R Ratio (Win/Loss)", format_optional_number(metrics.get("r_ratio"))),
            None,
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


config = load_config()


def add_current_draft() -> None:
    draft_from_state = draft_position(
        symbol=str(st.session_state.get("draft_symbol", "")).upper().strip(),
        buy_date=st.session_state.get("draft_buy_date"),
        share_price=st.session_state.get("draft_share_price"),
        stop_price=st.session_state.get("draft_stop_price"),
        portfolio_amount=st.session_state.get("draft_portfolio_amount"),
        risk_percent=st.session_state.get("draft_risk_percent"),
    )
    calculated_draft = calculate_positions(draft_from_state)
    validation_error = str(calculated_draft.iloc[0]["validation_error"] or "")
    if validation_error:
        return

    st.session_state.positions = append_committed_position(st.session_state.positions, draft_from_state)
    st.session_state.planned_stops = upsert_planned_stop(st.session_state.planned_stops, calculated_draft.iloc[0])
    save_positions(st.session_state.positions)
    save_planned_stops(st.session_state.planned_stops)
    st.session_state.draft_symbol = ""
    st.session_state.draft_buy_date = date.today()
    st.session_state.draft_share_price = 0.0
    st.session_state.draft_stop_price = 0.0
    st.session_state.draft_portfolio_amount = config.sizing_portfolio_amount
    st.session_state.draft_risk_percent = config.risk_percent


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
if "draft_portfolio_amount" not in st.session_state:
    st.session_state.draft_portfolio_amount = config.sizing_portfolio_amount
if "draft_risk_percent" not in st.session_state:
    st.session_state.draft_risk_percent = config.risk_percent
if "position_editor_revision" not in st.session_state:
    st.session_state.position_editor_revision = 0

robinhood_derivation = derive_fifo_trades(st.session_state.robinhood_transactions, st.session_state.planned_stops)
trade_metrics = calculate_trade_metrics(robinhood_derivation.closed_trades)
total_pnl = calculate_total_realized_pnl(robinhood_derivation.closed_trades)

apply_styles()
header_container = st.empty()
render_header(total_pnl, header_container)

render_section("New Position", "Enter a candidate position and review the live sizing before adding it.")

top_cols = st.columns([1.2, 1, 1, 1, 1, 0.8])
symbol = top_cols[0].text_input("Symbol", key="draft_symbol").upper().strip()
buy_date = top_cols[1].date_input("Buy Date", key="draft_buy_date")
share_price = top_cols[2].number_input("Share Price", min_value=0.0, step=0.01, format="%.2f", key="draft_share_price")
stop_price = top_cols[3].number_input("Stop Price", min_value=0.0, step=0.01, format="%.2f", key="draft_stop_price")
portfolio_amount = top_cols[4].number_input(
    "Portfolio",
    min_value=0.0,
    step=100.0,
    format="%.2f",
    key="draft_portfolio_amount",
)
risk_percent = top_cols[5].number_input(
    "Risk %",
    min_value=0.0,
    step=0.05,
    format="%.2f",
    key="draft_risk_percent",
)

draft = draft_position(
    symbol=symbol,
    buy_date=buy_date,
    share_price=share_price,
    stop_price=stop_price,
    portfolio_amount=portfolio_amount,
    risk_percent=risk_percent,
)
draft_result = calculate_positions(draft)
draft_row = draft_result.iloc[0]
draft_error = str(draft_row["validation_error"] or "")
draft_is_valid = bool(symbol) and not draft_error

preview_cols = st.columns(4)
preview_cols[0].metric("Stop Loss", format_percent(first_value(draft_result, "stop_loss_percent")))
preview_cols[1].metric("Shares", "" if pd.isna(draft_row["number_of_shares"]) else int(draft_row["number_of_shares"]))
preview_cols[2].metric("Position Size", format_currency(first_value(draft_result, "position_size")))
preview_cols[3].metric("Total Risk", format_currency(first_value(draft_result, "risk_amount")))

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
    st.info("No positions yet. Add a valid position from the calculator above to start the list.")
else:
    edited = st.data_editor(
        editor_frame(visible_calculated[PUBLIC_OUTPUT_COLUMNS]),
        column_config=positions_column_config(),
        disabled=[column for column in PUBLIC_OUTPUT_COLUMNS if column not in INPUT_COLUMNS],
        height=positions_editor_height(len(visible_calculated)),
        num_rows="fixed",
        width="stretch",
        hide_index=True,
        key=position_editor_key,
    )

    st.session_state.positions = committed_positions(edited[INPUT_COLUMNS])
    delete_rows = marked_delete_rows(edited)
    if delete_rows:
        delete_selected_positions(delete_rows)

    save_positions(st.session_state.positions)
    visible_calculated = filtered_output(calculate_positions(st.session_state.positions))
    invalid_positions = int((visible_calculated["validation_error"].fillna("") != "").sum())

if invalid_positions:
    invalid_rows = visible_calculated[visible_calculated["validation_error"].fillna("") != ""]
    messages = [
        f"{row['symbol'] or f'Row {index + 1}'}: {row['validation_error']}"
        for index, row in invalid_rows.iterrows()
    ]
    st.warning("Some rows need fixes before their calculated values can be used: " + "; ".join(messages))

with st.expander("Quip-style summary"):
    if visible_calculated.empty:
        st.info("Add one or more symbols to see the summary.")
    else:
        summary = pd.DataFrame(
            {
                row["symbol"]: {
                    "Buy Date": row["buy_date"],
                    "Share Price": format_currency(row["share_price"]),
                    "Stop Price": format_currency(row["stop_price"]),
                    "Stop Loss": format_percent(row["stop_loss_percent"]),
                    "No of Shares": "" if pd.isna(row["number_of_shares"]) else str(int(row["number_of_shares"])),
                    "Hold Count": "" if pd.isna(row["hold_count"]) else str(int(row["hold_count"])),
                    "Sell Lot": "" if pd.isna(row["sell_lot"]) else str(int(row["sell_lot"])),
                    "Position Size": format_currency(row["position_size"]),
                    "Risk %": format_percent(row["risk_percent"]),
                    "Total Risk": format_currency(row["risk_amount"]),
                }
                for _, row in visible_calculated.iterrows()
            }
        )
        st.dataframe(summary, width="stretch")

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
        render_feedback(
            f"Upload a Robinhood CSV report to add new transactions to {ROBINHOOD_TRANSACTIONS_PATH}. "
            f"Planned stop prices (SL) are read from {PLANNED_STOPS_PATH}.",
            "idle",
        )
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
        total_pnl = calculate_total_realized_pnl(robinhood_derivation.closed_trades)
        render_header(total_pnl, header_container)
        st.session_state.robinhood_import_result = import_result

        if import_result.transactions.empty:
            render_feedback("No buy or sell rows were found in the uploaded CSV.", "error")
        elif added_count:
            render_feedback(f"Added {added_count} new Robinhood transactions to the local dataset.", "ready")
        else:
            render_feedback("No new Robinhood transactions were added; all clean rows were already imported.", "idle")

    imported_result = st.session_state.get("robinhood_import_result")
    current_trade_rows = len(imported_result.transactions) if imported_result is not None else 0
    current_ignored_rows = len(imported_result.ignored_rows) if imported_result is not None else 0
    current_malformed_rows = len(imported_result.malformed_rows) if imported_result is not None else 0
    latest_added_rows = st.session_state.get("robinhood_last_added_count", 0) if imported_result is not None else 0
    latest_skipped_rows = st.session_state.get("robinhood_last_skipped_count", 0) if imported_result is not None else 0

    import_summary_cols = st.columns(9)
    import_summary_cols[0].metric("Upload Rows", current_trade_rows)
    import_summary_cols[1].metric("New Rows", latest_added_rows)
    import_summary_cols[2].metric("Skipped Rows", latest_skipped_rows)
    import_summary_cols[3].metric("Ignored Rows", current_ignored_rows)
    import_summary_cols[4].metric("Malformed Rows", current_malformed_rows)
    import_summary_cols[5].metric("Local Rows", len(st.session_state.robinhood_transactions))
    import_summary_cols[6].metric("Closed Trades", len(robinhood_derivation.closed_trades))
    import_summary_cols[7].metric("Open Positions", len(robinhood_derivation.open_positions))
    import_summary_cols[8].metric("Missing Stops", robinhood_derivation.missing_planned_stops)

    if robinhood_derivation.missing_planned_stops:
        render_feedback(
            f"{robinhood_derivation.missing_planned_stops} matched lot rows do not have planned stops yet. "
            f"Fill missing entry stops in {PLANNED_STOPS_PATH}.",
            "idle",
        )

    if imported_result is not None or not robinhood_derivation.unmatched_sells.empty:
        issue_parts = []
        if imported_result is not None and not imported_result.malformed_rows.empty:
            issue_parts.append(f"{len(imported_result.malformed_rows)} malformed rows in the latest upload")
        if not robinhood_derivation.unmatched_sells.empty:
            issue_parts.append(f"{len(robinhood_derivation.unmatched_sells)} unmatched sells in the local dataset")
        if issue_parts:
            render_feedback("Import issues found: " + ", ".join(issue_parts) + ".", "error")

    closed_tab, open_tab, issues_tab, raw_tab = st.tabs(
        ["Closed Trades", "Open Positions", "Import Issues", "Clean Transactions"]
    )

    with closed_tab:
        if robinhood_derivation.closed_trades.empty:
            render_feedback("No closed trades were derived from the local Robinhood dataset.", "idle")
        else:
            st.dataframe(
                display_date_frame(robinhood_derivation.closed_trades),
                column_config=closed_trades_column_config(),
                height=robinhood_dataframe_height(len(robinhood_derivation.closed_trades)),
                hide_index=True,
                width="stretch",
            )

    with open_tab:
        if robinhood_derivation.open_positions.empty:
            render_feedback("No open imported positions remain after FIFO matching.", "idle")
        else:
            st.dataframe(
                display_date_frame(robinhood_derivation.open_positions),
                column_config=open_positions_column_config(),
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
        if not shown_any_issue:
            render_feedback("No malformed rows or unmatched sells were found.", "idle")

    with raw_tab:
        if st.session_state.robinhood_transactions.empty:
            render_feedback("No clean Robinhood transactions are stored locally.", "idle")
        else:
            st.dataframe(
                display_date_frame(st.session_state.robinhood_transactions),
                height=robinhood_dataframe_height(len(st.session_state.robinhood_transactions)),
                hide_index=True,
                width="stretch",
            )

with trade_metrics_container:
    render_trade_metrics(trade_metrics)
