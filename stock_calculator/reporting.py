from __future__ import annotations

from datetime import date

import pandas as pd

from stock_calculator.robinhood import STRATEGY_METRIC_COLUMNS, calculate_strategy_metrics


ALL_TIME_LABEL = "All"
CURRENT_SIZING_SIGNAL_COLUMNS = ["rolling_mode_exp", "mode_adjusted_score", "mode", "action"]


def report_year_options(closed_trades: pd.DataFrame) -> list[str]:
    years = _valid_years(closed_trades, "sell_date")
    return [*[str(year) for year in sorted(years, reverse=True)], ALL_TIME_LABEL]


def default_report_year_label(closed_trades: pd.DataFrame, *, today: date | None = None) -> str:
    years = _valid_years(closed_trades, "sell_date")
    if not years:
        return ALL_TIME_LABEL

    current_year = (today or date.today()).year
    if current_year in years:
        return str(current_year)
    return str(max(years))


def report_year_value(label: str | int | None) -> int | None:
    if label is None or str(label) == ALL_TIME_LABEL:
        return None
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def report_scope_label(year: int | None) -> str:
    return ALL_TIME_LABEL if year is None else str(year)


def filter_frame_by_year(df: pd.DataFrame, year: int | None, date_column: str) -> pd.DataFrame:
    if year is None:
        return df.copy()
    if df.empty or date_column not in df.columns:
        return df.iloc[0:0].copy()

    dates = pd.to_datetime(df[date_column], errors="coerce")
    return df.loc[dates.dt.year == year].reset_index(drop=True)


def calculate_strategy_report_metrics(
    report_closed_trades: pd.DataFrame,
    all_closed_trades: pd.DataFrame,
) -> pd.DataFrame:
    report_metrics = calculate_strategy_metrics(report_closed_trades)
    if report_metrics.empty:
        return report_metrics

    all_time_metrics = calculate_strategy_metrics(all_closed_trades)
    if all_time_metrics.empty:
        return report_metrics

    signal_columns = [
        column
        for column in CURRENT_SIZING_SIGNAL_COLUMNS
        if column in report_metrics.columns and column in all_time_metrics.columns
    ]
    if not signal_columns:
        return report_metrics

    current_signals = all_time_metrics[["strategy", *signal_columns]]
    report_without_signals = report_metrics.drop(columns=signal_columns)
    merged = report_without_signals.merge(current_signals, on="strategy", how="left")
    return merged.reindex(columns=STRATEGY_METRIC_COLUMNS)


def _valid_years(df: pd.DataFrame, date_column: str) -> set[int]:
    if df.empty or date_column not in df.columns:
        return set()

    years = pd.to_datetime(df[date_column], errors="coerce").dt.year.dropna()
    return {int(year) for year in years}
