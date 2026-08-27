from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import date

import pandas as pd

from stock_calculator.robinhood import (
    STRATEGY_METRIC_COLUMNS,
    calculate_strategy_metrics,
    calculate_trade_metrics,
)


ALL_TIME_LABEL = "All"
CURRENT_SIZING_SIGNAL_COLUMNS = ["rolling_mode_exp", "mode_adjusted_score", "mode", "action"]
MONTHLY_TRACKER_COLUMNS = [
    "month",
    "average_gain_percent",
    "average_loss_percent",
    "win_loss_ratio",
    "win_rate",
    "adjusted_win_loss_ratio",
    "trade_count",
    "largest_gain_percent",
    "largest_loss_percent",
    "average_gain_hold_days",
    "average_loss_hold_days",
]
MONTHLY_TRACKER_METRIC_COLUMNS = MONTHLY_TRACKER_COLUMNS[1:]
MONTHLY_TRACKER_HIDDEN = "hidden"
MONTHLY_TRACKER_SELECT_YEAR = "select_year"
MONTHLY_TRACKER_READY = "ready"
RETURN_DISTRIBUTION_COLUMNS = ["bin_start", "bin_end", "bin_label", "trade_count", "overflow"]
RETURN_DISTRIBUTION_BIN_WIDTH = 2.0
RETURN_DISTRIBUTION_MIN_OUTLIER_SAMPLE = 4


@dataclass(frozen=True)
class TradeReturnDistribution:
    bins: pd.DataFrame
    valid_return_count: int
    completed_trade_count: int


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


def calculate_monthly_trade_tracker(closed_trades: pd.DataFrame) -> pd.DataFrame:
    sell_months = (
        pd.to_datetime(closed_trades["sell_date"], errors="coerce").dt.month
        if not closed_trades.empty and "sell_date" in closed_trades.columns
        else pd.Series(index=closed_trades.index, dtype="float64")
    )
    rows = []
    for month_number in range(1, 13):
        month_trades = closed_trades.loc[sell_months == month_number]
        metrics = calculate_trade_metrics(month_trades)
        realized_pnl = _numeric_column(month_trades, "realized_pnl")
        return_percent = _numeric_column(month_trades, "realized_pnl_percent")
        gain_returns = return_percent[realized_pnl > 0].dropna()
        loss_returns = return_percent[realized_pnl < 0].dropna()
        win_loss_ratio, adjusted_win_loss_ratio = _percentage_win_loss_ratios(metrics)
        rows.append(
            {
                "month": calendar.month_abbr[month_number].upper(),
                "average_gain_percent": metrics.get("average_win_percent"),
                "average_loss_percent": metrics.get("average_loss_percent"),
                "win_loss_ratio": win_loss_ratio,
                "win_rate": metrics.get("win_rate"),
                "adjusted_win_loss_ratio": adjusted_win_loss_ratio,
                "trade_count": metrics.get("trade_count", 0),
                "largest_gain_percent": gain_returns.max() if not gain_returns.empty else None,
                "largest_loss_percent": loss_returns.min() if not loss_returns.empty else None,
                "average_gain_hold_days": metrics.get("average_win_hold"),
                "average_loss_hold_days": metrics.get("average_loss_hold"),
            }
        )

    tracker = pd.DataFrame(rows, columns=MONTHLY_TRACKER_COLUMNS)
    for column in MONTHLY_TRACKER_METRIC_COLUMNS:
        tracker[column] = pd.to_numeric(tracker[column], errors="coerce")
    return tracker


def calculate_trade_return_distribution(closed_trades: pd.DataFrame) -> TradeReturnDistribution:
    completed_trade_count = int(calculate_trade_metrics(closed_trades).get("trade_count", 0))
    if closed_trades.empty:
        return _empty_trade_return_distribution(completed_trade_count)

    realized_pnl = _numeric_column(closed_trades, "realized_pnl")
    return_percent = _numeric_column(closed_trades, "realized_pnl_percent")
    finite_mask = realized_pnl.map(_is_finite) & return_percent.map(_is_finite)
    valid_returns = return_percent.loc[finite_mask]
    if valid_returns.empty:
        return _empty_trade_return_distribution(completed_trade_count)

    lower_boundary, upper_boundary = _return_distribution_core_boundaries(valid_returns)
    lower_overflow = valid_returns < lower_boundary if lower_boundary is not None else None
    upper_overflow = valid_returns >= upper_boundary if upper_boundary is not None else None
    has_lower_overflow = lower_overflow is not None and bool(lower_overflow.any())
    has_upper_overflow = upper_overflow is not None and bool(upper_overflow.any())

    core_returns = valid_returns
    if has_lower_overflow:
        core_returns = core_returns.loc[~lower_overflow]
    if has_upper_overflow:
        core_returns = core_returns.loc[~upper_overflow]

    bin_starts = core_returns.map(
        lambda value: math.floor(float(value) / RETURN_DISTRIBUTION_BIN_WIDTH) * RETURN_DISTRIBUTION_BIN_WIDTH
    )
    counts = {float(bin_start): int(count) for bin_start, count in bin_starts.value_counts().items()}
    rows = []
    if has_lower_overflow:
        rows.append(
            {
                "bin_start": lower_boundary - RETURN_DISTRIBUTION_BIN_WIDTH,
                "bin_end": lower_boundary,
                "bin_label": f"< {_format_percent_boundary(lower_boundary)}",
                "trade_count": int(lower_overflow.sum()),
                "overflow": "lower",
            }
        )
    core_bin_starts = _core_bin_starts(
        counts,
        lower_boundary if has_lower_overflow else None,
        upper_boundary if has_upper_overflow else None,
    )
    for bin_start in core_bin_starts:
        bin_end = bin_start + RETURN_DISTRIBUTION_BIN_WIDTH
        rows.append(
            {
                "bin_start": bin_start,
                "bin_end": bin_end,
                "bin_label": f"[{_format_percent_boundary(bin_start)}, {_format_percent_boundary(bin_end)})",
                "trade_count": counts.get(bin_start, 0),
                "overflow": None,
            }
        )
    if has_upper_overflow:
        rows.append(
            {
                "bin_start": upper_boundary,
                "bin_end": upper_boundary + RETURN_DISTRIBUTION_BIN_WIDTH,
                "bin_label": f"≥ {_format_percent_boundary(upper_boundary)}",
                "trade_count": int(upper_overflow.sum()),
                "overflow": "upper",
            }
        )
    bins = pd.DataFrame(rows, columns=RETURN_DISTRIBUTION_COLUMNS)
    return TradeReturnDistribution(
        bins=bins,
        valid_return_count=len(valid_returns),
        completed_trade_count=completed_trade_count,
    )


def trade_return_distribution_caption(distribution: TradeReturnDistribution) -> str:
    valid_count = distribution.valid_return_count
    trade_count = distribution.completed_trade_count
    if valid_count == 0:
        return "No completed trades have valid return data."
    explanation = "Each bar counts completed trades in a 2% point return range."
    has_overflow = "overflow" in distribution.bins.columns and distribution.bins["overflow"].notna().any()
    if has_overflow:
        explanation += " Extreme returns are grouped into labeled end bars."
    if valid_count == trade_count:
        return explanation

    # A trade with no finite return percent cannot be binned, so say so rather than let the
    # bars quietly total less than the completed trade count shown elsewhere.
    noun = "trade" if trade_count == 1 else "trades"
    return f"{explanation} Includes {valid_count} of {trade_count} completed {noun} with valid return data."


def monthly_tracker_state(closed_trades: pd.DataFrame, scope_label: str) -> str:
    if closed_trades.empty:
        return MONTHLY_TRACKER_HIDDEN
    if scope_label == ALL_TIME_LABEL:
        return MONTHLY_TRACKER_SELECT_YEAR
    return MONTHLY_TRACKER_READY


def _valid_years(df: pd.DataFrame, date_column: str) -> set[int]:
    if df.empty or date_column not in df.columns:
        return set()

    years = pd.to_datetime(df[date_column], errors="coerce").dt.year.dropna()
    return {int(year) for year in years}


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column] if column in frame.columns else pd.Series(index=frame.index, dtype="float64")
    return pd.to_numeric(values, errors="coerce")


def _percentage_win_loss_ratios(metrics: dict[str, float | int | None]) -> tuple[float | None, float | None]:
    average_gain = metrics.get("average_win_percent")
    average_loss = metrics.get("average_loss_percent")
    trade_count = int(metrics.get("trade_count") or 0)
    win_count = int(metrics.get("win_count") or 0)
    loss_count = int(metrics.get("loss_count") or 0)
    if (
        average_gain is None
        or pd.isna(average_gain)
        or average_loss is None
        or pd.isna(average_loss)
        or float(average_loss) == 0
    ):
        return None, None

    loss_magnitude = abs(float(average_loss))
    win_loss_ratio = round(float(average_gain) / loss_magnitude, 2)
    if trade_count == 0 or win_count == 0 or loss_count == 0:
        return win_loss_ratio, None

    win_percentage = win_count / trade_count
    loss_percentage = loss_count / trade_count
    adjusted_ratio = round(
        (float(average_gain) * win_percentage) / (loss_magnitude * loss_percentage),
        2,
    )
    return win_loss_ratio, adjusted_ratio


def _is_finite(value: object) -> bool:
    return value is not None and not pd.isna(value) and math.isfinite(float(value))


def _return_distribution_core_boundaries(valid_returns: pd.Series) -> tuple[float | None, float | None]:
    if len(valid_returns) < RETURN_DISTRIBUTION_MIN_OUTLIER_SAMPLE:
        return None, None

    first_quartile = float(valid_returns.quantile(0.25))
    third_quartile = float(valid_returns.quantile(0.75))
    interquartile_range = third_quartile - first_quartile
    if interquartile_range <= 0:
        return None, None

    lower_fence = first_quartile - 1.5 * interquartile_range
    upper_fence = third_quartile + 1.5 * interquartile_range
    lower_boundary = math.floor(lower_fence / RETURN_DISTRIBUTION_BIN_WIDTH) * RETURN_DISTRIBUTION_BIN_WIDTH
    upper_boundary = math.ceil(upper_fence / RETURN_DISTRIBUTION_BIN_WIDTH) * RETURN_DISTRIBUTION_BIN_WIDTH
    return float(lower_boundary), float(upper_boundary)


def _core_bin_starts(
    counts: dict[float, int],
    lower_boundary: float | None,
    upper_boundary: float | None,
) -> list[float]:
    """Every bin start across the core range, including bins no trade landed in.

    Sparse bins would let the chart's band axis close the gap and place non-adjacent
    return ranges side by side. The range is anchored to an overflow boundary when one
    exists, so the empty bin next to an end bar survives too.
    """
    if not counts:
        return []

    first_bin = lower_boundary if lower_boundary is not None else min(counts)
    last_bin = (
        upper_boundary - RETURN_DISTRIBUTION_BIN_WIDTH if upper_boundary is not None else max(counts)
    )
    bin_count = round((last_bin - first_bin) / RETURN_DISTRIBUTION_BIN_WIDTH) + 1
    return [first_bin + index * RETURN_DISTRIBUTION_BIN_WIDTH for index in range(max(bin_count, 0))]


def _format_percent_boundary(value: float) -> str:
    number = int(value) if float(value).is_integer() else value
    return f"{number:g}%"


def _empty_trade_return_distribution(completed_trade_count: int) -> TradeReturnDistribution:
    return TradeReturnDistribution(
        bins=pd.DataFrame(columns=RETURN_DISTRIBUTION_COLUMNS),
        valid_return_count=0,
        completed_trade_count=completed_trade_count,
    )
