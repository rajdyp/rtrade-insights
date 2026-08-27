from __future__ import annotations

from datetime import date

import pandas as pd
from pandas.api.types import is_numeric_dtype

from stock_calculator.reporting import (
    ALL_TIME_LABEL,
    MONTHLY_TRACKER_HIDDEN,
    MONTHLY_TRACKER_METRIC_COLUMNS,
    MONTHLY_TRACKER_READY,
    MONTHLY_TRACKER_SELECT_YEAR,
    calculate_monthly_trade_tracker,
    calculate_strategy_report_metrics,
    calculate_trade_return_distribution,
    default_report_year_label,
    filter_frame_by_year,
    monthly_tracker_state,
    report_year_options,
    trade_return_distribution_caption,
)
from stock_calculator.robinhood import (
    CLOSED_TRADE_COLUMNS,
    PLANNED_STOP_COLUMNS,
    calculate_trade_metrics,
    derive_fifo_trades,
)


def test_report_year_options_use_valid_closed_trade_sell_years_descending():
    closed_trades = pd.DataFrame(
        {
            "sell_date": ["2027-01-05", "not-a-date", "", "2026-12-30", "2027-02-01"],
        }
    )

    assert report_year_options(closed_trades) == ["2027", "2026", ALL_TIME_LABEL]


def test_default_report_year_prefers_current_year_then_latest_available_year():
    closed_trades = pd.DataFrame({"sell_date": ["2027-01-05", "2026-12-30"]})

    assert default_report_year_label(closed_trades, today=date(2026, 5, 23)) == "2026"
    assert default_report_year_label(closed_trades, today=date(2028, 5, 23)) == "2027"


def test_filter_frame_by_year_uses_requested_date_column_and_excludes_invalid_dates():
    rows = pd.DataFrame(
        {
            "symbol": ["OLD", "MATCH", "BAD", "OTHER"],
            "sell_date": ["2026-12-30", "2027-01-03", "not-a-date", "2028-01-04"],
        }
    )

    filtered = filter_frame_by_year(rows, 2027, "sell_date")

    assert filtered["symbol"].tolist() == ["MATCH"]


def test_filter_frame_by_year_keeps_all_time_when_year_is_none():
    rows = pd.DataFrame({"symbol": ["A", "B"], "activity_date": ["2027-01-01", "2028-01-01"]})

    filtered = filter_frame_by_year(rows, None, "activity_date")

    assert filtered.equals(rows)
    assert filtered is not rows


def test_cross_year_trade_counts_in_sell_date_year():
    closed_trades = pd.DataFrame(
        {
            "symbol": ["CROSS"],
            "buy_date": ["2026-12-28"],
            "sell_date": ["2027-01-03"],
            "realized_pnl": [125.0],
        }
    )

    assert filter_frame_by_year(closed_trades, 2026, "sell_date").empty
    assert filter_frame_by_year(closed_trades, 2027, "sell_date")["symbol"].tolist() == ["CROSS"]


def test_strategy_report_metrics_use_year_performance_and_full_history_sizing_signal():
    report_loss = _trade_row("EP", "2025-05-01", -50.0)
    current_winners = [_trade_row("EP", f"2026-01-{day:02d}", 10.0) for day in range(1, 16)]
    all_trades = pd.DataFrame([report_loss, *current_winners])
    report_trades = filter_frame_by_year(all_trades, 2025, "sell_date")

    metrics = calculate_strategy_report_metrics(report_trades, all_trades)
    row = metrics.loc[metrics["strategy"] == "EP"].iloc[0]

    assert row["trade_count"] == 1
    assert row["total_realized_pnl"] == -50.0
    assert row["win_rate"] == 0.0
    assert row["mode"] == "Working"
    assert row["action"] == "Use regime baseline"
    assert row["rolling_mode_exp"] == "+1.00R"


def test_monthly_trade_tracker_uses_existing_metrics_and_returns_all_months_without_summary():
    closed_trades = pd.DataFrame(
        [
            _closed_trade_row("JAN-WIN", "2026-01-05", 10.0, hold_days=3),
            _closed_trade_row("JAN-WIN-2", "2026-01-12", 20.0, hold_days=5),
            _closed_trade_row("JAN-LOSS", "2026-01-20", -5.0, hold_days=2),
            _closed_trade_row("JAN-FLAT", "2026-01-25", 0.0, hold_days=1),
            _closed_trade_row("FEB-FLAT", "2026-02-03", 0.0, hold_days=1),
        ],
        columns=CLOSED_TRADE_COLUMNS,
    )

    tracker = calculate_monthly_trade_tracker(closed_trades)
    january = tracker.iloc[0]
    february = tracker.iloc[1]
    january_metrics = calculate_trade_metrics(closed_trades.iloc[:4])

    assert tracker["month"].tolist() == [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]
    assert "TOTAL" not in tracker["month"].tolist()
    assert tracker.columns.get_loc("win_loss_ratio") + 1 == tracker.columns.get_loc("win_rate")
    assert tracker.columns.get_loc("adjusted_win_loss_ratio") == tracker.columns.get_loc("win_rate") + 1
    assert january["trade_count"] == january_metrics["trade_count"] == 4
    assert january["win_rate"] == january_metrics["win_rate"] == 50.0
    assert january["average_gain_percent"] == january_metrics["average_win_percent"] == 15.0
    assert january["average_loss_percent"] == january_metrics["average_loss_percent"] == -5.0
    assert january["win_loss_ratio"] == 3.0
    assert january["adjusted_win_loss_ratio"] == 6.0
    assert january["average_gain_hold_days"] == january_metrics["average_win_hold"] == 4.0
    assert january["average_loss_hold_days"] == january_metrics["average_loss_hold"] == 2.0
    assert january["largest_gain_percent"] == 20.0
    assert january["largest_loss_percent"] == -5.0
    assert february["trade_count"] == 1
    assert february["win_rate"] == 0.0
    assert pd.isna(february["win_loss_ratio"])
    assert pd.isna(february["adjusted_win_loss_ratio"])
    assert tracker["trade_count"].sum() == calculate_trade_metrics(closed_trades)["trade_count"]


def test_monthly_trade_tracker_coerces_all_empty_loss_columns_to_numeric():
    closed_trades = pd.DataFrame(
        [_closed_trade_row("WIN", "2026-03-05", 12.0)],
        columns=CLOSED_TRADE_COLUMNS,
    )

    tracker = calculate_monthly_trade_tracker(closed_trades)

    assert all(is_numeric_dtype(tracker[column]) for column in MONTHLY_TRACKER_METRIC_COLUMNS)
    assert tracker["average_loss_percent"].isna().all()
    assert tracker["largest_loss_percent"].isna().all()
    assert tracker["average_loss_hold_days"].isna().all()


def test_monthly_trade_tracker_assigns_partial_exit_trade_to_final_exit_month():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-12-20", "symbol": "CROSS", "trans_code": "Buy", "quantity": 10, "price": 100},
            {"activity_date": "2026-12-28", "symbol": "CROSS", "trans_code": "Sell", "quantity": 5, "price": 110},
            {"activity_date": "2027-01-04", "symbol": "CROSS", "trans_code": "Sell", "quantity": 5, "price": 90},
        ]
    )
    planned_stops = pd.DataFrame(
        [{"symbol": "CROSS", "buy_date": "2026-12-20", "quantity": 10, "planned_stop": 95, "strategy": "EP"}],
        columns=PLANNED_STOP_COLUMNS,
    )

    closed_trades = derive_fifo_trades(transactions, planned_stops).closed_trades
    tracker_2026 = calculate_monthly_trade_tracker(filter_frame_by_year(closed_trades, 2026, "sell_date"))
    tracker_2027 = calculate_monthly_trade_tracker(filter_frame_by_year(closed_trades, 2027, "sell_date"))

    assert len(closed_trades) == 1
    assert tracker_2026["trade_count"].sum() == 0
    assert tracker_2027.iloc[0]["trade_count"] == 1


def test_trade_return_distribution_uses_zero_aligned_bins_and_excludes_invalid_returns():
    returns = [-4.0, -2.0, -0.01, 0.0, 1.99, 2.0, 4.34, float("inf"), None]
    closed_trades = pd.DataFrame(
        [
            _closed_trade_row(
                f"T{index}",
                f"2026-04-{index + 1:02d}",
                return_percent,
                realized_pnl=10.0 if return_percent is None or return_percent >= 0 else -10.0,
            )
            for index, return_percent in enumerate(returns)
        ],
        columns=CLOSED_TRADE_COLUMNS,
    )

    distribution = calculate_trade_return_distribution(closed_trades)

    assert distribution.completed_trade_count == 9
    assert distribution.valid_return_count == 7
    assert distribution.bins.to_dict("records") == [
        {
            "bin_start": -4.0,
            "bin_end": -2.0,
            "bin_label": "[-4%, -2%)",
            "trade_count": 1,
            "overflow": None,
        },
        {
            "bin_start": -2.0,
            "bin_end": 0.0,
            "bin_label": "[-2%, 0%)",
            "trade_count": 2,
            "overflow": None,
        },
        {
            "bin_start": 0.0,
            "bin_end": 2.0,
            "bin_label": "[0%, 2%)",
            "trade_count": 2,
            "overflow": None,
        },
        {
            "bin_start": 2.0,
            "bin_end": 4.0,
            "bin_label": "[2%, 4%)",
            "trade_count": 1,
            "overflow": None,
        },
        {
            "bin_start": 4.0,
            "bin_end": 6.0,
            "bin_label": "[4%, 6%)",
            "trade_count": 1,
            "overflow": None,
        },
    ]


def test_trade_return_distribution_groups_automatic_outliers_into_end_bins():
    returns = [-72.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 46.0]
    closed_trades = pd.DataFrame(
        [
            _closed_trade_row(
                f"T{index}",
                f"2026-06-{index + 1:02d}",
                return_percent,
            )
            for index, return_percent in enumerate(returns)
        ],
        columns=CLOSED_TRADE_COLUMNS,
    )

    distribution = calculate_trade_return_distribution(closed_trades)

    lower_overflow = distribution.bins.iloc[0]
    upper_overflow = distribution.bins.iloc[-1]
    assert lower_overflow.to_dict() == {
        "bin_start": -20.0,
        "bin_end": -18.0,
        "bin_label": "< -18%",
        "trade_count": 1,
        "overflow": "lower",
    }
    assert upper_overflow.to_dict() == {
        "bin_start": 20.0,
        "bin_end": 22.0,
        "bin_label": "≥ 20%",
        "trade_count": 1,
        "overflow": "upper",
    }
    assert distribution.bins["trade_count"].sum() == distribution.valid_return_count == len(returns)


def test_trade_return_distribution_fills_empty_bins_inside_the_range():
    # A gap left as a missing row lets the chart's band axis close it and stand two
    # non-adjacent return ranges side by side.
    returns = [-10.0, -10.5, -9.0, -1.0, 0.0, 1.0, 0.5, -0.5]
    closed_trades = pd.DataFrame(
        [
            _closed_trade_row(f"T{index}", f"2026-10-{index + 1:02d}", return_percent)
            for index, return_percent in enumerate(returns)
        ],
        columns=CLOSED_TRADE_COLUMNS,
    )

    distribution = calculate_trade_return_distribution(closed_trades)

    bin_starts = distribution.bins["bin_start"].tolist()
    assert bin_starts == [-12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0]
    empty_bins = distribution.bins.loc[distribution.bins["trade_count"] == 0]
    assert empty_bins["bin_start"].tolist() == [-8.0, -6.0, -4.0]
    assert distribution.bins["trade_count"].sum() == distribution.valid_return_count == len(returns)


def test_trade_return_distribution_keeps_the_empty_bin_beside_a_lower_end_bar():
    returns = [-72.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 46.0]
    closed_trades = pd.DataFrame(
        [
            _closed_trade_row(f"T{index}", f"2026-11-{index + 1:02d}", return_percent)
            for index, return_percent in enumerate(returns)
        ],
        columns=CLOSED_TRADE_COLUMNS,
    )

    distribution = calculate_trade_return_distribution(closed_trades)

    lower_overflow = distribution.bins.iloc[0]
    first_core_bin = distribution.bins.iloc[1]
    assert lower_overflow["overflow"] == "lower"
    # The core range starts at the fence, not at the first bin that happens to hold a trade.
    assert first_core_bin["bin_start"] == lower_overflow["bin_end"] == -18.0
    assert first_core_bin["trade_count"] == 0
    assert distribution.bins["trade_count"].sum() == distribution.valid_return_count == len(returns)


def test_trade_return_distribution_keeps_full_range_for_small_or_zero_iqr_samples():
    small_sample = pd.DataFrame(
        [
            _closed_trade_row("LOW", "2026-07-01", -72.0),
            _closed_trade_row("MID", "2026-07-02", 0.0),
            _closed_trade_row("HIGH", "2026-07-03", 2.0),
        ],
        columns=CLOSED_TRADE_COLUMNS,
    )
    zero_iqr_sample = pd.DataFrame(
        [
            *[_closed_trade_row(f"FLAT-{index}", f"2026-08-{index + 1:02d}", 0.0) for index in range(4)],
            _closed_trade_row("HIGH", "2026-08-05", 72.0),
        ],
        columns=CLOSED_TRADE_COLUMNS,
    )

    small_distribution = calculate_trade_return_distribution(small_sample)
    zero_iqr_distribution = calculate_trade_return_distribution(zero_iqr_sample)

    assert small_distribution.bins["overflow"].isna().all()
    assert small_distribution.bins.iloc[0]["bin_start"] == -72.0
    assert zero_iqr_distribution.bins["overflow"].isna().all()
    assert zero_iqr_distribution.bins.iloc[-1]["bin_start"] == 72.0
    # Without a fence the raw extremes set the range, so filling it spans many empty bins.
    assert len(small_distribution.bins) == 38
    assert len(zero_iqr_distribution.bins) == 37
    for distribution in (small_distribution, zero_iqr_distribution):
        assert distribution.bins["trade_count"].sum() == distribution.valid_return_count


def test_trade_return_distribution_captions_cover_all_partial_and_no_valid_returns():
    all_valid = calculate_trade_return_distribution(
        pd.DataFrame([_closed_trade_row("ONE", "2026-05-01", 5.0)], columns=CLOSED_TRADE_COLUMNS)
    )
    partially_valid = calculate_trade_return_distribution(
        pd.DataFrame(
            [
                _closed_trade_row("VALID", "2026-05-01", 5.0),
                _closed_trade_row("MISSING", "2026-05-02", None, realized_pnl=10.0),
            ],
            columns=CLOSED_TRADE_COLUMNS,
        )
    )
    no_valid = calculate_trade_return_distribution(
        pd.DataFrame(
            [_closed_trade_row("MISSING", "2026-05-02", None, realized_pnl=10.0)],
            columns=CLOSED_TRADE_COLUMNS,
        )
    )

    explanation = "Each bar counts completed trades in a 2% point return range."
    # A complete distribution says nothing about counts; the bars already cover every trade.
    assert trade_return_distribution_caption(all_valid) == explanation
    assert trade_return_distribution_caption(partially_valid) == (
        f"{explanation} Includes 1 of 2 completed trades with valid return data."
    )
    assert trade_return_distribution_caption(no_valid) == "No completed trades have valid return data."


def test_trade_return_distribution_caption_explains_overflow_bins():
    closed_trades = pd.DataFrame(
        [
            _closed_trade_row(
                f"T{index}",
                f"2026-09-{index + 1:02d}",
                return_percent,
            )
            for index, return_percent in enumerate([-72.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
        ],
        columns=CLOSED_TRADE_COLUMNS,
    )

    caption = trade_return_distribution_caption(calculate_trade_return_distribution(closed_trades))

    assert "Extreme returns are grouped into labeled end bars." in caption
    assert "Includes" not in caption


def test_monthly_tracker_state_checks_empty_data_before_scope():
    empty = pd.DataFrame(columns=CLOSED_TRADE_COLUMNS)
    available = pd.DataFrame([_closed_trade_row("ONE", "2026-05-01", 5.0)], columns=CLOSED_TRADE_COLUMNS)

    assert monthly_tracker_state(empty, ALL_TIME_LABEL) == MONTHLY_TRACKER_HIDDEN
    assert monthly_tracker_state(available, ALL_TIME_LABEL) == MONTHLY_TRACKER_SELECT_YEAR
    assert monthly_tracker_state(available, "2026") == MONTHLY_TRACKER_READY


def _trade_row(strategy: str, sell_date: str, realized_pnl: float) -> dict[str, object]:
    return {
        "symbol": strategy,
        "buy_date": "2025-01-01",
        "sell_date": sell_date,
        "quantity": 10,
        "planned_stop": 9.0,
        "strategy": strategy,
        "atr": 2.0,
        "market_regime": "GO",
        "buy_price": 10.0,
        "buy_amount": 100.0,
        "sell_price": 15.0 if realized_pnl > 0 else 5.0,
        "sell_amount": 100.0 + realized_pnl,
        "realized_pnl": realized_pnl,
        "realized_pnl_percent": realized_pnl,
        "hold_days": 1,
    }


def _closed_trade_row(
    symbol: str,
    sell_date: str,
    return_percent: float | None,
    *,
    realized_pnl: float | None = None,
    hold_days: int = 1,
) -> dict[str, object]:
    realized_pnl = return_percent if realized_pnl is None else realized_pnl
    return {
        "symbol": symbol,
        "buy_date": "2026-01-01",
        "sell_date": sell_date,
        "quantity": 1,
        "planned_stop": 90.0,
        "strategy": "EP",
        "atr": 2.0,
        "market_regime": "GO",
        "buy_price": 100.0,
        "buy_amount": 100.0,
        "sell_price": 100.0 + float(realized_pnl or 0.0),
        "sell_amount": 100.0 + float(realized_pnl or 0.0),
        "realized_pnl": realized_pnl,
        "realized_pnl_percent": return_percent,
        "hold_days": hold_days,
        "num_buy_fills": 1,
        "is_pyramided": False,
        "entry_feature_basis": "single_buy_prior_bar",
    }
