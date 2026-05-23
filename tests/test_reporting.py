from __future__ import annotations

from datetime import date

import pandas as pd

from stock_calculator.reporting import (
    ALL_TIME_LABEL,
    calculate_strategy_report_metrics,
    default_report_year_label,
    filter_frame_by_year,
    report_year_options,
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
    assert row["action"] == "Normal size"
    assert row["rolling_mode_exp"] == "+1.00R"


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
