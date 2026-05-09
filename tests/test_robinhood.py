from datetime import date
from io import StringIO

import pandas as pd

from stock_calculator.robinhood import (
    PLANNED_STOP_COLUMNS,
    calculate_strategy_metrics,
    calculate_total_realized_pnl,
    calculate_trade_metrics,
    derive_fifo_trades,
    display_trade_context_frame,
    parse_robinhood_csv,
)


def robinhood_csv(body: str) -> StringIO:
    return StringIO(
        '"Activity Date","Process Date","Settle Date","Instrument","Description","Trans Code","Quantity","Price","Amount"\n'
        + body
    )


def test_parse_robinhood_csv_handles_multiline_descriptions_footer_and_money():
    result = parse_robinhood_csv(
        robinhood_csv(
            '"4/15/2026","4/15/2026","4/16/2026","OKLO","Oklo\n'
            'CUSIP: 02156V109","Buy","7","$63.84","($446.85)"\n'
            '"4/8/2026","4/8/2026","4/8/2026","SEI","Stock Lending","SLIP","","","$0.01"\n'
            '"2/4/2026","2/4/2026","2/5/2026","SION","Sionna Therapeutics\n'
            'CUSIP: 829401108","Sell","30","$42.54","$1,276.20"\n'
            '"","","","","","","","","","The data provided is for informational purposes only."\n'
        )
    )

    assert result.transactions["symbol"].tolist() == ["OKLO", "SION"]
    assert result.transactions["quantity"].tolist() == [7.0, 30.0]
    assert result.transactions["amount"].tolist() == [-446.85, 1276.20]
    assert len(result.ignored_rows) == 1
    assert result.malformed_rows.empty


def test_parse_robinhood_csv_reports_genuinely_malformed_non_footer_rows():
    result = parse_robinhood_csv(
        robinhood_csv(
            '"4/15/2026","4/15/2026","4/16/2026","OKLO","Oklo CUSIP: 02156V109","Buy","7","$63.84","($446.85)"\n'
            '"4/16/2026","4/16/2026","4/17/2026","BROKEN"\n'
            '"","","","","","","","","","The data provided is for informational purposes only."\n'
        )
    )

    assert result.transactions["symbol"].tolist() == ["OKLO"]
    assert len(result.malformed_rows) == 1
    assert result.malformed_rows.iloc[0]["row_number"] == 3


def test_display_trade_context_frame_shows_missing_atr_and_regime_as_na_without_mutating_source():
    source = pd.DataFrame(
        [
            {"symbol": "AAPL", "atr": None, "market_regime": ""},
            {"symbol": "MSFT", "atr": 2.5, "market_regime": "SELECTIVE GO"},
        ]
    )

    displayed = display_trade_context_frame(source)

    assert displayed[["atr", "market_regime"]].to_dict("records") == [
        {"atr": "N/A", "market_regime": "N/A"},
        {"atr": "2.50", "market_regime": "SELECTIVE GO"},
    ]
    assert pd.isna(source.iloc[0]["atr"])
    assert source.iloc[0]["market_regime"] == ""


def test_derive_fifo_trades_matches_full_sell_with_planned_stop():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-15", "symbol": "OKLO", "trans_code": "Buy", "quantity": 7, "price": 63.84},
            {"activity_date": "2026-04-16", "symbol": "OKLO", "trans_code": "Sell", "quantity": 7, "price": 60.90},
        ]
    )
    planned = pd.DataFrame(
        [
            {
                "symbol": "OKLO",
                "buy_date": "2026-04-15",
                "quantity": 7,
                "planned_stop": 58.25,
                "strategy": "EP",
                "atr": 4.5,
                "market_regime": "SELECTIVE GO",
            }
        ],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    row = result.closed_trades.iloc[0]
    assert row["planned_stop"] == 58.25
    assert row["strategy"] == "EP"
    assert row["atr"] == 4.5
    assert row["market_regime"] == "SELECTIVE GO"
    assert row["buy_price"] == 63.84
    assert row["sell_price"] == 60.90
    assert row["realized_pnl"] == -20.58
    assert row["realized_pnl_percent"] == -4.61
    assert result.open_lots.empty
    assert result.missing_planned_stops == 0


def test_derive_fifo_trades_carries_planned_atr_and_market_regime_into_closed_trades_and_open_lots():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-01", "symbol": "AAPL", "trans_code": "Buy", "quantity": 10, "price": 100},
            {"activity_date": "2026-04-02", "symbol": "AAPL", "trans_code": "Buy", "quantity": 5, "price": 101},
            {"activity_date": "2026-04-03", "symbol": "AAPL", "trans_code": "Sell", "quantity": 10, "price": 95},
        ]
    )
    planned = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "buy_date": "2026-04-01",
                "quantity": 10,
                "planned_stop": 96,
                "strategy": "EP",
                "atr": 2.5,
                "market_regime": "GO",
            },
            {
                "symbol": "AAPL",
                "buy_date": "2026-04-02",
                "quantity": 5,
                "planned_stop": 97,
                "strategy": "BO",
                "atr": 3.5,
                "market_regime": "NO-GO",
            },
        ],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    assert result.closed_trades.iloc[0]["atr"] == 2.5
    assert result.closed_trades.iloc[0]["market_regime"] == "GO"
    assert result.exit_matches.iloc[0]["atr"] == 2.5
    assert result.exit_matches.iloc[0]["market_regime"] == "GO"
    assert result.open_lots.iloc[0]["atr"] == 3.5
    assert result.open_lots.iloc[0]["market_regime"] == "NO-GO"


def test_derive_fifo_trades_supports_partial_sells_and_open_lots():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-08", "symbol": "AEHR", "trans_code": "Buy", "quantity": 4, "price": 59.78},
            {"activity_date": "2026-04-08", "symbol": "AEHR", "trans_code": "Buy", "quantity": 4, "price": 59.38},
            {"activity_date": "2026-04-09", "symbol": "AEHR", "trans_code": "Sell", "quantity": 6, "price": 67.43},
        ]
    )

    result = derive_fifo_trades(transactions, pd.DataFrame())

    assert result.exit_matches["quantity"].tolist() == [4, 2]
    assert result.exit_matches["buy_price"].tolist() == [59.78, 59.38]
    assert result.closed_trades["quantity"].tolist() == [4]
    assert result.open_lots.iloc[0]["quantity"] == 2
    assert result.open_lots.iloc[0]["buy_price"] == 59.38
    assert result.missing_planned_stops == 2


def test_derive_fifo_trades_aggregates_split_exits_into_one_closed_trade():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-01", "symbol": "AAPL", "trans_code": "Buy", "quantity": 10, "price": 100},
            {"activity_date": "2026-04-03", "symbol": "AAPL", "trans_code": "Sell", "quantity": 5, "price": 110},
            {"activity_date": "2026-04-06", "symbol": "AAPL", "trans_code": "Sell", "quantity": 5, "price": 90},
        ]
    )
    planned = pd.DataFrame(
        [{"symbol": "AAPL", "buy_date": "2026-04-01", "quantity": 10, "planned_stop": 95, "strategy": "EP"}],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)
    metrics = calculate_trade_metrics(result.closed_trades)

    assert len(result.exit_matches) == 2
    assert len(result.closed_trades) == 1
    row = result.closed_trades.iloc[0]
    assert row["quantity"] == 10
    assert row["buy_amount"] == 1000
    assert row["sell_amount"] == 1000
    assert row["realized_pnl"] == 0
    assert row["realized_pnl_percent"] == 0
    assert row["sell_price"] == 100
    assert row["sell_date"] == "2026-04-06"
    assert row["hold_days"] == 3
    assert row["strategy"] == "EP"
    assert metrics["trade_count"] == 1
    assert metrics["breakeven_count"] == 1


def test_derive_fifo_trades_excludes_partially_open_lots_from_closed_trades():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-01", "symbol": "AAPL", "trans_code": "Buy", "quantity": 10, "price": 100},
            {"activity_date": "2026-04-03", "symbol": "AAPL", "trans_code": "Sell", "quantity": 5, "price": 110},
        ]
    )

    result = derive_fifo_trades(transactions, pd.DataFrame())

    assert len(result.exit_matches) == 1
    assert result.closed_trades.empty
    assert result.open_lots.iloc[0]["quantity"] == 5


def test_derive_fifo_trades_matches_same_day_buys_by_exact_quantity():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-08", "symbol": "AEHR", "trans_code": "Buy", "quantity": 3, "price": 59.78},
            {"activity_date": "2026-04-08", "symbol": "AEHR", "trans_code": "Buy", "quantity": 4, "price": 59.38},
            {"activity_date": "2026-04-09", "symbol": "AEHR", "trans_code": "Sell", "quantity": 7, "price": 67.43},
        ]
    )
    planned = pd.DataFrame(
        [
            {"symbol": "AEHR", "buy_date": "2026-04-08", "quantity": 4, "planned_stop": 55.25, "strategy": "5% BO"},
            {"symbol": "AEHR", "buy_date": "2026-04-08", "quantity": 3, "planned_stop": 56.75, "strategy": "BO"},
        ],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    assert result.closed_trades["planned_stop"].tolist() == [56.75, 55.25]
    assert result.closed_trades["strategy"].tolist() == ["BO", "5% BO"]
    assert result.exit_matches["planned_stop"].tolist() == [56.75, 55.25]
    assert result.exit_matches["strategy"].tolist() == ["BO", "5% BO"]
    assert result.missing_planned_stops == 0


def test_derive_fifo_trades_treats_duplicate_planned_stop_keys_as_missing():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-08", "symbol": "AEHR", "trans_code": "Buy", "quantity": 4, "price": 59.78},
            {"activity_date": "2026-04-09", "symbol": "AEHR", "trans_code": "Sell", "quantity": 4, "price": 67.43},
        ]
    )
    planned = pd.DataFrame(
        [
            {"symbol": "AEHR", "buy_date": "2026-04-08", "quantity": 4, "planned_stop": 55.25},
            {"symbol": "AEHR", "buy_date": "2026-04-08", "quantity": 4, "planned_stop": 56.75},
        ],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    assert pd.isna(result.closed_trades.iloc[0]["planned_stop"])
    assert result.missing_planned_stops == 1


def test_derive_fifo_trades_accepts_duplicate_planned_stop_keys_when_stop_matches():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-02-04", "symbol": "SMCI", "trans_code": "Buy", "quantity": 10, "price": 34.24},
            {"activity_date": "2026-02-04", "symbol": "SMCI", "trans_code": "Buy", "quantity": 10, "price": 34.48},
            {"activity_date": "2026-02-05", "symbol": "SMCI", "trans_code": "Sell", "quantity": 20, "price": 31.64},
        ]
    )
    planned = pd.DataFrame(
        [
            {"symbol": "SMCI", "buy_date": "2026-02-04", "quantity": 10, "planned_stop": 31.65},
            {"symbol": "SMCI", "buy_date": "2026-02-04", "quantity": 10, "planned_stop": 31.65},
        ],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    assert result.closed_trades["planned_stop"].tolist() == [31.65, 31.65]
    assert result.missing_planned_stops == 0


def test_derive_fifo_trades_treats_conflicting_strategy_keys_as_unclassified():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-08", "symbol": "AEHR", "trans_code": "Buy", "quantity": 4, "price": 59.78},
            {"activity_date": "2026-04-09", "symbol": "AEHR", "trans_code": "Sell", "quantity": 4, "price": 67.43},
        ]
    )
    planned = pd.DataFrame(
        [
            {"symbol": "AEHR", "buy_date": "2026-04-08", "quantity": 4, "planned_stop": 55.25, "strategy": "EP"},
            {"symbol": "AEHR", "buy_date": "2026-04-08", "quantity": 4, "planned_stop": 55.25, "strategy": "BO"},
        ],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    assert result.closed_trades.iloc[0]["planned_stop"] == 55.25
    assert result.closed_trades.iloc[0]["strategy"] == ""


def test_derive_fifo_trades_processes_same_day_buys_before_sells():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-03-25", "symbol": "SPRB", "trans_code": "Sell", "quantity": 6, "price": 67.32},
            {"activity_date": "2026-03-25", "symbol": "SPRB", "trans_code": "Buy", "quantity": 6, "price": 73.49},
        ]
    )

    result = derive_fifo_trades(transactions, pd.DataFrame())

    assert len(result.closed_trades) == 1
    assert result.unmatched_sells.empty
    assert result.closed_trades.iloc[0]["realized_pnl"] == -37.02


def test_derive_fifo_trades_reports_unmatched_sells():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-16", "symbol": "OKLO", "trans_code": "Sell", "quantity": 7, "price": 60.90},
        ]
    )

    result = derive_fifo_trades(transactions, pd.DataFrame())

    assert result.closed_trades.empty
    assert result.unmatched_sells.iloc[0]["symbol"] == "OKLO"
    assert result.unmatched_sells.iloc[0]["quantity"] == 7


def test_derive_fifo_trades_calculates_open_hold_days():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-15", "symbol": "IONQ", "trans_code": "Buy", "quantity": 9, "price": 39.68},
        ]
    )

    result = derive_fifo_trades(transactions, pd.DataFrame(), as_of=date(2026, 4, 20))

    assert result.open_lots.iloc[0]["cost_basis"] == 357.12
    assert result.open_lots.iloc[0]["hold_days"] == 3


def test_derive_fifo_trades_copies_strategy_to_open_lots():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-15", "symbol": "IONQ", "trans_code": "Buy", "quantity": 9, "price": 39.68},
        ]
    )
    planned = pd.DataFrame(
        [{"symbol": "IONQ", "buy_date": "2026-04-15", "quantity": 9, "planned_stop": 35.50, "strategy": "5% BO"}],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    assert result.open_lots.iloc[0]["strategy"] == "5% BO"


def test_calculate_trade_metrics_from_closed_trades():
    closed_trades = pd.DataFrame(
        [
            {
                "sell_date": "2026-04-01",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": 20,
                "realized_pnl_percent": 10,
                "hold_days": 3,
            },
            {
                "sell_date": "2026-04-02",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": 40,
                "realized_pnl_percent": 20,
                "hold_days": 5,
            },
            {
                "sell_date": "2026-04-03",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": -10,
                "realized_pnl_percent": -5,
                "hold_days": 2,
            },
            {
                "sell_date": "2026-04-04",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": -30,
                "realized_pnl_percent": -15,
                "hold_days": 4,
            },
            {
                "sell_date": "2026-04-05",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": 10,
                "realized_pnl_percent": 5,
                "hold_days": 1,
            },
        ]
    )

    metrics = calculate_trade_metrics(closed_trades)

    assert metrics["trade_count"] == 5
    assert metrics["win_count"] == 3
    assert metrics["loss_count"] == 2
    assert metrics["breakeven_count"] == 0
    assert metrics["win_rate"] == 60.0
    assert metrics["win_loss_ratio"] == 1.17
    assert metrics["average_win_r"] == 2.33
    assert metrics["average_loss_r"] == -2.0
    assert metrics["r_ratio"] == 1.17
    assert metrics["expectancy_r"] == 0.6
    assert metrics["expectancy"] == 6.0
    assert metrics["profit_factor"] == 1.75
    assert metrics["average_win"] == 23.33
    assert metrics["average_win_percent"] == 11.67
    assert metrics["average_win_hold"] == 3.0
    assert metrics["win_streak"] == 2
    assert metrics["top_win"] == 40.0
    assert metrics["average_loss"] == -20.0
    assert metrics["average_loss_percent"] == -10.0
    assert metrics["average_loss_hold"] == 3.0
    assert metrics["loss_streak"] == 2
    assert metrics["top_loss"] == -30.0


def test_calculate_trade_metrics_excludes_invalid_stop_data_from_r_metrics():
    closed_trades = pd.DataFrame(
        [
            {
                "sell_date": "2026-04-01",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": 20,
                "realized_pnl_percent": 20,
                "hold_days": 1,
            },
            {
                "sell_date": "2026-04-02",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": -10,
                "realized_pnl_percent": -10,
                "hold_days": 1,
            },
            {
                "sell_date": "2026-04-03",
                "quantity": 1,
                "planned_stop": None,
                "buy_price": 100,
                "realized_pnl": 100,
                "realized_pnl_percent": 100,
                "hold_days": 1,
            },
            {
                "sell_date": "2026-04-04",
                "quantity": 1,
                "planned_stop": 101,
                "buy_price": 100,
                "realized_pnl": -50,
                "realized_pnl_percent": -50,
                "hold_days": 1,
            },
        ]
    )

    metrics = calculate_trade_metrics(closed_trades)

    assert metrics["trade_count"] == 4
    assert metrics["average_win"] == 60.0
    assert metrics["average_loss"] == -30.0
    assert metrics["average_win_r"] == 2.0
    assert metrics["average_loss_r"] == -1.0
    assert metrics["r_ratio"] == 2.0
    assert metrics["expectancy_r"] == 0.5


def test_calculate_trade_metrics_counts_wins_losses_and_breakevens_from_numeric_pnl():
    closed_trades = pd.DataFrame(
        [
            {
                "sell_date": "2026-04-01",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": 20,
                "realized_pnl_percent": 10,
                "hold_days": 1,
            },
            {
                "sell_date": "2026-04-02",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": -5,
                "realized_pnl_percent": -2.5,
                "hold_days": 1,
            },
            {
                "sell_date": "2026-04-03",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": 0,
                "realized_pnl_percent": 0,
                "hold_days": 1,
            },
            {
                "sell_date": "2026-04-04",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": "0.00",
                "realized_pnl_percent": 0,
                "hold_days": 1,
            },
            {"sell_date": "2026-04-05", "realized_pnl": None, "realized_pnl_percent": None, "hold_days": 1},
            {"sell_date": "2026-04-06", "realized_pnl": "not a number", "realized_pnl_percent": None, "hold_days": 1},
        ]
    )

    metrics = calculate_trade_metrics(closed_trades)

    assert metrics["trade_count"] == 4
    assert metrics["win_count"] == 1
    assert metrics["loss_count"] == 1
    assert metrics["breakeven_count"] == 2
    assert metrics["expectancy_r"] == 0.38


def test_calculate_trade_metrics_empty_closed_trades_are_safe():
    metrics = calculate_trade_metrics(pd.DataFrame())

    assert metrics["trade_count"] == 0
    assert metrics["win_count"] == 0
    assert metrics["loss_count"] == 0
    assert metrics["breakeven_count"] == 0
    assert metrics["win_rate"] is None
    assert metrics["profit_factor"] is None
    assert metrics["average_win_r"] is None
    assert metrics["average_loss_r"] is None
    assert metrics["r_ratio"] is None
    assert metrics["expectancy_r"] is None
    assert metrics["win_streak"] == 0
    assert metrics["loss_streak"] == 0


def test_calculate_total_realized_pnl_sums_numeric_closed_trade_pnl():
    closed_trades = pd.DataFrame(
        [
            {"realized_pnl": "20.125"},
            {"realized_pnl": -10},
            {"realized_pnl": "not a number"},
        ]
    )

    assert calculate_total_realized_pnl(closed_trades) == 10.12


def test_calculate_total_realized_pnl_empty_or_missing_column_is_zero():
    assert calculate_total_realized_pnl(pd.DataFrame()) == 0.0
    assert calculate_total_realized_pnl(pd.DataFrame([{"symbol": "AAPL"}])) == 0.0


def test_calculate_strategy_metrics_groups_closed_trades_by_strategy():
    closed_trades = pd.DataFrame(
        [
            {
                "strategy": "EP",
                "sell_date": "2026-04-01",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": 20,
                "realized_pnl_percent": 20,
                "hold_days": 1,
            },
            {
                "strategy": "EP",
                "sell_date": "2026-04-02",
                "quantity": 1,
                "planned_stop": 90,
                "buy_price": 100,
                "realized_pnl": -10,
                "realized_pnl_percent": -10,
                "hold_days": 3,
            },
            {
                "strategy": "",
                "sell_date": "2026-04-03",
                "quantity": 1,
                "planned_stop": 40,
                "buy_price": 50,
                "realized_pnl": 10,
                "realized_pnl_percent": 20,
                "hold_days": 2,
            },
        ]
    )

    result = calculate_strategy_metrics(closed_trades)

    assert result[["strategy", "trade_count", "total_realized_pnl", "win_rate", "expectancy"]].to_dict("records") == [
        {"strategy": "EP", "trade_count": 2, "total_realized_pnl": 10.0, "win_rate": 50.0, "expectancy": 5.0},
        {
            "strategy": "Unclassified",
            "trade_count": 1,
            "total_realized_pnl": 10.0,
            "win_rate": 100.0,
            "expectancy": 10.0,
        },
    ]


def _closed_trade(strategy="EP", sell_date="2026-04-01", r_multiple=0.0, planned_stop=90, buy_price=100):
    return {
        "strategy": strategy,
        "sell_date": sell_date,
        "quantity": 1,
        "planned_stop": planned_stop,
        "buy_price": buy_price,
        "realized_pnl": (buy_price - planned_stop) * r_multiple,
        "realized_pnl_percent": r_multiple * 10,
        "hold_days": 1,
    }


def test_calculate_strategy_metrics_adds_rolling_10r_mode_and_action_from_r_multiples():
    closed_trades = pd.DataFrame(
        [_closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=0.35) for day in range(1, 11)]
    )

    result = calculate_strategy_metrics(closed_trades)

    assert result[["rolling_10r_exp", "mode", "action"]].to_dict("records") == [
        {"rolling_10r_exp": "+0.35R", "mode": "Working", "action": "Normal size"}
    ]


def test_calculate_strategy_metrics_uses_latest_10_closed_trades_by_sell_date():
    old_large_winner = _closed_trade(sell_date="2026-04-01", r_multiple=10)
    latest_weak_trades = [
        _closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=-0.1)
        for day in range(2, 12)
    ]
    closed_trades = pd.DataFrame([*latest_weak_trades, old_large_winner])

    result = calculate_strategy_metrics(closed_trades)

    assert result.iloc[0]["rolling_10r_exp"] == "-0.10R"
    assert result.iloc[0]["mode"] == "Weak"
    assert result.iloc[0]["action"] == "Quarter size"


def test_calculate_strategy_metrics_requires_10_closed_trades_for_rolling_10r():
    closed_trades = pd.DataFrame(
        [_closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=1) for day in range(1, 10)]
    )

    result = calculate_strategy_metrics(closed_trades)

    assert result.iloc[0]["rolling_10r_exp"] == "N/A"
    assert result.iloc[0]["mode"] == "Unknown"
    assert result.iloc[0]["action"] == "Tiny size only"


def test_calculate_strategy_metrics_requires_latest_10_trades_to_have_valid_r_risk():
    trades = [_closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=0.5) for day in range(1, 11)]
    trades[-1]["planned_stop"] = 100
    closed_trades = pd.DataFrame(trades)

    result = calculate_strategy_metrics(closed_trades)

    assert result.iloc[0]["rolling_10r_exp"] == "N/A"
    assert result.iloc[0]["mode"] == "Unknown"
    assert result.iloc[0]["action"] == "Tiny size only"


def test_calculate_strategy_metrics_maps_rolling_10r_thresholds():
    cases = [
        (0.30, "+0.30R", "Caution", "Half size"),
        (0.00, "+0.00R", "Caution", "Half size"),
        (-0.25, "-0.25R", "Weak", "Quarter size"),
        (-0.30, "-0.30R", "Failing", "Probe only / pause"),
    ]

    for r_multiple, expected_exp, expected_mode, expected_action in cases:
        closed_trades = pd.DataFrame(
            [_closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=r_multiple) for day in range(1, 11)]
        )

        result = calculate_strategy_metrics(closed_trades)

        assert result.iloc[0]["rolling_10r_exp"] == expected_exp
        assert result.iloc[0]["mode"] == expected_mode
        assert result.iloc[0]["action"] == expected_action


def test_calculate_strategy_metrics_calculates_rolling_10r_per_strategy():
    ep_trades = [_closed_trade(strategy="EP", sell_date=f"2026-04-{day:02d}", r_multiple=0.4) for day in range(1, 11)]
    bo_trades = [_closed_trade(strategy="BO", sell_date=f"2026-04-{day:02d}", r_multiple=-0.3) for day in range(1, 11)]

    result = calculate_strategy_metrics(pd.DataFrame([*bo_trades, *ep_trades]))

    assert result[["strategy", "rolling_10r_exp", "mode", "action"]].to_dict("records") == [
        {"strategy": "EP", "rolling_10r_exp": "+0.40R", "mode": "Working", "action": "Normal size"},
        {"strategy": "BO", "rolling_10r_exp": "-0.30R", "mode": "Failing", "action": "Probe only / pause"},
    ]
