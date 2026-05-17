from datetime import date
from io import StringIO

import pandas as pd

from stock_calculator.robinhood import (
    PLANNED_STOP_COLUMNS,
    calculate_strategy_attribution,
    calculate_strategy_metrics,
    calculate_total_realized_pnl,
    calculate_trade_metrics,
    derive_fifo_trades,
    display_trade_context_frame,
    parse_robinhood_csv,
    portfolio_attribution_note,
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


def test_derive_fifo_trades_groups_same_price_split_buys_into_one_logical_closed_trade():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-05-11", "symbol": "ROIV", "trans_code": "Buy", "quantity": 8, "price": 29.44},
            {"activity_date": "2026-05-11", "symbol": "ROIV", "trans_code": "Buy", "quantity": 1, "price": 29.44},
            {"activity_date": "2026-05-11", "symbol": "ROIV", "trans_code": "Buy", "quantity": 11, "price": 29.44},
            {"activity_date": "2026-05-12", "symbol": "ROIV", "trans_code": "Sell", "quantity": 20, "price": 32.00},
        ]
    )
    planned = pd.DataFrame(
        [
            {
                "symbol": "ROIV",
                "buy_date": "2026-05-11",
                "quantity": 20,
                "planned_stop": 27.25,
                "strategy": "EP",
                "atr": 2.4,
                "market_regime": "SELECTIVE GO",
            }
        ],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)
    metrics = calculate_trade_metrics(result.closed_trades)

    assert result.exit_matches["quantity"].tolist() == [8, 1, 11]
    assert len(result.closed_trades) == 1
    row = result.closed_trades.iloc[0]
    assert row["symbol"] == "ROIV"
    assert row["quantity"] == 20
    assert row["planned_stop"] == 27.25
    assert row["strategy"] == "EP"
    assert row["atr"] == 2.4
    assert row["market_regime"] == "SELECTIVE GO"
    assert row["buy_price"] == 29.44
    assert row["sell_price"] == 32.00
    assert row["buy_amount"] == 588.80
    assert row["sell_amount"] == 640.00
    assert row["realized_pnl"] == 51.20
    assert metrics["trade_count"] == 1
    assert metrics["win_count"] == 1
    assert result.missing_planned_stops == 0
    assert result.lot_grouping_audit.to_dict("records") == [
        {
            "symbol": "ROIV",
            "buy_date": "2026-05-11",
            "planned_quantity": 20,
            "split_quantities": "8, 1, 11",
            "buy_price": 29.44,
            "planned_stop": 27.25,
            "strategy": "EP",
            "atr": 2.4,
            "market_regime": "SELECTIVE GO",
            "reason": "Same symbol/date/price buy lots summed to one planned position.",
        }
    ]


def test_derive_fifo_trades_exact_matches_win_before_split_grouping():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-05-11", "symbol": "AEHR", "trans_code": "Buy", "quantity": 3, "price": 59.00},
            {"activity_date": "2026-05-11", "symbol": "AEHR", "trans_code": "Buy", "quantity": 4, "price": 59.00},
            {"activity_date": "2026-05-11", "symbol": "AEHR", "trans_code": "Buy", "quantity": 6, "price": 59.00},
            {"activity_date": "2026-05-12", "symbol": "AEHR", "trans_code": "Sell", "quantity": 13, "price": 62.00},
        ]
    )
    planned = pd.DataFrame(
        [
            {"symbol": "AEHR", "buy_date": "2026-05-11", "quantity": 3, "planned_stop": 56.75, "strategy": "BO"},
            {"symbol": "AEHR", "buy_date": "2026-05-11", "quantity": 10, "planned_stop": 55.25, "strategy": "5% BO"},
        ],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    assert result.closed_trades["quantity"].tolist() == [3, 10]
    assert result.closed_trades["planned_stop"].tolist() == [56.75, 55.25]
    assert result.closed_trades["strategy"].tolist() == ["BO", "5% BO"]
    assert result.lot_grouping_audit["split_quantities"].tolist() == ["4, 6"]


def test_derive_fifo_trades_does_not_group_same_day_buys_with_different_prices():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-05-11", "symbol": "ROIV", "trans_code": "Buy", "quantity": 8, "price": 29.44},
            {"activity_date": "2026-05-11", "symbol": "ROIV", "trans_code": "Buy", "quantity": 12, "price": 29.45},
            {"activity_date": "2026-05-12", "symbol": "ROIV", "trans_code": "Sell", "quantity": 20, "price": 32.00},
        ]
    )
    planned = pd.DataFrame(
        [{"symbol": "ROIV", "buy_date": "2026-05-11", "quantity": 20, "planned_stop": 27.25}],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    assert result.closed_trades["quantity"].tolist() == [8, 12]
    assert result.closed_trades["planned_stop"].isna().tolist() == [True, True]
    assert result.missing_planned_stops == 2
    assert result.lot_grouping_audit.empty
    assert result.missing_planned_stop_rows[["status", "symbol", "quantity", "buy_price"]].to_dict("records") == [
        {"status": "Closed", "symbol": "ROIV", "quantity": 8, "buy_price": 29.44},
        {"status": "Closed", "symbol": "ROIV", "quantity": 12, "buy_price": 29.45},
    ]


def test_derive_fifo_trades_keeps_partially_exited_split_group_out_of_closed_trades():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-05-11", "symbol": "ROIV", "trans_code": "Buy", "quantity": 8, "price": 29.44},
            {"activity_date": "2026-05-11", "symbol": "ROIV", "trans_code": "Buy", "quantity": 1, "price": 29.44},
            {"activity_date": "2026-05-11", "symbol": "ROIV", "trans_code": "Buy", "quantity": 11, "price": 29.44},
            {"activity_date": "2026-05-12", "symbol": "ROIV", "trans_code": "Sell", "quantity": 10, "price": 32.00},
        ]
    )
    planned = pd.DataFrame(
        [{"symbol": "ROIV", "buy_date": "2026-05-11", "quantity": 20, "planned_stop": 27.25}],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    assert result.closed_trades.empty
    assert result.exit_matches["quantity"].tolist() == [8, 1, 1]
    assert result.open_lots["quantity"].tolist() == [10]
    assert result.open_lots.iloc[0]["planned_stop"] == 27.25
    assert result.missing_planned_stops == 0
    assert len(result.lot_grouping_audit) == 1


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
    assert result.planned_stop_issues.to_dict("records") == [
        {
            "symbol": "AEHR",
            "buy_date": "2026-04-08",
            "quantity": 4,
            "issue": "Conflicting planned stops for this lot key.",
            "detail": "planned_stop values: 55.25, 56.75",
        }
    ]


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
    assert result.planned_stop_issues.empty


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


def test_calculate_trade_metrics_profit_factor_zero_loss_cases():
    all_winners = pd.DataFrame(
        [
            {"sell_date": "2026-04-01", "realized_pnl": 10, "realized_pnl_percent": 10, "hold_days": 1},
            {"sell_date": "2026-04-02", "realized_pnl": 20, "realized_pnl_percent": 20, "hold_days": 2},
        ]
    )
    only_breakevens = pd.DataFrame(
        [
            {"sell_date": "2026-04-01", "realized_pnl": 0, "realized_pnl_percent": 0, "hold_days": 1},
            {"sell_date": "2026-04-02", "realized_pnl": 0, "realized_pnl_percent": 0, "hold_days": 2},
        ]
    )
    only_losses = pd.DataFrame(
        [
            {"sell_date": "2026-04-01", "realized_pnl": -10, "realized_pnl_percent": -10, "hold_days": 1},
            {"sell_date": "2026-04-02", "realized_pnl": -20, "realized_pnl_percent": -20, "hold_days": 2},
        ]
    )

    assert calculate_trade_metrics(all_winners)["profit_factor"] is None
    assert calculate_trade_metrics(only_breakevens)["profit_factor"] is None
    assert calculate_trade_metrics(only_losses)["profit_factor"] == 0.0


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


def _closed_trade(
    strategy="EP",
    sell_date="2026-04-01",
    r_multiple=0.0,
    planned_stop=90,
    buy_price=100,
    atr=None,
    market_regime="",
    hold_days=1,
):
    return {
        "strategy": strategy,
        "sell_date": sell_date,
        "quantity": 1,
        "planned_stop": planned_stop,
        "atr": atr,
        "market_regime": market_regime,
        "buy_price": buy_price,
        "realized_pnl": (buy_price - planned_stop) * r_multiple,
        "realized_pnl_percent": r_multiple * 10,
        "hold_days": hold_days,
    }


def _closed_trade_window(r_multiples, *, strategy="EP", start="2026-04-01", regimes=None):
    dates = pd.date_range(start=start, periods=len(r_multiples), freq="D")
    regimes = regimes or [""] * len(r_multiples)
    return [
        _closed_trade(
            strategy=strategy,
            sell_date=date.date().isoformat(),
            r_multiple=r_multiple,
            market_regime=regime,
        )
        for date, r_multiple, regime in zip(dates, r_multiples, regimes, strict=True)
    ]


def _held_trade_window(r_multiples, hold_days, *, strategy="EP", start="2026-04-01"):
    dates = pd.date_range(start=start, periods=len(r_multiples), freq="D")
    return [
        _closed_trade(
            strategy=strategy,
            sell_date=date.date().isoformat(),
            r_multiple=r_multiple,
            hold_days=hold_day,
        )
        for date, r_multiple, hold_day in zip(dates, r_multiples, hold_days, strict=True)
    ]


def test_calculate_strategy_metrics_adds_rolling_mode_and_action_from_r_multiples():
    closed_trades = pd.DataFrame(
        [_closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=0.35) for day in range(1, 16)]
    )

    result = calculate_strategy_metrics(closed_trades)

    columns = list(result.columns)
    assert columns.index("mode_adjusted_score") == columns.index("rolling_mode_exp") + 1
    assert result[["rolling_mode_exp", "mode_adjusted_score", "mode", "action"]].to_dict("records") == [
        {
            "rolling_mode_exp": "+0.35R",
            "mode_adjusted_score": "+0.35R",
            "mode": "Working",
            "action": "Normal size",
        }
    ]


def test_calculate_strategy_metrics_keeps_adjusted_score_as_reference_only():
    noisy_positive_window = [1.0] * 8 + [-0.4] * 7
    closed_trades = pd.DataFrame(_closed_trade_window(noisy_positive_window))

    result = calculate_strategy_metrics(closed_trades)

    row = result.iloc[0]
    assert row["rolling_mode_exp"] == "+0.35R"
    assert row["mode_adjusted_score"] == "+0.25R"
    assert row["mode"] == "Working"
    assert row["action"] == "Normal size"


def test_calculate_strategy_metrics_uses_latest_15_closed_trades_by_sell_date():
    old_large_winner = _closed_trade(sell_date="2026-04-01", r_multiple=10)
    latest_weak_trades = [
        _closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=-0.1)
        for day in range(2, 17)
    ]
    closed_trades = pd.DataFrame([*latest_weak_trades, old_large_winner])

    result = calculate_strategy_metrics(closed_trades)

    assert result.iloc[0]["rolling_mode_exp"] == "-0.10R"
    assert result.iloc[0]["mode_adjusted_score"] == "-0.10R"
    assert result.iloc[0]["mode"] == "Weak"
    assert result.iloc[0]["action"] == "Quarter size"


def test_calculate_strategy_metrics_requires_15_closed_trades_for_rolling_mode():
    closed_trades = pd.DataFrame(
        [_closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=1) for day in range(1, 15)]
    )

    result = calculate_strategy_metrics(closed_trades)

    assert result.iloc[0]["rolling_mode_exp"] == "N/A"
    assert result.iloc[0]["mode_adjusted_score"] == "N/A"
    assert result.iloc[0]["mode"] == "Unknown"
    assert result.iloc[0]["action"] == "Tiny size only"


def test_calculate_strategy_metrics_uses_latest_15_valid_r_trades_within_20_trade_lookback():
    trades = [_closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=0.5) for day in range(1, 17)]
    trades[-2]["planned_stop"] = 100
    closed_trades = pd.DataFrame(trades)

    metrics = calculate_strategy_metrics(closed_trades)
    attribution = calculate_strategy_attribution(closed_trades)

    assert metrics.iloc[0]["rolling_mode_exp"] == "+0.50R"
    assert metrics.iloc[0]["mode_adjusted_score"] == "+0.50R"
    assert metrics.iloc[0]["mode"] == "Working"
    assert metrics.iloc[0]["action"] == "Normal size"
    assert attribution.iloc[0]["mode_basis"] == (
        "15 valid R trades from latest 16 closed trades; skipped 1 missing/invalid stop | 15R +0.50R | Adj +0.50R"
    )


def test_calculate_strategy_metrics_stays_unknown_when_20_trade_lookback_has_too_few_valid_r_trades():
    old_valid_trades = [_closed_trade(sell_date=f"2026-03-{day:02d}", r_multiple=2) for day in range(1, 6)]
    recent_valid_trades = [_closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=0.5) for day in range(1, 15)]
    recent_invalid_trades = [_closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=0.5) for day in range(15, 21)]
    for trade in recent_invalid_trades:
        trade["planned_stop"] = 100
    closed_trades = pd.DataFrame([*old_valid_trades, *recent_valid_trades, *recent_invalid_trades])

    metrics = calculate_strategy_metrics(closed_trades)
    attribution = calculate_strategy_attribution(closed_trades)

    assert metrics.iloc[0]["rolling_mode_exp"] == "N/A"
    assert metrics.iloc[0]["mode_adjusted_score"] == "N/A"
    assert metrics.iloc[0]["mode"] == "Unknown"
    assert metrics.iloc[0]["action"] == "Tiny size only"
    assert attribution.iloc[0]["mode_basis"] == "Need 15 valid R trades; found 14 in latest 20 closed trades"


def test_calculate_strategy_metrics_maps_rolling_mode_thresholds():
    cases = [
        (0.30, "+0.30R", "Caution", "Half size"),
        (0.00, "+0.00R", "Caution", "Half size"),
        (-0.10, "-0.10R", "Weak", "Quarter size"),
        (-0.11, "-0.11R", "Failing", "Probe only / pause"),
    ]

    for r_multiple, expected_exp, expected_mode, expected_action in cases:
        closed_trades = pd.DataFrame(
            [_closed_trade(sell_date=f"2026-04-{day:02d}", r_multiple=r_multiple) for day in range(1, 16)]
        )

        result = calculate_strategy_metrics(closed_trades)

        assert result.iloc[0]["rolling_mode_exp"] == expected_exp
        assert result.iloc[0]["mode_adjusted_score"] == expected_exp
        assert result.iloc[0]["mode"] == expected_mode
        assert result.iloc[0]["action"] == expected_action


def test_calculate_strategy_metrics_calculates_rolling_mode_per_strategy():
    ep_trades = [_closed_trade(strategy="EP", sell_date=f"2026-04-{day:02d}", r_multiple=0.4) for day in range(1, 16)]
    bo_trades = [_closed_trade(strategy="BO", sell_date=f"2026-04-{day:02d}", r_multiple=-0.3) for day in range(1, 16)]

    result = calculate_strategy_metrics(pd.DataFrame([*bo_trades, *ep_trades]))

    assert result[["strategy", "rolling_mode_exp", "mode_adjusted_score", "mode", "action"]].to_dict("records") == [
        {
            "strategy": "EP",
            "rolling_mode_exp": "+0.40R",
            "mode_adjusted_score": "+0.40R",
            "mode": "Working",
            "action": "Normal size",
        },
        {
            "strategy": "BO",
            "rolling_mode_exp": "-0.30R",
            "mode_adjusted_score": "-0.30R",
            "mode": "Failing",
            "action": "Probe only / pause",
        },
    ]


def test_calculate_trade_metrics_computes_hold_time_ratio_from_loss_and_win_holds():
    trades = pd.DataFrame(
        [
            _closed_trade(r_multiple=1.0, hold_days=2),
            _closed_trade(r_multiple=1.0, hold_days=4),
            _closed_trade(r_multiple=-0.5, hold_days=6),
            _closed_trade(r_multiple=-0.5, hold_days=12),
        ]
    )

    metrics = calculate_trade_metrics(trades)

    assert metrics["average_win_hold"] == 3.0
    assert metrics["average_loss_hold"] == 9.0
    assert metrics["hold_time_ratio"] == 3.0


def test_calculate_strategy_attribution_uses_same_mode_as_strategy_metrics():
    trades = _closed_trade_window([-0.11] * 15)

    strategy_metrics = calculate_strategy_metrics(pd.DataFrame(trades))
    attribution = calculate_strategy_attribution(pd.DataFrame(trades))

    assert attribution.iloc[0]["mode"] == strategy_metrics.iloc[0]["mode"] == "Failing"
    assert attribution.iloc[0]["mode_basis"] == "15R -0.11R | Adj -0.11R"
    assert attribution.iloc[0]["trend"] == "Need 30 trades"
    assert "interpretation" not in attribution.columns


def test_calculate_strategy_attribution_mode_basis_shows_reference_adjusted_score():
    trades = _closed_trade_window([1.0] * 8 + [-0.4] * 7)

    strategy_metrics = calculate_strategy_metrics(pd.DataFrame(trades))
    attribution = calculate_strategy_attribution(pd.DataFrame(trades))

    row = attribution.iloc[0]
    assert row["mode"] == strategy_metrics.iloc[0]["mode"] == "Working"
    assert row["mode_basis"] == "15R +0.35R | Adj +0.25R"
    assert row["trend"] == "Need 30 trades"


def test_calculate_strategy_attribution_uses_latest_15_against_prior_15_disjoint_window():
    old_trades = _closed_trade_window([-5.0] * 15, start="2026-01-01")
    prior_trades = _closed_trade_window([0.2] * 15, start="2026-02-01")
    recent_trades = _closed_trade_window([0.6] * 15, start="2026-03-01")

    result = calculate_strategy_attribution(pd.DataFrame([*old_trades, *prior_trades, *recent_trades]))

    row = result.iloc[0]
    assert row["mode"] == "Working"
    assert row["mode_basis"] == "15R +0.60R | Adj +0.60R"
    assert row["trend"] == "Improving (+0.40R)"
    assert row["trend_driver"] == "Winner size + Expectancy R"
    assert "Exp R 0.60 vs 0.20" in row["evidence"]
    assert "Recent: 15 trades over 15 days" in row["evidence"]
    assert "Prior: 15 trades over 15 days" in row["evidence"]


def test_calculate_strategy_attribution_detects_weakening_strategy():
    prior_trades = _closed_trade_window([1.0] * 15, start="2026-02-01")
    recent_trades = _closed_trade_window([0.5] * 15, start="2026-03-01")

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    row = result.iloc[0]
    assert row["mode"] == "Working"
    assert row["trend"] == "Weakening (-0.50R)"
    assert row["trend_driver"] == "Winner size + Expectancy R"
    assert "Exp R 0.50 vs 1.00" in row["evidence"]
    assert row["playbook"] == (
        "Early warning: expectancy is declining (Winner size + Expectancy R). "
        "Monitor closely and reduce size proactively if trend continues."
    )


def test_calculate_strategy_attribution_detects_failing_strategy():
    prior_trades = _closed_trade_window([1.0] * 15, start="2026-02-01")
    recent_trades = _closed_trade_window([-0.3] * 15, start="2026-03-01")

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    row = result.iloc[0]
    assert row["mode"] == "Failing"
    assert row["trend"] == "Weakening (-1.30R)"
    assert row["trend_driver"] == "Hit rate + Loss streak + Expectancy R"


def test_calculate_strategy_attribution_marks_flat_when_no_clear_driver():
    trades = _closed_trade_window([0.2] * 30, start="2026-02-01")

    result = calculate_strategy_attribution(pd.DataFrame(trades))

    row = result.iloc[0]
    assert row["mode"] == "Caution"
    assert row["trend"] == "Flat (+0.00R)"
    assert row["trend_driver"] == "No clear driver"
    assert row["playbook"] == "Keep using Market Regime and Strategy Mode sizing while monitoring the listed drivers."


def test_calculate_strategy_attribution_requires_15_trades_per_strategy():
    trades = _closed_trade_window([0.4] * 14)

    result = calculate_strategy_attribution(pd.DataFrame(trades))

    row = result.iloc[0]
    assert row[["mode", "mode_basis", "trend", "trend_driver"]].to_dict() == {
        "mode": "Unknown",
        "mode_basis": "Need 15 valid R trades",
        "trend": "Need 15 trades",
        "trend_driver": "Need 15 trades",
    }
    assert row["evidence"] == "14 closed trades | Exp R 0.40 | directional only until 15 valid R trades"
    assert row["playbook"] == (
        "Insufficient valid R-trade history; maintain minimum sizing until 15 valid R trades are available."
    )


def test_calculate_strategy_attribution_reports_regime_driver_when_enough_tagged_trades():
    prior_trades = _closed_trade_window([0.3] * 15, start="2026-02-01", regimes=["GO"] * 15)
    recent_trades = _closed_trade_window(
        [-1.0, -1.0, -1.0, -1.0, -1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        start="2026-03-01",
        regimes=["NO-GO", "NO-GO", "NO-GO", "NO-GO", "NO-GO", "GO", "GO", "GO", "GO", "GO", "GO", "GO", "GO", "GO", "GO"],
    )

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    row = result.iloc[0]
    assert row["mode"] == "Failing"
    assert row["trend_driver"] == "Regime filter: NO-GO"
    assert row["playbook"] == "Tighten entry regime filter before resuming normal activity."


def test_calculate_strategy_attribution_ignores_regime_driver_without_enough_regime_trades():
    prior_trades = _closed_trade_window([0.3] * 15, start="2026-02-01", regimes=["GO"] * 15)
    recent_trades = _closed_trade_window(
        [-2.0, *([0.1] * 14)],
        start="2026-03-01",
        regimes=["NO-GO", *(["GO"] * 14)],
    )

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    row = result.iloc[0]
    assert not row["trend_driver"].startswith("Regime")
    assert "Regime attribution: need at least 3 tagged trades in 2 regimes" in row["evidence"]


def test_calculate_strategy_attribution_detects_recovering_trend():
    prior_trades = _closed_trade_window([-0.44] * 15, start="2026-02-01")
    recent_trades = _closed_trade_window([-0.01] * 15, start="2026-03-01")

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    row = result.iloc[0]
    assert row["mode"] == "Weak"
    assert row["mode_basis"] == "15R -0.01R | Adj -0.01R"
    assert row["trend"] == "Recovering (+0.43R)"
    assert row["trend_driver"] == "Loss control + Expectancy R"
    assert "Exp R -0.01 vs -0.44" in row["evidence"]
    assert "Avg loss R -0.01 vs -0.44 (+0.43)" in row["evidence"]


def test_calculate_strategy_attribution_reports_regime_warmup_when_tagged_trades_are_sparse():
    prior_trades = _closed_trade_window([0.1] * 15, start="2026-02-01")
    recent_trades = _closed_trade_window(
        [0.1] * 15,
        start="2026-03-01",
        regimes=["GO", "NO-GO", *([""] * 13)],
    )

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    assert "Regime attribution: need 4 more tagged trades" in result.iloc[0]["evidence"]


def test_calculate_strategy_attribution_playbook_respects_no_go_regime_for_working_strategy():
    trades = _closed_trade_window([0.6] * 30, start="2026-02-01")

    result = calculate_strategy_attribution(pd.DataFrame(trades), market_regime="NO-GO")

    row = result.iloc[0]
    assert row["mode"] == "Working"
    assert row["playbook"] == (
        "Strategy is working, but NO-GO regime limits sizing. Wait for GO conditions for full deployment."
    )


def test_calculate_strategy_attribution_playbook_reports_caution_recovery():
    prior_trades = _closed_trade_window([-0.2] * 15, start="2026-02-01")
    recent_trades = _closed_trade_window([0.1] * 15, start="2026-03-01")

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    row = result.iloc[0]
    assert row["mode"] == "Caution"
    assert row["trend"] == "Improving (+0.30R)"
    assert row["playbook"] == "Trend is recovering. Watch for Working status before resuming normal sizing."


def test_calculate_strategy_attribution_playbook_reports_caution_weakening():
    prior_trades = _closed_trade_window([0.4] * 15, start="2026-02-01")
    recent_trades = _closed_trade_window([0.1] * 15, start="2026-03-01")

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    row = result.iloc[0]
    assert row["mode"] == "Caution"
    assert row["trend"] == "Weakening (-0.30R)"
    assert row["playbook"] == (
        "Trend weakening from Caution. Reduce to Weak-level sizing proactively rather than waiting for mode to drop."
    )


def test_calculate_strategy_attribution_playbook_combines_multi_driver_advice():
    prior_trades = _closed_trade_window([1.0] * 10 + [-0.2] * 5, start="2026-02-01")
    recent_trades = _closed_trade_window([1.0] * 5 + [-0.6] * 10, start="2026-03-01")

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    row = result.iloc[0]
    assert row["mode"] == "Weak"
    assert "Loss control" in row["trend_driver"]
    assert "Hit rate" in row["trend_driver"]
    assert "Tighten stop discipline and avoid entries where risk can expand." in row["playbook"]
    assert "Increase entry selectivity and reduce marginal setups." in row["playbook"]


def test_calculate_strategy_attribution_flags_hold_time_when_losses_are_held_longer_and_worsening():
    prior_trades = _held_trade_window(
        [1.0] * 10 + [-0.2] * 5,
        [4] * 15,
        start="2026-02-01",
    )
    recent_trades = _held_trade_window(
        [1.0] * 5 + [-0.6] * 10,
        [1] * 5 + [3] * 10,
        start="2026-03-01",
    )

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    row = result.iloc[0]
    assert "Hold time" in row["trend_driver"]
    assert "Hold ratio 3.00 vs 1.00 (+2.00)" in row["evidence"]
    assert (
        "Losses are being held significantly longer than wins. "
        "Enforce time-based or rule-based exits on losing positions."
    ) in row["playbook"]


def test_calculate_strategy_attribution_ignores_hold_time_when_ratio_is_high_but_not_worsening():
    prior_trades = _held_trade_window(
        [1.0] * 10 + [-0.2] * 5,
        [2] * 10 + [4] * 5,
        start="2026-02-01",
    )
    recent_trades = _held_trade_window(
        [1.0] * 5 + [-0.6] * 10,
        [2] * 5 + [4] * 10,
        start="2026-03-01",
    )

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    assert "Hold time" not in result.iloc[0]["trend_driver"]


def test_calculate_strategy_attribution_ignores_hold_time_when_ratio_worsens_but_stays_below_threshold():
    prior_trades = _held_trade_window(
        [1.0] * 10 + [-0.2] * 5,
        [4] * 10 + [3] * 5,
        start="2026-02-01",
    )
    recent_trades = _held_trade_window(
        [1.0] * 5 + [-0.6] * 10,
        [4] * 5 + [5] * 10,
        start="2026-03-01",
    )

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    row = result.iloc[0]
    assert "Hold ratio 1.25 vs 0.75 (+0.50)" in row["evidence"]
    assert "Hold time" not in row["trend_driver"]


def test_calculate_strategy_attribution_labels_no_prior_positive_metrics_as_current_strengths():
    trades = _closed_trade_window([0.6] * 15)

    result = calculate_strategy_attribution(pd.DataFrame(trades))

    row = result.iloc[0]
    assert row["mode"] == "Working"
    assert row["trend"] == "Need 30 trades"
    assert row["trend_driver"].startswith("Current strengths:")


def test_calculate_strategy_attribution_surfaces_risk_in_atr_shift():
    prior_trades = [
        _closed_trade(sell_date=date.date().isoformat(), r_multiple=0.2, atr=10)
        for date in pd.date_range(start="2026-02-01", periods=15, freq="D")
    ]
    recent_trades = [
        _closed_trade(sell_date=date.date().isoformat(), r_multiple=0.2, atr=20)
        for date in pd.date_range(start="2026-03-01", periods=15, freq="D")
    ]

    result = calculate_strategy_attribution(pd.DataFrame([*prior_trades, *recent_trades]))

    assert "Risk/ATR 0.50 vs 1.00 (-0.50)" in result.iloc[0]["evidence"]


def test_portfolio_attribution_note_is_blank_with_one_deteriorating_strategy():
    attribution = pd.DataFrame(
        [
            {"strategy": "EP", "mode": "Weak", "trend": "Flat (+0.00R)"},
            {"strategy": "BO", "mode": "Working", "trend": "Improving (+0.40R)"},
        ]
    )

    assert portfolio_attribution_note(attribution) == ""


def test_portfolio_attribution_note_ignores_unknown_and_warmup_rows():
    attribution = pd.DataFrame(
        [
            {"strategy": "EP", "mode": "Weak", "trend": "Flat (+0.00R)"},
            {"strategy": "5% BO", "mode": "Unknown", "trend": "Need 15 trades"},
            {"strategy": "BO", "mode": "Failing", "trend": "Need 30 trades"},
        ]
    )

    assert portfolio_attribution_note(attribution) == ""


def test_portfolio_attribution_note_reports_multiple_deteriorating_strategies():
    attribution = pd.DataFrame(
        [
            {"strategy": "EP", "mode": "Caution", "trend": "Weakening (-0.30R)"},
            {"strategy": "5% BO", "mode": "Unknown", "trend": "Need 15 trades"},
            {"strategy": "BO", "mode": "Weak", "trend": "Recovering (+0.40R)"},
        ]
    )

    assert portfolio_attribution_note(attribution) == (
        "Multiple strategies deteriorating simultaneously: EP, BO. "
        "Review market regime and trading behavior before setup-specific adjustments."
    )
