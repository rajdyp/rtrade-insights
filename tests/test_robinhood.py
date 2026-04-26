from datetime import date
from io import StringIO

import pandas as pd

from stock_calculator.robinhood import (
    PLANNED_STOP_COLUMNS,
    calculate_total_realized_pnl,
    calculate_trade_metrics,
    derive_fifo_trades,
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
            }
        ],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    row = result.closed_trades.iloc[0]
    assert row["planned_stop"] == 58.25
    assert row["buy_price"] == 63.84
    assert row["sell_price"] == 60.90
    assert row["realized_pnl"] == -20.58
    assert row["realized_pnl_percent"] == -4.61
    assert result.open_positions.empty
    assert result.missing_planned_stops == 0


def test_derive_fifo_trades_supports_partial_sells_and_open_lots():
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-04-08", "symbol": "AEHR", "trans_code": "Buy", "quantity": 4, "price": 59.78},
            {"activity_date": "2026-04-08", "symbol": "AEHR", "trans_code": "Buy", "quantity": 4, "price": 59.38},
            {"activity_date": "2026-04-09", "symbol": "AEHR", "trans_code": "Sell", "quantity": 6, "price": 67.43},
        ]
    )

    result = derive_fifo_trades(transactions, pd.DataFrame())

    assert result.closed_trades["quantity"].tolist() == [4, 2]
    assert result.closed_trades["buy_price"].tolist() == [59.78, 59.38]
    assert result.open_positions.iloc[0]["quantity"] == 2
    assert result.open_positions.iloc[0]["buy_price"] == 59.38
    assert result.missing_planned_stops == 3


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
            {"symbol": "AEHR", "buy_date": "2026-04-08", "quantity": 4, "planned_stop": 55.25},
            {"symbol": "AEHR", "buy_date": "2026-04-08", "quantity": 3, "planned_stop": 56.75},
        ],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned)

    assert result.closed_trades["planned_stop"].tolist() == [56.75, 55.25]
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

    assert result.open_positions.iloc[0]["cost_basis"] == 357.12
    assert result.open_positions.iloc[0]["hold_days"] == 3


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
    assert metrics["win_rate"] == 60.0
    assert metrics["win_loss_ratio"] == 1.17
    assert metrics["average_win_r"] == 2.33
    assert metrics["average_loss_r"] == -2.0
    assert metrics["r_ratio"] == 1.17
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


def test_calculate_trade_metrics_empty_closed_trades_are_safe():
    metrics = calculate_trade_metrics(pd.DataFrame())

    assert metrics["trade_count"] == 0
    assert metrics["win_rate"] is None
    assert metrics["profit_factor"] is None
    assert metrics["average_win_r"] is None
    assert metrics["average_loss_r"] is None
    assert metrics["r_ratio"] is None
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
