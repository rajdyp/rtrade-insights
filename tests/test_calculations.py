from datetime import date

import pandas as pd

from stock_calculator.calculations import (
    POSITION_ID_COLUMN,
    POSITION_SOURCE_COLUMNS,
    PUBLIC_OUTPUT_COLUMNS,
    append_committed_position,
    calculate_positions,
    committed_positions,
    delete_positions_by_index,
    draft_position,
    empty_positions,
    percent_of_portfolio,
    prospective_symbol_exposure_breach,
    symbol_exposure_breaches,
    weekday_hold_count,
)


def test_calculates_position_size_from_screenshot_values():
    result = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "AEIS",
                    "buy_date": "2026-01-06",
                    "share_price": 228.50,
                    "stop_price": 221.00,
                    "portfolio_amount": 20_000.00,
                    "risk_percent": 0.50,
                }
            ]
        )
    )

    row = result.iloc[0]
    assert row["stop_loss_percent"] == 3.28
    assert pd.isna(row["risk_in_atr"])
    assert row["risk_amount"] == 100.00
    assert row["number_of_shares"] == 13
    assert row["sell_lot"] == 4
    assert row["position_size"] == 2970.50
    assert row["validation_error"] == ""


def test_rejects_stop_price_above_share_price():
    result = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "BAD",
                    "share_price": 100,
                    "stop_price": 101,
                    "portfolio_amount": 20_000,
                    "risk_percent": 0.5,
                }
            ]
        )
    )

    row = result.iloc[0]
    assert pd.isna(row["number_of_shares"])
    assert row["validation_error"] == "Stop price must be below share price."


def test_blank_rows_are_allowed_without_validation_noise():
    result = calculate_positions(empty_positions(1))

    row = result.iloc[0]
    assert row["symbol"] == ""
    assert row["validation_error"] == ""


def test_risk_percent_changes_total_risk_and_shares():
    result = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "TEST",
                    "share_price": 50,
                    "stop_price": 48,
                    "portfolio_amount": 10_000,
                    "risk_percent": 1.0,
                }
            ]
        )
    )

    row = result.iloc[0]
    assert row["risk_amount"] == 100
    assert row["number_of_shares"] == 50
    assert row["position_size"] == 2500


def test_calculates_risk_in_atr_from_atr_percent():
    result = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "MKSI",
                    "share_price": 280.44,
                    "stop_price": 268.00,
                    "atr": 5.04,
                    "portfolio_amount": 19_250,
                    "risk_percent": 0.12,
                }
            ]
        )
    )

    row = result.iloc[0]
    assert row["stop_loss_percent"] == 4.44
    assert row["risk_in_atr"] == 0.88
    assert row["number_of_shares"] == 1


def test_missing_atr_keeps_position_valid_with_blank_risk_in_atr():
    result = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "TEST",
                    "share_price": 100,
                    "stop_price": 95,
                    "portfolio_amount": 20_000,
                    "risk_percent": 0.5,
                }
            ]
        )
    )

    row = result.iloc[0]
    assert pd.isna(row["risk_in_atr"])
    assert row["validation_error"] == ""


def test_non_positive_atr_keeps_position_valid_with_blank_risk_in_atr():
    result = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "ZERO",
                    "share_price": 100,
                    "stop_price": 95,
                    "atr": 0,
                    "portfolio_amount": 20_000,
                    "risk_percent": 0.5,
                },
                {
                    "symbol": "NEG",
                    "share_price": 100,
                    "stop_price": 95,
                    "atr": -1,
                    "portfolio_amount": 20_000,
                    "risk_percent": 0.5,
                },
            ]
        )
    )

    assert result["risk_in_atr"].isna().all()
    assert result["validation_error"].tolist() == ["", ""]


def test_draft_position_builds_normalized_single_row():
    draft = draft_position(
        symbol=" aapl ",
        buy_date=date(2026, 4, 24),
        share_price=100,
        stop_price=95,
        portfolio_amount=20_000,
        risk_percent=0.5,
    )

    row = draft.iloc[0]
    assert row["symbol"] == "AAPL"
    assert row["buy_date"] == "2026-04-24"
    assert row["share_price"] == 100
    assert pd.isna(row["atr"])
    assert list(draft.columns) == POSITION_SOURCE_COLUMNS
    assert row[POSITION_ID_COLUMN] == ""


def test_append_committed_position_adds_valid_draft():
    existing = draft_position(
        symbol="EXISTING",
        buy_date="2026-04-23",
        share_price=50,
        stop_price=45,
        portfolio_amount=20_000,
        risk_percent=0.5,
    )
    draft = draft_position(
        symbol="TEST",
        buy_date="2026-04-24",
        share_price=100,
        stop_price=95,
        portfolio_amount=20_000,
        risk_percent=0.5,
    )

    result = append_committed_position(existing, draft)

    assert len(result) == 2
    assert result.iloc[0]["symbol"] == "TEST"
    assert result.iloc[1]["symbol"] == "EXISTING"
    assert result[POSITION_ID_COLUMN].str.startswith("pos_").all()
    assert result[POSITION_ID_COLUMN].nunique() == 2


def test_append_committed_position_rejects_invalid_draft():
    existing = pd.DataFrame(columns=["symbol"])
    draft = draft_position(
        symbol="BAD",
        buy_date="2026-04-24",
        share_price=100,
        stop_price=101,
        portfolio_amount=20_000,
        risk_percent=0.5,
    )

    result = append_committed_position(existing, draft)

    assert result.empty


def test_committed_positions_filters_blank_rows():
    result = committed_positions(
        pd.DataFrame(
            [
                {"symbol": "", "share_price": 100},
                {"symbol": "KEEP", "share_price": 100},
            ]
        )
    )

    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "KEEP"
    assert result.iloc[0][POSITION_ID_COLUMN].startswith("pos_")


def test_delete_positions_by_index_deletes_single_row():
    positions = pd.DataFrame(
        [
            {"symbol": "AAA", "share_price": 100},
            {"symbol": "BBB", "share_price": 100},
            {"symbol": "CCC", "share_price": 100},
        ]
    )

    result = delete_positions_by_index(positions, [1])

    assert result["symbol"].tolist() == ["AAA", "CCC"]
    assert result.index.tolist() == [0, 1]


def test_delete_positions_by_index_deletes_multiple_rows():
    positions = pd.DataFrame(
        [
            {"symbol": "AAA", "share_price": 100},
            {"symbol": "BBB", "share_price": 100},
            {"symbol": "CCC", "share_price": 100},
            {"symbol": "DDD", "share_price": 100},
        ]
    )

    result = delete_positions_by_index(positions, [0, 2])

    assert result["symbol"].tolist() == ["BBB", "DDD"]
    assert result.index.tolist() == [0, 1]


def test_delete_positions_by_index_ignores_empty_selection():
    positions = pd.DataFrame(
        [
            {"symbol": "AAA", "share_price": 100},
            {"symbol": "BBB", "share_price": 100},
        ]
    )

    result = delete_positions_by_index(positions, [])

    assert result["symbol"].tolist() == ["AAA", "BBB"]
    assert result.index.tolist() == [0, 1]


def test_delete_positions_by_index_ignores_out_of_range_rows():
    positions = pd.DataFrame(
        [
            {"symbol": "AAA", "share_price": 100},
            {"symbol": "BBB", "share_price": 100},
        ]
    )

    result = delete_positions_by_index(positions, [9])

    assert result["symbol"].tolist() == ["AAA", "BBB"]
    assert result.index.tolist() == [0, 1]


def test_hold_count_excludes_same_day_buy_date():
    assert weekday_hold_count("2026-04-24", as_of=date(2026, 4, 24)) == 0


def test_hold_count_counts_weekdays_after_buy_date():
    assert weekday_hold_count("2026-04-24", as_of=date(2026, 4, 27)) == 1
    assert weekday_hold_count("2026-04-24", as_of=date(2026, 4, 28)) == 2


def test_hold_count_does_not_count_weekends():
    assert weekday_hold_count("2026-04-24", as_of=date(2026, 4, 26)) == 0


def test_calculate_positions_uses_weekday_hold_count():
    result = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "TEST",
                    "buy_date": "2026-04-24",
                    "share_price": 100,
                    "stop_price": 95,
                    "portfolio_amount": 20_000,
                    "risk_percent": 0.5,
                }
            ]
        ),
        as_of=date(2026, 4, 28),
    )

    assert result.iloc[0]["hold_count"] == 2


def test_sell_lot_rounds_down_to_one_third_of_shares():
    result = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "ONE",
                    "share_price": 100,
                    "stop_price": 90,
                    "portfolio_amount": 100,
                    "risk_percent": 10,
                },
                {
                    "symbol": "TWO",
                    "share_price": 100,
                    "stop_price": 90,
                    "portfolio_amount": 200,
                    "risk_percent": 10,
                },
                {
                    "symbol": "THREE",
                    "share_price": 100,
                    "stop_price": 90,
                    "portfolio_amount": 300,
                    "risk_percent": 10,
                },
                {
                    "symbol": "THIRTEEN",
                    "share_price": 228.50,
                    "stop_price": 221.00,
                    "portfolio_amount": 20_000,
                    "risk_percent": 0.5,
                },
                {
                    "symbol": "SIXTEEN",
                    "share_price": 164.00,
                    "stop_price": 157.90,
                    "portfolio_amount": 20_000,
                    "risk_percent": 0.5,
                },
            ]
        )
    )

    assert result.iloc[0]["number_of_shares"] == 1
    assert result.iloc[0]["sell_lot"] == 1
    assert result.iloc[1]["number_of_shares"] == 2
    assert result.iloc[1]["sell_lot"] == 1
    assert result.iloc[2]["number_of_shares"] == 3
    assert result.iloc[2]["sell_lot"] == 1
    assert result.iloc[3]["number_of_shares"] == 13
    assert result.iloc[3]["sell_lot"] == 4
    assert result.iloc[4]["number_of_shares"] == 16
    assert result.iloc[4]["sell_lot"] == 5


def test_invalid_position_has_blank_sell_lot():
    result = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "BAD",
                    "share_price": 100,
                    "stop_price": 101,
                    "portfolio_amount": 20_000,
                    "risk_percent": 0.5,
                }
            ]
        )
    )

    assert pd.isna(result.iloc[0]["sell_lot"])


def test_public_output_columns_exclude_internal_fields():
    assert "portfolio_after_position" not in PUBLIC_OUTPUT_COLUMNS
    assert "validation_error" not in PUBLIC_OUTPUT_COLUMNS


def test_percent_of_portfolio_uses_current_portfolio_value():
    assert percent_of_portfolio(319.55, 19_250) == 1.66


def test_percent_of_portfolio_is_blank_for_invalid_portfolio_value():
    assert percent_of_portfolio(319.55, 0) is None
    assert percent_of_portfolio(319.55, None) is None


def test_symbol_exposure_breaches_sum_multiple_rows_for_same_symbol():
    positions = pd.DataFrame(
        [
            {"symbol": "NVDA", "position_size": 2_500},
            {"symbol": "nvda", "position_size": 1_600},
            {"symbol": "AAPL", "position_size": 2_000},
        ]
    )

    breaches = symbol_exposure_breaches(
        positions,
        portfolio_amount=19_250,
        max_symbol_exposure_percent=20.0,
    )

    assert breaches.to_dict("records") == [
        {"symbol": "NVDA", "position_size": 4_100, "exposure_percent": 21.3},
    ]


def test_symbol_exposure_breaches_uses_strictly_greater_than_limit():
    positions = pd.DataFrame(
        [
            {"symbol": "AAPL", "position_size": 4_000},
            {"symbol": "MSFT", "position_size": 4_002},
        ]
    )

    breaches = symbol_exposure_breaches(
        positions,
        portfolio_amount=20_000,
        max_symbol_exposure_percent=20.0,
    )

    assert breaches.to_dict("records") == [
        {"symbol": "MSFT", "position_size": 4_002, "exposure_percent": 20.01},
    ]


def test_symbol_exposure_breaches_returns_empty_when_all_symbols_are_within_limit():
    positions = pd.DataFrame(
        [
            {"symbol": "AAPL", "position_size": 3_900},
            {"symbol": "MSFT", "position_size": 2_000},
        ]
    )

    breaches = symbol_exposure_breaches(
        positions,
        portfolio_amount=20_000,
        max_symbol_exposure_percent=20.0,
    )

    assert breaches.empty


def test_prospective_symbol_exposure_breach_sums_existing_and_draft_same_symbol():
    positions = pd.DataFrame(
        [
            {"symbol": "ABC", "position_size": 3_000},
            {"symbol": "XYZ", "position_size": 4_500},
        ]
    )
    draft = pd.Series({"symbol": "ABC", "position_size": 1_000})

    breach = prospective_symbol_exposure_breach(
        positions,
        draft,
        portfolio_amount=19_250,
        max_symbol_exposure_percent=20.0,
    )

    assert breach is not None
    assert breach.to_dict() == {"symbol": "ABC", "position_size": 4_000, "exposure_percent": 20.78}


def test_prospective_symbol_exposure_breach_ignores_unrelated_existing_breaches():
    positions = pd.DataFrame(
        [
            {"symbol": "XYZ", "position_size": 4_500},
        ]
    )
    draft = pd.Series({"symbol": "ABC", "position_size": 1_000})

    breach = prospective_symbol_exposure_breach(
        positions,
        draft,
        portfolio_amount=19_250,
        max_symbol_exposure_percent=20.0,
    )

    assert breach is None


def test_prospective_symbol_exposure_breach_returns_none_at_or_below_limit():
    positions = pd.DataFrame(
        [
            {"symbol": "ABC", "position_size": 3_000},
        ]
    )
    draft = pd.Series({"symbol": "ABC", "position_size": 850})

    breach = prospective_symbol_exposure_breach(
        positions,
        draft,
        portfolio_amount=19_250,
        max_symbol_exposure_percent=20.0,
    )

    assert breach is None
