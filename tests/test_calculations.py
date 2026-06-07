from datetime import date

import pandas as pd

from stock_calculator.calculations import (
    CAMPAIGN_OVERRIDE_COLUMNS,
    CAMPAIGN_TRIM_VIEW_COLUMNS,
    CAMPAIGN_VIEW_COLUMNS,
    POSITION_ID_COLUMN,
    POSITION_CAMPAIGN_COLUMNS,
    POSITION_SOURCE_COLUMNS,
    PUBLIC_OUTPUT_COLUMNS,
    ProfitProtectedAddOn,
    RiskNeutralAddOn,
    append_committed_position,
    apply_campaign_overrides,
    campaign_free_roll_summary,
    campaign_overrides_from_editor,
    campaign_trim_view,
    campaign_view_positions,
    calculate_profit_protected_add_on,
    calculate_stop_itm,
    calculate_trim_to_free_roll,
    calculate_positions,
    committed_positions,
    delete_positions_by_index,
    draft_position,
    empty_positions,
    normalize_campaign_overrides,
    normalize_position_campaigns,
    percent_of_portfolio,
    prospective_symbol_exposure_breach,
    risk_neutral_add_on,
    risk_neutral_add_on_message,
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


def test_campaign_view_uses_positions_when_robinhood_open_lots_are_missing():
    positions = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "DNTH",
                    "buy_date": "2026-05-29",
                    "share_price": 91.50,
                    "stop_price": 90.00,
                    "portfolio_amount": 19_250,
                    "risk_percent": 0.12,
                },
                {
                    "symbol": "SMCI",
                    "buy_date": "2026-05-26",
                    "share_price": 36.11,
                    "stop_price": 35.25,
                    "portfolio_amount": 19_250,
                    "risk_percent": 0.12,
                },
                {
                    "symbol": "SMCI",
                    "buy_date": "2026-05-29",
                    "share_price": 47.95,
                    "stop_price": 44.00,
                    "portfolio_amount": 19_250,
                    "risk_percent": 1.00,
                },
            ]
        )
    )
    positions["strategy"] = ["BO", "4% BO", "EP"]

    result = campaign_view_positions(pd.DataFrame(), positions)

    assert result.columns.tolist() == CAMPAIGN_VIEW_COLUMNS
    assert result.to_dict("records") == [
        {
            "symbol": "DNTH",
            "lots": 1,
            "current_shares": 15,
            "avg_entry": 91.5,
            "campaign_stop": 90.0,
            "sell_lot": 5,
            "position_size": 1372.5,
            "risk_at_campaign_stop": 22.5,
            "planned_lot_risk": 23.1,
            "strategy": "BO",
            "source": "Positions",
        },
        {
            "symbol": "SMCI",
            "lots": 2,
            "current_shares": 74,
            "avg_entry": 43.79,
            "campaign_stop": 44.0,
            "sell_lot": 24,
            "position_size": 3240.46,
            "risk_at_campaign_stop": 0.0,
            "planned_lot_risk": 215.6,
            "strategy": "Mixed",
            "source": "Positions",
        }
    ]


def test_campaign_view_uses_robinhood_open_lots_before_positions():
    open_lots = pd.DataFrame(
        [
            {"symbol": "DNTH", "buy_date": "2026-05-29", "quantity": 15, "planned_stop": 90, "strategy": "BO", "buy_price": 91.49},
            {
                "symbol": "SMCI",
                "buy_date": "2026-05-26",
                "quantity": 18,
                "planned_stop": 35.25,
                "strategy": "4% BO",
                "buy_price": 36.03,
            },
            {
                "symbol": "SMCI",
                "buy_date": "2026-05-29",
                "quantity": 48,
                "planned_stop": 44.00,
                "strategy": "EP",
                "buy_price": 48.23,
            },
        ]
    )
    positions = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "SMCI",
                    "buy_date": "2026-05-26",
                    "share_price": 36.11,
                    "stop_price": 35.25,
                    "portfolio_amount": 19_250,
                    "risk_percent": 0.12,
                },
                {
                    "symbol": "SMCI",
                    "buy_date": "2026-05-29",
                    "share_price": 47.95,
                    "stop_price": 44.00,
                    "portfolio_amount": 19_250,
                    "risk_percent": 1.00,
                },
            ]
        )
    )
    positions["strategy"] = "EP"

    result = campaign_view_positions(open_lots, positions)

    assert result.to_dict("records") == [
        {
            "symbol": "DNTH",
            "lots": 1,
            "current_shares": 15,
            "avg_entry": 91.49,
            "campaign_stop": 90.0,
            "sell_lot": 5,
            "position_size": 1372.35,
            "risk_at_campaign_stop": 22.35,
            "planned_lot_risk": 22.35,
            "strategy": "BO",
            "source": "Robinhood",
        },
        {
            "symbol": "SMCI",
            "lots": 2,
            "current_shares": 66,
            "avg_entry": 44.9,
            "campaign_stop": 44.0,
            "sell_lot": 22,
            "position_size": 2963.58,
            "risk_at_campaign_stop": 59.58,
            "planned_lot_risk": 217.08,
            "strategy": "Mixed",
            "source": "Robinhood",
        }
    ]


def test_campaign_view_falls_back_to_position_context_for_missing_robinhood_metadata():
    open_lots = pd.DataFrame(
        [
            {"symbol": "SMCI", "buy_date": "2026-05-26", "quantity": 18, "buy_price": 36.03},
            {"symbol": "SMCI", "buy_date": "2026-05-29", "quantity": 48, "buy_price": 48.23},
        ]
    )
    positions = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "SMCI",
                    "buy_date": "2026-05-26",
                    "share_price": 36.11,
                    "stop_price": 35.25,
                    "portfolio_amount": 19_250,
                    "risk_percent": 0.12,
                },
                {
                    "symbol": "SMCI",
                    "buy_date": "2026-05-29",
                    "share_price": 47.95,
                    "stop_price": 44.00,
                    "portfolio_amount": 19_250,
                    "risk_percent": 1.00,
                },
            ]
        )
    )
    positions["strategy"] = ["4% BO", "EP"]

    result = campaign_view_positions(open_lots, positions)

    row = result.iloc[0]
    assert row["current_shares"] == 66
    assert row["campaign_stop"] == 44
    assert row["sell_lot"] == 22
    assert row["planned_lot_risk"] == 215.6
    assert row["strategy"] == "Mixed"
    assert row["source"] == "Hybrid"


def test_normalize_position_campaigns_cleans_override_rows():
    result = normalize_position_campaigns(
        pd.DataFrame(
            [
                {"symbol": " smci ", "current_shares": "60"},
                {"symbol": "", "current_shares": "10"},
            ]
        )
    )

    assert result["symbol"].tolist() == ["SMCI"]
    assert result["current_shares"].tolist() == [60]
    assert result.columns.tolist() == POSITION_CAMPAIGN_COLUMNS


def test_normalize_campaign_overrides_keeps_only_symbol_current_shares_and_campaign_stop():
    result = normalize_campaign_overrides(
        pd.DataFrame(
            [
                {"symbol": " smci ", "current_shares": "66", "campaign_stop": "44.90", "live_price": "55.20"},
                {"symbol": "", "current_shares": "10"},
                {"symbol": "SKYT", "current_shares": None, "campaign_stop": "36.30"},
            ]
        )
    )

    assert result.columns.tolist() == CAMPAIGN_OVERRIDE_COLUMNS
    assert result["symbol"].tolist() == ["SMCI", "SKYT"]
    assert result["current_shares"].iloc[0] == 66.0
    assert pd.isna(result["current_shares"].iloc[1])
    assert result["campaign_stop"].tolist() == [44.9, 36.3]


def test_apply_campaign_overrides_updates_visible_rows_only_and_recalculates_risk():
    campaigns = pd.DataFrame(
        [
            {
                "symbol": "SMCI",
                "lots": 2,
                "current_shares": 74,
                "avg_entry": 43.79,
                "campaign_stop": 44.0,
                "sell_lot": 24,
                "position_size": 3240.46,
                "risk_at_campaign_stop": 0.0,
            },
            {
                "symbol": "SKYT",
                "lots": 1,
                "current_shares": 11,
                "avg_entry": 36.30,
                "campaign_stop": 35.31,
                "sell_lot": 3,
                "position_size": 399.30,
                "risk_at_campaign_stop": 10.89,
            },
        ]
    )
    overrides = pd.DataFrame(
        [
            {"symbol": "SMCI", "current_shares": 66, "campaign_stop": 44.90},
            {"symbol": "REMOVED", "current_shares": 99},
        ]
    )

    result = apply_campaign_overrides(campaigns, overrides)

    assert result["symbol"].tolist() == ["SMCI", "SKYT"]
    assert result[["symbol", "current_shares", "campaign_stop", "sell_lot", "position_size", "risk_at_campaign_stop"]].to_dict("records") == [
        {
            "symbol": "SMCI",
            "current_shares": 66,
            "campaign_stop": 44.9,
            "sell_lot": 22,
            "position_size": 2890.14,
            "risk_at_campaign_stop": 0.0,
        },
        {"symbol": "SKYT", "current_shares": 11, "campaign_stop": 35.31, "sell_lot": 3, "position_size": 399.3, "risk_at_campaign_stop": 10.89},
    ]


def test_apply_campaign_overrides_can_update_current_shares_and_campaign_stop_independently():
    campaigns = pd.DataFrame(
        [
            {"symbol": "SMCI", "current_shares": 74, "avg_entry": 43.79, "campaign_stop": 44.0},
            {"symbol": "SKYT", "current_shares": 11, "avg_entry": 36.30, "campaign_stop": 35.31},
        ]
    )
    overrides = pd.DataFrame(
        [
            {"symbol": "SMCI", "current_shares": 66},
            {"symbol": "SKYT", "campaign_stop": 36.30},
        ]
    )

    result = apply_campaign_overrides(campaigns, overrides)

    assert result[["symbol", "current_shares", "campaign_stop", "risk_at_campaign_stop"]].to_dict("records") == [
        {"symbol": "SMCI", "current_shares": 66, "campaign_stop": 44.0, "risk_at_campaign_stop": 0.0},
        {"symbol": "SKYT", "current_shares": 11, "campaign_stop": 36.3, "risk_at_campaign_stop": 0.0},
    ]


def test_campaign_risk_at_stop_clamps_manual_stop_above_entry_to_zero():
    campaigns = pd.DataFrame(
        [
            {
                "symbol": "KRYS",
                "lots": 1,
                "current_shares": 3,
                "avg_entry": 304.84,
                "campaign_stop": 298.00,
                "sell_lot": 1,
                "position_size": 914.52,
                "risk_at_campaign_stop": 20.52,
            }
        ]
    )
    overrides = pd.DataFrame([{"symbol": "KRYS", "campaign_stop": 305.00}])

    result = apply_campaign_overrides(campaigns, overrides)

    assert result[["symbol", "current_shares", "campaign_stop", "risk_at_campaign_stop"]].to_dict("records") == [
        {"symbol": "KRYS", "current_shares": 3, "campaign_stop": 305.0, "risk_at_campaign_stop": 0.0}
    ]


def test_campaign_stop_override_can_create_stop_only_free_roll():
    campaigns = pd.DataFrame(
        [
            {
                "symbol": "SMCI",
                "lots": 2,
                "current_shares": 66,
                "avg_entry": 44.90,
                "campaign_stop": 44.00,
                "sell_lot": 22,
                "position_size": 2963.40,
                "risk_at_campaign_stop": 59.40,
                "planned_lot_risk": 217.08,
            }
        ]
    )
    overridden = apply_campaign_overrides(campaigns, pd.DataFrame([{"symbol": "SMCI", "campaign_stop": 44.90}]))

    result = campaign_trim_view(overridden, {})

    assert result[["symbol", "campaign_stop", "risk_at_campaign_stop", "trim_count", "free_roll"]].to_dict("records") == [
        {"symbol": "SMCI", "campaign_stop": 44.9, "risk_at_campaign_stop": 0.0, "trim_count": 0, "free_roll": "Yes"}
    ]


def test_campaign_overrides_from_editor_excludes_trim_columns_and_saves_only_manual_diffs():
    base = pd.DataFrame(
        [
            {"symbol": "SMCI", "current_shares": 74, "campaign_stop": 44.00, "avg_entry": 44.90},
            {"symbol": "SKYT", "current_shares": 11, "campaign_stop": 35.31, "avg_entry": 36.30},
        ]
    )
    edited = pd.DataFrame(
        [
            {
                "symbol": "SMCI",
                "current_shares": 66,
                "campaign_stop": 44.90,
                "avg_entry": 44.90,
                "live_price": 55.20,
                "trim_count": 6,
                "free_roll": "No",
            },
            {
                "symbol": "SKYT",
                "current_shares": 11,
                "campaign_stop": 35.31,
                "avg_entry": 36.30,
                "live_price": 50.00,
                "trim_count": 1,
                "free_roll": "No",
            },
        ]
    )

    result = campaign_overrides_from_editor(base, edited)

    assert result.columns.tolist() == CAMPAIGN_OVERRIDE_COLUMNS
    assert result.to_dict("records") == [{"symbol": "SMCI", "current_shares": 66, "campaign_stop": 44.9}]


def test_campaign_overrides_from_editor_saves_stop_only_diff_and_drops_reverted_values():
    base = pd.DataFrame(
        [
            {"symbol": "SMCI", "current_shares": 66, "campaign_stop": 44.00},
            {"symbol": "SKYT", "current_shares": 11, "campaign_stop": 35.31},
        ]
    )
    edited = pd.DataFrame(
        [
            {"symbol": "SMCI", "current_shares": 66, "campaign_stop": 44.90},
            {"symbol": "SKYT", "current_shares": 11, "campaign_stop": 35.31},
        ]
    )

    result = campaign_overrides_from_editor(base, edited)

    assert result.columns.tolist() == CAMPAIGN_OVERRIDE_COLUMNS
    assert result["symbol"].tolist() == ["SMCI"]
    assert pd.isna(result["current_shares"].iloc[0])
    assert result["campaign_stop"].tolist() == [44.9]


def test_campaign_trim_view_adds_display_only_trim_columns_for_examples():
    campaigns = pd.DataFrame(
        [
            {
                "symbol": "SMCI",
                "lots": 2,
                "current_shares": 66,
                "avg_entry": 44.90,
                "campaign_stop": 44.00,
                "sell_lot": 22,
                "position_size": 2963.40,
                "risk_at_campaign_stop": 59.40,
                "planned_lot_risk": 217.08,
            },
            {
                "symbol": "SKYT",
                "lots": 1,
                "current_shares": 11,
                "avg_entry": 36.30,
                "campaign_stop": 35.31,
                "sell_lot": 3,
                "position_size": 399.30,
                "risk_at_campaign_stop": 10.89,
                "planned_lot_risk": 11.55,
            },
        ]
    )

    result = campaign_trim_view(campaigns, {"SMCI": 55.20, "SKYT": 50.00})

    assert result.columns.tolist() == CAMPAIGN_TRIM_VIEW_COLUMNS
    assert result[
        [
            "symbol",
            "stop_itm",
            "live_price",
            "trim_count",
            "free_roll",
            "max_add",
            "add_risk",
            "profit_at_stop",
        ]
    ].to_dict("records") == [
        {
            "symbol": "SMCI",
            "stop_itm": 0.0,
            "live_price": "$55.20",
            "trim_count": 6,
            "free_roll": "No",
            "max_add": 0,
            "add_risk": "N/A",
            "profit_at_stop": "N/A",
        },
        {
            "symbol": "SKYT",
            "stop_itm": 0.0,
            "live_price": "$50.00",
            "trim_count": 1,
            "free_roll": "No",
            "max_add": 0,
            "add_risk": "N/A",
            "profit_at_stop": "N/A",
        },
    ]


def test_campaign_trim_view_shows_na_when_add_is_inapplicable():
    campaigns = pd.DataFrame(
        [
            {
                "symbol": "NTAP",
                "lots": 1,
                "current_shares": 12,
                "avg_entry": 187.32,
                "campaign_stop": 172.00,
                "sell_lot": 4,
                "position_size": 2247.84,
                "risk_at_campaign_stop": 183.84,
                "planned_lot_risk": 183.84,
            }
        ]
    )

    result = campaign_trim_view(campaigns, {"NTAP": 173.94})

    assert result[["symbol", "max_add", "add_risk", "profit_at_stop"]].to_dict("records") == [
        {"symbol": "NTAP", "max_add": "N/A", "add_risk": "N/A", "profit_at_stop": "N/A"}
    ]


def test_campaign_trim_view_shows_placeholder_when_live_price_is_missing():
    campaigns = pd.DataFrame(
        [
            {
                "symbol": "SMCI",
                "lots": 2,
                "current_shares": 66,
                "avg_entry": 44.90,
                "campaign_stop": 44.00,
                "sell_lot": 22,
                "position_size": 2963.40,
                "risk_at_campaign_stop": 59.40,
                "planned_lot_risk": 217.08,
            }
        ]
    )

    result = campaign_trim_view(campaigns, {})

    assert result[
        ["symbol", "stop_itm", "live_price", "trim_count", "max_add", "add_risk", "profit_at_stop"]
    ].to_dict("records") == [
        {
            "symbol": "SMCI",
            "stop_itm": 0.0,
            "live_price": "-",
            "trim_count": "-",
            "max_add": "-",
            "add_risk": "-",
            "profit_at_stop": "-",
        }
    ]


def test_calculate_trim_to_free_roll_returns_zero_when_already_free_roll():
    row = pd.Series({"current_shares": 74, "avg_entry": 43.79, "campaign_stop": 44.0})

    assert calculate_trim_to_free_roll(row, None) == (0, True)


def test_calculate_trim_to_free_roll_rounds_up_to_break_even():
    row = pd.Series({"current_shares": 66, "avg_entry": 44.90, "campaign_stop": 44.00})

    assert calculate_trim_to_free_roll(row, 55.20) == (6, False)


def test_calculate_trim_to_free_roll_blanks_when_live_price_is_missing_or_below_stop():
    row = pd.Series({"current_shares": 66, "avg_entry": 44.90, "campaign_stop": 44.00})

    assert calculate_trim_to_free_roll(row, None) == (None, False)
    assert calculate_trim_to_free_roll(row, 44.00) == (None, False)
    assert calculate_trim_to_free_roll(row, 43.50) == (None, False)


def test_calculate_trim_to_free_roll_uses_trim_credit_to_mark_free_roll():
    row = pd.Series({"current_shares": 4, "avg_entry": 29.08, "campaign_stop": 27.87})

    assert calculate_trim_to_free_roll(row, None, 6.36) == (0, True)


def test_calculate_trim_to_free_roll_uses_uncovered_risk_after_trim_credit():
    row = pd.Series({"current_shares": 10, "avg_entry": 29.08, "campaign_stop": 27.87})

    assert calculate_trim_to_free_roll(row, 30.14, 6.36) == (3, False)


def test_calculate_stop_itm_uses_stop_above_average_entry():
    row = pd.Series({"current_shares": 10, "avg_entry": 29.08, "campaign_stop": 30.00})

    assert calculate_stop_itm(row) == 9.2


def test_calculate_stop_itm_clamps_stop_below_average_entry_to_zero():
    row = pd.Series({"current_shares": 3, "avg_entry": 151.60, "campaign_stop": 149.00})

    assert calculate_stop_itm(row) == 0.0


def test_calculate_stop_itm_includes_realized_trim_credit():
    row = pd.Series({"current_shares": 4, "avg_entry": 29.08, "campaign_stop": 27.87})

    assert calculate_stop_itm(row, 10.00) == 5.16


def test_calculate_stop_itm_exact_breakeven_is_zero_while_free_roll_is_yes():
    row = pd.Series({"current_shares": 4, "avg_entry": 29.08, "campaign_stop": 27.87})

    assert calculate_stop_itm(row, 4.84) == 0.0
    assert calculate_trim_to_free_roll(row, None, 4.84) == (0, True)


def test_campaign_trim_view_uses_campaign_trim_credit_by_symbol():
    campaigns = pd.DataFrame(
        [
            {
                "symbol": "ATEN",
                "lots": 1,
                "current_shares": 4,
                "avg_entry": 29.08,
                "campaign_stop": 27.87,
                "sell_lot": 1,
                "position_size": 116.32,
                "risk_at_campaign_stop": 4.84,
                "planned_lot_risk": 12.10,
            }
        ]
    )
    trim_credits = pd.DataFrame([{"symbol": "ATEN", "realized_trim_credit": 6.36}])

    result = campaign_trim_view(campaigns, {}, trim_credits)

    assert result[["symbol", "stop_itm", "trim_count", "free_roll"]].to_dict("records") == [
        {"symbol": "ATEN", "stop_itm": 1.52, "trim_count": 0, "free_roll": "Yes"}
    ]


def test_campaign_free_roll_summary_counts_yes_rows():
    frame = pd.DataFrame({"free_roll": ["Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes", "No", "No"]})

    assert campaign_free_roll_summary(frame) == "Free Roll: 5 / 10 (50%)"


def test_campaign_free_roll_summary_handles_empty_frame():
    assert campaign_free_roll_summary(pd.DataFrame(columns=["free_roll"])) == "Free Roll: 0 / 0 (0%)"


def test_campaign_free_roll_summary_counts_only_normalized_yes_values():
    frame = pd.DataFrame({"free_roll": ["Yes", " yes ", "YES", "", None, "No"]})

    assert campaign_free_roll_summary(frame) == "Free Roll: 1 / 6 (17%)"


def test_calculate_profit_protected_add_on_reproduces_smci_guardrail():
    row = pd.Series({"current_shares": 18, "avg_entry": 36.03, "campaign_stop": 44.00})

    result = calculate_profit_protected_add_on(row, 48.23, 14.80, preserve_percent=50.0)

    assert result == ProfitProtectedAddOn(
        applicable=True,
        max_add_shares=7,
        add_risk=29.61,
        profit_at_stop=128.65,
    )


def test_calculate_profit_protected_add_on_uses_ftnt_campaign_values():
    row = pd.Series({"current_shares": 4, "avg_entry": 109.84, "campaign_stop": 137.00})

    result = calculate_profit_protected_add_on(row, 144.71, 38.04, preserve_percent=50.0)

    assert result == ProfitProtectedAddOn(
        applicable=True,
        max_add_shares=5,
        add_risk=38.55,
        profit_at_stop=108.13,
    )


def test_calculate_profit_protected_add_on_returns_zero_when_floor_has_no_capacity():
    row = pd.Series({"current_shares": 10, "avg_entry": 100.00, "campaign_stop": 105.00})

    result = calculate_profit_protected_add_on(row, 120.00, preserve_percent=50.0)

    assert result == ProfitProtectedAddOn(
        applicable=True,
        max_add_shares=0,
        add_risk=0.0,
        profit_at_stop=50.0,
    )


def test_calculate_profit_protected_add_on_is_inapplicable_at_or_below_entry_or_stop():
    row = pd.Series({"current_shares": 10, "avg_entry": 100.00, "campaign_stop": 95.00})

    assert calculate_profit_protected_add_on(row, 100.00) == ProfitProtectedAddOn(False, 0, 0.0, None)

    stop_above_entry = pd.Series({"current_shares": 10, "avg_entry": 90.00, "campaign_stop": 105.00})
    assert calculate_profit_protected_add_on(stop_above_entry, 105.00) == ProfitProtectedAddOn(False, 0, 0.0, None)


def test_calculate_profit_protected_add_on_requires_price_and_valid_inputs():
    row = pd.Series({"current_shares": 10, "avg_entry": 100.00, "campaign_stop": 95.00})

    assert calculate_profit_protected_add_on(row, None) is None
    assert calculate_profit_protected_add_on(row, 110.00, preserve_percent=-1) is None
    assert calculate_profit_protected_add_on(row, 110.00, preserve_percent=101) is None


def test_calculate_profit_protected_add_on_floors_without_breaching_required_floor():
    row = pd.Series({"current_shares": 18, "avg_entry": 36.03, "campaign_stop": 44.00})
    live_price = 48.23
    realized_trim_credit = 14.80

    result = calculate_profit_protected_add_on(
        row,
        live_price,
        realized_trim_credit,
        preserve_percent=50.0,
    )

    required_floor = realized_trim_credit + (0.50 * (live_price - 36.03) * 18)
    assert result is not None
    assert result.profit_at_stop is not None
    assert result.profit_at_stop >= required_floor
    assert result.profit_at_stop - (live_price - 44.00) < required_floor


def test_campaign_trim_view_keeps_trim_columns_out_of_campaign_storage_columns():
    assert not set(CAMPAIGN_TRIM_VIEW_COLUMNS) - set(
        [
            *CAMPAIGN_VIEW_COLUMNS,
            "stop_itm",
            "live_price",
            "trim_count",
            "free_roll",
            "max_add",
            "add_risk",
            "profit_at_stop",
        ]
    )
    assert "live_price" not in CAMPAIGN_OVERRIDE_COLUMNS
    assert "trim_count" not in CAMPAIGN_OVERRIDE_COLUMNS
    assert "free_roll" not in CAMPAIGN_OVERRIDE_COLUMNS
    assert "max_add" not in CAMPAIGN_OVERRIDE_COLUMNS
    assert "add_risk" not in CAMPAIGN_OVERRIDE_COLUMNS
    assert "profit_at_stop" not in CAMPAIGN_OVERRIDE_COLUMNS


def test_risk_neutral_add_on_triggers_for_single_robinhood_open_lot():
    open_lots = pd.DataFrame(
        [
            {
                "symbol": "SMCI",
                "buy_date": "2026-05-26",
                "quantity": 18,
                "planned_stop": 35.25,
                "strategy": "4% BO",
                "buy_price": 36.03,
            }
        ]
    )
    draft = pd.Series(
        {
            "symbol": "SMCI",
            "share_price": 48.23,
            "stop_price": 44.00,
            "number_of_shares": 48,
            "portfolio_amount": 19_250,
        }
    )

    result = risk_neutral_add_on(symbol="SMCI", draft=draft, open_lots=open_lots, positions=pd.DataFrame())

    assert result is not None
    assert result.source == "Robinhood"
    assert result.current_shares == 18
    assert result.draft_shares == 48
    assert result.combined_shares == 66
    assert result.combined_avg_entry == 44.9
    assert result.combined_risk_at_stop == 59.58
    assert result.max_risk_neutral_shares == 33
    assert result.max_risk_neutral_risk_percent == 0.73
    assert result.risk_neutral is False


def test_risk_neutral_add_on_reports_safe_draft_within_cap():
    open_lots = pd.DataFrame(
        [{"symbol": "SMCI", "buy_date": "2026-05-26", "quantity": 18, "buy_price": 36.03}]
    )
    draft = pd.Series(
        {
            "symbol": "SMCI",
            "share_price": 48.23,
            "stop_price": 44.00,
            "number_of_shares": 30,
            "portfolio_amount": 19_250,
        }
    )

    result = risk_neutral_add_on(symbol="SMCI", draft=draft, open_lots=open_lots, positions=pd.DataFrame())

    assert result is not None
    assert result.max_risk_neutral_shares == 33
    assert result.combined_risk_at_stop == -16.56
    assert result.risk_neutral is True


def test_risk_neutral_add_on_uses_active_positions_when_robinhood_has_no_symbol():
    positions = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "SMCI",
                    "buy_date": "2026-05-26",
                    "share_price": 36.11,
                    "stop_price": 35.25,
                    "portfolio_amount": 19_250,
                    "risk_percent": 0.12,
                }
            ]
        )
    )
    draft = pd.Series(
        {
            "symbol": "SMCI",
            "share_price": 47.95,
            "stop_price": 44.00,
            "number_of_shares": 48,
            "portfolio_amount": 19_250,
        }
    )

    result = risk_neutral_add_on(symbol="SMCI", draft=draft, open_lots=pd.DataFrame(), positions=positions)

    assert result is not None
    assert result.source == "Positions"
    assert result.current_shares == 26
    assert result.max_risk_neutral_shares == 51
    assert result.combined_risk_at_stop == -15.54
    assert result.risk_neutral is True


def test_risk_neutral_add_on_returns_none_for_unowned_symbol():
    draft = pd.Series(
        {
            "symbol": "SMCI",
            "share_price": 48.23,
            "stop_price": 44.00,
            "number_of_shares": 48,
            "portfolio_amount": 19_250,
        }
    )

    result = risk_neutral_add_on(symbol="SMCI", draft=draft, open_lots=pd.DataFrame(), positions=pd.DataFrame())

    assert result is None


def test_risk_neutral_add_on_message_formats_positive_result_compactly():
    message = risk_neutral_add_on_message(
        RiskNeutralAddOn(
            symbol="FTNT",
            source="Robinhood",
            current_shares=10,
            avg_entry=120,
            draft_shares=6,
            combined_shares=16,
            combined_avg_entry=126,
            combined_risk_at_stop=-10,
            max_risk_neutral_shares=11,
            max_risk_neutral_risk_percent=0.31,
            risk_neutral=True,
        )
    )

    assert message == ("Risk-neutral: Yes. Max size 11; draft 6.", "ready")


def test_risk_neutral_add_on_message_formats_negative_result_with_cap_compactly():
    message = risk_neutral_add_on_message(
        RiskNeutralAddOn(
            symbol="FTNT",
            source="Robinhood",
            current_shares=10,
            avg_entry=120,
            draft_shares=26,
            combined_shares=36,
            combined_avg_entry=132,
            combined_risk_at_stop=50,
            max_risk_neutral_shares=11,
            max_risk_neutral_risk_percent=0.31,
            risk_neutral=False,
        )
    )

    assert message == ("Risk-neutral: No. Max size 11 (cap 0.31%); draft 26.", "idle")


def test_risk_neutral_add_on_message_formats_negative_result_without_cap_compactly():
    message = risk_neutral_add_on_message(
        RiskNeutralAddOn(
            symbol="FTNT",
            source="Robinhood",
            current_shares=10,
            avg_entry=120,
            draft_shares=26,
            combined_shares=36,
            combined_avg_entry=132,
            combined_risk_at_stop=50,
            max_risk_neutral_shares=11,
            max_risk_neutral_risk_percent=None,
            risk_neutral=False,
        )
    )

    assert message == ("Risk-neutral: No. Max size 11; draft 26.", "idle")


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
