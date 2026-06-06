import pandas as pd
import pytest

from stock_calculator.calculations import (
    CAMPAIGN_OVERRIDE_COLUMNS,
    POSITION_ID_COLUMN,
    POSITION_SOURCE_COLUMNS,
    calculate_positions,
)
from stock_calculator.robinhood import PLANNED_STOP_COLUMNS, TRANSACTION_COLUMNS, calculate_trade_metrics, derive_fifo_trades
from stock_calculator.storage import (
    GoogleSheetsStorage,
    LocalCsvStorage,
    POSITION_ARCHIVE_COLUMNS,
    StorageConfigurationError,
    append_robinhood_transactions,
    build_storage_backend,
    generate_planned_stops_from_transactions,
    load_campaign_overrides,
    load_planned_stops,
    load_positions,
    load_positions_archive,
    load_robinhood_transactions,
    save_campaign_overrides,
    save_planned_stops,
    save_positions,
    save_positions_archive,
    save_robinhood_transactions,
    upsert_positions_archive,
    upsert_planned_stop,
)


def test_save_positions_filters_blank_rows(tmp_path):
    path = tmp_path / "positions.csv"
    save_positions(
        pd.DataFrame(
            [
                {"symbol": "", "share_price": 100},
                {
                    "symbol": "KEEP",
                    "buy_date": "2026-04-24",
                    "share_price": 100,
                    "stop_price": 95,
                    "atr": 2.5,
                    "portfolio_amount": 20_000,
                    "risk_percent": 0.5,
                },
            ]
        ),
        path,
    )

    loaded = pd.read_csv(path)

    assert loaded["symbol"].tolist() == ["KEEP"]
    assert loaded.columns.tolist() == POSITION_SOURCE_COLUMNS
    assert loaded[POSITION_ID_COLUMN].str.startswith("pos_").all()
    assert loaded["atr"].tolist() == [2.5]


def test_load_positions_accepts_old_files_without_atr(tmp_path):
    path = tmp_path / "positions.csv"
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "buy_date": "2026-04-01",
                "share_price": 200,
                "stop_price": 190,
                "portfolio_amount": 20_000,
                "risk_percent": 0.5,
            }
        ]
    ).to_csv(path, index=False)

    loaded = load_positions(path)

    assert loaded.columns.tolist() == POSITION_SOURCE_COLUMNS
    assert loaded["symbol"].tolist() == ["AAPL"]
    assert loaded[POSITION_ID_COLUMN].str.startswith("pos_").all()
    assert loaded["atr"].isna().all()


def test_load_positions_returns_empty_frame_for_missing_file(tmp_path):
    loaded = load_positions(tmp_path / "missing.csv")

    assert loaded.empty
    assert "symbol" in loaded.columns
    assert POSITION_ID_COLUMN in loaded.columns


def test_save_and_load_positions_archive_preserves_visible_table_fields(tmp_path):
    path = tmp_path / "positions_archive.csv"
    calculated = calculate_positions(
        pd.DataFrame(
            [
                {
                    "symbol": "aapl",
                    "buy_date": "2026-04-01",
                    "share_price": 100,
                    "stop_price": 95,
                    "atr": 2.5,
                    "portfolio_amount": 20_000,
                    "risk_percent": 0.5,
                }
            ]
        )
    )
    calculated["strategy"] = "EP"

    save_positions_archive(calculated, path)
    loaded = load_positions_archive(path)

    assert loaded.columns.tolist() == POSITION_ARCHIVE_COLUMNS
    assert loaded[POSITION_ID_COLUMN].str.startswith("pos_").all()
    assert loaded["symbol"].tolist() == ["AAPL"]
    assert loaded["strategy"].tolist() == ["EP"]
    assert loaded["number_of_shares"].tolist() == [20]
    assert loaded["position_size"].tolist() == [2000.0]


def test_upsert_positions_archive_updates_matches_and_keeps_deleted_rows():
    existing = pd.DataFrame(
        [
            {
                POSITION_ID_COLUMN: "pos_keep",
                "symbol": "KEEP",
                "buy_date": "2026-04-01",
                "share_price": 100,
                "stop_price": 95,
                "strategy": "EP",
            },
            {
                POSITION_ID_COLUMN: "pos_deleted",
                "symbol": "OLD",
                "buy_date": "2026-04-01",
                "share_price": 50,
                "stop_price": 45,
                "strategy": "BO",
            },
        ]
    )
    updated = calculate_positions(
        pd.DataFrame(
            [
                {
                    POSITION_ID_COLUMN: "pos_keep",
                    "symbol": "KEEP",
                    "buy_date": "2026-04-01",
                    "share_price": 110,
                    "stop_price": 95,
                    "portfolio_amount": 20_000,
                    "risk_percent": 0.5,
                }
            ]
        )
    )
    updated["strategy"] = "4% BO"

    result = upsert_positions_archive(existing, updated)

    assert result[POSITION_ID_COLUMN].tolist() == ["pos_keep", "pos_deleted"]
    assert result.loc[0, "share_price"] == 110
    assert result.loc[0, "strategy"] == "4% BO"
    assert result.loc[1, "symbol"] == "OLD"


def test_load_robinhood_transactions_returns_empty_frame_for_missing_file(tmp_path):
    loaded = load_robinhood_transactions(tmp_path / "missing.csv")

    assert loaded.empty
    assert loaded.columns.tolist() == TRANSACTION_COLUMNS


def test_save_and_load_campaign_overrides_preserves_only_editable_campaign_fields(tmp_path):
    path = tmp_path / "campaign_overrides.csv"
    save_campaign_overrides(
        pd.DataFrame(
            [
                {
                    "symbol": " smci ",
                    "lots": "2",
                    "current_shares": "66",
                    "avg_entry": "44.90",
                    "campaign_stop": "44.00",
                    "sell_lot": "22",
                    "position_size": "2963.58",
                    "risk_at_campaign_stop": "59.58",
                    "planned_lot_risk": "217.08",
                    "live_price": "55.20",
                    "trim_count": "6",
                    "free_roll": "No",
                    "strategy": "Mixed",
                    "source": "Robinhood",
                },
                {"symbol": "", "current_shares": "10"},
            ]
        ),
        path,
    )

    loaded = load_campaign_overrides(path)
    saved_columns = pd.read_csv(path).columns.tolist()

    assert saved_columns == CAMPAIGN_OVERRIDE_COLUMNS
    assert loaded.columns.tolist() == CAMPAIGN_OVERRIDE_COLUMNS
    assert loaded.to_dict("records") == [
        {
            "symbol": "SMCI",
            "current_shares": 66,
            "campaign_stop": 44.0,
        }
    ]


def test_load_campaign_overrides_returns_empty_frame_for_missing_file(tmp_path):
    loaded = load_campaign_overrides(tmp_path / "missing.csv")

    assert loaded.empty
    assert loaded.columns.tolist() == CAMPAIGN_OVERRIDE_COLUMNS


def test_load_campaign_overrides_accepts_override_only_file(tmp_path):
    path = tmp_path / "campaign_overrides.csv"
    pd.DataFrame([{"symbol": "smci", "current_shares": 60}]).to_csv(path, index=False)

    loaded = load_campaign_overrides(path)

    assert loaded.columns.tolist() == CAMPAIGN_OVERRIDE_COLUMNS
    assert loaded["symbol"].tolist() == ["SMCI"]
    assert loaded["current_shares"].tolist() == [60]
    assert loaded["campaign_stop"].isna().tolist() == [True]


def test_load_planned_stops_returns_empty_frame_for_missing_file(tmp_path):
    loaded = load_planned_stops(tmp_path / "missing.csv")

    assert loaded.empty
    assert loaded.columns.tolist() == PLANNED_STOP_COLUMNS


def test_save_and_load_planned_stops_preserves_cleaned_columns(tmp_path):
    path = tmp_path / "planned_stops.csv"
    save_planned_stops(
        pd.DataFrame(
            [
                {
                    "symbol": "aapl",
                    "buy_date": "2026-04-01",
                    "quantity": "2",
                    "planned_stop": "190.50",
                    "strategy": "EP",
                    "atr": "4.25",
                    "market_regime": "selective go",
                }
            ]
        ),
        path,
    )

    loaded = load_planned_stops(path)

    assert loaded.columns.tolist() == PLANNED_STOP_COLUMNS
    assert loaded["symbol"].tolist() == ["AAPL"]
    assert loaded["quantity"].tolist() == [2.0]
    assert loaded["planned_stop"].tolist() == [190.50]
    assert loaded["strategy"].tolist() == ["EP"]
    assert loaded["atr"].tolist() == [4.25]
    assert loaded["market_regime"].tolist() == ["SELECTIVE GO"]


def test_load_planned_stops_accepts_old_files_without_strategy(tmp_path):
    path = tmp_path / "planned_stops.csv"
    pd.DataFrame([{"symbol": "AAPL", "buy_date": "2026-04-01", "quantity": 2, "planned_stop": 190.50}]).to_csv(
        path,
        index=False,
    )

    loaded = load_planned_stops(path)

    assert loaded.columns.tolist() == PLANNED_STOP_COLUMNS
    assert loaded["strategy"].tolist() == [""]


def test_load_planned_stops_accepts_old_files_without_atr(tmp_path):
    path = tmp_path / "planned_stops.csv"
    pd.DataFrame(
        [{"symbol": "AAPL", "buy_date": "2026-04-01", "quantity": 2, "planned_stop": 190.50, "strategy": "EP"}]
    ).to_csv(path, index=False)

    loaded = load_planned_stops(path)

    assert loaded.columns.tolist() == PLANNED_STOP_COLUMNS
    assert loaded["atr"].isna().all()


def test_load_planned_stops_accepts_old_files_without_market_regime(tmp_path):
    path = tmp_path / "planned_stops.csv"
    pd.DataFrame(
        [{"symbol": "AAPL", "buy_date": "2026-04-01", "quantity": 2, "planned_stop": 190.50, "strategy": "EP"}]
    ).to_csv(path, index=False)

    loaded = load_planned_stops(path)

    assert loaded.columns.tolist() == PLANNED_STOP_COLUMNS
    assert loaded["market_regime"].tolist() == [""]


def test_load_planned_stops_normalizes_invalid_market_regime_to_blank(tmp_path):
    path = tmp_path / "planned_stops.csv"
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "buy_date": "2026-04-01",
                "quantity": 2,
                "planned_stop": 190.50,
                "market_regime": "BAD",
            }
        ]
    ).to_csv(path, index=False)

    loaded = load_planned_stops(path)

    assert loaded["market_regime"].tolist() == [""]


def test_save_and_load_robinhood_transactions_preserves_cleaned_columns(tmp_path):
    path = tmp_path / "robinhood_transactions.csv"
    save_robinhood_transactions(_transactions([{"symbol": "aapl", "quantity": "2", "price": "190.50"}]), path)

    loaded = load_robinhood_transactions(path)

    assert loaded.columns.tolist() == TRANSACTION_COLUMNS
    assert loaded["symbol"].tolist() == ["AAPL"]
    assert loaded["quantity"].tolist() == [2.0]
    assert loaded["price"].tolist() == [190.50]


def test_append_robinhood_transactions_adds_unseen_rows_and_skips_duplicates():
    existing = _transactions(
        [
            {"activity_date": "2026-04-01", "symbol": "AAPL", "trans_code": "Buy", "quantity": 1, "price": 100},
        ]
    )
    incoming = _transactions(
        [
            {"activity_date": "2026-04-01", "symbol": "aapl", "trans_code": "Buy", "quantity": 1, "price": 100},
            {"activity_date": "2026-04-02", "symbol": "AAPL", "trans_code": "Sell", "quantity": 1, "price": 110},
        ]
    )

    result, added_count = append_robinhood_transactions(existing, incoming)

    assert added_count == 1
    assert result["trans_code"].tolist() == ["Buy", "Sell"]


def test_append_robinhood_transactions_preserves_identical_fills_in_initial_report():
    identical_fills = _transactions(
        [
            {"symbol": "BHE", "trans_code": "Sell", "quantity": 2, "price": 85.49},
            {"symbol": "BHE", "trans_code": "Sell", "quantity": 2, "price": 85.49},
        ]
    )

    result, added_count = append_robinhood_transactions(pd.DataFrame(columns=TRANSACTION_COLUMNS), identical_fills)

    assert added_count == 2
    assert result[["symbol", "trans_code", "quantity", "price"]].to_dict("records") == [
        {"symbol": "BHE", "trans_code": "Sell", "quantity": 2, "price": 85.49},
        {"symbol": "BHE", "trans_code": "Sell", "quantity": 2, "price": 85.49},
    ]


def test_append_robinhood_transactions_reupload_preserves_identical_fill_count():
    report = _transactions(
        [
            {"symbol": "BHE", "trans_code": "Sell", "quantity": 2, "price": 85.49},
            {"symbol": "BHE", "trans_code": "Sell", "quantity": 2, "price": 85.49},
        ]
    )

    persisted, first_count = append_robinhood_transactions(pd.DataFrame(columns=TRANSACTION_COLUMNS), report)
    persisted, second_count = append_robinhood_transactions(persisted, report)

    assert first_count == 2
    assert second_count == 0
    assert len(persisted) == 2


def test_append_robinhood_transactions_adds_only_missing_identical_occurrences():
    existing = _transactions(
        [
            {
                "symbol": "BHE",
                "description": "Benchmark ElectronicsCUSIP: 08160H101",
                "trans_code": "Sell",
                "quantity": 2,
                "price": 85.49,
            }
        ]
    )
    incoming = _transactions(
        [
            {
                "symbol": "BHE",
                "description": "Benchmark Electronics\nCUSIP: 08160H101",
                "trans_code": "Sell",
                "quantity": 2,
                "price": 85.49,
            },
            {
                "symbol": "BHE",
                "description": "Benchmark Electronics\nCUSIP: 08160H101",
                "trans_code": "Sell",
                "quantity": 2,
                "price": 85.49,
            },
        ]
    )

    result, added_count = append_robinhood_transactions(existing, incoming)

    assert added_count == 1
    assert len(result) == 2


def test_append_robinhood_transactions_does_not_remove_excess_stored_occurrences():
    existing = _transactions(
        [
            {"symbol": "BHE", "trans_code": "Sell", "quantity": 2, "price": 85.49},
            {"symbol": "BHE", "trans_code": "Sell", "quantity": 2, "price": 85.49},
        ]
    )
    incoming = existing.iloc[[0]].reset_index(drop=True)

    result, added_count = append_robinhood_transactions(existing, incoming)

    assert added_count == 0
    assert len(result) == 2


def test_identical_bhe_sell_fills_close_the_full_position():
    report = _transactions(
        [
            {
                "activity_date": "2026-06-02",
                "process_date": "2026-06-02",
                "settle_date": "2026-06-03",
                "symbol": "BHE",
                "description": "Benchmark Electronics\nCUSIP: 08160H101",
                "trans_code": "Buy",
                "quantity": 9,
                "price": 87.97,
            },
            {
                "activity_date": "2026-06-04",
                "process_date": "2026-06-04",
                "settle_date": "2026-06-05",
                "symbol": "BHE",
                "description": "Benchmark Electronics\nCUSIP: 08160H101",
                "trans_code": "Sell",
                "quantity": 5,
                "price": 85.49,
            },
            {
                "activity_date": "2026-06-04",
                "process_date": "2026-06-04",
                "settle_date": "2026-06-05",
                "symbol": "BHE",
                "description": "Benchmark Electronics\nCUSIP: 08160H101",
                "trans_code": "Sell",
                "quantity": 2,
                "price": 85.49,
            },
            {
                "activity_date": "2026-06-04",
                "process_date": "2026-06-04",
                "settle_date": "2026-06-05",
                "symbol": "BHE",
                "description": "Benchmark Electronics\nCUSIP: 08160H101",
                "trans_code": "Sell",
                "quantity": 2,
                "price": 85.49,
            },
        ]
    )

    persisted, added_count = append_robinhood_transactions(pd.DataFrame(columns=TRANSACTION_COLUMNS), report)
    derived = derive_fifo_trades(persisted)

    assert added_count == 4
    assert derived.open_lots.empty
    assert derived.closed_trades[["symbol", "quantity"]].to_dict("records") == [{"symbol": "BHE", "quantity": 9}]
    assert derived.exit_matches["quantity"].tolist() == [5, 2, 2]


def test_reuploading_same_robinhood_transactions_does_not_double_count_metrics():
    first_upload = _transactions(
        [
            {"activity_date": "2026-04-01", "symbol": "AAPL", "trans_code": "Buy", "quantity": 1, "price": 100},
            {"activity_date": "2026-04-02", "symbol": "AAPL", "trans_code": "Sell", "quantity": 1, "price": 110},
        ]
    )
    second_upload = _transactions(
        [
            {"activity_date": "2026-04-01", "symbol": "aapl", "trans_code": "Buy", "quantity": 1, "price": 100},
            {"activity_date": "2026-04-02", "symbol": "AAPL", "trans_code": "Sell", "quantity": 1, "price": 110},
            {"activity_date": "2026-04-03", "symbol": "MSFT", "trans_code": "Buy", "quantity": 2, "price": 50},
            {"activity_date": "2026-04-04", "symbol": "MSFT", "trans_code": "Sell", "quantity": 2, "price": 55},
        ]
    )

    persisted, first_count = append_robinhood_transactions(pd.DataFrame(columns=TRANSACTION_COLUMNS), first_upload)
    persisted, second_count = append_robinhood_transactions(persisted, second_upload)
    metrics = calculate_trade_metrics(derive_fifo_trades(persisted).closed_trades)

    assert first_count == 2
    assert second_count == 2
    assert metrics["trade_count"] == 2
    assert metrics["expectancy"] == 10.0


def test_generate_planned_stops_from_transactions_uses_buy_rows_with_blank_stops():
    transactions = _transactions(
        [
            {"activity_date": "2026-04-01", "symbol": "AAPL", "trans_code": "Buy", "quantity": 1, "price": 100},
            {"activity_date": "2026-04-02", "symbol": "AAPL", "trans_code": "Sell", "quantity": 1, "price": 110},
            {"activity_date": "2026-04-03", "symbol": "MSFT", "trans_code": "Buy", "quantity": 2, "price": 50},
        ]
    )

    result = generate_planned_stops_from_transactions(transactions)

    assert result.columns.tolist() == PLANNED_STOP_COLUMNS
    assert result[["symbol", "buy_date", "quantity"]].to_dict("records") == [
        {"symbol": "AAPL", "buy_date": "2026-04-01", "quantity": 1.0},
        {"symbol": "MSFT", "buy_date": "2026-04-03", "quantity": 2.0},
    ]
    assert result["planned_stop"].isna().all()
    assert result["strategy"].tolist() == ["", ""]
    assert result["market_regime"].tolist() == ["", ""]


def test_upsert_planned_stop_records_calculated_entry_stop():
    planned_stops = pd.DataFrame(columns=PLANNED_STOP_COLUMNS)
    calculated_position = pd.Series(
        {
            "symbol": "aapl",
            "buy_date": "2026-04-01",
            "number_of_shares": 2,
            "stop_price": 190.5,
            "strategy": "BO",
            "atr": 4.25,
            "market_regime": "NO-GO",
        }
    )

    result = upsert_planned_stop(planned_stops, calculated_position)

    assert result.to_dict("records") == [
        {
            "symbol": "AAPL",
            "buy_date": "2026-04-01",
            "quantity": 2,
            "planned_stop": 190.5,
            "strategy": "BO",
            "atr": 4.25,
            "market_regime": "NO-GO",
        }
    ]


def test_upsert_planned_stop_preserves_existing_market_regime_when_missing_from_update():
    planned_stops = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "buy_date": "2026-04-01",
                "quantity": 2,
                "planned_stop": 180,
                "strategy": "EP",
                "atr": 4.0,
                "market_regime": "SELECTIVE GO",
            }
        ],
        columns=PLANNED_STOP_COLUMNS,
    )
    calculated_position = pd.Series(
        {
            "symbol": "aapl",
            "buy_date": "2026-04-01",
            "number_of_shares": 2,
            "stop_price": 190.5,
            "strategy": "BO",
            "atr": 4.25,
        }
    )

    result = upsert_planned_stop(planned_stops, calculated_position)

    assert result.iloc[0]["market_regime"] == "SELECTIVE GO"


def test_robinhood_planned_stop_lookup_does_not_depend_on_active_positions():
    transactions = _transactions(
        [
            {"activity_date": "2026-04-01", "symbol": "AAPL", "trans_code": "Buy", "quantity": 1, "price": 100},
            {"activity_date": "2026-04-02", "symbol": "AAPL", "trans_code": "Sell", "quantity": 1, "price": 110},
        ]
    )
    planned_stops = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "buy_date": "2026-04-01",
                "quantity": 1,
                "planned_stop": 95,
                "strategy": "EP",
                "atr": 2.5,
                "market_regime": "GO",
            }
        ],
        columns=PLANNED_STOP_COLUMNS,
    )

    result = derive_fifo_trades(transactions, planned_stops)

    assert result.closed_trades.iloc[0]["planned_stop"] == 95
    assert result.closed_trades.iloc[0]["strategy"] == "EP"
    assert result.closed_trades.iloc[0]["atr"] == 2.5
    assert result.closed_trades.iloc[0]["market_regime"] == "GO"
    assert result.missing_planned_stops == 0


def test_build_storage_backend_uses_local_csv_when_sheets_secrets_are_missing():
    backend = build_storage_backend({})

    assert isinstance(backend, LocalCsvStorage)


def test_build_storage_backend_rejects_partial_google_sheets_secrets():
    with pytest.raises(StorageConfigurationError, match="requires both"):
        build_storage_backend({"google_sheets": {"spreadsheet_id": "sheet-id"}})


def test_build_storage_backend_uses_google_sheets_for_complete_secrets():
    spreadsheet = FakeSpreadsheet()
    backend = build_storage_backend(
        {
            "google_sheets": {"spreadsheet_id": "sheet-id"},
            "gcp_service_account": {"client_email": "bot@example.com"},
        },
        gspread_module=FakeGspreadModule(spreadsheet),
    )

    assert isinstance(backend, GoogleSheetsStorage)


def test_google_sheets_empty_sheet_returns_normalized_empty_frame_and_header():
    worksheet = FakeWorksheet([])
    backend = GoogleSheetsStorage(FakeSpreadsheet({"positions": worksheet}))

    loaded = backend.load_positions()

    assert loaded.empty
    assert loaded.columns.tolist() == POSITION_SOURCE_COLUMNS
    assert worksheet.values == [POSITION_SOURCE_COLUMNS]


def test_google_sheets_save_writes_headers_and_normalized_rows():
    worksheet = FakeWorksheet([])
    backend = GoogleSheetsStorage(FakeSpreadsheet({"planned_stops": worksheet}))

    backend.save_planned_stops(
        pd.DataFrame(
            [{"symbol": "aapl", "buy_date": "2026-04-01", "quantity": "2", "planned_stop": "190.50", "strategy": "EP"}]
        )
    )

    assert worksheet.values == [
        PLANNED_STOP_COLUMNS,
        ["AAPL", "2026-04-01", 2.0, 190.5, "EP", "", ""],
    ]


def test_google_sheets_positions_archive_is_created_and_normalized():
    spreadsheet = FakeSpreadsheet()
    backend = GoogleSheetsStorage(spreadsheet)
    archive = pd.DataFrame(
        [
            {
                "symbol": "aapl",
                "buy_date": "2026-04-01",
                "share_price": 100,
                "stop_price": 95,
                "strategy": "EP",
                "number_of_shares": 20,
            }
        ]
    )

    backend.save_positions_archive(archive)

    assert spreadsheet.worksheets["positions_archive"].values[0] == POSITION_ARCHIVE_COLUMNS
    assert spreadsheet.worksheets["positions_archive"].values[1][1:5] == ["AAPL", "2026-04-01", 100, 95]


def test_google_sheets_campaign_overrides_is_created_and_normalized():
    spreadsheet = FakeSpreadsheet()
    backend = GoogleSheetsStorage(spreadsheet)

    backend.save_campaign_overrides(
        pd.DataFrame(
            [
                {
                    "symbol": "smci",
                    "lots": "2",
                    "current_shares": "66",
                    "avg_entry": "44.90",
                    "campaign_stop": "44.00",
                    "sell_lot": "22",
                    "position_size": "2963.58",
                    "risk_at_campaign_stop": "59.58",
                    "planned_lot_risk": "217.08",
                    "strategy": "Mixed",
                    "source": "Robinhood",
                }
            ]
        )
    )

    assert spreadsheet.worksheets["campaign_overrides"].values == [
        CAMPAIGN_OVERRIDE_COLUMNS,
        ["SMCI", 66, 44],
    ]


def test_google_sheets_load_preserves_expected_column_order():
    worksheet = FakeWorksheet(
        [
            ["price", "symbol", "activity_date", "trans_code", "quantity"],
            ["100", "aapl", "2026-04-01", "Buy", "1"],
        ]
    )
    backend = GoogleSheetsStorage(FakeSpreadsheet({"robinhood_transactions": worksheet}))

    loaded = backend.load_robinhood_transactions()

    assert loaded.columns.tolist() == TRANSACTION_COLUMNS
    assert loaded[["symbol", "activity_date", "trans_code", "quantity", "price"]].to_dict("records") == [
        {"symbol": "AAPL", "activity_date": "2026-04-01", "trans_code": "Buy", "quantity": 1, "price": 100}
    ]


def test_google_sheets_robinhood_append_deduplicates_before_save():
    worksheet = FakeWorksheet([])
    backend = GoogleSheetsStorage(FakeSpreadsheet({"robinhood_transactions": worksheet}))
    existing = _transactions([{"activity_date": "2026-04-01", "symbol": "AAPL", "trans_code": "Buy"}])
    incoming = _transactions(
        [
            {"activity_date": "2026-04-01", "symbol": "aapl", "trans_code": "Buy"},
            {"activity_date": "2026-04-02", "symbol": "MSFT", "trans_code": "Buy"},
        ]
    )

    persisted, added_count = append_robinhood_transactions(existing, incoming)
    backend.save_robinhood_transactions(persisted)

    assert added_count == 1
    assert [row[3] for row in worksheet.values[1:]] == ["AAPL", "MSFT"]


def test_google_sheets_planned_stop_upsert_replaces_by_symbol_date_and_quantity():
    worksheet = FakeWorksheet([])
    backend = GoogleSheetsStorage(FakeSpreadsheet({"planned_stops": worksheet}))
    planned_stops = pd.DataFrame(
        [{"symbol": "AAPL", "buy_date": "2026-04-01", "quantity": 2, "planned_stop": 180, "strategy": "EP"}],
        columns=PLANNED_STOP_COLUMNS,
    )
    calculated_position = pd.Series(
        {
            "symbol": "aapl",
            "buy_date": "2026-04-01",
            "number_of_shares": 2,
            "stop_price": 190.5,
            "strategy": "4% BO",
            "atr": 4.25,
            "market_regime": "SELECTIVE GO",
        }
    )

    updated = upsert_planned_stop(planned_stops, calculated_position)
    backend.save_planned_stops(updated)

    assert worksheet.values == [
        PLANNED_STOP_COLUMNS,
        ["AAPL", "2026-04-01", 2, 190.5, "4% BO", 4.25, "SELECTIVE GO"],
    ]


class WorksheetNotFound(Exception):
    pass


class FakeWorksheet:
    def __init__(self, values: list[list[object]]):
        self.values = values

    def get_all_values(self):
        return self.values

    def clear(self):
        self.values = []

    def update(self, values, value_input_option=None):
        self.values = values


class FakeSpreadsheet:
    def __init__(self, worksheets: dict[str, FakeWorksheet] | None = None):
        self.worksheets = worksheets or {}

    def worksheet(self, title: str):
        if title not in self.worksheets:
            raise WorksheetNotFound(title)
        return self.worksheets[title]

    def add_worksheet(self, title: str, rows: int, cols: int):
        worksheet = FakeWorksheet([])
        self.worksheets[title] = worksheet
        return worksheet


class FakeGspreadModule:
    def __init__(self, spreadsheet: FakeSpreadsheet):
        self.spreadsheet = spreadsheet

    def service_account_from_dict(self, service_account):
        return FakeGspreadClient(self.spreadsheet)


class FakeGspreadClient:
    def __init__(self, spreadsheet: FakeSpreadsheet):
        self.spreadsheet = spreadsheet

    def open_by_key(self, spreadsheet_id: str):
        return self.spreadsheet


def _transactions(overrides: list[dict]) -> pd.DataFrame:
    rows = []
    for override in overrides:
        row = {
            "activity_date": "2026-04-01",
            "process_date": "2026-04-01",
            "settle_date": "2026-04-02",
            "symbol": "AAPL",
            "description": "Apple",
            "trans_code": "Buy",
            "quantity": 1,
            "price": 100,
            "amount": -100,
        }
        row.update(override)
        if "amount" not in override:
            multiplier = -1 if row["trans_code"] == "Buy" else 1
            row["amount"] = multiplier * float(row["quantity"]) * float(row["price"])
        rows.append(row)
    return pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)
