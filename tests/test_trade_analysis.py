from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stock_calculator import cli
from stock_calculator.trade_analysis import (
    BarCache,
    MetadataLookup,
    _entry_features,
    _quality_labels,
    _risk_features,
    _build_analysis_bundle,
    _empty_enriched_trades,
    _render_summary,
    _strategy_breakdown,
    _strategy_folder_slug,
    _threshold_tests,
    _winner_loser_summary,
    enrich_trades,
    load_bigquery_metadata,
    prepare_trades,
)


def test_entry_features_use_prior_completed_bar_not_entry_day_bar():
    bars = pd.DataFrame(
        [
            {"symbol": "TEST", "bar_date": date(2026, 1, 8), "open": 99, "high": 101, "low": 98, "close": 100, "volume": 1000},
            {"symbol": "TEST", "bar_date": date(2026, 1, 9), "open": 100, "high": 106, "low": 99, "close": 105, "volume": 1200},
            {"symbol": "TEST", "bar_date": date(2026, 1, 12), "open": 900, "high": 1000, "low": 800, "close": 999, "volume": 999999},
        ]
    )

    features = _entry_features(bars, pd.DataFrame(), date(2026, 1, 12), 110)

    assert features["entry_gap_pct"] == 4.76


def test_entry_features_use_previous_trading_bar_for_non_trading_buy_date():
    bars = pd.DataFrame(
        [
            {"symbol": "TEST", "bar_date": date(2026, 1, 8), "open": 99, "high": 101, "low": 98, "close": 100, "volume": 1000},
            {"symbol": "TEST", "bar_date": date(2026, 1, 9), "open": 100, "high": 106, "low": 99, "close": 105, "volume": 1200},
        ]
    )

    features = _entry_features(bars, pd.DataFrame(), date(2026, 1, 10), 110)

    assert features["entry_gap_pct"] == 4.76


def test_entry_features_compute_deterministic_indicators_from_prior_bars_only():
    bars = _constant_bars("TEST", close=100, high=102, low=98, volume=100)
    spy_bars = _constant_bars("SPY", close=200, high=202, low=198, volume=100)

    features = _entry_features(bars, spy_bars, date(2026, 1, 12), 105)

    assert features["entry_gap_pct"] == 5.0
    assert features["distance_from_10ema_pct"] == 5.0
    assert features["distance_from_20ema_pct"] == 5.0
    assert features["distance_from_50sma_pct"] == 5.0
    assert features["distance_from_52w_high_pct"] == 2.94
    assert features["entry_volume_vs_50d_avg"] == 1.0
    assert features["computed_atr_21_pct"] == 4.0
    assert features["momentum_1m_pct"] == 0.0
    assert features["momentum_3m_pct"] == 0.0
    assert features["momentum_6m_pct"] == 0.0
    assert features["rs_vs_spy_1m_pct"] == 0.0
    assert features["rs_vs_spy_3m_pct"] == 0.0
    assert features["rs_vs_spy_6m_pct"] == 0.0


def test_quality_labels_keep_near_20ema_and_pullback_distinct():
    assert _quality_labels({"distance_from_20ema_pct": 2})["near_20ema"] is True
    assert _quality_labels({"distance_from_20ema_pct": 2})["pullback_entry"] is False

    assert _quality_labels({"distance_from_20ema_pct": -4})["near_20ema"] is False
    assert _quality_labels({"distance_from_20ema_pct": -4})["pullback_entry"] is True

    labels = _quality_labels({"distance_from_20ema_pct": -2})
    assert labels["near_20ema"] is True
    assert labels["pullback_entry"] is True


def test_enrich_trades_blanks_invalid_r_multiple_and_marks_missing_history():
    trades = prepare_trades(
        pd.DataFrame(
            [
                {
                    "symbol": "TEST",
                    "buy_date": "2026-01-12",
                    "sell_date": "2026-01-13",
                    "quantity": 10,
                    "planned_stop": 101,
                    "strategy": "EP",
                    "atr": 2.5,
                    "market_regime": "",
                    "buy_price": 100,
                    "buy_amount": 1000,
                    "sell_price": 95,
                    "sell_amount": 950,
                    "realized_pnl": -50,
                    "realized_pnl_percent": -5,
                    "hold_days": 1,
                    "num_buy_fills": 2,
                }
            ]
        )
    )
    bars = pd.DataFrame(
        [
            {"symbol": "TEST", "bar_date": date(2026, 1, 9), "open": 99, "high": 101, "low": 98, "close": 100, "volume": 1000},
            {"symbol": "TEST", "bar_date": date(2026, 1, 12), "open": 100, "high": 103, "low": 97, "close": 99, "volume": 2000},
            {"symbol": "TEST", "bar_date": date(2026, 1, 13), "open": 99, "high": 102, "low": 94, "close": 95, "volume": 2200},
        ]
    )

    enriched = enrich_trades(trades, BarCache({"TEST": bars}), MetadataLookup({}, skipped=True))
    row = enriched.iloc[0]

    assert pd.isna(row["r_multiple"])
    assert row["outcome"] == "loss"
    assert bool(row["is_pyramided"]) is True
    assert row["entry_feature_basis"] == "avg_cost_first_date_prior_bar"
    assert row["market_regime"] == "Unknown"
    assert pd.isna(row["distance_from_52w_high_pct"])
    assert row["max_gain_during_trade_pct"] == 3.0
    assert row["max_loss_during_trade_pct"] == -6.0


def test_risk_features_use_exact_zero_pnl_for_breakeven_outcome():
    result = _risk_features({"realized_pnl": 0, "buy_price": 100, "planned_stop": 95, "quantity": 10})

    assert result["outcome"] == "breakeven"
    assert result["winner"] is False
    assert result["r_multiple"] == 0.0


def test_bigquery_skip_returns_blank_metadata_lookup():
    lookup = load_bigquery_metadata(["TEST"], gcp_project_id=None, bigquery_dataset=None, skip_bigquery=True)

    assert lookup.skipped is True
    values = lookup.get("TEST", date(2026, 1, 1))
    assert pd.isna(values["sector"])
    assert values["classification_metadata_status"] == "skipped"
    assert values["signal_metadata_status"] == "skipped"


def test_bigquery_missing_project_has_clear_error(monkeypatch):
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    with pytest.raises(ValueError, match="BigQuery project is required"):
        load_bigquery_metadata(["TEST"], gcp_project_id=None, bigquery_dataset=None, skip_bigquery=False)


def test_metadata_lookup_splits_latest_classification_from_point_in_time_signals():
    rows = pd.DataFrame(
        [
            {
                "symbol": "TEST",
                "snapshot_date": date(2026, 1, 1),
                "sector": "Old",
                "industry_name": "Old Industry",
                "rs_rating": 40,
            },
            {
                "symbol": "TEST",
                "snapshot_date": date(2026, 1, 10),
                "sector": "Current",
                "industry_name": "Current Industry",
                "rs_rating": 70,
            },
            {
                "symbol": "TEST",
                "snapshot_date": date(2026, 1, 20),
                "sector": "Future",
                "industry_name": "Future Industry",
                "rs_rating": 90,
            },
        ]
    )
    lookup = MetadataLookup({"TEST": rows})

    values = lookup.get("TEST", date(2026, 1, 15))
    assert values["sector"] == "Future"
    assert values["industry_name"] == "Future Industry"
    assert values["classification_snapshot_date"] == date(2026, 1, 20)
    assert values["classification_metadata_status"] == "present"
    assert values["rs_rating"] == 70
    assert values["signal_snapshot_date"] == date(2026, 1, 10)
    assert values["signal_metadata_status"] == "present"

    before_all = lookup.get("TEST", date(2025, 12, 31))
    assert before_all["sector"] == "Future"
    assert pd.isna(before_all["rs_rating"])
    assert before_all["signal_metadata_status"] == "no_snapshot_before_buy"

    no_buy_date = lookup.get("TEST", None)
    assert no_buy_date["sector"] == "Future"
    assert pd.isna(no_buy_date["rs_rating"])
    assert no_buy_date["signal_metadata_status"] == "no_buy_date"

    missing = lookup.get("MISSING", date(2026, 1, 15))
    assert pd.isna(missing["sector"])
    assert missing["classification_metadata_status"] == "missing_symbol"
    assert missing["signal_metadata_status"] == "missing_symbol"


def test_bigquery_missing_schema_has_clear_error(monkeypatch):
    _patch_bigquery_client(monkeypatch, _FakeBigQueryClient(schema=["snapshot_date", "universe_type", "symbol", "sector"]))

    with pytest.raises(ValueError, match="missing required columns: industry_name"):
        load_bigquery_metadata(["TEST"], gcp_project_id="project", bigquery_dataset="dataset", skip_bigquery=False)


def test_bigquery_access_failure_has_clear_error(monkeypatch):
    _patch_bigquery_client(monkeypatch, _FakeBigQueryClient(access_error=RuntimeError("no auth")))

    with pytest.raises(ValueError, match="Could not access BigQuery table project.dataset.universe_snapshots"):
        load_bigquery_metadata(["TEST"], gcp_project_id="project", bigquery_dataset="dataset", skip_bigquery=False)


def test_bigquery_empty_query_results_return_blank_metadata(monkeypatch):
    _patch_bigquery_client(monkeypatch, _FakeBigQueryClient(query_rows=[]))

    lookup = load_bigquery_metadata(["TEST"], gcp_project_id="project", bigquery_dataset="dataset", skip_bigquery=False)

    assert lookup.skipped is False
    values = lookup.get("TEST", date(2026, 1, 1))
    assert pd.isna(values["sector"])
    assert values["classification_metadata_status"] == "missing_symbol"
    assert values["signal_metadata_status"] == "missing_symbol"


def test_bigquery_query_is_symbol_bounded_without_date_cap(monkeypatch):
    fake_client = _FakeBigQueryClient(query_rows=[])
    _patch_bigquery_client(monkeypatch, fake_client)

    load_bigquery_metadata(["TEST"], gcp_project_id="project", bigquery_dataset="dataset", skip_bigquery=False)

    assert "snapshot_date <= @max_buy_date" not in fake_client.query_sql
    assert [parameter.name for parameter in fake_client.query_parameters] == ["symbols"]


def test_empty_enriched_schema_includes_metadata_provenance_columns():
    frame = _empty_enriched_trades()

    for column in [
        "classification_snapshot_date",
        "signal_snapshot_date",
        "classification_metadata_status",
        "signal_metadata_status",
    ]:
        assert column in frame.columns


def test_report_aggregations_handle_outcomes_strategy_columns_and_thresholds():
    rows = []
    for index in range(10):
        rows.append(_analysis_row("EP", "win", True, 0.0, 10.0, 1.0 + index / 10))
    rows.append(_analysis_row("EP", "loss", False, 3.0, -5.0, -0.5))
    rows.append(_analysis_row("EP", "breakeven", False, 6.0, 0.0, 0.0))
    enriched = pd.DataFrame(rows)

    winners_vs_losers = _winner_loser_summary(enriched)
    strategy = _strategy_breakdown(enriched)
    thresholds = _threshold_tests(enriched)

    assert set(winners_vs_losers["group"]) == {"Winner", "Loser", "Breakeven"}
    breakeven_return = winners_vs_losers[
        (winners_vs_losers["group"] == "Breakeven") & (winners_vs_losers["feature"] == "realized_pnl_percent")
    ].iloc[0]
    assert breakeven_return["trade_count"] == 1
    assert breakeven_return["average"] == 0.0
    assert "average_r_multiple" in strategy.columns
    assert "expectancy_r" not in strategy.columns

    distance_buckets = thresholds[thresholds["feature"] == "distance_from_20ema_pct"]
    bucket_0_3 = distance_buckets[distance_buckets["bucket"] == "0-3"].iloc[0]
    bucket_3_6 = distance_buckets[distance_buckets["bucket"] == "3-6"].iloc[0]
    assert bucket_0_3["trade_count"] == 10
    assert bool(bucket_0_3["directional_only"]) is False
    assert bucket_3_6["trade_count"] == 1
    assert bool(bucket_3_6["directional_only"]) is True


def test_strategy_folder_slug_uses_known_mapping_and_safe_fallback():
    assert _strategy_folder_slug("EP") == "ep"
    assert _strategy_folder_slug("4% BO") == "4pct-bo"
    assert _strategy_folder_slug("Pullback") == "pullback"
    assert _strategy_folder_slug("") == "unclassified"
    assert _strategy_folder_slug("Mean Reversion / Swing") == "mean-reversion-swing"


def test_strategy_bundle_reuses_existing_aggregation_logic(tmp_path):
    rows = []
    for index in range(10):
        rows.append(_analysis_row("EP", "win", True, 0.0, 10.0, 1.0 + index / 10))
    rows.append(_analysis_row("4% BO", "loss", False, 3.0, -5.0, -0.5))
    enriched = pd.DataFrame(rows)

    ep_subset = enriched[enriched["strategy"] == "EP"].copy()
    bundle = _build_analysis_bundle(
        ep_subset,
        output_dir=tmp_path / "ep",
        bigquery_skipped=True,
        title="Trade Analysis - EP",
    )

    pd.testing.assert_frame_equal(bundle.winners_vs_losers, _winner_loser_summary(ep_subset))
    pd.testing.assert_frame_equal(bundle.threshold_tests, _threshold_tests(ep_subset))
    assert bundle.summary.startswith("# Trade Analysis - EP")


def test_summary_uses_metadata_status_counts_and_handles_blank_snapshot_dates():
    enriched = pd.DataFrame(
        [
            {
                "buy_date": "2026-01-01",
                "strategy": "EP",
                "market_regime": "Unknown",
                "is_pyramided": False,
                "classification_metadata_status": "skipped",
                "signal_metadata_status": "skipped",
                "classification_snapshot_date": pd.NA,
                "signal_snapshot_date": pd.NA,
            }
        ]
    )

    summary = _render_summary(
        enriched,
        pd.DataFrame(columns=["group", "feature", "trade_count", "average", "median"]),
        pd.DataFrame(),
        pd.DataFrame(columns=["trade_count"]),
        True,
    )

    assert "Classification metadata status:" in summary
    assert "  - skipped: 1" in summary
    assert "Signal metadata status:" in summary
    assert "Classification snapshot date range" not in summary
    assert summary.startswith("# Trade Analysis")


def test_trade_analysis_cli_smoke_writes_reports(tmp_path, monkeypatch, capsys):
    market_root = tmp_path / "market"
    month_dir = market_root / "datasets" / "day_aggs_v1" / "year=2026" / "month=01"
    month_dir.mkdir(parents=True)
    bars = pd.DataFrame(
        [
            {"symbol": "TEST", "bar_date": "2026-01-08", "open": 99, "high": 101, "low": 98, "close": 100, "volume": 1000},
            {"symbol": "FOUR", "bar_date": "2026-01-08", "open": 49, "high": 51, "low": 48, "close": 50, "volume": 1000},
            {"symbol": "BLANK", "bar_date": "2026-01-08", "open": 29, "high": 31, "low": 28, "close": 30, "volume": 1000},
            {"symbol": "SPY", "bar_date": "2026-01-08", "open": 499, "high": 501, "low": 498, "close": 500, "volume": 1000},
            {"symbol": "TEST", "bar_date": "2026-01-09", "open": 100, "high": 106, "low": 99, "close": 105, "volume": 1200},
            {"symbol": "FOUR", "bar_date": "2026-01-09", "open": 50, "high": 56, "low": 49, "close": 55, "volume": 1200},
            {"symbol": "BLANK", "bar_date": "2026-01-09", "open": 30, "high": 36, "low": 29, "close": 35, "volume": 1200},
            {"symbol": "SPY", "bar_date": "2026-01-09", "open": 500, "high": 506, "low": 499, "close": 505, "volume": 1200},
            {"symbol": "TEST", "bar_date": "2026-01-12", "open": 105, "high": 116, "low": 104, "close": 115, "volume": 2000},
            {"symbol": "FOUR", "bar_date": "2026-01-12", "open": 55, "high": 62, "low": 54, "close": 60, "volume": 2000},
            {"symbol": "BLANK", "bar_date": "2026-01-12", "open": 35, "high": 40, "low": 34, "close": 38, "volume": 2000},
            {"symbol": "SPY", "bar_date": "2026-01-12", "open": 505, "high": 511, "low": 504, "close": 510, "volume": 1300},
        ]
    )
    bars.to_parquet(month_dir / "bars-through-2026-01-12-test.parquet", index=False)
    transactions = pd.DataFrame(
        [
            {"activity_date": "2026-01-09", "symbol": "TEST", "trans_code": "Buy", "quantity": 10, "price": 105},
            {"activity_date": "2026-01-12", "symbol": "TEST", "trans_code": "Sell", "quantity": 10, "price": 115},
            {"activity_date": "2026-01-09", "symbol": "FOUR", "trans_code": "Buy", "quantity": 10, "price": 55},
            {"activity_date": "2026-01-12", "symbol": "FOUR", "trans_code": "Sell", "quantity": 10, "price": 60},
            {"activity_date": "2026-01-09", "symbol": "BLANK", "trans_code": "Buy", "quantity": 10, "price": 35},
            {"activity_date": "2026-01-12", "symbol": "BLANK", "trans_code": "Sell", "quantity": 10, "price": 38},
        ]
    )
    planned = pd.DataFrame(
        [
            {"symbol": "TEST", "buy_date": "2026-01-09", "quantity": 10, "planned_stop": 100, "strategy": "EP"},
            {"symbol": "FOUR", "buy_date": "2026-01-09", "quantity": 10, "planned_stop": 50, "strategy": "4% BO"},
            {"symbol": "BLANK", "buy_date": "2026-01-09", "quantity": 10, "planned_stop": 30, "strategy": ""},
        ]
    )
    monkeypatch.setattr("stock_calculator.trade_analysis.load_robinhood_transactions", lambda: transactions)
    monkeypatch.setattr("stock_calculator.trade_analysis.load_planned_stops", lambda: planned)

    output_dir = tmp_path / "reports"
    exit_code = cli.main(
        [
            "trade-analysis",
            "--market-data-root",
            market_root.as_posix(),
            "--output-dir",
            output_dir.as_posix(),
            "--skip-bigquery",
        ]
    )

    assert exit_code == 0
    assert "Wrote trade analysis reports" in capsys.readouterr().out
    assert (output_dir / "enriched_trades.csv").exists()
    assert (output_dir / "winners_vs_losers.csv").exists()
    assert (output_dir / "strategy_breakdown.csv").exists()
    assert (output_dir / "threshold_tests.csv").exists()
    assert (output_dir / "summary.md").exists()
    report_files = ["enriched_trades.csv", "winners_vs_losers.csv", "strategy_breakdown.csv", "threshold_tests.csv", "summary.md"]
    for strategy_slug in ["ep", "4pct-bo", "unclassified"]:
        for report_file in report_files:
            assert (output_dir / "strategies" / strategy_slug / report_file).exists()
    enriched_header = (output_dir / "enriched_trades.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "classification_snapshot_date" in enriched_header
    assert "signal_snapshot_date" in enriched_header
    assert "classification_metadata_status" in enriched_header
    assert "signal_metadata_status" in enriched_header
    top_level_enriched = pd.read_csv(output_dir / "enriched_trades.csv")
    ep_enriched = pd.read_csv(output_dir / "strategies" / "ep" / "enriched_trades.csv")
    assert len(top_level_enriched) == 3
    assert list(top_level_enriched.columns) == list(ep_enriched.columns)
    assert len(ep_enriched) == 1
    assert (output_dir / "strategies" / "4pct-bo" / "summary.md").read_text(encoding="utf-8").startswith(
        "# Trade Analysis - 4% BO"
    )


def _constant_bars(symbol: str, *, close: float, high: float, low: float, volume: float) -> pd.DataFrame:
    prior_dates = pd.bdate_range(end="2026-01-09", periods=253)
    rows = [
        {"symbol": symbol, "bar_date": bar_date.date(), "open": close, "high": high, "low": low, "close": close, "volume": volume}
        for bar_date in prior_dates
    ]
    rows.append(
        {
            "symbol": symbol,
            "bar_date": date(2026, 1, 12),
            "open": close * 10,
            "high": high * 10,
            "low": low * 10,
            "close": close * 10,
            "volume": volume * 100,
        }
    )
    return pd.DataFrame(rows)


def _analysis_row(
    strategy: str,
    outcome: str,
    winner: bool,
    distance_20: float,
    realized_pnl_percent: float,
    r_multiple: float,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "buy_date": "2026-01-09",
        "outcome": outcome,
        "winner": winner,
        "realized_pnl_percent": realized_pnl_percent,
        "r_multiple": r_multiple,
        "distance_from_20ema_pct": distance_20,
        "computed_atr_21_pct": 4.0,
        "entry_volume_vs_50d_avg": 1.2,
        "momentum_6m_pct": 25.0,
        "rs_vs_spy_6m_pct": 5.0,
        "hold_days": 2,
    }


class _FakeField:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeTable:
    def __init__(self, schema: list[str]) -> None:
        self.schema = [_FakeField(name) for name in schema]


class _FakeJob:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def result(self) -> list[dict[str, object]]:
        return self._rows


class _FakeBigQueryClient:
    def __init__(
        self,
        *,
        schema: list[str] | None = None,
        query_rows: list[dict[str, object]] | None = None,
        access_error: Exception | None = None,
    ) -> None:
        self.schema = schema or ["snapshot_date", "universe_type", "symbol", "sector", "industry_name"]
        self.query_rows = query_rows or []
        self.access_error = access_error
        self.query_sql = ""
        self.query_parameters = []

    def get_table(self, table_id: str) -> _FakeTable:
        if self.access_error is not None:
            raise self.access_error
        return _FakeTable(self.schema)

    def query(self, query: str, *, job_config: object) -> _FakeJob:
        self.query_sql = query
        self.query_parameters = list(job_config.query_parameters)
        return _FakeJob(self.query_rows)


def _patch_bigquery_client(monkeypatch: pytest.MonkeyPatch, fake_client: _FakeBigQueryClient) -> None:
    from google.cloud import bigquery

    monkeypatch.setattr(bigquery, "Client", lambda project: fake_client)
