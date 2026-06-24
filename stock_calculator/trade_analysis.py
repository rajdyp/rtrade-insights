from __future__ import annotations

import os
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stock_calculator.robinhood import derive_fifo_trades
from stock_calculator.storage import load_planned_stops, load_robinhood_transactions


DEFAULT_OUTPUT_DIR = Path("reports/trade_analysis")
DEFAULT_MARKET_DATA_ROOT = Path("/home/rajdyp/market-data/massive")
DEFAULT_BIGQUERY_DATASET = "rtrade_analytics"
BIGQUERY_REQUIRED_COLUMNS = {"snapshot_date", "universe_type", "symbol", "sector", "industry_name"}
BIGQUERY_OPTIONAL_COLUMNS = [
    "rs_rating",
    "eps_rating",
    "comp_rating",
    "percent_off_high",
    "col_21_day_atr_percent",
]
ENTRY_FEATURE_COLUMNS = [
    "distance_from_10ema_pct",
    "distance_from_20ema_pct",
    "distance_from_50sma_pct",
    "distance_from_52w_high_pct",
    "entry_volume_vs_50d_avg",
    "computed_atr_21_pct",
    "entry_gap_pct",
    "momentum_1m_pct",
    "momentum_3m_pct",
    "momentum_6m_pct",
    "rs_vs_spy_1m_pct",
    "rs_vs_spy_3m_pct",
    "rs_vs_spy_6m_pct",
]
METADATA_COLUMNS = [
    "sector",
    "industry_name",
    "rs_rating",
    "eps_rating",
    "comp_rating",
    "marketsurge_percent_off_high",
    "marketsurge_21_day_atr_percent",
    "classification_snapshot_date",
    "signal_snapshot_date",
    "classification_metadata_status",
    "signal_metadata_status",
]
LABEL_COLUMNS = [
    "near_20ema",
    "extended_from_20ema",
    "near_52w_high",
    "high_volume_entry",
    "high_atr_risk",
    "late_breakout",
    "pullback_entry",
]
STRATEGY_FOLDER_SLUGS = {
    "EP": "ep",
    "BO": "bo",
    "4% BO": "4pct-bo",
    "Pullback": "pullback",
    "Unclassified": "unclassified",
}


@dataclass(frozen=True)
class AnalysisBundle:
    output_dir: Path
    enriched_trades: pd.DataFrame
    winners_vs_losers: pd.DataFrame
    strategy_breakdown: pd.DataFrame
    threshold_tests: pd.DataFrame
    summary: str


@dataclass(frozen=True)
class TradeAnalysisResult:
    output_dir: Path
    enriched_trades: pd.DataFrame
    winners_vs_losers: pd.DataFrame
    strategy_breakdown: pd.DataFrame
    threshold_tests: pd.DataFrame
    summary: str
    strategy_results: dict[str, AnalysisBundle] = field(default_factory=dict)


@dataclass(frozen=True)
class BarCache:
    bars_by_symbol: dict[str, pd.DataFrame]

    def get(self, symbol: str) -> pd.DataFrame:
        return self.bars_by_symbol.get(normalize_symbol(symbol), _empty_bars())


@dataclass(frozen=True)
class MetadataLookup:
    rows_by_symbol: dict[str, pd.DataFrame]
    skipped: bool = False

    def get(self, symbol: str, as_of: date | None) -> dict[str, Any]:
        if self.skipped:
            return _blank_metadata("skipped", "skipped")
        rows = self.rows_by_symbol.get(normalize_symbol(symbol))
        if rows is None or rows.empty:
            return _blank_metadata("missing_symbol", "missing_symbol")

        classification_row = rows.iloc[-1]
        values = _blank_metadata("present", "no_buy_date" if as_of is None else "no_snapshot_before_buy")
        values.update(
            {
                "sector": classification_row.get("sector"),
                "industry_name": classification_row.get("industry_name"),
                "classification_snapshot_date": classification_row.get("snapshot_date"),
            }
        )
        if as_of is None:
            return values

        dates = rows["snapshot_date"].tolist()
        index = bisect_right(dates, as_of) - 1
        if index < 0:
            return values

        signal_row = rows.iloc[index]
        values.update(
            {
                "rs_rating": signal_row.get("rs_rating"),
                "eps_rating": signal_row.get("eps_rating"),
                "comp_rating": signal_row.get("comp_rating"),
                "marketsurge_percent_off_high": signal_row.get("percent_off_high"),
                "marketsurge_21_day_atr_percent": signal_row.get("col_21_day_atr_percent"),
                "signal_snapshot_date": signal_row.get("snapshot_date"),
                "signal_metadata_status": "present",
            }
        )
        return values


def run_trade_analysis(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    market_data_root: Path = DEFAULT_MARKET_DATA_ROOT,
    gcp_project_id: str | None = None,
    bigquery_dataset: str | None = None,
    skip_bigquery: bool = False,
) -> TradeAnalysisResult:
    trades = derive_fifo_trades(load_robinhood_transactions(), load_planned_stops()).closed_trades
    return analyze_trades(
        trades,
        output_dir=output_dir,
        market_data_root=market_data_root,
        gcp_project_id=gcp_project_id,
        bigquery_dataset=bigquery_dataset,
        skip_bigquery=skip_bigquery,
    )


def analyze_trades(
    trades: pd.DataFrame,
    *,
    output_dir: Path,
    market_data_root: Path,
    gcp_project_id: str | None = None,
    bigquery_dataset: str | None = None,
    skip_bigquery: bool = False,
) -> TradeAnalysisResult:
    output_dir = Path(output_dir)
    prepared = prepare_trades(trades)
    if prepared.empty:
        enriched = _empty_enriched_trades()
        bundle = _build_analysis_bundle(enriched, output_dir=output_dir, bigquery_skipped=skip_bigquery)
        return TradeAnalysisResult(
            output_dir,
            bundle.enriched_trades,
            bundle.winners_vs_losers,
            bundle.strategy_breakdown,
            bundle.threshold_tests,
            bundle.summary,
            {},
        )

    symbols = sorted({normalize_symbol(symbol) for symbol in prepared["symbol"].dropna()} | {"SPY"})
    min_date = min(prepared["buy_date_parsed"].dropna())
    max_date = max(prepared["sell_date_parsed"].dropna())
    bar_cache = load_massive_bar_cache(
        market_data_root,
        symbols,
        start_date=min_date - timedelta(days=520),
        end_date=max_date,
    )
    metadata = load_bigquery_metadata(
        prepared["symbol"].dropna().tolist(),
        gcp_project_id=gcp_project_id,
        bigquery_dataset=bigquery_dataset,
        skip_bigquery=skip_bigquery,
    )
    enriched = enrich_trades(prepared, bar_cache, metadata)
    bundle = _build_analysis_bundle(enriched, output_dir=output_dir, bigquery_skipped=metadata.skipped)
    strategy_results = _build_strategy_analysis_bundles(enriched, output_dir=output_dir, bigquery_skipped=metadata.skipped)
    return TradeAnalysisResult(
        output_dir,
        bundle.enriched_trades,
        bundle.winners_vs_losers,
        bundle.strategy_breakdown,
        bundle.threshold_tests,
        bundle.summary,
        strategy_results,
    )


def prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    if frame.empty:
        return frame
    frame["buy_date_parsed"] = pd.to_datetime(frame.get("buy_date"), errors="coerce").dt.date
    frame["sell_date_parsed"] = pd.to_datetime(frame.get("sell_date"), errors="coerce").dt.date
    frame["symbol"] = frame.get("symbol", "").fillna("").astype(str).str.upper().str.strip()
    frame["market_regime"] = frame.get("market_regime", "").fillna("").astype(str).str.strip()
    frame.loc[frame["market_regime"] == "", "market_regime"] = "Unknown"
    if "num_buy_fills" not in frame.columns:
        frame["num_buy_fills"] = 1
    frame["num_buy_fills"] = pd.to_numeric(frame["num_buy_fills"], errors="coerce").fillna(1).astype(int)
    if "is_pyramided" not in frame.columns:
        frame["is_pyramided"] = frame["num_buy_fills"] > 1
    frame["is_pyramided"] = frame["is_pyramided"].fillna(False).astype(bool)
    frame["entry_feature_basis"] = frame["is_pyramided"].map(
        {True: "avg_cost_first_date_prior_bar", False: "single_buy_prior_bar"}
    )
    return frame


def load_massive_bar_cache(
    market_data_root: Path,
    symbols: Iterable[str],
    *,
    start_date: date,
    end_date: date,
) -> BarCache:
    wanted = {normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)}
    if not wanted:
        return BarCache({})

    paths = _monthly_bar_paths(Path(market_data_root), start_date, end_date)
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_parquet(path, filters=[("symbol", "in", sorted(wanted))])
        except Exception:
            frame = pd.read_parquet(path)
            frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
            frame = frame[frame["symbol"].isin(wanted)]
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return BarCache({})

    bars = pd.concat(frames, ignore_index=True)
    bars["symbol"] = bars["symbol"].astype(str).str.upper().str.strip()
    bars["bar_date"] = pd.to_datetime(bars["bar_date"], errors="coerce").dt.date
    bars = bars.dropna(subset=["symbol", "bar_date"]).sort_values(["symbol", "bar_date"], kind="mergesort")
    bars = bars[(bars["bar_date"] >= start_date) & (bars["bar_date"] <= end_date)]
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        bars[column] = pd.to_numeric(bars.get(column), errors="coerce")
    return BarCache({symbol: group.reset_index(drop=True) for symbol, group in bars.groupby("symbol", sort=False)})


def load_bigquery_metadata(
    symbols: list[str],
    *,
    gcp_project_id: str | None,
    bigquery_dataset: str | None,
    skip_bigquery: bool,
) -> MetadataLookup:
    if skip_bigquery:
        return MetadataLookup({}, skipped=True)

    project_id = (gcp_project_id or os.getenv("GCP_PROJECT_ID") or "").strip()
    if not project_id:
        raise ValueError("BigQuery project is required. Pass --gcp-project-id, set GCP_PROJECT_ID, or use --skip-bigquery.")
    dataset = (bigquery_dataset or os.getenv("BIGQUERY_DATASET") or DEFAULT_BIGQUERY_DATASET).strip()
    table_id = f"{project_id}.{dataset}.universe_snapshots"

    try:
        from google.api_core.exceptions import GoogleAPIError, NotFound
        from google.cloud import bigquery
    except ImportError as exc:
        raise ValueError("BigQuery enrichment requires google-cloud-bigquery. Install requirements or use --skip-bigquery.") from exc

    try:
        client = bigquery.Client(project=project_id)
        table = client.get_table(table_id)
    except Exception as exc:
        raise ValueError(f"Could not access BigQuery table {table_id}. Check ADC/GOOGLE_APPLICATION_CREDENTIALS or use --skip-bigquery.") from exc

    schema_names = {field.name for field in table.schema}
    missing = sorted(BIGQUERY_REQUIRED_COLUMNS - schema_names)
    if missing:
        raise ValueError(f"BigQuery table {table_id} is missing required columns: {', '.join(missing)}")

    optional = [column for column in BIGQUERY_OPTIONAL_COLUMNS if column in schema_names]
    select_columns = ["snapshot_date", "symbol", "sector", "industry_name", *optional]
    normalized_symbols = sorted({normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)})
    if not normalized_symbols:
        return MetadataLookup({})

    query = f"""
        SELECT {", ".join(select_columns)}
        FROM `{table_id}`
        WHERE universe_type = 'stock'
          AND symbol IN UNNEST(@symbols)
        ORDER BY symbol, snapshot_date
    """
    try:
        job = client.query(
            query,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("symbols", "STRING", normalized_symbols),
                ]
            ),
        )
        rows = [dict(row.items()) for row in job.result()]
    except (GoogleAPIError, NotFound) as exc:
        raise ValueError(f"BigQuery metadata query failed for {table_id}.") from exc

    if not rows:
        return MetadataLookup({})
    frame = pd.DataFrame(rows)
    for column in BIGQUERY_OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["symbol", "snapshot_date"]).sort_values(["symbol", "snapshot_date"], kind="mergesort")
    return MetadataLookup({symbol: group.reset_index(drop=True) for symbol, group in frame.groupby("symbol", sort=False)})


def enrich_trades(trades: pd.DataFrame, bar_cache: BarCache, metadata: MetadataLookup) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spy_bars = bar_cache.get("SPY")
    for _, trade in trades.iterrows():
        row = trade.to_dict()
        symbol = row.get("symbol")
        buy_date = row.get("buy_date_parsed")
        sell_date = row.get("sell_date_parsed")
        buy_price = _to_float(row.get("buy_price"))
        bars = bar_cache.get(str(symbol))
        row.update(_entry_features(bars, spy_bars, buy_date, buy_price))
        row.update(_outcome_features(bars, buy_date, sell_date, buy_price))
        row.update(metadata.get(str(symbol), buy_date))
        row.update(_risk_features(row))
        row.update(_quality_labels(row))
        rows.append(row)
    enriched = pd.DataFrame(rows)
    drop_columns = [column for column in ["buy_date_parsed", "sell_date_parsed"] if column in enriched.columns]
    return enriched.drop(columns=drop_columns)


def _entry_features(bars: pd.DataFrame, spy_bars: pd.DataFrame, buy_date: date | None, buy_price: float | None) -> dict[str, Any]:
    values = {column: pd.NA for column in ENTRY_FEATURE_COLUMNS}
    if buy_date is None or buy_price is None or buy_price <= 0 or bars.empty:
        return values

    prior = _prior_bars(bars, buy_date)
    if prior.empty:
        return values
    closes = prior["close"].dropna()
    if closes.empty:
        return values

    prior_close = _last_positive(closes)
    values["entry_gap_pct"] = _pct_change(buy_price, prior_close)
    values["distance_from_10ema_pct"] = _distance_from_average(buy_price, _ema(closes, 10))
    values["distance_from_20ema_pct"] = _distance_from_average(buy_price, _ema(closes, 20))
    values["distance_from_50sma_pct"] = _distance_from_average(buy_price, _sma(closes, 50))
    values["distance_from_52w_high_pct"] = _distance_from_52w_high(buy_price, prior)
    values["entry_volume_vs_50d_avg"] = _volume_vs_average(prior, 50)
    values["computed_atr_21_pct"] = _atr_pct(prior, 21)
    for label, window in [("1m", 21), ("3m", 63), ("6m", 126)]:
        momentum = _momentum_pct(prior, window)
        values[f"momentum_{label}_pct"] = momentum
        spy_momentum = _momentum_pct(_prior_bars(spy_bars, buy_date), window)
        if momentum is not pd.NA and spy_momentum is not pd.NA:
            values[f"rs_vs_spy_{label}_pct"] = round(float(momentum) - float(spy_momentum), 2)
    return values


def _outcome_features(
    bars: pd.DataFrame,
    buy_date: date | None,
    sell_date: date | None,
    buy_price: float | None,
) -> dict[str, Any]:
    values = {"max_gain_during_trade_pct": pd.NA, "max_loss_during_trade_pct": pd.NA}
    if buy_date is None or sell_date is None or buy_price is None or buy_price <= 0 or bars.empty:
        return values
    window = bars[(bars["bar_date"] >= buy_date) & (bars["bar_date"] <= sell_date)]
    if window.empty:
        return values
    # Daily bars can include pre-entry intraday extremes on the buy date.
    high = _max_positive(window["high"])
    low = _min_positive(window["low"])
    values["max_gain_during_trade_pct"] = _pct_change(high, buy_price)
    values["max_loss_during_trade_pct"] = _pct_change(low, buy_price)
    return values


def _risk_features(row: dict[str, Any]) -> dict[str, Any]:
    realized_pnl = _to_float(row.get("realized_pnl"))
    buy_price = _to_float(row.get("buy_price"))
    planned_stop = _to_float(row.get("planned_stop"))
    quantity = _to_float(row.get("quantity"))
    winner = realized_pnl is not None and realized_pnl > 0
    outcome = _outcome_label(realized_pnl)
    r_multiple = pd.NA
    if (
        realized_pnl is not None
        and buy_price is not None
        and planned_stop is not None
        and quantity is not None
        and quantity > 0
    ):
        initial_risk = (buy_price - planned_stop) * quantity
        if initial_risk > 0:
            r_multiple = round(realized_pnl / initial_risk, 2)
    return {"winner": winner, "outcome": outcome, "r_multiple": r_multiple}


def _quality_labels(row: dict[str, Any]) -> dict[str, bool]:
    distance_20 = _to_float(row.get("distance_from_20ema_pct"))
    distance_high = _to_float(row.get("distance_from_52w_high_pct"))
    volume = _to_float(row.get("entry_volume_vs_50d_avg"))
    atr = _to_float(row.get("computed_atr_21_pct"))
    near_20 = distance_20 is not None and abs(distance_20) <= 3
    extended_20 = distance_20 is not None and distance_20 > 10
    near_high = distance_high is not None and distance_high >= -10
    return {
        "near_20ema": near_20,
        "extended_from_20ema": extended_20,
        "near_52w_high": near_high,
        "high_volume_entry": volume is not None and volume >= 2,
        "high_atr_risk": atr is not None and atr > 8,
        "late_breakout": extended_20 and near_high,
        "pullback_entry": distance_20 is not None and -8 <= distance_20 < 0,
    }


def _winner_loser_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "realized_pnl_percent",
        "r_multiple",
        "distance_from_20ema_pct",
        "computed_atr_21_pct",
        "entry_volume_vs_50d_avg",
        "momentum_6m_pct",
        "rs_vs_spy_6m_pct",
        "hold_days",
    ]
    rows = []
    if enriched.empty or "outcome" not in enriched.columns:
        return pd.DataFrame(columns=["group", "feature", "trade_count", "average", "median"])
    groups = [
        ("Winner", enriched[enriched["outcome"] == "win"]),
        ("Loser", enriched[enriched["outcome"] == "loss"]),
        ("Breakeven", enriched[enriched["outcome"] == "breakeven"]),
    ]
    for group_label, group_frame in groups:
        for column in columns:
            series = pd.to_numeric(group_frame.get(column), errors="coerce").dropna()
            rows.append(
                {
                    "group": group_label,
                    "feature": column,
                    "trade_count": int(series.count()),
                    "average": _round(series.mean()),
                    "median": _round(series.median()),
                }
            )
    return pd.DataFrame(rows)


def _strategy_breakdown(enriched: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "strategy",
        "trade_count",
        "win_rate",
        "average_return",
        "median_return",
        "average_r_multiple",
        "average_hold_days",
    ]
    if enriched.empty:
        return pd.DataFrame(columns=columns)
    frame = enriched.copy()
    frame["strategy"] = _strategy_display_series(frame)
    rows = []
    for strategy, group in frame.groupby("strategy", sort=False, dropna=False):
        returns = pd.to_numeric(group.get("realized_pnl_percent"), errors="coerce")
        r_values = pd.to_numeric(group.get("r_multiple"), errors="coerce")
        holds = pd.to_numeric(group.get("hold_days"), errors="coerce")
        rows.append(
            {
                "strategy": strategy,
                "trade_count": len(group),
                "win_rate": _round(group["winner"].mean() * 100),
                "average_return": _round(returns.mean()),
                "median_return": _round(returns.median()),
                "average_r_multiple": _round(r_values.mean()),
                "average_hold_days": _round(holds.mean(), 1),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_analysis_bundle(
    enriched: pd.DataFrame,
    *,
    output_dir: Path,
    bigquery_skipped: bool,
    title: str = "Trade Analysis",
) -> AnalysisBundle:
    winners_vs_losers = _winner_loser_summary(enriched)
    strategy_breakdown = _strategy_breakdown(enriched)
    threshold_tests = _threshold_tests(enriched)
    summary = _render_summary(
        enriched,
        winners_vs_losers,
        strategy_breakdown,
        threshold_tests,
        bigquery_skipped,
        title=title,
    )
    _write_reports(output_dir, enriched, winners_vs_losers, strategy_breakdown, threshold_tests, summary)
    return AnalysisBundle(output_dir, enriched, winners_vs_losers, strategy_breakdown, threshold_tests, summary)


def _build_strategy_analysis_bundles(
    enriched: pd.DataFrame,
    *,
    output_dir: Path,
    bigquery_skipped: bool,
) -> dict[str, AnalysisBundle]:
    if enriched.empty:
        return {}

    frame = enriched.copy()
    frame["strategy"] = _strategy_display_series(frame)
    results: dict[str, AnalysisBundle] = {}
    for strategy, group in frame.groupby("strategy", sort=False, dropna=False):
        strategy_name = str(strategy)
        strategy_dir = output_dir / "strategies" / _strategy_folder_slug(strategy_name)
        results[strategy_name] = _build_analysis_bundle(
            group.copy(),
            output_dir=strategy_dir,
            bigquery_skipped=bigquery_skipped,
            title=f"Trade Analysis - {strategy_name}",
        )
    return results


def _strategy_display_series(frame: pd.DataFrame) -> pd.Series:
    if "strategy" in frame.columns:
        values = frame["strategy"]
    else:
        values = pd.Series([""] * len(frame), index=frame.index)
    strategies = values.fillna("").astype(str).str.strip()
    return strategies.mask(strategies == "", "Unclassified")


def _strategy_folder_slug(strategy: Any) -> str:
    display = str(strategy or "").strip() or "Unclassified"
    if display in STRATEGY_FOLDER_SLUGS:
        return STRATEGY_FOLDER_SLUGS[display]
    slug = re.sub(r"[^a-z0-9]+", "-", display.lower().replace("%", "pct")).strip("-")
    return slug or "unclassified"


def _threshold_tests(enriched: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature",
        "bucket",
        "trade_count",
        "win_rate",
        "average_return",
        "median_return",
        "average_r_multiple",
        "directional_only",
    ]
    if enriched.empty:
        return pd.DataFrame(columns=columns)
    specs = [
        ("momentum_6m_pct", [("<0", None, 0), ("0-20", 0, 20), ("20-40", 20, 40), ("40-60", 40, 60), ("60-100", 60, 100), ("100+", 100, None)]),
        ("distance_from_20ema_pct", [("<0", None, 0), ("0-3", 0, 3), ("3-6", 3, 6), ("6-10", 6, 10), ("10-15", 10, 15), ("15+", 15, None)]),
        ("computed_atr_21_pct", [("<3", None, 3), ("3-5", 3, 5), ("5-8", 5, 8), ("8-12", 8, 12), ("12+", 12, None)]),
        ("entry_volume_vs_50d_avg", [("<1", None, 1), ("1-1.5", 1, 1.5), ("1.5-2", 1.5, 2), ("2-3", 2, 3), ("3+", 3, None)]),
        ("rs_vs_spy_6m_pct", [("<0", None, 0), ("0-10", 0, 10), ("10-25", 10, 25), ("25-50", 25, 50), ("50+", 50, None)]),
    ]
    rows = []
    for feature, buckets in specs:
        values = pd.to_numeric(enriched.get(feature), errors="coerce")
        for label, low, high in buckets:
            mask = values.notna()
            if low is not None:
                mask &= values >= low
            if high is not None:
                mask &= values < high
            group = enriched[mask]
            returns = pd.to_numeric(group.get("realized_pnl_percent"), errors="coerce")
            r_values = pd.to_numeric(group.get("r_multiple"), errors="coerce")
            rows.append(
                {
                    "feature": feature,
                    "bucket": label,
                    "trade_count": len(group),
                    "win_rate": _round(group["winner"].mean() * 100) if len(group) else pd.NA,
                    "average_return": _round(returns.mean()),
                    "median_return": _round(returns.median()),
                    "average_r_multiple": _round(r_values.mean()),
                    "directional_only": len(group) < 10,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _render_summary(
    enriched: pd.DataFrame,
    winners_vs_losers: pd.DataFrame,
    strategy_breakdown: pd.DataFrame,
    threshold_tests: pd.DataFrame,
    bigquery_skipped: bool,
    *,
    title: str = "Trade Analysis",
) -> str:
    lines = [f"# {title}", ""]
    if enriched.empty:
        return f"# {title}\n\nNo closed trades were available.\n"

    buy_dates = pd.to_datetime(enriched.get("buy_date"), errors="coerce")
    lines.extend(
        [
            "## Coverage",
            "",
            f"- Trades: {len(enriched)}",
            f"- Buy date range: {buy_dates.min().date()} to {buy_dates.max().date()}",
            f"- Pyramided trades: {int(enriched.get('is_pyramided', pd.Series(dtype=bool)).fillna(False).sum())}",
            f"- BigQuery metadata: {'skipped' if bigquery_skipped else 'enabled'}",
            "- Strategy breakdown `average_r_multiple` is the same mean-R statistic the dashboard calls `expectancy_r`.",
            "",
            "## Strategy Counts",
            "",
        ]
    )
    for strategy, count in _strategy_display_series(enriched).value_counts().items():
        lines.append(f"- {strategy}: {count}")
    lines.extend(["", "## Market Regime Counts", ""])
    for regime, count in enriched.get("market_regime", pd.Series(dtype=str)).fillna("Unknown").replace("", "Unknown").value_counts().items():
        lines.append(f"- {regime}: {count}")

    lines.extend(["", "## Missing Data", ""])
    lines.append("- Classification metadata status:")
    for status, count in _status_counts(enriched, "classification_metadata_status").items():
        lines.append(f"  - {status}: {count}")
    classification_range = _date_range_text(enriched.get("classification_snapshot_date"))
    if classification_range is not None:
        lines.append(f"- Classification snapshot date range: {classification_range}")
    lines.append("- Signal metadata status:")
    for status, count in _status_counts(enriched, "signal_metadata_status").items():
        lines.append(f"  - {status}: {count}")
    signal_range = _date_range_text(enriched.get("signal_snapshot_date"))
    if signal_range is not None:
        lines.append(f"- Signal snapshot date range: {signal_range}")
    for feature in ENTRY_FEATURE_COLUMNS:
        if feature in enriched.columns:
            lines.append(f"- Missing {feature}: {int(pd.to_numeric(enriched[feature], errors='coerce').isna().sum())}")

    lines.extend(["", "## Outcome Highlights", ""])
    for feature in ["distance_from_20ema_pct", "computed_atr_21_pct", "entry_volume_vs_50d_avg", "momentum_6m_pct"]:
        rows = winners_vs_losers[winners_vs_losers["feature"] == feature]
        if rows.empty:
            continue
        pieces = []
        for _, row in rows.iterrows():
            pieces.append(f"{row['group']} avg {row['average']}")
        lines.append(f"- {feature}: {'; '.join(pieces)}")

    eligible = threshold_tests[threshold_tests["trade_count"] >= 10].copy()
    lines.extend(["", "## Candidate Rules", ""])
    if eligible.empty:
        lines.append("- No threshold bucket has at least 10 trades; all bucket findings are directional only.")
    else:
        eligible["median_return_sort"] = pd.to_numeric(eligible["median_return"], errors="coerce")
        best = eligible.dropna(subset=["median_return_sort"]).sort_values(
            ["median_return_sort", "trade_count"], ascending=[False, False]
        ).head(5)
        positive = best[best["median_return_sort"] > 0]
        if positive.empty:
            lines.append("- No tested threshold bucket with at least 10 trades has a positive median return.")
            lines.append("- Best observed buckets are listed as context, not as entry rules:")
            best_to_render = best
        else:
            best_to_render = positive
        for _, row in best_to_render.iterrows():
            verb = "Favor" if row["median_return_sort"] > 0 else "Review"
            lines.append(
                f"- {verb} {row['feature']} bucket {row['bucket']} only as evidence: "
                f"n={row['trade_count']}, median return={row['median_return']}, win rate={row['win_rate']}%."
            )
    return "\n".join(lines) + "\n"


def _write_reports(
    output_dir: Path,
    enriched: pd.DataFrame,
    winners_vs_losers: pd.DataFrame,
    strategy_breakdown: pd.DataFrame,
    threshold_tests: pd.DataFrame,
    summary: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_dir / "enriched_trades.csv", index=False)
    winners_vs_losers.to_csv(output_dir / "winners_vs_losers.csv", index=False)
    strategy_breakdown.to_csv(output_dir / "strategy_breakdown.csv", index=False)
    threshold_tests.to_csv(output_dir / "threshold_tests.csv", index=False)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def _monthly_bar_paths(root: Path, start_date: date, end_date: date) -> list[Path]:
    paths: list[Path] = []
    current = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)
    while current <= end_month:
        month_dir = root / "datasets" / "day_aggs_v1" / f"year={current.year:04d}" / f"month={current.month:02d}"
        paths.extend(sorted(month_dir.glob("*.parquet")))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return paths


def _prior_bars(bars: pd.DataFrame, buy_date: date) -> pd.DataFrame:
    if bars.empty:
        return bars
    return bars[bars["bar_date"] < buy_date]


def _ema(closes: pd.Series, span: int) -> float | None:
    valid = pd.to_numeric(closes, errors="coerce").dropna()
    if len(valid) < span:
        return None
    return float(valid.ewm(span=span, adjust=False).mean().iloc[-1])


def _sma(closes: pd.Series, window: int) -> float | None:
    valid = pd.to_numeric(closes, errors="coerce").dropna()
    if len(valid) < window:
        return None
    return float(valid.tail(window).mean())


def _distance_from_average(value: float, average: float | None) -> float | Any:
    if average is None or average <= 0:
        return pd.NA
    return round(((value / average) - 1) * 100, 2)


def _distance_from_52w_high(value: float, bars: pd.DataFrame) -> float | Any:
    if len(bars) < 252:
        return pd.NA
    high = _max_positive(bars["high"].tail(252))
    if high is None:
        return pd.NA
    return _pct_change(value, high)


def _volume_vs_average(bars: pd.DataFrame, window: int) -> float | Any:
    if len(bars) < window + 1:
        return pd.NA
    volumes = pd.to_numeric(bars["volume"], errors="coerce").dropna()
    if len(volumes) < window + 1:
        return pd.NA
    latest = float(volumes.iloc[-1])
    average = float(volumes.iloc[-window - 1 : -1].mean())
    if average <= 0:
        return pd.NA
    return round(latest / average, 2)


def _atr_pct(bars: pd.DataFrame, window: int) -> float | Any:
    if len(bars) < window + 1:
        return pd.NA
    frame = bars.tail(window + 1).copy()
    for column in ["high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1).dropna()
    if len(tr) < window:
        return pd.NA
    latest_close = _last_positive(frame["close"])
    if latest_close is None:
        return pd.NA
    return round((float(tr.tail(window).mean()) / latest_close) * 100, 2)


def _momentum_pct(bars: pd.DataFrame, window: int) -> float | Any:
    if bars.empty or "close" not in bars.columns:
        return pd.NA
    closes = pd.to_numeric(bars["close"], errors="coerce").dropna()
    if len(closes) < window + 1:
        return pd.NA
    return _pct_change(float(closes.iloc[-1]), float(closes.iloc[-window - 1]))


def _outcome_label(realized_pnl: float | None) -> str | Any:
    if realized_pnl is None:
        return pd.NA
    if realized_pnl > 0:
        return "win"
    if realized_pnl < 0:
        return "loss"
    return "breakeven"


def _pct_change(value: float | None, base: float | None) -> float | Any:
    if value is None or base is None or base <= 0:
        return pd.NA
    return round(((value / base) - 1) * 100, 2)


def _last_positive(values: pd.Series) -> float | None:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def _max_positive(values: pd.Series) -> float | None:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return None
    return float(valid.max())


def _min_positive(values: pd.Series) -> float | None:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return None
    return float(valid.min())


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _round(value: Any, digits: int = 2) -> float | Any:
    try:
        if pd.isna(value):
            return pd.NA
        return round(float(value), digits)
    except (TypeError, ValueError):
        return pd.NA


def _status_counts(enriched: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in enriched.columns:
        return {"missing_column": len(enriched)}
    return {str(status): int(count) for status, count in enriched[column].fillna("unknown").value_counts().items()}


def _date_range_text(values: pd.Series | None) -> str | None:
    if values is None:
        return None
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None
    return f"{dates.min().date()} to {dates.max().date()}"


def _blank_metadata(classification_status: str, signal_status: str) -> dict[str, Any]:
    values = {column: pd.NA for column in METADATA_COLUMNS}
    values["classification_metadata_status"] = classification_status
    values["signal_metadata_status"] = signal_status
    return values


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "bar_date", "open", "high", "low", "close", "volume"])


def _empty_enriched_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=[*ENTRY_FEATURE_COLUMNS, *METADATA_COLUMNS, "winner", "outcome", "r_multiple", *LABEL_COLUMNS])


def normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").upper().strip()
