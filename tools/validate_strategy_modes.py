from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_calculator.risk import RISK_PERCENT_MATRIX
from stock_calculator.robinhood import (
    STRATEGY_MODE_ADJUSTMENT_K,
    STRATEGY_MODE_FAILING_THRESHOLD,
    STRATEGY_MODE_WINDOW,
    STRATEGY_MODE_WORKING_THRESHOLD,
    derive_fifo_trades,
)
from stock_calculator.storage import load_planned_stops, load_robinhood_transactions


MODE_ORDER = ("Working", "Caution", "Weak", "Failing")
CURRENT_THRESHOLDS = (
    STRATEGY_MODE_WINDOW,
    STRATEGY_MODE_FAILING_THRESHOLD,
    STRATEGY_MODE_WORKING_THRESHOLD,
)
DEFAULT_WINDOWS = (5, 10, 15, 20)
DEFAULT_FAILING_THRESHOLDS = (-0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50)
DEFAULT_WORKING_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
DEFAULT_ADJUSTMENT_KS = (0.0, 0.25, STRATEGY_MODE_ADJUSTMENT_K, 0.75, 1.0)
GO_MODE_MULTIPLIERS = {
    mode: RISK_PERCENT_MATRIX["GO"][mode]
    for mode in MODE_ORDER
}


@dataclass(frozen=True)
class ModeThresholds:
    window: int
    failing_threshold: float
    working_threshold: float
    adjustment_k: float = 0.0


@dataclass(frozen=True)
class ValidationResult:
    thresholds: ModeThresholds
    total_scaled_r: float
    max_drawdown: float
    score: float
    eligible_trades: int
    mode_counts: dict[str, int]


@dataclass(frozen=True)
class SimulationRun:
    thresholds: ModeThresholds
    returns_by_mode: list[tuple[float, str]]


@dataclass(frozen=True)
class SizingComparison:
    label: str
    total_scaled_r: float
    max_drawdown: float
    score: float
    eligible_trades: int


@dataclass(frozen=True)
class AdjustmentComparison:
    adjustment_k: float
    total_scaled_r: float
    max_drawdown: float
    score: float
    eligible_trades: int
    mode_counts: dict[str, int]
    mode_changes: dict[str, int]


def classify_mode(expectancy_r: float, thresholds: ModeThresholds) -> str:
    if expectancy_r > thresholds.working_threshold:
        return "Working"
    if expectancy_r >= 0:
        return "Caution"
    if expectancy_r >= thresholds.failing_threshold:
        return "Weak"
    return "Failing"


def prepare_r_trades(closed_trades: pd.DataFrame) -> pd.DataFrame:
    if closed_trades.empty:
        return pd.DataFrame(columns=["strategy", "sell_date", "r_multiple"])

    trades = closed_trades.copy()
    for column in ["realized_pnl", "buy_price", "planned_stop", "quantity"]:
        trades[column] = pd.to_numeric(trades.get(column), errors="coerce")

    trades["sell_date"] = pd.to_datetime(trades.get("sell_date"), errors="coerce")
    trades["initial_risk"] = (trades["buy_price"] - trades["planned_stop"]) * trades["quantity"]
    trades["r_multiple"] = trades["realized_pnl"] / trades["initial_risk"]
    trades["strategy"] = trades.get("strategy", "").fillna("").astype(str).str.strip()
    trades.loc[trades["strategy"] == "", "strategy"] = "Unclassified"

    valid = trades.loc[
        (trades["initial_risk"] > 0)
        & trades["r_multiple"].notna()
        & trades["sell_date"].notna(),
        ["strategy", "sell_date", "r_multiple"],
    ].copy()
    valid["_original_order"] = range(len(valid))
    valid = valid.sort_values(["sell_date", "_original_order"], kind="mergesort").drop(columns="_original_order")
    return valid.reset_index(drop=True)


def calculate_max_drawdown(returns: Iterable[float]) -> float:
    peak = 0.0
    equity = 0.0
    max_drawdown = 0.0
    for value in returns:
        equity += float(value)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return round(max_drawdown, 4)


def return_drawdown_score(total_return: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return math.inf if total_return > 0 else 0.0
    return round(total_return / max_drawdown, 4)


def simulate_thresholds(
    trades: pd.DataFrame,
    thresholds: ModeThresholds,
    *,
    mode_multipliers: dict[str, float] | None = None,
) -> ValidationResult:
    mode_multipliers = GO_MODE_MULTIPLIERS if mode_multipliers is None else mode_multipliers
    run = collect_simulation_run(trades, thresholds)
    scaled_returns = [
        r_multiple * mode_multipliers[mode]
        for r_multiple, mode in run.returns_by_mode
    ]
    mode_counts = Counter(mode for _, mode in run.returns_by_mode)
    return _validation_result(thresholds, scaled_returns, mode_counts)


def collect_simulation_run(trades: pd.DataFrame, thresholds: ModeThresholds) -> SimulationRun:
    history_by_strategy: dict[str, list[float]] = defaultdict(list)
    returns_by_mode: list[tuple[float, str]] = []

    for row in trades.itertuples(index=False):
        strategy = str(row.strategy)
        r_multiple = float(row.r_multiple)
        history = history_by_strategy[strategy]

        if len(history) >= thresholds.window:
            prior_window = history[-thresholds.window :]
            score = adjusted_mode_score(prior_window, thresholds.adjustment_k)
            mode = classify_mode(score, thresholds)
            returns_by_mode.append((r_multiple, mode))

        history.append(r_multiple)

    return SimulationRun(thresholds=thresholds, returns_by_mode=returns_by_mode)


def adjusted_mode_score(r_multiples: Iterable[float], adjustment_k: float) -> float:
    values = [float(value) for value in r_multiples]
    mean_r = sum(values) / len(values)
    if adjustment_k == 0 or len(values) < 2:
        return mean_r
    stddev_r = _sample_stddev(values, mean_r)
    return mean_r - (adjustment_k * stddev_r / math.sqrt(len(values)))


def _sample_stddev(values: list[float], mean_r: float) -> float:
    return math.sqrt(sum((value - mean_r) ** 2 for value in values) / (len(values) - 1))


def compare_sizing_methods(run: SimulationRun) -> list[SizingComparison]:
    return [
        _constant_sizing_comparison("Full risk", run, 1.00),
        _constant_sizing_comparison("Half risk", run, 0.50),
        _mode_sizing_comparison(run),
    ]


def _constant_sizing_comparison(label: str, run: SimulationRun, multiplier: float) -> SizingComparison:
    scaled_returns = [r_multiple * multiplier for r_multiple, _ in run.returns_by_mode]
    return _sizing_comparison(label, scaled_returns)


def _mode_sizing_comparison(run: SimulationRun) -> SizingComparison:
    scaled_returns = [
        r_multiple * GO_MODE_MULTIPLIERS[mode]
        for r_multiple, mode in run.returns_by_mode
    ]
    return _sizing_comparison("Mode sizing", scaled_returns)


def _sizing_comparison(label: str, scaled_returns: list[float]) -> SizingComparison:
    total_scaled_r = round(sum(scaled_returns), 4)
    max_drawdown = calculate_max_drawdown(scaled_returns)
    return SizingComparison(
        label=label,
        total_scaled_r=total_scaled_r,
        max_drawdown=max_drawdown,
        score=return_drawdown_score(total_scaled_r, max_drawdown),
        eligible_trades=len(scaled_returns),
    )


def _validation_result(
    thresholds: ModeThresholds,
    scaled_returns: list[float],
    mode_counts: Counter[str],
) -> ValidationResult:
    total_scaled_r = round(sum(scaled_returns), 4)
    max_drawdown = calculate_max_drawdown(scaled_returns)
    return ValidationResult(
        thresholds=thresholds,
        total_scaled_r=total_scaled_r,
        max_drawdown=max_drawdown,
        score=return_drawdown_score(total_scaled_r, max_drawdown),
        eligible_trades=len(scaled_returns),
        mode_counts={mode: mode_counts.get(mode, 0) for mode in MODE_ORDER},
    )


def threshold_grid(
    *,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    failing_thresholds: Iterable[float] = DEFAULT_FAILING_THRESHOLDS,
    working_thresholds: Iterable[float] = DEFAULT_WORKING_THRESHOLDS,
    adjustment_ks: Iterable[float] = (0.0,),
) -> list[ModeThresholds]:
    candidates = []
    for window in windows:
        for failing_threshold in failing_thresholds:
            for working_threshold in working_thresholds:
                for adjustment_k in adjustment_ks:
                    if failing_threshold < 0 < working_threshold:
                        candidates.append(ModeThresholds(window, failing_threshold, working_threshold, adjustment_k))
    return candidates


def rank_thresholds(trades: pd.DataFrame, candidates: Iterable[ModeThresholds]) -> list[ValidationResult]:
    results = [simulate_thresholds(trades, candidate) for candidate in candidates]
    return sorted(
        results,
        key=lambda result: (
            result.score,
            result.total_scaled_r,
            -result.max_drawdown,
            result.eligible_trades,
        ),
        reverse=True,
    )


def compare_adjustment_factors(
    trades: pd.DataFrame,
    thresholds: ModeThresholds,
    adjustment_ks: Iterable[float] = DEFAULT_ADJUSTMENT_KS,
) -> list[AdjustmentComparison]:
    raw_thresholds = ModeThresholds(
        thresholds.window,
        thresholds.failing_threshold,
        thresholds.working_threshold,
        0.0,
    )
    raw_run = collect_simulation_run(trades, raw_thresholds)
    raw_modes = [mode for _, mode in raw_run.returns_by_mode]
    comparisons = []

    for adjustment_k in adjustment_ks:
        adjusted_thresholds = ModeThresholds(
            thresholds.window,
            thresholds.failing_threshold,
            thresholds.working_threshold,
            adjustment_k,
        )
        run = collect_simulation_run(trades, adjusted_thresholds)
        scaled_returns = [
            r_multiple * GO_MODE_MULTIPLIERS[mode]
            for r_multiple, mode in run.returns_by_mode
        ]
        mode_counts = Counter(mode for _, mode in run.returns_by_mode)
        mode_changes = Counter(
            f"{raw_mode}->{adjusted_mode}"
            for raw_mode, (_, adjusted_mode) in zip(raw_modes, run.returns_by_mode, strict=True)
            if raw_mode != adjusted_mode
        )
        total_scaled_r = round(sum(scaled_returns), 4)
        max_drawdown = calculate_max_drawdown(scaled_returns)
        comparisons.append(
            AdjustmentComparison(
                adjustment_k=adjustment_k,
                total_scaled_r=total_scaled_r,
                max_drawdown=max_drawdown,
                score=return_drawdown_score(total_scaled_r, max_drawdown),
                eligible_trades=len(scaled_returns),
                mode_counts={mode: mode_counts.get(mode, 0) for mode in MODE_ORDER},
                mode_changes=dict(mode_changes),
            )
        )

    return comparisons


def load_closed_trades(transactions_path: Path | None, planned_stops_path: Path | None) -> pd.DataFrame:
    transactions = load_robinhood_transactions(transactions_path)
    planned_stops = load_planned_stops(planned_stops_path)
    return derive_fifo_trades(transactions, planned_stops).closed_trades


def render_results_for_trades(
    trades: pd.DataFrame,
    current: ValidationResult,
    ranked: list[ValidationResult],
    *,
    top: int,
) -> str:
    best = ranked[0]
    lines = [
        "Current production comparison",
        _format_thresholds(current.thresholds),
        _render_comparison_table(compare_sizing_methods(collect_simulation_run(trades, current.thresholds))),
        "",
        "Adjustment comparison",
        _render_adjustment_comparison_table(compare_adjustment_factors(trades, current.thresholds)),
        "",
        "Best candidate comparison",
        _format_thresholds(best.thresholds),
        _render_comparison_table(compare_sizing_methods(collect_simulation_run(trades, best.thresholds))),
        "",
        f"Top {top} candidate thresholds",
        _render_table(ranked[:top]),
    ]
    return "\n".join(lines)


def _format_thresholds(thresholds: ModeThresholds) -> str:
    text = (
        f"window={thresholds.window} "
        f"failing={thresholds.failing_threshold:+.2f} "
        f"working={thresholds.working_threshold:+.2f}"
    )
    if thresholds.adjustment_k:
        text = f"{text} k={thresholds.adjustment_k:g}"
    return text


def _render_comparison_table(comparisons: list[SizingComparison]) -> str:
    headers = ["sizing", "trades", "total_r", "max_dd", "score"]
    rows = [
        [
            comparison.label,
            str(comparison.eligible_trades),
            f"{comparison.total_scaled_r:+.2f}",
            f"{comparison.max_drawdown:.2f}",
            _format_score(comparison.score),
        ]
        for comparison in comparisons
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    rendered = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    rendered.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(rendered)


def _render_adjustment_comparison_table(comparisons: list[AdjustmentComparison]) -> str:
    headers = ["k", "trades", "total_r", "max_dd", "score", "modes", "changes_vs_raw"]
    rows = [
        [
            f"{comparison.adjustment_k:g}",
            str(comparison.eligible_trades),
            f"{comparison.total_scaled_r:+.2f}",
            f"{comparison.max_drawdown:.2f}",
            _format_score(comparison.score),
            _format_mode_counts(comparison.mode_counts),
            _format_mode_changes(comparison.mode_changes),
        ]
        for comparison in comparisons
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    rendered = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    rendered.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(rendered)


def _render_table(results: list[ValidationResult]) -> str:
    headers = ["window", "failing", "working", "k", "trades", "total_r", "max_dd", "score", "modes"]
    rows = []
    for result in results:
        thresholds = result.thresholds
        rows.append(
            [
                str(thresholds.window),
                f"{thresholds.failing_threshold:+.2f}",
                f"{thresholds.working_threshold:+.2f}",
                f"{thresholds.adjustment_k:g}",
                str(result.eligible_trades),
                f"{result.total_scaled_r:+.2f}",
                f"{result.max_drawdown:.2f}",
                _format_score(result.score),
                _format_mode_counts(result.mode_counts),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    rendered = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    rendered.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(rendered)


def _format_score(score: float) -> str:
    if math.isinf(score):
        return "inf"
    return f"{score:.2f}"


def _format_mode_counts(mode_counts: dict[str, int]) -> str:
    labels = {
        "Working": "Working",
        "Caution": "Caution",
        "Weak": "Weak",
        "Failing": "Failing",
    }
    return ", ".join(f"{labels[mode]}={mode_counts.get(mode, 0)}" for mode in MODE_ORDER)


def _format_mode_changes(mode_changes: dict[str, int]) -> str:
    if not mode_changes:
        return "none"
    return ", ".join(f"{change}={count}" for change, count in sorted(mode_changes.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate strategy mode thresholds with walk-forward simulation.")
    parser.add_argument("--transactions", type=Path, help="Robinhood transactions CSV. Defaults to configured storage.")
    parser.add_argument("--planned-stops", type=Path, help="Planned stops CSV. Defaults to configured storage.")
    parser.add_argument("--top", type=int, default=10, help="Number of candidate rows to print. Default: 10")
    args = parser.parse_args(argv)

    closed_trades = load_closed_trades(args.transactions, args.planned_stops)
    trades = prepare_r_trades(closed_trades)
    if trades.empty:
        print("No valid R-multiple trades found.")
        return 1

    current = simulate_thresholds(trades, ModeThresholds(*CURRENT_THRESHOLDS))
    ranked = rank_thresholds(trades, threshold_grid())
    print(render_results_for_trades(trades, current, ranked, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
