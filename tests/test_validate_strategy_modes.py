import pandas as pd

from tools.validate_strategy_modes import (
    ModeThresholds,
    adjusted_mode_score,
    calculate_max_drawdown,
    classify_mode,
    compare_adjustment_factors,
    collect_simulation_run,
    compare_sizing_methods,
    rank_thresholds,
    render_results_for_trades,
    return_drawdown_score,
    simulate_thresholds,
)


def test_classify_mode_uses_expected_threshold_boundaries():
    thresholds = ModeThresholds(window=10, failing_threshold=-0.25, working_threshold=0.30)

    assert classify_mode(0.31, thresholds) == "Working"
    assert classify_mode(0.30, thresholds) == "Caution"
    assert classify_mode(0.00, thresholds) == "Caution"
    assert classify_mode(-0.25, thresholds) == "Weak"
    assert classify_mode(-0.26, thresholds) == "Failing"


def test_adjusted_mode_score_penalizes_noisy_windows():
    assert adjusted_mode_score([1.0, -1.0], 0.0) == 0.0
    assert adjusted_mode_score([1.0, -1.0], 0.5) == -0.5


def test_simulate_thresholds_uses_only_prior_strategy_trades_for_mode():
    trades = pd.DataFrame(
        [
            {"strategy": "BO", "sell_date": "2026-01-01", "r_multiple": -1.0},
            {"strategy": "BO", "sell_date": "2026-01-02", "r_multiple": -1.0},
            {"strategy": "BO", "sell_date": "2026-01-03", "r_multiple": 10.0},
            {"strategy": "BO", "sell_date": "2026-01-04", "r_multiple": 1.0},
        ]
    )

    result = simulate_thresholds(trades, ModeThresholds(window=2, failing_threshold=-0.25, working_threshold=0.30))

    assert result.eligible_trades == 2
    assert result.mode_counts == {"Working": 1, "Caution": 0, "Weak": 0, "Failing": 1}
    assert result.total_scaled_r == 2.2


def test_compare_adjustment_factors_reports_mode_changes_against_raw():
    trades = pd.DataFrame(
        [
            {"strategy": "BO", "sell_date": "2026-01-01", "r_multiple": -1.0},
            {"strategy": "BO", "sell_date": "2026-01-02", "r_multiple": -1.0},
            {"strategy": "BO", "sell_date": "2026-01-03", "r_multiple": 10.0},
            {"strategy": "BO", "sell_date": "2026-01-04", "r_multiple": 1.0},
        ]
    )
    thresholds = ModeThresholds(window=2, failing_threshold=-0.25, working_threshold=0.30)

    raw, adjusted = compare_adjustment_factors(trades, thresholds, adjustment_ks=[0.0, 1.0])

    assert raw.adjustment_k == 0.0
    assert raw.mode_changes == {}
    assert raw.total_scaled_r == 2.2
    assert adjusted.adjustment_k == 1.0
    assert adjusted.total_scaled_r == 1.32
    assert adjusted.mode_counts == {"Working": 0, "Caution": 0, "Weak": 0, "Failing": 2}
    assert adjusted.mode_changes == {"Working->Failing": 1}


def test_simulate_thresholds_keeps_strategy_histories_separate():
    trades = pd.DataFrame(
        [
            {"strategy": "EP", "sell_date": "2026-01-01", "r_multiple": 1.0},
            {"strategy": "BO", "sell_date": "2026-01-02", "r_multiple": -1.0},
            {"strategy": "EP", "sell_date": "2026-01-03", "r_multiple": 1.0},
            {"strategy": "BO", "sell_date": "2026-01-04", "r_multiple": -1.0},
            {"strategy": "EP", "sell_date": "2026-01-05", "r_multiple": 1.0},
            {"strategy": "BO", "sell_date": "2026-01-06", "r_multiple": 1.0},
        ]
    )

    result = simulate_thresholds(trades, ModeThresholds(window=2, failing_threshold=-0.25, working_threshold=0.30))

    assert result.eligible_trades == 2
    assert result.mode_counts == {"Working": 1, "Caution": 0, "Weak": 0, "Failing": 1}


def test_calculate_max_drawdown_and_return_drawdown_score():
    returns = [1.0, -0.5, 2.0, -3.0, 1.0]

    assert calculate_max_drawdown(returns) == 3.0
    assert return_drawdown_score(0.5, 3.0) == 0.1667


def test_constant_risk_baselines_use_same_eligible_trades_as_mode_sizing():
    trades = pd.DataFrame(
        [
            {"strategy": "BO", "sell_date": "2026-01-01", "r_multiple": 1.0},
            {"strategy": "BO", "sell_date": "2026-01-02", "r_multiple": -1.0},
            {"strategy": "BO", "sell_date": "2026-01-03", "r_multiple": 2.0},
            {"strategy": "BO", "sell_date": "2026-01-04", "r_multiple": -2.0},
        ]
    )
    run = collect_simulation_run(trades, ModeThresholds(window=2, failing_threshold=-0.25, working_threshold=0.30))

    comparisons = compare_sizing_methods(run)

    assert [comparison.label for comparison in comparisons] == ["Full risk", "Half risk", "Mode sizing"]
    assert [comparison.eligible_trades for comparison in comparisons] == [2, 2, 2]
    assert comparisons[0].total_scaled_r == 0.0
    assert comparisons[1].total_scaled_r == 0.0


def test_full_and_half_risk_totals_scale_predictably():
    trades = pd.DataFrame(
        [
            {"strategy": "EP", "sell_date": "2026-01-01", "r_multiple": 1.0},
            {"strategy": "EP", "sell_date": "2026-01-02", "r_multiple": 1.0},
            {"strategy": "EP", "sell_date": "2026-01-03", "r_multiple": 2.0},
            {"strategy": "EP", "sell_date": "2026-01-04", "r_multiple": 4.0},
        ]
    )
    run = collect_simulation_run(trades, ModeThresholds(window=2, failing_threshold=-0.25, working_threshold=0.30))

    full_risk, half_risk, _ = compare_sizing_methods(run)

    assert full_risk.total_scaled_r == 6.0
    assert half_risk.total_scaled_r == 3.0
    assert half_risk.total_scaled_r == full_risk.total_scaled_r * 0.5


def test_render_results_shows_comparisons_before_ranked_thresholds():
    trades = pd.DataFrame(
        [
            {"strategy": "EP", "sell_date": "2026-01-01", "r_multiple": 1.0},
            {"strategy": "EP", "sell_date": "2026-01-02", "r_multiple": 1.0},
            {"strategy": "EP", "sell_date": "2026-01-03", "r_multiple": -1.0},
            {"strategy": "EP", "sell_date": "2026-01-04", "r_multiple": 2.0},
        ]
    )
    current_thresholds = ModeThresholds(window=2, failing_threshold=-0.25, working_threshold=0.30)
    current = simulate_thresholds(trades, current_thresholds)
    ranked = rank_thresholds(
        trades,
        [
            current_thresholds,
            ModeThresholds(window=2, failing_threshold=-0.10, working_threshold=0.30),
        ],
    )

    output = render_results_for_trades(trades, current, ranked, top=2)

    assert output.index("Current production comparison") < output.index("Best candidate comparison")
    assert output.index("Adjustment comparison") < output.index("Best candidate comparison")
    assert output.index("Best candidate comparison") < output.index("Top 2 candidate thresholds")
    assert "Full risk" in output
    assert "Half risk" in output
    assert "Mode sizing" in output
    assert "changes_vs_raw" in output
