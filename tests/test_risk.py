import pandas as pd

from stock_calculator.risk import (
    normalize_market_regime,
    strategy_mode_for_selection,
    suggested_risk_percent,
)


def test_suggested_risk_percent_uses_market_regime_and_strategy_mode_matrix():
    assert suggested_risk_percent("GO", "Working", fallback=0.5) == 1.00
    assert suggested_risk_percent("SELECTIVE GO", "Weak", fallback=0.5) == 0.12
    assert suggested_risk_percent("NO-GO", "Failing", fallback=0.5) == 0.00
    assert suggested_risk_percent("NO-GO", "Unknown", fallback=0.5) == 0.00


def test_strategy_mode_for_selection_reads_existing_strategy_metrics_mode_column():
    strategy_metrics = pd.DataFrame(
        [
            {"strategy": "EP", "mode": "Working"},
            {"strategy": "BO", "mode": "Weak"},
        ]
    )

    assert strategy_mode_for_selection(strategy_metrics, "EP") == "Working"
    assert strategy_mode_for_selection(strategy_metrics, "BO") == "Weak"


def test_strategy_mode_for_selection_uses_unknown_for_missing_strategy_row():
    strategy_metrics = pd.DataFrame([{"strategy": "EP", "mode": "Working"}])

    assert strategy_mode_for_selection(strategy_metrics, "4% BO") == "Unknown"


def test_strategy_mode_for_selection_uses_unknown_for_unrecognized_mode():
    strategy_metrics = pd.DataFrame([{"strategy": "EP", "mode": "Experimental"}])

    assert strategy_mode_for_selection(strategy_metrics, "EP") == "Unknown"


def test_normalize_market_regime_accepts_only_supported_regimes():
    assert normalize_market_regime("selective go") == "SELECTIVE GO"
    assert normalize_market_regime("bad", fallback="NO-GO") == "NO-GO"
    assert normalize_market_regime("bad", fallback="bad") == "GO"
