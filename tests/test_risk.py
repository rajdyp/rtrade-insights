import pandas as pd
import pytest

from stock_calculator.risk import (
    normalize_market_regime,
    strategy_mode_for_selection,
    suggested_exposure,
)


@pytest.mark.parametrize(
    ("regime", "mode", "expected"),
    [
        ("GO", "Working", "Full"),
        ("GO", "Caution", "Half"),
        ("GO", "Weak", "Quarter"),
        ("GO", "Failing", "Probe"),
        ("GO", "Unknown", "Probe"),
        ("SELECTIVE GO", "Working", "Half"),
        ("SELECTIVE GO", "Caution", "Quarter"),
        ("SELECTIVE GO", "Weak", "Probe"),
        ("SELECTIVE GO", "Failing", "No Trade"),
        ("SELECTIVE GO", "Unknown", "No Trade"),
        ("NO-GO", "Working", "Quarter"),
        ("NO-GO", "Caution", "Probe"),
        ("NO-GO", "Weak", "No Trade"),
        ("NO-GO", "Failing", "No Trade"),
        ("NO-GO", "Unknown", "No Trade"),
    ],
)
def test_suggested_exposure_uses_market_regime_and_strategy_mode_matrix(regime, mode, expected):
    assert suggested_exposure(regime, mode) == expected


def test_suggested_exposure_normalizes_unknown_inputs():
    assert suggested_exposure("invalid", "Experimental") == "Probe"


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
