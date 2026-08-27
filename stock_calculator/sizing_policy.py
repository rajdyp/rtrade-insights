from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_EXPOSURE = "Full"
NO_TRADE_EXPOSURE = "No Trade"
NO_TRADE_VALIDATION_ERROR = 'Exposure matrix recommends "No Trade".'

EXPOSURE_LEVELS = {
    "Full": 20.0,
    "Half": 10.0,
    "Quarter": 5.0,
    "Probe": 2.5,
}

MARKET_REGIME_OPTIONS = ["GO", "SELECTIVE GO", "NO-GO"]
STRATEGY_MODE_UNKNOWN = "Unknown"
STRATEGY_MODES = ("Working", "Caution", "Weak", "Failing", STRATEGY_MODE_UNKNOWN)

EXPOSURE_RECOMMENDATION_MATRIX = {
    "GO": {
        "Working": "Full",
        "Caution": "Half",
        "Weak": "Quarter",
        "Failing": "Probe",
        "Unknown": "Probe",
    },
    "SELECTIVE GO": {
        "Working": "Half",
        "Caution": "Quarter",
        "Weak": "Probe",
        "Failing": NO_TRADE_EXPOSURE,
        "Unknown": NO_TRADE_EXPOSURE,
    },
    "NO-GO": {
        "Working": "Quarter",
        "Caution": "Probe",
        "Weak": NO_TRADE_EXPOSURE,
        "Failing": NO_TRADE_EXPOSURE,
        "Unknown": NO_TRADE_EXPOSURE,
    },
}


@dataclass(frozen=True)
class DraftExposureState:
    context: tuple[str, str, str]
    recommendation: str
    selection: str
    options: tuple[str, ...]
    overridden: bool


def normalize_market_regime(value: Any, fallback: str = "GO") -> str:
    regime = str(value or "").strip().upper()
    if regime in MARKET_REGIME_OPTIONS:
        return regime
    return fallback if fallback in MARKET_REGIME_OPTIONS else "GO"


def normalize_exposure(value: Any, *, blank_default: str = "") -> str:
    if value is None:
        return blank_default

    text = str(value).strip()
    if not text:
        return blank_default

    options = (*EXPOSURE_LEVELS, NO_TRADE_EXPOSURE)
    canonical = {option.casefold(): option for option in options}
    return canonical.get(text.casefold(), text)


def suggested_exposure(market_regime: Any, strategy_mode: Any) -> str:
    regime = normalize_market_regime(market_regime)
    mode = str(strategy_mode or "").strip()
    if mode not in STRATEGY_MODES:
        mode = STRATEGY_MODE_UNKNOWN
    return EXPOSURE_RECOMMENDATION_MATRIX[regime][mode]


def resolve_draft_exposure(
    *,
    previous_context: tuple[str, str, str] | None,
    current_selection: Any,
    market_regime: Any,
    strategy: Any,
    strategy_mode: Any,
) -> DraftExposureState:
    regime = normalize_market_regime(market_regime)
    strategy_name = str(strategy or "").strip()
    mode = str(strategy_mode or "").strip()
    if mode not in STRATEGY_MODES:
        mode = STRATEGY_MODE_UNKNOWN

    context = (regime, strategy_name, mode)
    recommendation = suggested_exposure(regime, mode)
    selection = normalize_exposure(current_selection)
    valid_selections = {*EXPOSURE_LEVELS, NO_TRADE_EXPOSURE}
    if previous_context != context or selection not in valid_selections:
        selection = recommendation

    options = tuple(EXPOSURE_LEVELS)
    if recommendation == NO_TRADE_EXPOSURE or selection == NO_TRADE_EXPOSURE:
        options = (NO_TRADE_EXPOSURE, *options)

    return DraftExposureState(
        context=context,
        recommendation=recommendation,
        selection=selection,
        options=options,
        overridden=selection != recommendation,
    )
