from stock_calculator.sizing_policy import (
    NO_TRADE_EXPOSURE,
    resolve_draft_exposure,
)


def test_draft_exposure_initializes_from_recommendation():
    state = resolve_draft_exposure(
        previous_context=None,
        current_selection=None,
        market_regime="SELECTIVE GO",
        strategy="EP",
        strategy_mode="Working",
    )

    assert state.context == ("SELECTIVE GO", "EP", "Working")
    assert state.recommendation == "Half"
    assert state.selection == "Half"
    assert not state.overridden
    assert state.options == ("Full", "Half", "Quarter", "Probe")


def test_draft_exposure_preserves_override_while_context_is_unchanged():
    state = resolve_draft_exposure(
        previous_context=("SELECTIVE GO", "EP", "Working"),
        current_selection="Full",
        market_regime="SELECTIVE GO",
        strategy="EP",
        strategy_mode="Working",
    )

    assert state.selection == "Full"
    assert state.overridden


def test_draft_exposure_resets_when_regime_strategy_or_mode_changes():
    previous = ("GO", "EP", "Working")
    for regime, strategy, mode, expected in [
        ("SELECTIVE GO", "EP", "Working", "Half"),
        ("GO", "BO", "Working", "Full"),
        ("GO", "EP", "Weak", "Quarter"),
    ]:
        state = resolve_draft_exposure(
            previous_context=previous,
            current_selection="Half",
            market_regime=regime,
            strategy=strategy,
            strategy_mode=mode,
        )
        assert state.selection == expected
        assert not state.overridden


def test_no_trade_recommendation_adds_option_and_explains_override():
    recommended = resolve_draft_exposure(
        previous_context=None,
        current_selection=None,
        market_regime="NO-GO",
        strategy="EP",
        strategy_mode="Weak",
    )
    assert recommended.selection == NO_TRADE_EXPOSURE
    assert recommended.options[0] == NO_TRADE_EXPOSURE

    overridden = resolve_draft_exposure(
        previous_context=recommended.context,
        current_selection="Probe",
        market_regime="NO-GO",
        strategy="EP",
        strategy_mode="Weak",
    )
    assert overridden.selection == "Probe"
    assert overridden.options[0] == NO_TRADE_EXPOSURE


def test_cleared_post_add_state_reinitializes_directly_to_recommendation():
    state = resolve_draft_exposure(
        previous_context=None,
        current_selection=None,
        market_regime="SELECTIVE GO",
        strategy="EP",
        strategy_mode="Caution",
    )
    assert state.selection == "Quarter"
