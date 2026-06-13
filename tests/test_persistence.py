import pandas as pd
import pytest

from stock_calculator.persistence import frames_equal, rows_for_ids, save_if_changed


def test_frames_equal_ignores_index_and_harmless_dtype_differences():
    left = pd.DataFrame([{"symbol": "AAPL", "quantity": 2}], index=[4])
    right = pd.DataFrame([{"symbol": "AAPL", "quantity": 2.0}], index=[9])

    assert frames_equal(left, right)


def test_save_if_changed_skips_equivalent_frame():
    calls = []
    last_saved = pd.DataFrame([{"symbol": "AAPL", "quantity": 2}])

    snapshot, saved = save_if_changed(
        pd.DataFrame([{"symbol": "AAPL", "quantity": 2.0}]),
        last_saved,
        calls.append,
    )

    assert not saved
    assert calls == []
    assert snapshot is last_saved


def test_save_if_changed_saves_once_and_returns_independent_snapshot():
    calls = []
    current = pd.DataFrame([{"symbol": "MSFT", "quantity": 3}])

    snapshot, saved = save_if_changed(
        current,
        pd.DataFrame([{"symbol": "AAPL", "quantity": 2}]),
        lambda frame: calls.append(frame.copy(deep=True)),
    )
    current.loc[0, "quantity"] = 4

    assert saved
    assert calls[0].to_dict("records") == [{"symbol": "MSFT", "quantity": 3}]
    assert snapshot.to_dict("records") == [{"symbol": "MSFT", "quantity": 3}]


def test_save_if_changed_does_not_advance_snapshot_when_save_fails():
    last_saved = pd.DataFrame([{"symbol": "AAPL"}])

    with pytest.raises(RuntimeError, match="quota exceeded"):
        save_if_changed(
            pd.DataFrame([{"symbol": "MSFT"}]),
            last_saved,
            lambda _frame: (_ for _ in ()).throw(RuntimeError("quota exceeded")),
        )

    assert last_saved.to_dict("records") == [{"symbol": "AAPL"}]


def test_rows_for_ids_builds_archive_baseline_in_active_position_order():
    archive = pd.DataFrame(
        [
            {"position_id": "pos_old", "symbol": "OLD"},
            {"position_id": "pos_b", "symbol": "MSFT"},
            {"position_id": "pos_a", "symbol": "AAPL"},
        ]
    )

    baseline = rows_for_ids(
        archive,
        ["pos_a", "pos_b"],
        id_column="position_id",
        columns=["position_id", "symbol"],
    )

    assert baseline.to_dict("records") == [
        {"position_id": "pos_a", "symbol": "AAPL"},
        {"position_id": "pos_b", "symbol": "MSFT"},
    ]
