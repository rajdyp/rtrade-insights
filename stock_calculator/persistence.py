from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError:
        return False
    return True


def save_if_changed(
    current: pd.DataFrame,
    last_saved: pd.DataFrame,
    save: Callable[[pd.DataFrame], None],
) -> tuple[pd.DataFrame, bool]:
    if frames_equal(current, last_saved):
        return last_saved, False

    save(current)
    return current.copy(deep=True), True


def rows_for_ids(
    frame: pd.DataFrame,
    ids: list[str],
    *,
    id_column: str,
    columns: list[str],
) -> pd.DataFrame:
    if not ids or frame.empty:
        return pd.DataFrame(columns=columns)

    rows_by_id = {
        str(row[id_column]): row
        for _, row in frame.iterrows()
        if str(row.get(id_column) or "")
    }
    rows = [rows_by_id[row_id] for row_id in ids if row_id in rows_by_id]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)
