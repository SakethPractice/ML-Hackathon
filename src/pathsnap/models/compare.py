"""Model comparison and selection records."""

from __future__ import annotations

import pandas as pd


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Convert named model results into a sortable comparison table."""
    rows = []
    for name, result in results.items():
        row = {"model": name}
        row.update(result)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
        ignore_index=True,
    )