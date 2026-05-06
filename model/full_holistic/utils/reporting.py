from __future__ import annotations

import pandas as pd


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "_No rows available._"
    clean = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    for column in clean.columns:
        if pd.api.types.is_float_dtype(clean[column]):
            clean[column] = clean[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
        else:
            clean[column] = clean[column].map(lambda value: "" if pd.isna(value) else str(value).replace("|", "/"))
    headers = clean.columns.tolist()
    rows = clean.astype(str).values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]

    def format_row(values) -> str:
        return "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def not_run() -> str:
    return "This analysis was not run."
