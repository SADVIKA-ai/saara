
"""Reusable, non-destructive OHLC data validation for StockAI India."""

from __future__ import annotations

from collections import Counter

import pandas as pd


DATE_CANDIDATES = ("date", "datetime", "timestamp", "time")
OHLC_NAMES = ("open", "high", "low", "close")


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    """Map normalized column names to their original names."""
    return {
        str(column).strip().lower(): column
        for column in frame.columns
    }


def validate_ohlc(
    frame: pd.DataFrame,
    *,
    date_column: str | None = None,
) -> pd.DataFrame:
    """Return a copy annotated with OHLC validity and explicit issue codes.

    The input is never modified. Original columns and values are preserved.
    Rows with invalid prices, missing dates, or duplicate dates are flagged.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    lookup = _column_lookup(frame)

    missing = [name for name in OHLC_NAMES if name not in lookup]
    if missing:
        raise ValueError(
            f"Missing required OHLC columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    if date_column is not None:
        if date_column not in frame.columns:
            raise ValueError(
                f"Date column {date_column!r} not found."
            )
        actual_date_column = date_column
    else:
        actual_date_column = next(
            (
                lookup[name]
                for name in DATE_CANDIDATES
                if name in lookup
            ),
            None,
        )

    result = frame.copy(deep=True)
    issues_by_row: list[list[str]] = [[] for _ in range(len(result))]

    date_values = None
    if actual_date_column is not None:
        date_values = pd.to_datetime(
            result[actual_date_column],
            errors="coerce",
            utc=True,
        )

        for position, value in enumerate(date_values):
            if pd.isna(value):
                issues_by_row[position].append("INVALID_DATE")

        # Flag every row involved in a duplicated calendar date.
        normalized_dates = date_values.dt.strftime("%Y-%m-%d")
        duplicate_mask = normalized_dates.notna() & normalized_dates.duplicated(
            keep=False
        )
        for position, is_duplicate in enumerate(duplicate_mask.tolist()):
            if is_duplicate:
                issues_by_row[position].append("DUPLICATE_DATE")

    numeric: dict[str, pd.Series] = {}
    for name in OHLC_NAMES:
        numeric[name] = pd.to_numeric(
            result[lookup[name]], errors="coerce"
        )

    for position in range(len(result)):
        values = {
            name: numeric[name].iloc[position]
            for name in OHLC_NAMES
        }

        if any(pd.isna(value) for value in values.values()):
            issues_by_row[position].append("MISSING_OR_NONNUMERIC_OHLC")
            continue

        open_price = float(values["open"])
        high = float(values["high"])
        low = float(values["low"])
        close = float(values["close"])

        if any(value <= 0 for value in values.values()):
            issues_by_row[position].append("NONPOSITIVE_OHLC")

        if high < low:
            issues_by_row[position].append("HIGH_BELOW_LOW")
        if high < open_price:
            issues_by_row[position].append("OPEN_ABOVE_HIGH")
        if high < close:
            issues_by_row[position].append("CLOSE_ABOVE_HIGH")
        if low > open_price:
            issues_by_row[position].append("OPEN_BELOW_LOW")
        if low > close:
            issues_by_row[position].append("CLOSE_BELOW_LOW")

    result["_quality_issues"] = [
        ";".join(dict.fromkeys(row_issues))
        for row_issues in issues_by_row
    ]
    result["_quality_valid"] = result["_quality_issues"].eq("")

    return result


def valid_ohlc_rows(
    frame: pd.DataFrame,
    *,
    date_column: str | None = None,
) -> pd.DataFrame:
    """Return only valid rows, without the temporary quality columns."""
    checked = validate_ohlc(frame, date_column=date_column)
    valid = checked.loc[checked["_quality_valid"]].copy()
    return valid.drop(
        columns=["_quality_issues", "_quality_valid"]
    )


def quality_summary(checked: pd.DataFrame) -> dict[str, object]:
    """Summarize a frame previously returned by validate_ohlc()."""
    required = {"_quality_valid", "_quality_issues"}
    if not required.issubset(checked.columns):
        raise ValueError(
            "Pass a DataFrame returned by validate_ohlc()."
        )

    issue_counts: Counter[str] = Counter()
    for value in checked["_quality_issues"].fillna(""):
        for issue in str(value).split(";"):
            if issue:
                issue_counts[issue] += 1

    return {
        "total_rows": int(len(checked)),
        "valid_rows": int(checked["_quality_valid"].sum()),
        "invalid_rows": int((~checked["_quality_valid"]).sum()),
        "issue_counts": dict(sorted(issue_counts.items())),
    }
