
import pandas as pd
import pytest

from src.data_quality import (
    quality_summary,
    valid_ohlc_rows,
    validate_ohlc,
)


def test_valid_candle_passes():
    data = pd.DataFrame({
        "Date": ["2026-01-02"],
        "Open": [100.0],
        "High": [110.0],
        "Low": [95.0],
        "Close": [105.0],
    })

    checked = validate_ohlc(data)

    assert checked.loc[0, "_quality_valid"]
    assert checked.loc[0, "_quality_issues"] == ""


@pytest.mark.parametrize(
    ("open_price", "high", "low", "close", "expected_issue"),
    [
        (22.0, 21.0, 20.0, 20.5, "OPEN_ABOVE_HIGH"),
        (10.0, 12.0, 11.0, 11.5, "OPEN_BELOW_LOW"),
        (10.0, 12.0, 9.0, 12.5, "CLOSE_ABOVE_HIGH"),
        (10.0, 12.0, 9.0, 8.5, "CLOSE_BELOW_LOW"),
        (10.0, 8.0, 9.0, 9.5, "HIGH_BELOW_LOW"),
    ],
)
def test_invalid_candles_are_flagged(
    open_price, high, low, close, expected_issue
):
    data = pd.DataFrame({
        "Date": ["2026-01-02"],
        "Open": [open_price],
        "High": [high],
        "Low": [low],
        "Close": [close],
    })

    checked = validate_ohlc(data)

    assert not checked.loc[0, "_quality_valid"]
    assert expected_issue in checked.loc[0, "_quality_issues"]


def test_raw_input_is_not_modified():
    data = pd.DataFrame({
        "Date": ["2026-01-02", "2026-01-03"],
        "Open": [10.0, 22.0],
        "High": [12.0, 21.0],
        "Low": [9.0, 20.0],
        "Close": [11.0, 20.5],
    })
    original = data.copy(deep=True)

    validate_ohlc(data)

    pd.testing.assert_frame_equal(data, original)


def test_filter_keeps_only_valid_rows():
    data = pd.DataFrame({
        "Date": ["2026-01-02", "2026-01-03"],
        "Open": [10.0, 22.0],
        "High": [12.0, 21.0],
        "Low": [9.0, 20.0],
        "Close": [11.0, 20.5],
    })

    result = valid_ohlc_rows(data)

    assert len(result) == 1
    assert result.iloc[0]["Date"] == "2026-01-02"
    assert "_quality_valid" not in result.columns


def test_duplicate_dates_are_flagged():
    data = pd.DataFrame({
        "Date": ["2026-01-02", "2026-01-02"],
        "Open": [10.0, 10.5],
        "High": [12.0, 12.0],
        "Low": [9.0, 9.0],
        "Close": [11.0, 11.5],
    })

    checked = validate_ohlc(data)

    assert not checked["_quality_valid"].any()
    assert checked["_quality_issues"].str.contains(
        "DUPLICATE_DATE"
    ).all()


def test_missing_ohlc_column_fails_clearly():
    data = pd.DataFrame({
        "Date": ["2026-01-02"],
        "Open": [10.0],
        "High": [12.0],
        "Close": [11.0],
    })

    with pytest.raises(ValueError, match="Missing required OHLC"):
        validate_ohlc(data)


def test_quality_summary_counts_rows_and_issues():
    data = pd.DataFrame({
        "Date": ["2026-01-02", "2026-01-03"],
        "Open": [10.0, 22.0],
        "High": [12.0, 21.0],
        "Low": [9.0, 20.0],
        "Close": [11.0, 20.5],
    })

    summary = quality_summary(validate_ohlc(data))

    assert summary["total_rows"] == 2
    assert summary["valid_rows"] == 1
    assert summary["invalid_rows"] == 1
    assert summary["issue_counts"]["OPEN_ABOVE_HIGH"] == 1
