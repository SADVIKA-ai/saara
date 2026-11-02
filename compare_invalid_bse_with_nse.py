
from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "master" / "companies_master.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "master"
    / "invalid_bse_nse_reconciliation.csv"
)

TARGET_BSE_SYMBOLS = {
    "3IINFOLTD",
    "AFFLE",
    "BSOFT",
    "COFORGE",
    "CYBERTECH",
    "DEVIT",
    "ECLERX",
    "HAPPSTMNDS",
    "NETWEB",
    "OMNIAX",
    "SASKEN",
    "SONATSOFTW",
    "STARCOM",
    "TANLA",
}

OHLC_COLUMNS = ("open", "high", "low", "close")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_symbol(value: object) -> str:
    symbol = normalize_text(value).upper()
    symbol = re.sub(r"\.(NS|BO)$", "", symbol)
    return symbol


def normalize_exchange(value: object) -> str:
    value = normalize_text(value).upper()
    if "NSE" in value:
        return "NSE"
    if "BSE" in value:
        return "BSE"
    return value


def find_column(frame: pd.DataFrame, names: set[str]) -> str | None:
    for column in frame.columns:
        if str(column).strip().lower() in names:
            return column
    return None


def read_master() -> pd.DataFrame:
    if not MASTER_PATH.is_file():
        raise FileNotFoundError(f"Master file not found: {MASTER_PATH}")

    frame = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    required = {"exchange", "isin"}
    actual = {str(c).strip().lower() for c in frame.columns}
    missing = required - actual
    if missing:
        raise ValueError(
            f"Master is missing required columns {sorted(missing)}. "
            f"Found: {list(frame.columns)}"
        )

    # Normalize column names without changing the source file.
    frame.columns = [str(c).strip().lower() for c in frame.columns]

    if "provider_symbol" not in frame.columns:
        frame["provider_symbol"] = ""
    if "ticker" not in frame.columns:
        frame["ticker"] = ""
    if "company_name" not in frame.columns:
        frame["company_name"] = ""

    frame["_exchange"] = frame["exchange"].map(normalize_exchange)
    frame["_isin"] = frame["isin"].map(
        lambda x: normalize_text(x).upper()
    )
    frame["_provider_base"] = frame["provider_symbol"].map(normalize_symbol)
    frame["_ticker_base"] = frame["ticker"].map(normalize_symbol)

    return frame


def candidate_history_paths(record: pd.Series, exchange: str) -> list[Path]:
    suffix = ".NS" if exchange == "NSE" else ".BO"
    candidates: list[Path] = []

    for field in ("provider_symbol", "ticker"):
        value = normalize_text(record.get(field, ""))
        if not value:
            continue

        candidates.append(RAW_DIR / f"{value}.csv")

        base = normalize_symbol(value)
        if base:
            candidates.append(RAW_DIR / f"{base}{suffix}.csv")

    # De-duplicate while preserving priority.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()).casefold()
        if key not in seen:
            unique.append(path)
            seen.add(key)

    return unique


def read_history(record: pd.Series, exchange: str) -> tuple[pd.DataFrame | None, str]:
    paths = candidate_history_paths(record, exchange)
    existing = [p for p in paths if p.is_file()]

    if not existing:
        return None, "HISTORY_FILE_MISSING"

    # Use the first matching path only; do not silently combine files.
    path = existing[0]

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return None, f"READ_ERROR:{type(exc).__name__}:{exc}"

    date_col = find_column(
        frame,
        {"date", "datetime", "timestamp", "time", "unnamed: 0"},
    )
    if date_col is None:
        return None, f"DATE_COLUMN_MISSING:{path.name}"

    mapped: dict[str, str] = {}
    for name in OHLC_COLUMNS:
        column = find_column(frame, {name})
        if column is None:
            return None, f"OHLC_COLUMN_MISSING:{name}:{path.name}"
        mapped[name] = column

    output = pd.DataFrame()
    output["date"] = pd.to_datetime(
        frame[date_col], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()

    for name, column in mapped.items():
        output[name] = pd.to_numeric(frame[column], errors="coerce")

    output = output.dropna(subset=["date"]).sort_values("date")
    output = output.drop_duplicates(subset=["date"], keep="last")
    output.attrs["source_path"] = str(path)
    return output, "OK"


def invalid_rules(row: pd.Series) -> list[str]:
    values = {name: row.get(name) for name in OHLC_COLUMNS}
    if any(pd.isna(value) for value in values.values()):
        return ["MISSING_OR_NONNUMERIC_OHLC"]

    open_price = float(values["open"])
    high = float(values["high"])
    low = float(values["low"])
    close = float(values["close"])

    rules = []
    if high < low:
        rules.append("HIGH_BELOW_LOW")
    if open_price > high:
        rules.append("OPEN_ABOVE_HIGH")
    if open_price < low:
        rules.append("OPEN_BELOW_LOW")
    if close > high:
        rules.append("CLOSE_ABOVE_HIGH")
    if close < low:
        rules.append("CLOSE_BELOW_LOW")

    return rules


def relative_difference_pct(left: object, right: object) -> float | None:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return None

    if pd.isna(a) or pd.isna(b) or a == 0:
        return None

    return round((b - a) / abs(a) * 100.0, 4)


def main() -> int:
    if not RAW_DIR.is_dir():
        print(f"Raw data directory not found: {RAW_DIR}")
        return 2

    master = read_master()
    bse_records = master[
        (master["_exchange"] == "BSE")
        & (
            master["_provider_base"].isin(TARGET_BSE_SYMBOLS)
            | master["_ticker_base"].isin(TARGET_BSE_SYMBOLS)
        )
    ]

    if bse_records.empty:
        print("No target BSE symbols matched the master file.")
        print("Check ticker/provider_symbol values; no files were changed.")
        return 2

    output_rows: list[dict[str, object]] = []

    for _, bse in bse_records.iterrows():
        bse_symbol = (
            normalize_text(bse.get("provider_symbol", ""))
            or normalize_text(bse.get("ticker", ""))
        )
        base_symbol = normalize_symbol(bse_symbol)
        isin = normalize_text(bse.get("isin", "")).upper()
        company_name = normalize_text(bse.get("company_name", ""))

        bse_history, bse_status = read_history(bse, "BSE")
        if bse_history is None:
            output_rows.append({
                "bse_symbol": bse_symbol,
                "company_name": company_name,
                "isin": isin,
                "bse_date": "",
                "bse_open": None,
                "bse_high": None,
                "bse_low": None,
                "bse_close": None,
                "bse_violations": "",
                "nse_symbol": "",
                "nse_date": "",
                "nse_open": None,
                "nse_high": None,
                "nse_low": None,
                "nse_close": None,
                "nse_violations": "",
                "close_difference_pct_nse_vs_bse": None,
                "status": f"BSE_{bse_status}",
                "bse_source_file": "",
                "nse_source_file": "",
            })
            continue

        bse_invalid = bse_history[
            bse_history.apply(
                lambda row: bool(invalid_rules(row)), axis=1
            )
        ]

        # Cross-exchange matching uses ISIN, not a guessed ticker mapping.
        nse_matches = master[
            (master["_exchange"] == "NSE")
            & (master["_isin"] == isin)
        ] if isin else master.iloc[0:0]

        if nse_matches.empty:
            nse_match_records = []
        else:
            nse_match_records = [
                record for _, record in nse_matches.iterrows()
            ]

        if bse_invalid.empty:
            continue

        for _, bse_candle in bse_invalid.iterrows():
            violations = invalid_rules(bse_candle)
            base_result = {
                "bse_symbol": bse_symbol,
                "company_name": company_name,
                "isin": isin,
                "bse_date": bse_candle["date"].date().isoformat(),
                "bse_open": bse_candle["open"],
                "bse_high": bse_candle["high"],
                "bse_low": bse_candle["low"],
                "bse_close": bse_candle["close"],
                "bse_violations": ";".join(violations),
                "bse_source_file": bse_history.attrs.get("source_path", ""),
            }

            if not nse_match_records:
                output_rows.append({
                    **base_result,
                    "nse_symbol": "",
                    "nse_date": "",
                    "nse_open": None,
                    "nse_high": None,
                    "nse_low": None,
                    "nse_close": None,
                    "nse_violations": "",
                    "close_difference_pct_nse_vs_bse": None,
                    "status": (
                        "NO_ISIN" if not isin else "NO_NSE_LISTING_SAME_ISIN"
                    ),
                    "nse_source_file": "",
                })
                continue

            for nse in nse_match_records:
                nse_symbol = (
                    normalize_text(nse.get("provider_symbol", ""))
                    or normalize_text(nse.get("ticker", ""))
                )
                nse_history, nse_status = read_history(nse, "NSE")

                if nse_history is None:
                    output_rows.append({
                        **base_result,
                        "nse_symbol": nse_symbol,
                        "nse_date": "",
                        "nse_open": None,
                        "nse_high": None,
                        "nse_low": None,
                        "nse_close": None,
                        "nse_violations": "",
                        "close_difference_pct_nse_vs_bse": None,
                        "status": f"NSE_{nse_status}",
                        "nse_source_file": "",
                    })
                    continue

                matches = nse_history[
                    nse_history["date"] == bse_candle["date"]
                ]

                if matches.empty:
                    output_rows.append({
                        **base_result,
                        "nse_symbol": nse_symbol,
                        "nse_date": "",
                        "nse_open": None,
                        "nse_high": None,
                        "nse_low": None,
                        "nse_close": None,
                        "nse_violations": "",
                        "close_difference_pct_nse_vs_bse": None,
                        "status": "NSE_DATE_MISSING",
                        "nse_source_file": nse_history.attrs.get(
                            "source_path", ""
                        ),
                    })
                    continue

                nse_candle = matches.iloc[-1]
                nse_violations = invalid_rules(nse_candle)
                output_rows.append({
                    **base_result,
                    "nse_symbol": nse_symbol,
                    "nse_date": nse_candle["date"].date().isoformat(),
                    "nse_open": nse_candle["open"],
                    "nse_high": nse_candle["high"],
                    "nse_low": nse_candle["low"],
                    "nse_close": nse_candle["close"],
                    "nse_violations": ";".join(nse_violations),
                    "close_difference_pct_nse_vs_bse": (
                        relative_difference_pct(
                            bse_candle["close"], nse_candle["close"]
                        )
                    ),
                    "status": (
                        "BOTH_INVALID"
                        if nse_violations
                        else "NSE_MATCH_FOUND_REVIEW_BSE"
                    ),
                    "nse_source_file": nse_history.attrs.get(
                        "source_path", ""
                    ),
                })

    if not output_rows:
        print(
            "No invalid OHLC rows were found for the target symbols in the "
            "currently available raw files. This does not prove the earlier "
            "audit was wrong; files may have changed."
        )
        return 1

    report = pd.DataFrame(output_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print(f"Report saved: {OUTPUT_PATH}")
    print(f"Reconciliation rows: {len(report)}")
    print("\nStatus counts:")
    print(report["status"].value_counts(dropna=False).to_string())
    print(
        "\nImportant: matching NSE data is cross-exchange corroboration, "
        "not independent-provider verification if both files came from "
        "the same data vendor."
    )
    print("Raw CSVs and the master file were not modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Reconciliation failed: {type(exc).__name__}: {exc}")
        raise
