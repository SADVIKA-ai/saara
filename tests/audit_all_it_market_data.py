
import pandas as pd

from src.market_data import history

MASTER_PATH = "data/master/companies_master.csv"
OUTPUT_PATH = "data/master/it_market_data_audit_all.csv"

master = pd.read_csv(MASTER_PATH)

required_columns = ["company_name", "exchange", "provider_symbol"]
missing_columns = [c for c in required_columns if c not in master.columns]

if missing_columns:
    raise ValueError(f"Missing required master columns: {missing_columns}")

results = []

for _, company in master.iterrows():
    symbol = str(company["provider_symbol"]).strip()

    row = {
        "company_name": company["company_name"],
        "exchange": company["exchange"],
        "ticker": company["ticker"],
        "provider_symbol": symbol,
        "rows": 0,
        "invalid_dates": None,
        "duplicate_dates": None,
        "dates_sorted": None,
        "missing_values": None,
        "non_positive_prices": None,
        "invalid_ohlc_rows": None,
        "largest_abs_close_change_pct": None,
        "status": "NOT_CHECKED",
        "error": "",
    }

    try:
        df = history(symbol).copy()

        if df.empty:
            row["status"] = "EMPTY_HISTORY"
            results.append(row)
            continue

        if "Date" not in df.columns:
            raise ValueError("Missing Date column")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        row["rows"] = len(df)
        row["invalid_dates"] = int(df["Date"].isna().sum())
        row["duplicate_dates"] = int(df["Date"].duplicated().sum())
        row["dates_sorted"] = bool(df["Date"].is_monotonic_increasing)
        row["missing_values"] = int(df.isna().sum().sum())

        price_columns = ["Open", "High", "Low", "Close", "Adj_Close"]
        missing_prices = [c for c in price_columns if c not in df.columns]

        if missing_prices:
            raise ValueError(f"Missing price columns: {missing_prices}")

        prices = df[price_columns].apply(pd.to_numeric, errors="coerce")
        row["non_positive_prices"] = int((prices <= 0).sum().sum())

        ohlc = df[["Open", "High", "Low", "Close"]].apply(
            pd.to_numeric, errors="coerce"
        )
        invalid_ohlc = (
            (ohlc["High"] < ohlc[["Open", "Close"]].max(axis=1))
            | (ohlc["Low"] > ohlc[["Open", "Close"]].min(axis=1))
            | (ohlc["High"] < ohlc["Low"])
        )
        row["invalid_ohlc_rows"] = int(invalid_ohlc.sum())

        close = pd.to_numeric(df["Close"], errors="coerce")
        changes = close.pct_change(fill_method=None).abs() * 100
        row["largest_abs_close_change_pct"] = (
            round(float(changes.max()), 4)
            if changes.notna().any()
            else None
        )

        if (
            row["invalid_dates"] > 0
            or row["duplicate_dates"] > 0
            or not row["dates_sorted"]
            or row["missing_values"] > 0
            or row["non_positive_prices"] > 0
            or row["invalid_ohlc_rows"] > 0
        ):
            row["status"] = "REVIEW"
        elif row["rows"] < 120:
            row["status"] = "SHORT_HISTORY"
        else:
            row["status"] = "PASSED_BASIC_CHECKS"

    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = f"{type(exc).__name__}: {exc}"

    results.append(row)

    print(
        f"[{len(results)}/{len(master)}] "
        f"{symbol}: {row['status']} "
        f"(rows={row['rows']})"
    )

audit = pd.DataFrame(results)
audit.to_csv(OUTPUT_PATH, index=False)

print("\n=== AUDIT SUMMARY ===")
print("Master records:", len(master))
print("Audited records:", len(audit))
print("\nStatus counts:")
print(audit["status"].value_counts(dropna=False).to_string())

print("\nOutput file:", OUTPUT_PATH)
print("\nRecords requiring review:")
print(
    audit.loc[
        audit["status"].isin(
            ["REVIEW", "SHORT_HISTORY", "EMPTY_HISTORY", "ERROR"]
        ),
        ["provider_symbol", "exchange", "rows", "status", "error"],
    ].to_string(index=False)
)
