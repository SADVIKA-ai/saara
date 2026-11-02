from pathlib import Path
from datetime import datetime, timedelta
import inspect
import re
import shutil
import sys

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from src.config import PRED, INITIAL_YEARS
from src.universe import load_master, provider_symbol
from src.market_data import update, history
from src.sentiment import fetch, aggregate
from src.predict import predict

if INITIAL_YEARS != 10:
    raise SystemExit(f"STOP: INITIAL_YEARS={INITIAL_YEARS}, expected 10.")

latest = PRED / "predictions_latest.csv"
if not latest.exists():
    raise SystemExit(f"STOP: prediction report not found: {latest}")

sig = inspect.signature(update)
if "backfill_years" not in sig.parameters:
    raise SystemExit("STOP: update() does not support backfill_years.")

original = pd.read_csv(latest, dtype={"ticker": str})
if not {"ticker", "status"}.issubset(original.columns):
    raise SystemExit("STOP: report must contain ticker and status columns.")

master = load_master()
if not {"ticker", "exchange"}.issubset(master.columns):
    raise SystemExit("STOP: master must contain ticker and exchange columns.")

def norm(x):
    return str(x).strip().upper()

def is_rights(ticker):
    return bool(re.search(r"-RE\d*$", norm(ticker), re.I))

def pick(row, names):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return str(row[name]).strip()
    return ""

# Backup the existing report before any work.
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = latest.with_name(f"predictions_latest_before_recovery_{stamp}.csv")
shutil.copy2(latest, backup)

master = master.copy()
master["_ticker_key"] = master["ticker"].map(norm)
master["_exchange_key"] = master["exchange"].map(norm)

work = original.copy()
work["_ticker_key"] = work["ticker"].map(norm)
if "exchange" not in work.columns:
    work["exchange"] = ""
work["_exchange_key"] = work["exchange"].map(norm)

statuses = work["status"].fillna("").astype(str).str.strip().str.upper()
mask = (statuses != "OK") & (~work["ticker"].map(is_rights))
mask &= ~statuses.str.startswith("SKIPPED_")
targets = work.index[mask].tolist()

print(f"INITIAL_YEARS={INITIAL_YEARS}; recovery window=3 years")
print(f"Existing OK rows protected: {(statuses == 'OK').sum()}")
print(f"Non-OK ordinary-share candidates: {len(targets)}")
print(f"Backup: {backup}")

log = []
updated_count = 0
cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365.25 * 3)

for n, idx in enumerate(targets, 1):
    row = work.loc[idx]
    ticker = norm(row["ticker"])
    exchange = norm(row.get("exchange", ""))

    candidates = master[master["_ticker_key"] == ticker]
    if exchange:
        exact = candidates[candidates["_exchange_key"] == exchange]
        if not exact.empty:
            candidates = exact

    if candidates.empty:
        log.append({"ticker": ticker, "status": "MASTER_MATCH_NOT_FOUND"})
        print(f"[{n}/{len(targets)}] {ticker}: master match missing")
        continue

    company = candidates.iloc[0]
    try:
        symbol = provider_symbol(company)
        if not symbol:
            raise ValueError("No provider symbol")

        kwargs = {"force_backfill": True, "backfill_years": 3}
        if "exchange" in sig.parameters:
            kwargs["exchange"] = str(company["exchange"])
        if "ticker" in sig.parameters:
            kwargs["ticker"] = ticker
        if "security_code" in sig.parameters:
            kwargs["security_code"] = pick(
                company,
                ["security_code", "securityCode", "scrip_code",
                 "scripCode", "bse_code", "Security Code"]
            )

        update_result = update(symbol, **kwargs)
        prices = history(symbol)

        if prices is None or prices.empty or "Date" not in prices.columns:
            raise ValueError(f"No readable history after refresh: {update_result}")

        dates = pd.to_datetime(prices["Date"], errors="coerce").dropna()
        if dates.empty:
            raise ValueError("No valid historical dates")
        if dates.duplicated().any():
            raise ValueError("Duplicate dates in historical data")
        if dates.min().normalize() > cutoff + pd.Timedelta(days=45):
            raise ValueError(
                f"Less than approximately 3 years of coverage; "
                f"first date={dates.min().date()}"
            )
        if len(prices) < 120:
            raise ValueError(f"Only {len(prices)} history rows")

        sentiment = aggregate(fetch(str(company.get("company_name", ticker))))
        prediction = predict(prices, sentiment.get("sentiment_score", 0.0))
        new_status = str(prediction.get("status", "")).upper()

        if new_status != "OK":
            raise ValueError(f"Prediction status is {new_status or 'missing'}")

        # Update only this non-OK row; leave all other rows unchanged.
        replacement = {
            **{k: v for k, v in sentiment.items()},
            **{k: v for k, v in prediction.items()},
        }
        for key, value in replacement.items():
            if key not in work.columns:
                work[key] = pd.NA
            work.at[idx, key] = value

        updated_count += 1
        log.append({
            "ticker": ticker, "provider_symbol": symbol,
            "status": "RECOVERED", "rows": len(prices),
            "first_date": str(dates.min().date()),
            "last_date": str(dates.max().date()),
            "provider_update": str(update_result)[:300],
        })
        print(f"[{n}/{len(targets)}] {ticker}: RECOVERED")

    except Exception as exc:
        log.append({
            "ticker": ticker, "provider_symbol": locals().get("symbol", ""),
            "status": "NOT_RECOVERED",
            "detail": f"{type(exc).__name__}: {exc}",
        })
        print(f"[{n}/{len(targets)}] {ticker}: not recovered — {exc}")

# Remove helper columns and protect previously-OK rows with a final equality check.
work.drop(columns=["_ticker_key", "_exchange_key"], errors="ignore", inplace=True)
before_ok = original[original["status"].fillna("").astype(str).str.upper() == "OK"]
after = work.loc[before_ok.index]
for col in original.columns:
    if col in after.columns:
        a = before_ok[col].fillna("").astype(str).reset_index(drop=True)
        b = after[col].fillna("").astype(str).reset_index(drop=True)
        if not a.equals(b):
            raise SystemExit(f"STOP: existing OK rows changed in column {col}; report not written.")

report = PRED / f"recovery_results_{stamp}.csv"
pd.DataFrame(log).to_csv(report, index=False)

# Atomic write to avoid leaving a partial latest report.
temp = latest.with_suffix(".recovery.tmp")
work.to_csv(temp, index=False)
pd.read_csv(temp)
temp.replace(latest)

print("\nRECOVERY SUMMARY")
print(f"Successful recoveries: {updated_count}")
print(f"Original OK rows preserved: {len(before_ok)}")
print(f"Recovery log: {report}")
print(f"Updated report: {latest}")
print("INITIAL_YEARS remains 10.")
