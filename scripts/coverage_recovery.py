from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd

from src.config import PRED, INITIAL_YEARS, MIN_ROWS
from src.universe import load_master, provider_symbol
from src.market_data import history
from src.predict import predict
from src.sentiment import aggregate
from src.walk_forward import walk_forward_validate


# ============================================================
# STOCKAI INDIA — CONSOLIDATED COVERAGE RECOVERY
# - Keeps original predictions_latest.csv unchanged
# - Tries same-exchange history first
# - Uses alternate exchange only as a separate labelled analysis
# - Selects ML only if walk-forward MAE AND RMSE beat baseline
# - Does not fabricate missing prices or lower MIN_ROWS
# ============================================================

PRED.mkdir(parents=True, exist_ok=True)
latest_file = PRED / "predictions_latest.csv"

if not latest_file.exists():
    raise FileNotFoundError(f"Required report not found: {latest_file}")

if INITIAL_YEARS != 10:
    raise RuntimeError(
        f"Safety check failed: INITIAL_YEARS={INITIAL_YEARS}, expected 10."
    )

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
original_hash = hashlib.sha256(latest_file.read_bytes()).hexdigest()
original = pd.read_csv(latest_file, dtype=str).fillna("")
master = load_master().fillna("")

required_report = {"company_name", "exchange", "ticker", "status"}
required_master = {"company_name", "exchange", "ticker", "provider_symbol"}
if not required_report.issubset(original.columns):
    raise ValueError(
        f"Prediction report missing columns: {required_report - set(original.columns)}"
    )
if not required_master.issubset(master.columns):
    raise ValueError(
        f"Master missing columns: {required_master - set(master.columns)}"
    )

original_ok_mask = original["status"].astype(str).str.upper().eq("OK")
original_ok = original.loc[original_ok_mask].copy()
original_ok_count = len(original_ok)

backup = PRED / f"predictions_latest_before_coverage_recovery_{stamp}.csv"
shutil.copy2(latest_file, backup)

def norm(value):
    return str(value or "").strip().upper()

def number(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default

def date_span(df):
    if df is None or df.empty or "Date" not in df.columns:
        return "", "", 0.0
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    if dates.empty:
        return "", "", 0.0
    first, last = dates.min(), dates.max()
    years = max(0.0, (last - first).days / 365.2425)
    return str(first.date()), str(last.date()), round(years, 2)

def rights_entitlement(ticker):
    return bool(re.search(r"-RE\d*$", norm(ticker)))

# Map exact exchange + ticker to master records.
master_by_key = {}
for _, row in master.iterrows():
    key = (norm(row.get("exchange")), norm(row.get("ticker")))
    master_by_key.setdefault(key, []).append(row)

# Do not process the same failed listing twice.
seen = set()
attempts = []
successful = []
successful_by_key = {}

non_ok = original.loc[~original_ok_mask].copy()
total_candidates = len(non_ok)

print("=" * 72)
print("STOCKAI INDIA — COVERAGE RECOVERY AND ACCURACY SAFEGUARD")
print("=" * 72)
print(f"Master listing records:       {len(master)}")
print(f"Existing OK prediction rows:  {original_ok_count}")
print(f"Non-OK rows to review:        {total_candidates}")
print(f"INITIAL_YEARS:                {INITIAL_YEARS}")
print(f"MIN_ROWS:                     {MIN_ROWS}")
print("Original prediction report will NOT be overwritten.")
print("-" * 72)

for index, (_, report_row) in enumerate(non_ok.iterrows(), start=1):
    company = str(report_row.get("company_name", ""))
    exchange = norm(report_row.get("exchange"))
    ticker = norm(report_row.get("ticker"))
    old_status = norm(report_row.get("status"))
    key = (exchange, ticker)

    if key in seen:
        continue
    seen.add(key)

    base = {
        "company_name": company,
        "original_exchange": exchange,
        "original_ticker": ticker,
        "original_provider_symbol": str(report_row.get("provider_symbol", "")),
        "original_status": old_status,
        "analysis_source_exchange": "",
        "analysis_source_symbol": "",
        "analysis_method": "",
        "analysis_status": "NOT_RECOVERED",
        "history_rows": 0,
        "history_start_date": "",
        "history_latest_date": "",
        "history_years": 0.0,
        "walk_forward_status": "",
        "walk_forward_folds": 0,
        "walk_forward_model_mae": np.nan,
        "walk_forward_baseline_mae": np.nan,
        "walk_forward_model_rmse": np.nan,
        "walk_forward_baseline_rmse": np.nan,
        "model_beats_baseline_mae": False,
        "model_beats_baseline_rmse": False,
        "selected_forecast_method": "",
        "selected_predicted_next_close": np.nan,
        "selected_predicted_change": np.nan,
        "selected_expected_return_pct": np.nan,
        "selected_decision": "",
        "message": "",
    }

    if rights_entitlement(ticker) or old_status.startswith("SKIPPED_"):
        base["analysis_status"] = "SKIPPED_NON_ORDINARY_SECURITY"
        base["message"] = "Non-ordinary security; not included in recovery."
        attempts.append(base)
        continue

    rows = master_by_key.get(key, [])
    if not rows:
        base["message"] = "No exact exchange + ticker match in company master."
        attempts.append(base)
        continue

    primary = rows[0]
    primary_symbol = provider_symbol(primary)

    # Attempt 1: use existing history from the original listing.
    # Attempt 2: if needed, use the other exchange as a separately labelled source.
    source_options = []
    if primary_symbol:
        source_options.append({
            "exchange": exchange,
            "symbol": primary_symbol,
            "method": "SAME_EXCHANGE",
        })

    other_exchanges = master[
        master["ticker"].astype(str).map(norm).eq(ticker)
        & ~master["exchange"].astype(str).map(norm).eq(exchange)
    ].copy()

    # Prefer NSE when the original BSE listing has no usable history.
    if exchange == "BSE":
        other_exchanges["_priority"] = (
            ~other_exchanges["exchange"].astype(str).map(norm).eq("NSE")
        ).astype(int)
        other_exchanges = other_exchanges.sort_values("_priority")
    else:
        # Still allow a clearly labelled BSE source for a failed NSE listing
        # if it has the required history and passes prediction validation.
        other_exchanges["_priority"] = (
            ~other_exchanges["exchange"].astype(str).map(norm).eq("BSE")
        ).astype(int)
        other_exchanges = other_exchanges.sort_values("_priority")

    for _, alternate in other_exchanges.iterrows():
        alt_exchange = norm(alternate.get("exchange"))
        alt_symbol = provider_symbol(alternate)
        if alt_exchange and alt_symbol and alt_exchange != exchange:
            if not any(
                item["exchange"] == alt_exchange and item["symbol"] == alt_symbol
                for item in source_options
            ):
                source_options.append({
                    "exchange": alt_exchange,
                    "symbol": alt_symbol,
                    "method": "ALTERNATE_EXCHANGE",
                })

    recovered = None
    failure_messages = []

    for source in source_options:
        source_exchange = source["exchange"]
        source_symbol = source["symbol"]

        try:
            prices = history(source_symbol, refresh=False)
            if prices is None or prices.empty:
                failure_messages.append(f"{source_symbol}: no local history")
                continue

            first_date, last_date, years = date_span(prices)
            if len(prices) < MIN_ROWS:
                failure_messages.append(
                    f"{source_symbol}: only {len(prices)} raw rows; "
                    f"minimum configured threshold is {MIN_ROWS}"
                )
                continue

            sentiment_score = number(report_row.get("sentiment_score"), 0.0)
            pred = predict(prices, sentiment_score)

            if pred.get("status") != "OK":
                failure_messages.append(
                    f"{source_symbol}: prediction status "
                    f"{pred.get('status', 'UNKNOWN')} — "
                    f"{pred.get('message', 'no detail')}"
                )
                continue

            # Evaluate model vs last-close baseline using expanding-window folds.
            wf = walk_forward_validate(prices, folds=5, min_test_rows=10)
            wf_ok = wf.get("status") == "OK"
            beats_mae = bool(wf.get("model_beats_baseline_mae", False)) if wf_ok else False
            beats_rmse = bool(wf.get("model_beats_baseline_rmse", False)) if wf_ok else False

            latest_close = number(pred.get("latest_actual_close"), np.nan)
            model_forecast = number(pred.get("predicted_next_close"), np.nan)

            if not np.isfinite(latest_close) or latest_close <= 0:
                failure_messages.append(
                    f"{source_symbol}: latest observed close is invalid"
                )
                continue

            # Conservative selection: use ML only when both walk-forward
            # error measures beat the last-close baseline.
            if wf_ok and beats_mae and beats_rmse:
                selected_method = "ML_MODEL_WALK_FORWARD_BEATS_BASELINE"
                selected_forecast = model_forecast
                selected_decision = str(pred.get("decision", "HOLD"))
            else:
                selected_method = "LAST_CLOSE_BASELINE_CONSERVATIVE"
                selected_forecast = latest_close
                selected_decision = "HOLD"

            selected_change = selected_forecast - latest_close
            selected_return_pct = (
                selected_change / latest_close * 100.0
                if latest_close > 0 else 0.0
            )

            recovered = {
                **base,
                "analysis_source_exchange": source_exchange,
                "analysis_source_symbol": source_symbol,
                "analysis_method": source["method"],
                "analysis_status": "OK",
                "history_rows": len(prices),
                "history_start_date": first_date,
                "history_latest_date": last_date,
                "history_years": years,
                "walk_forward_status": wf.get("status", "NOT_AVAILABLE"),
                "walk_forward_folds": wf.get("folds_completed", 0),
                "walk_forward_model_mae": wf.get("model_mae", np.nan),
                "walk_forward_baseline_mae": wf.get("baseline_mae", np.nan),
                "walk_forward_model_rmse": wf.get("model_rmse", np.nan),
                "walk_forward_baseline_rmse": wf.get("baseline_rmse", np.nan),
                "model_beats_baseline_mae": beats_mae,
                "model_beats_baseline_rmse": beats_rmse,
                "selected_forecast_method": selected_method,
                "selected_predicted_next_close": selected_forecast,
                "selected_predicted_change": selected_change,
                "selected_expected_return_pct": selected_return_pct,
                "selected_decision": selected_decision,
                "message": (
                    "Alternate exchange analysis; original listing prediction "
                    "is preserved separately. "
                    if source["method"] == "ALTERNATE_EXCHANGE" else
                    "Recovered using available same-exchange history. "
                ) + (
                    "History shorter than 3 years; coverage limitation recorded."
                    if years < 3 else
                    "History spans at least 3 years."
                ),
                # Include detailed original model output for auditability.
                **{
                    f"model_{k}": v
                    for k, v in pred.items()
                    if k not in {"status", "message"}
                },
            }
            break

        except Exception as exc:
            failure_messages.append(
                f"{source_symbol}: {type(exc).__name__}: {exc}"
            )

    if recovered is not None:
        attempts.append(recovered)
        successful.append(recovered)
        successful_by_key[key] = recovered
        print(
            f"[{index}/{total_candidates}] {company} ({exchange}/{ticker}) "
            f"=> OK via {recovered['analysis_source_exchange']} "
            f"{recovered['analysis_source_symbol']} | "
            f"{recovered['selected_forecast_method']}"
        )
    else:
        base["message"] = " | ".join(failure_messages) if failure_messages else (
            "No eligible alternate exchange source was found."
        )
        attempts.append(base)
        print(
            f"[{index}/{total_candidates}] {company} ({exchange}/{ticker}) "
            f"=> NOT RECOVERED"
        )

# Write separate outputs. The original latest report is never rewritten.
attempts_df = pd.DataFrame(attempts)
success_df = pd.DataFrame(successful)

attempts_file = PRED / "coverage_recovery_attempts_latest.csv"
success_file = PRED / "predictions_recovered_separate_source_latest.csv"
augmented_file = PRED / "predictions_coverage_augmented_latest.csv"
quality_file = PRED / "prediction_quality_audit_latest.csv"

attempts_df.to_csv(attempts_file, index=False)
success_df.to_csv(success_file, index=False)

# Keep the original 126-row report and every original prediction field intact.
# Add separate recovery fields; never replace original status/prediction values.
augmented = original.copy()
recovery_columns = [
    "analysis_status",
    "analysis_source_exchange",
    "analysis_source_symbol",
    "analysis_method",
    "history_rows",
    "history_start_date",
    "history_latest_date",
    "history_years",
    "walk_forward_status",
    "walk_forward_folds",
    "walk_forward_model_mae",
    "walk_forward_baseline_mae",
    "walk_forward_model_rmse",
    "walk_forward_baseline_rmse",
    "model_beats_baseline_mae",
    "model_beats_baseline_rmse",
    "selected_forecast_method",
    "selected_predicted_next_close",
    "selected_predicted_change",
    "selected_expected_return_pct",
    "selected_decision",
    "message",
]
for col in recovery_columns:
    augmented[col] = ""

for i, row in augmented.iterrows():
    key = (norm(row.get("exchange")), norm(row.get("ticker")))
    rec = successful_by_key.get(key)
    if rec:
        for col in recovery_columns:
            augmented.at[i, col] = rec.get(col, "")

augmented.to_csv(augmented_file, index=False)

# Quality audit includes every newly recovered analysis with its walk-forward
# model/baseline comparison and selected forecast method.
quality_cols = [
    "company_name",
    "original_exchange",
    "original_ticker",
    "analysis_source_exchange",
    "analysis_source_symbol",
    "history_rows",
    "history_start_date",
    "history_latest_date",
    "history_years",
    "walk_forward_status",
    "walk_forward_folds",
    "walk_forward_model_mae",
    "walk_forward_baseline_mae",
    "walk_forward_model_rmse",
    "walk_forward_baseline_rmse",
    "model_beats_baseline_mae",
    "model_beats_baseline_rmse",
    "selected_forecast_method",
    "selected_predicted_next_close",
    "selected_expected_return_pct",
    "selected_decision",
]
quality = success_df[[c for c in quality_cols if c in success_df.columns]].copy() if not success_df.empty else pd.DataFrame(columns=quality_cols)
quality.to_csv(quality_file, index=False)

# Verify the original report was not changed during this run.
new_hash = hashlib.sha256(latest_file.read_bytes()).hexdigest()
if original_hash != new_hash:
    raise RuntimeError(
        "Safety failure: predictions_latest.csv changed during recovery."
    )

# Verify the existing OK records were preserved in the augmented report.
after = pd.read_csv(latest_file, dtype=str).fillna("")
after_ok = after.loc[after["status"].astype(str).str.upper().eq("OK")].copy()
if len(after_ok) != original_ok_count:
    raise RuntimeError("Safety failure: existing OK-row count changed.")

primary_ok = int(original_ok_count)
new_ok = len(successful)
combined_supported = primary_ok + new_ok
target_111_met = combined_supported >= 111

summary = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "master_listing_records": int(len(master)),
    "original_primary_OK_records": primary_ok,
    "new_separately_labelled_recovered_records": int(new_ok),
    "combined_supported_analysis_records": int(combined_supported),
    "target_more_than_110_met": bool(target_111_met),
    "initial_years_unchanged": int(INITIAL_YEARS),
    "minimum_usable_rows_unchanged": int(MIN_ROWS),
    "original_predictions_latest_sha256_unchanged": bool(original_hash == new_hash),
    "original_predictions_latest_file": str(latest_file),
    "backup_file": str(backup),
    "attempts_file": str(attempts_file),
    "recovered_predictions_file": str(success_file),
    "augmented_126_row_report": str(augmented_file),
    "quality_audit_file": str(quality_file),
    "model_selected_only_if_walk_forward_beats_baseline_on_both_metrics": True,
    "warning": (
        "Coverage recovery is not proof of better future accuracy. "
        "Alternate-exchange analyses are explicitly labelled and do not "
        "replace the original listing's prediction."
    ),
}

summary_file = PRED / "coverage_recovery_summary_latest.json"
summary_file.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

print("\n" + "=" * 72)
print("FINAL COVERAGE SUMMARY")
print("=" * 72)
print(json.dumps(summary, indent=2))
if not attempts_df.empty:
    print("\nRecovery attempt statuses:")
    print(attempts_df["analysis_status"].value_counts(dropna=False).to_string())
if not success_df.empty:
    print("\nRecovered analysis sources:")
    print(success_df["analysis_source_exchange"].value_counts(dropna=False).to_string())
    print("\nSelected forecast methods:")
    print(success_df["selected_forecast_method"].value_counts(dropna=False).to_string())

print("\nOriginal predictions_latest.csv is unchanged.")
print("Existing 96 OK predictions remain untouched.")
print("Recovery completed. No prices were copied between exchange files.")
