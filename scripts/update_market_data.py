
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Allow execution from the project root using:
# python -m scripts.update_market_data
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LOGS
from src.market_data import read, update
from src.universe import load_master, provider_symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update market data for the configured company master."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N eligible master records.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between processed records.",
    )
    parser.add_argument(
        "--force-backfill",
        action="store_true",
        help="Force a historical backfill using the market-data updater.",
    )
    return parser.parse_args()


def is_rights_entitlement(row: dict) -> bool:
    """Identify rights-entitlement records that must not be price-updated."""
    values = [
        row.get("ticker", ""),
        row.get("company_name", ""),
        row.get("category", ""),
        row.get("security_type", ""),
    ]

    text = " ".join(
        str(value).strip().upper()
        for value in values
        if value is not None
    )

    return (
        "RIGHTS_ENTITLEMENT" in text
        or "RIGHTS ENTITLEMENT" in text
        or "-RE" in str(row.get("ticker", "")).strip().upper()
    )


def safe_int(value, default: int = 0) -> int:
    """Convert a value to int without allowing summary generation to crash."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def get_history_quality(symbol: str, result: dict) -> str:
    """
    Use the updater's quality result when available.
    Otherwise, inspect the stored history without modifying it.
    """
    quality = result.get("history_quality")

    if quality in {"ADEQUATE", "SHORT_HISTORY", "UNAVAILABLE"}:
        return quality

    try:
        frame = read(symbol)
        if frame is None or frame.empty:
            return "UNAVAILABLE"

        # Keep the same broad classification used in the project:
        # 252+ rows = adequate history; fewer rows = short history.
        return "ADEQUATE" if len(frame) >= 252 else "SHORT_HISTORY"

    except Exception:
        return "UNAVAILABLE"


def build_summary(
    master_records: int,
    results: list[dict],
    skipped_rights: list[dict],
) -> dict:
    """Build counts from the actual statuses returned by the updater."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "master_records_considered": master_records,
        "processed_records": len(results),
        "skipped_rights_entitlements": len(skipped_rights),

        # Update outcomes
        "updated": sum(
            result.get("status") == "UPDATED"
            for result in results
        ),
        "refreshed_no_new_dates": sum(
            result.get("status") == "REFRESHED_NO_NEW_DATES"
            for result in results
        ),
        "failed": sum(
            result.get("status") == "FAILED"
            for result in results
        ),
        "no_new_data": sum(
            result.get("status") == "NO_NEW_DATA"
            for result in results
        ),

        # Historical data availability
        "adequate_history": sum(
            result.get("history_quality") == "ADEQUATE"
            for result in results
        ),
        "short_history": sum(
            result.get("history_quality") == "SHORT_HISTORY"
            for result in results
        ),
        "unavailable_history": sum(
            result.get("history_quality") == "UNAVAILABLE"
            for result in results
        ),

        "skipped_rights_details": skipped_rights,
    }


def main() -> int:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        print("ERROR: --limit must be at least 1.", file=sys.stderr)
        return 2

    if args.delay < 0:
        print("ERROR: --delay cannot be negative.", file=sys.stderr)
        return 2

    LOGS.mkdir(parents=True, exist_ok=True)

    master = load_master()

    if master is None or master.empty:
        print("ERROR: The company master is empty or could not be loaded.")
        return 1

    records = master.to_dict(orient="records")

    skipped_rights: list[dict] = []
    eligible_records: list[dict] = []

    for row in records:
        if is_rights_entitlement(row):
            skipped_rights.append(
                {
                    "company_name": row.get("company_name", ""),
                    "exchange": row.get("exchange", ""),
                    "ticker": row.get("ticker", ""),
                    "reason": "RIGHTS_ENTITLEMENT",
                }
            )
        else:
            eligible_records.append(row)

    if args.limit is not None:
        eligible_records = eligible_records[: args.limit]

    results: list[dict] = []
    total = len(eligible_records)

    for index, row in enumerate(eligible_records, start=1):
        ticker = str(row.get("ticker", "")).strip()
        exchange = str(row.get("exchange", "")).strip()
        company_name = str(row.get("company_name", "")).strip()

        try:
            symbol = provider_symbol(row)

            if not symbol:
                raise ValueError(
                    f"Could not resolve provider symbol for ticker {ticker!r}"
                )

            result = update(
                symbol,
                force_backfill=args.force_backfill,
            )

            if not isinstance(result, dict):
                raise TypeError(
                    f"Updater returned {type(result).__name__}, expected dict."
                )

            result = dict(result)
            result.setdefault("symbol", symbol)
            result.setdefault("ticker", ticker)
            result.setdefault("exchange", exchange)
            result.setdefault("company_name", company_name)

            # Ensure each result has a known history-quality value.
            result["history_quality"] = get_history_quality(
                symbol,
                result,
            )

            result.setdefault("error", "")
            result.setdefault("status", "FAILED")

        except Exception as exc:
            result = {
                "symbol": ticker,
                "ticker": ticker,
                "exchange": exchange,
                "company_name": company_name,
                "status": "FAILED",
                "mode": "FORCE_BACKFILL" if args.force_backfill else "INCREMENTAL",
                "previous_rows": 0,
                "rows": 0,
                "new_dates": 0,
                "latest_date": "",
                "history_quality": "UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

        results.append(result)

        print(
            f"[{index}/{total}] "
            f"{result.get('symbol', ticker)}: "
            f"{result.get('status', 'UNKNOWN')} | "
            f"{result.get('history_quality', 'UNAVAILABLE')} | "
            f"rows={safe_int(result.get('rows'))}"
        )

        if args.delay > 0 and index < total:
            time.sleep(args.delay)

    # Keep skipped records separate from processed updater results.
    summary = build_summary(
        master_records=len(master),
        results=results,
        skipped_rights=skipped_rights,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    detailed_csv = LOGS / f"market_update_{timestamp}.csv"
    latest_csv = LOGS / "market_update_latest.csv"
    summary_json = LOGS / "market_update_summary_latest.json"

    results_frame = pd.DataFrame(results)

    # Save an empty CSV with no rows if no eligible records were processed.
    results_frame.to_csv(detailed_csv, index=False)
    results_frame.to_csv(latest_csv, index=False)

    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDetailed log: {detailed_csv}")
    print(f"Latest log:   {latest_csv}")
    print(f"Summary JSON: {summary_json}")

    # Any FAILED market update makes this script exit unsuccessfully.
    if summary["failed"] > 0:
        print(
            f"\nERROR: {summary['failed']} market-data update(s) failed."
        )
        return 1

    print("\nMarket-data update completed without reported failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
