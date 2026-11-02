from bseindia import equity

tests = [
    ("HCL-INSYS", "01-10-2026", "29-10-2026"),
    ("500179", "01-10-2026", "29-10-2026"),
    ("SBIN", "01-10-2026", "29-10-2026"),
]

for symbol, start, end in tests:
    print(f"\nTEST: {symbol}")
    try:
        result = equity.historical_stock_data(
            symbol=symbol,
            from_date=start,
            to_date=end,
        )
        print("Type:", type(result).__name__)
        print("Rows:", len(result))
        print(result.head(2) if hasattr(result, "head") else str(result)[:400])
    except Exception as exc:
        print("ERROR:", type(exc).__name__, str(exc)[:300])
