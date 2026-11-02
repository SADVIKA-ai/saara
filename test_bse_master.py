try:
    import bseindia
    result = bseindia.all_listed_securities(refresh=True)
    print("Refresh succeeded")
    print("Rows:", len(result))
    print(result.head(3).to_string() if hasattr(result, "head") else str(result)[:800])
except Exception as exc:
    print("Refresh failed:", type(exc).__name__, str(exc)[:500])
