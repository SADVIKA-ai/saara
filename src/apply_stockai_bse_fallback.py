    """Fetch public BSE historical data; never fail the whole pipeline on provider errors."""
    try:
        from bseindia import equity
    except Exception as exc:
        return pd.DataFrame(), "", f"bseindia import: {type(exc).__name__}: {exc}"
    candidates = []
    if str(ticker or "").strip():
        candidates.append(str(ticker).strip())
    code = str(security_code or "").strip()
    if code.endswith(".0"):
        code = code[:-2]
    if code.isdigit() and len(code) == 6 and code not in candidates:
        candidates.append(code)
    errors = []
    for candidate in candidates:
        try:
            raw = equity.historical_stock_data(symbol=candidate, from_date=start.strftime("%d-%m-%Y"), to_date=end.strftime("%d-%m-%Y"))
            data = _normalize_bse_history(raw)
            if not data.empty:
                return data, candidate, ""
            errors.append(f"{candidate}: no usable rows")
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(), "", "; ".join(errors)
'''
m = replace_once(m, 'def update(symbol, retries=3, force_backfill=False):', helpers + '\ndef update(symbol, retries=3, force_backfill=False, exchange=None, ticker=None, security_code=None):', 'helpers/signature')
m = m.replace('needs_backfill = old.empty or force_backfill', 'needs_backfill = old.empty or force_backfill or len(old) < MIN_ROWS')
marker = '    # Do not overwrite existing data when the download is empty.\n    if fresh.empty:'
fallback = '''    # Yahoo may return only a few rows for BSE listings; use BSE history as fallback.
    if str(exchange or "").strip().upper() == "BSE" and len(fresh) < MIN_ROWS:
        bse_data, used_symbol, bse_error = _fetch_bse_history(ticker, security_code, start, today)
        if not bse_data.empty:
            fresh = pd.concat([fresh, bse_data], ignore_index=True, sort=False)
            fresh = fresh.dropna(subset=["Date"]).drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
            errors.append(f"BSE fallback used ({used_symbol})")
        elif bse_error:
            errors.append(f"BSE fallback unavailable: {bse_error}")

'''
m = replace_once(m, marker, fallback + marker, 'BSE fallback insertion')
old_call = '''            result = update(
                symbol,
                force_backfill=args.force_backfill,
            )'''
new_call = '''            result = update(
                symbol,
                force_backfill=args.force_backfill,
                exchange=row.get("exchange", ""),
                ticker=row.get("ticker", ""),
                security_code=row.get("security_code", ""),
            )'''
u = replace_once(u, old_call, new_call, 'updater arguments')
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
shutil.copy2(market, market.with_name(market.name + f'.before_bse_fallback_{stamp}.bak'))
shutil.copy2(updater, updater.with_name(updater.name + f'.before_bse_fallback_{stamp}.bak'))
market.write_text(m, encoding='utf-8')
updater.write_text(u, encoding='utf-8')
print('BSE fallback patch applied; backups saved beside both modified files.')
