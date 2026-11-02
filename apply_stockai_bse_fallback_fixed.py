from pathlib import Path
import shutil
from datetime import datetime

root = Path(__file__).resolve().parent
market = root / 'src' / 'market_data.py'
updater = root / 'scripts' / 'update_market_data.py'

def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'Safe patch stopped at {label}: expected 1 exact match, found {count}. No files written.')
    return text.replace(old, new, 1)

if not market.exists() or not updater.exists():
    raise SystemExit('Place this script in the project root beside src/ and scripts/, then run it there.')
m = market.read_text(encoding='utf-8')
u = updater.read_text(encoding='utf-8')
if 'def _fetch_bse_history(' in m:
    raise SystemExit('BSE fallback already exists; no changes made.')
if '\nimport re\n' not in m:
    m = replace_once(m, 'import time\n', 'import time\nimport re\n', 'import re')

helpers = '''\n\ndef _normalize_bse_history(data):
    """Normalize BSE-wrapper history to this project's OHLCV schema."""
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    d = data.copy()
    def key(value):
        return re.sub(r"[^a-z0-9]", "", str(value).lower())
    aliases = {
        "Date": {"date", "tradedate", "tradingdate", "datetime"},
        "Open": {"open", "openprice"}, "High": {"high", "highprice"},
        "Low": {"low", "lowprice"}, "Close": {"close", "closeprice", "last", "lastprice"},
        "Volume": {"volume", "noofshares", "noofshrs", "tradedquantity", "totaltradedquantity", "quantity", "qty", "trdqty"},
    }
    cols = {key(c): c for c in d.columns}
    rename = {}
    for target, names in aliases.items():
        for name in names:
            if name in cols:
                rename[cols[name]] = target
                break
    d = d.rename(columns=rename)
    if "Date" not in d.columns:
        d = d.reset_index()
        cols = {key(c): c for c in d.columns}
        for name in ("date", "tradedate", "tradingdate", "datetime"):
            if name in cols:
                d = d.rename(columns={cols[name]: "Date"})
                break
    if not {"Date", "Open", "High", "Low", "Close"}.issubset(d.columns):
        return pd.DataFrame()
    if "Volume" not in d.columns:
        d["Volume"] = 0
    d["Date"] = trading_dates(d["Date"])
    for col in ("Open", "High", "Low", "Close", "Volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    d = d[(d[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    return d.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)


def _fetch_bse_history(ticker, security_code, start, end):
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
