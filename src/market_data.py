import os
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime

logger = logging.getLogger(__name__)

def update_market_data_file(symbol: str) -> pd.DataFrame:
    """
    Automated Daily Market Refresher:
    Fetches latest available trading data up to today from Yahoo Finance
    and merges it cleanly into local raw storage.
    """
    if not symbol or pd.isna(symbol):
        return pd.DataFrame()

    symbol_str = str(symbol).strip()
    raw_path = os.path.join("data", "raw", f"{symbol_str}.csv")
    
    existing_df = pd.DataFrame()
    if os.path.exists(raw_path):
        try:
            existing_df = pd.read_csv(raw_path)
            if 'Date' in existing_df.columns:
                existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                existing_df = existing_df.set_index('Date')
            elif not isinstance(existing_df.index, pd.DatetimeIndex):
                existing_df.index = pd.to_datetime(existing_df.index)
        except Exception as e:
            logger.warning(f"Could not read existing raw CSV for {symbol_str}: {e}")

    # Automated live update fetch
    try:
        ticker_obj = yf.Ticker(symbol_str)
        # Fetch recent 30 days to catch missing trading sessions up to today
        df_new = ticker_obj.history(period="1mo")
        if not df_new.empty:
            df_new = df_new[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            if df_new.index.tz is not None:
                df_new.index = df_new.index.tz_localize(None)
            df_new.index = pd.to_datetime(df_new.index)
            
            if not existing_df.empty:
                # Merge and drop duplicate dates, keeping latest updated prices
                combined = pd.concat([existing_df, df_new])
                combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            else:
                combined = df_new.sort_index()
                
            # Save updated data back to disk automatically
            os.makedirs(os.path.join("data", "raw"), exist_ok=True)
            combined.to_csv(raw_path)
            return combined
    except Exception as e:
        logger.warning(f"Live market refresh for {symbol_str} failed: {e}. Falling back to cached history.")

    return existing_df

def get_market_data(symbol: str, refresh: bool = True) -> pd.DataFrame:
    """Loads historical market OHLCV data, refreshing automatically if requested."""
    if refresh:
        df = update_market_data_file(symbol)
        if not df.empty:
            return df

    symbol_str = str(symbol).strip()
    raw_path = os.path.join("data", "raw", f"{symbol_str}.csv")
    
    if os.path.exists(raw_path):
        try:
            df = pd.read_csv(raw_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            return df
        except Exception:
            pass

    return pd.DataFrame()

load_market_data = get_market_data
get_history = get_market_data
market_data = get_market_data
