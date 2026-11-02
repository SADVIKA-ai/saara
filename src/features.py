import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "stockai-features-v7-explainable-params"

def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    if 'Date' in clean_df.columns:
        clean_df['Date'] = pd.to_datetime(clean_df['Date'])
        clean_df = clean_df.set_index('Date')
    elif not isinstance(clean_df.index, pd.DatetimeIndex):
        clean_df.index = pd.to_datetime(clean_df.index)
        
    valid_mask = (
        (clean_df['High'] >= clean_df['Low']) &
        (clean_df['High'] >= clean_df['Open']) &
        (clean_df['High'] >= clean_df['Close']) &
        (clean_df['Low'] <= clean_df['Open']) &
        (clean_df['Low'] <= clean_df['Close'])
    )
    return clean_df[valid_mask].copy()

def generate_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Generates historical technical features, indicator setups, price-action resistance/support bounds,
    and target returns for arbitrary dynamic horizons.
    """
    if df.empty or len(df) < 15:
        return pd.DataFrame()
        
    df = validate_ohlc(df)
    df = df.sort_index()
    
    n_rows = len(df)
    win_20 = min(20, max(2, n_rows - 1))
    win_14 = min(14, max(2, n_rows - 1))
    win_50 = min(50, max(2, n_rows - 1))
    
    # 1. Exponential Moving Averages & Trend
    df['ema_9'] = df['Close'].ewm(span=min(9, n_rows), adjust=False).mean()
    df['ema_21'] = df['Close'].ewm(span=min(21, n_rows), adjust=False).mean()
    df['ema_ratio'] = df['ema_9'] / df['ema_21']
    
    # 2. Price Action: High/Low Rolling Bounds (Resistance & Support Proximity)
    df['high_20d'] = df['High'].rolling(window=win_20, min_periods=1).max()
    df['low_20d'] = df['Low'].rolling(window=win_20, min_periods=1).min()
    df['dist_to_resistance_pct'] = ((df['high_20d'] - df['Close']) / df['Close']) * 100
    df['dist_to_support_pct'] = ((df['Close'] - df['low_20d']) / df['Close']) * 100
    
    # 3. Pattern Recognition Proxies (Double Top Resistance Breakdown & Inverse H&S Reversals)
    # Double Top Proxy: Close rejected near 20d High and breaking below EMA 21
    df['double_top_bearish_breakout'] = np.where(
        (df['Close'] < df['ema_21']) & (df['high_20d'].shift(1) > df['ema_21'].shift(1)) & (df['dist_to_resistance_pct'] < 3.0),
        True, False
    )
    
    # Inverse Head & Shoulders Proxy: Close breaking above EMA 21 after bouncing off 20d Low
    df['inverse_hs_bullish_reversal'] = np.where(
        (df['Close'] > df['ema_21']) & (df['low_20d'].shift(1) < df['ema_21'].shift(1)) & (df['dist_to_support_pct'] < 3.0),
        True, False
    )

    # 4. Returns & Volatility
    df['return_1d'] = df['Close'].pct_change(periods=1, fill_method=None)
    df['return_5d'] = df['Close'].pct_change(periods=min(5, max(1, n_rows-1)), fill_method=None)
    df['rolling_std_20'] = df['return_1d'].rolling(window=win_20, min_periods=1).std()
    
    # 5. Volume Indicators
    df['vol_pct_change'] = df['Volume'].pct_change(periods=1, fill_method=None)
    df['vol_ma_20'] = df['Volume'].rolling(window=win_20, min_periods=1).mean()
    df['vol_ratio'] = np.where(df['vol_ma_20'] > 0, df['Volume'] / df['vol_ma_20'], 1.0)
    
    # 6. MACD Oscillator
    ema_12 = df['Close'].ewm(span=min(12, n_rows), adjust=False).mean()
    ema_26 = df['Close'].ewm(span=min(26, n_rows), adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=min(9, n_rows), adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 7. Relative Strength Index (RSI 14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=win_14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=win_14, min_periods=1).mean()
    rs = np.where(loss == 0, 100, gain / loss)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Target Return over dynamic horizon
    future_close = df['Close'].shift(-horizon)
    df['target_return'] = (future_close - df['Close']) / df['Close']
    
    return df
