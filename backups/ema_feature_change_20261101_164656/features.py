
import numpy as np
import pandas as pd


FEATURES = [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_ratio_5",
    "ma_ratio_20",
    "ma_ratio_50",
    "volatility_10",
    "volatility_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "volume_change",
    "volume_ratio",
    "range_pct",
    "body_pct",
]


def features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build technical features from historical OHLCV market data."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    d = raw.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]

    # Normalize common date-column names.
    if "date" not in d.columns:
        if "datetime" in d.columns:
            d = d.rename(columns={"datetime": "date"})
        elif "index" in d.columns:
            d = d.rename(columns={"index": "date"})
        else:
            d = d.reset_index()
            d.columns = [str(c).strip().lower() for c in d.columns]
            if "date" not in d.columns:
                if "datetime" in d.columns:
                    d = d.rename(columns={"datetime": "date"})
                else:
                    d = d.rename(columns={d.columns[0]: "date"})

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in d.columns]
    if missing:
        raise ValueError(
            f"Market data is missing required columns: {missing}"
        )

    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
    d["date"] = (
        d["date"]
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    for column in ["open", "high", "low", "close", "volume"]:
        d[column] = pd.to_numeric(d[column], errors="coerce")

    d = (
        d.dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    close = d["close"]
    volume = d["volume"]

    # Price returns.
    d["return_1d"] = close.pct_change(fill_method=None)
    d["return_5d"] = close.pct_change(periods=5, fill_method=None)
    d["return_10d"] = close.pct_change(periods=10, fill_method=None)

    # Moving-average ratios.
    ma_5 = close.rolling(window=5).mean()
    ma_20 = close.rolling(window=20).mean()
    ma_50 = close.rolling(window=50).mean()

    d["ma_ratio_5"] = close / ma_5 - 1
    d["ma_ratio_20"] = close / ma_20 - 1
    d["ma_ratio_50"] = close / ma_50 - 1

    # Rolling volatility.
    d["volatility_10"] = d["return_1d"].rolling(window=10).std()
    d["volatility_20"] = d["return_1d"].rolling(window=20).std()

    # RSI (14-period).
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.rolling(window=14).mean()
    average_loss = losses.rolling(window=14).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    d["rsi_14"] = 100 - (100 / (1 + relative_strength))

    # Handle periods with gains but no losses.
    d.loc[
        (average_loss == 0) & (average_gain > 0), "rsi_14"
    ] = 100

    # MACD (12-period EMA minus 26-period EMA).
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    d["macd"] = ema_12 - ema_26
    d["macd_signal"] = d["macd"].ewm(
        span=9, adjust=False
    ).mean()

    # Volume features.
    # Explicitly disable implicit forward filling to avoid the pandas warning.
    d["volume_change"] = volume.pct_change(fill_method=None)
    d["volume_ratio"] = volume / volume.rolling(window=20).mean()

    # Daily high-low range and open-close body.
    d["range_pct"] = (d["high"] - d["low"]) / close
    d["body_pct"] = (d["close"] - d["open"]) / d["open"].replace(
        0, np.nan
    )

    return d


def training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Create features and next-observation prediction targets."""
    d = features(raw)

    if d.empty:
        return d

    d["target_next_close"] = d["close"].shift(-1)
    d["target_next_return"] = (
        d["target_next_close"] / d["close"] - 1
    )

    required = FEATURES + ["target_next_close", "target_next_return"]
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=required).reset_index(drop=True)

    return d
