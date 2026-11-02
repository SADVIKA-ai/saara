
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import FEATURES, training_frame, features
from .config import (
    MIN_ROWS,
    TEST_FRACTION,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
)


MODEL_NAME = "HistGradientBoostingRegressor"
FEATURE_SCHEMA_VERSION = "stockai-features-v1"


def decision(expected_return):
    """Convert a decimal expected return into a configured signal."""
    if expected_return >= BUY_THRESHOLD:
        return "BUY"
    if expected_return <= SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def history_metadata(raw):
    """Report observed history coverage."""
    empty = {
        "raw_history_rows": 0,
        "history_start_date": "",
        "raw_history_latest_date": "",
    }

    if raw is None or raw.empty:
        return empty

    date_column = next(
        (
            c for c in ("Date", "Datetime", "date", "datetime")
            if c in raw.columns
        ),
        None,
    )

    if date_column is None:
        return {**empty, "raw_history_rows": len(raw)}

    dates = pd.to_datetime(
        raw[date_column], errors="coerce", utc=True
    ).dropna()

    if dates.empty:
        return {**empty, "raw_history_rows": len(raw)}

    dates = (
        dates.dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    return {
        "raw_history_rows": len(raw),
        "history_start_date": str(dates.min().date()),
        "raw_history_latest_date": str(dates.max().date()),
    }


def _make_model():
    """Build a fresh model pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            HistGradientBoostingRegressor(
                max_iter=160,
                learning_rate=0.06,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ])


def _price_metrics(actual, predicted):
    """Calculate errors in price units and relative to mean actual price."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mean_actual = float(np.mean(actual))

    return {
        "mae": mae,
        "rmse": rmse,
        "mae_pct_of_mean_actual": (
            mae / abs(mean_actual) * 100
            if np.isfinite(mean_actual) and mean_actual != 0
            else None
        ),
    }


def _directional_accuracy(actual_returns, predicted_returns):
    """Percentage of correct up/down calls, excluding flat actual returns."""
    actual_returns = np.asarray(actual_returns, dtype=float)
    predicted_returns = np.asarray(predicted_returns, dtype=float)

    valid = (
        np.isfinite(actual_returns)
        & np.isfinite(predicted_returns)
        & (actual_returns != 0)
    )

    if not np.any(valid):
        return None

    return float(
        np.mean(
            np.sign(actual_returns[valid])
            == np.sign(predicted_returns[valid])
        ) * 100
    )


def predict(raw, sentiment_score=0.0):
    """
    Predict the next usable observation's close from predicted return.

    This function does not generate one-month or multi-year targets.
    """
    metadata = history_metadata(raw)

    if raw is None or raw.empty:
        return {
            "status": "NO_PRICE_HISTORY",
            "message": "No historical market data is available.",
            **metadata,
        }

    f = training_frame(raw)

    if f.empty:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "message": "No usable feature and target rows are available.",
            "training_rows": 0,
            **metadata,
        }

    X = f[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(f["target_next_return"], errors="coerce")
    current_close = pd.to_numeric(f["close"], errors="coerce")
    target_close = pd.to_numeric(f["target_next_close"], errors="coerce")

    valid = (
        y.notna()
        & current_close.notna()
        & target_close.notna()
        & np.isfinite(y)
        & np.isfinite(current_close)
        & np.isfinite(target_close)
        & (current_close > 0)
        & (target_close > 0)
    )

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    current_close = current_close.loc[valid].reset_index(drop=True)
    target_close = target_close.loc[valid].reset_index(drop=True)

    if len(X) < MIN_ROWS:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "message": (
                f"Need at least {MIN_ROWS} usable training rows; "
                f"got {len(X)}."
            ),
            "training_rows": len(X),
            "minimum_training_rows": MIN_ROWS,
            **metadata,
        }

    split = int(len(X) * (1 - TEST_FRACTION))
    split = max(split, MIN_ROWS)

    if split >= len(X):
        split = len(X) - 1

    if split < 60 or len(X) - split < 1:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "message": "Not enough observations for chronological validation.",
            "training_rows": len(X),
            **metadata,
        }

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = _make_model()
    model.fit(X_train, y_train)

    test_returns = model.predict(X_test)

    if not np.isfinite(test_returns).all():
        return {
            "status": "INVALID_FORECAST",
            "message": "Validation produced invalid return predictions.",
            "training_rows": len(X),
            **metadata,
        }

    test_current_close = current_close.iloc[split:].to_numpy(dtype=float)
    test_actual_close = target_close.iloc[split:].to_numpy(dtype=float)

    # Convert each predicted return into a predicted closing price.
    test_predicted_close = test_current_close * (1 + test_returns)

    # Naive baseline: next close equals the current close.
    baseline_predicted_close = test_current_close.copy()

    model_metrics = _price_metrics(
        test_actual_close, test_predicted_close
    )
    baseline_metrics = _price_metrics(
        test_actual_close, baseline_predicted_close
    )

    actual_returns = y_test.to_numpy(dtype=float)

    directional_accuracy = _directional_accuracy(
        actual_returns, test_returns
    )

    # Refit on all labelled observations for the latest estimate.
    model.fit(X, y)

    latest_features = features(raw)
    if latest_features.empty:
        return {
            "status": "INSUFFICIENT_FEATURES",
            "message": "No technical features could be calculated.",
            "training_rows": len(X),
            **metadata,
        }

    latest_features = latest_features.replace(
        [np.inf, -np.inf], np.nan
    )
    latest_features = latest_features.dropna(
        subset=FEATURES + ["close", "date"]
    )

    if latest_features.empty:
        return {
            "status": "INSUFFICIENT_FEATURES",
            "message": "The latest row lacks required features.",
            "training_rows": len(X),
            **metadata,
        }

    latest = latest_features.iloc[-1]
    actual = float(latest["close"])

    if not np.isfinite(actual) or actual <= 0:
        return {
            "status": "INVALID_LATEST_CLOSE",
            "message": "Latest close is missing or non-positive.",
            "training_rows": len(X),
            **metadata,
        }

    model_return = float(
        model.predict(latest_features[FEATURES].iloc[[-1]])[0]
    )

    if not np.isfinite(model_return):
        return {
            "status": "INVALID_FORECAST",
            "message": "The model returned an invalid predicted return.",
            "training_rows": len(X),
            **metadata,
        }

    forecast = actual * (1 + model_return)

    if not np.isfinite(forecast) or forecast <= 0:
        return {
            "status": "INVALID_FORECAST",
            "message": (
                "Predicted return implies a non-positive or invalid price."
            ),
            "predicted_return": model_return,
            "training_rows": len(X),
            **metadata,
        }

    try:
        score = float(sentiment_score)
    except (TypeError, ValueError):
        score = 0.0

    if not np.isfinite(score):
        score = 0.0

    adjusted_return = model_return + 0.0025 * score

    market_date = pd.to_datetime(latest["date"], errors="coerce")
    if pd.isna(market_date):
        return {
            "status": "INVALID_MARKET_DATE",
            "message": "Latest feature row has an invalid date.",
            "training_rows": len(X),
            **metadata,
        }

    return {
        "status": "OK",
        "latest_actual_close": actual,
        "latest_market_date": str(market_date.date()),
        "predicted_next_close": forecast,
        "predicted_change": forecast - actual,
        "model_expected_return_pct": model_return * 100,
        "sentiment_score": score,
        "sentiment_adjusted_return_pct": adjusted_return * 100,
        "decision": decision(adjusted_return),

        "mae_holdout": model_metrics["mae"],
        "rmse_holdout": model_metrics["rmse"],
        "mae_holdout_pct_of_mean_actual": (
            model_metrics["mae_pct_of_mean_actual"]
        ),
        "directional_accuracy_pct": directional_accuracy,

        "baseline_name": "LAST_CLOSE",
        "baseline_mae_holdout": baseline_metrics["mae"],
        "baseline_rmse_holdout": baseline_metrics["rmse"],
        "baseline_mae_pct_of_mean_actual": (
            baseline_metrics["mae_pct_of_mean_actual"]
        ),
        "model_beats_baseline_mae": (
            model_metrics["mae"] < baseline_metrics["mae"]
        ),
        "model_beats_baseline_rmse": (
            model_metrics["rmse"] < baseline_metrics["rmse"]
        ),

        "holdout_rows": len(X_test),
        "training_rows": len(X),
        "model": MODEL_NAME,
        "forecast_horizon": "NEXT_OBSERVATION",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "prediction_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "warning": (
            "Research estimate for the next usable observation only. "
            "The model is not validated for long-term targets. Compare "
            "holdout errors with the last-close baseline. Validation "
            "results do not guarantee future accuracy. Sentiment is a "
            "heuristic overlay, not a learned causal effect. "
            "Not financial advice."
        ),
        **metadata,
    }
