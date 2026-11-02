
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import MIN_ROWS
from .features import FEATURES, training_frame
from .predict import _make_model, _directional_accuracy


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def walk_forward_validate(raw, folds=5, min_test_rows=10):
    """Evaluate next-observation predictions with expanding training windows."""

    if folds < 2 or min_test_rows < 1:
        return {
            "status": "INVALID_SETTINGS",
            "message": "Use at least 2 folds and 1 test row per fold.",
        }

    if raw is None or raw.empty:
        return {
            "status": "NO_PRICE_HISTORY",
            "message": "No historical market data is available.",
        }

    frame = training_frame(raw)

    if frame.empty:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "message": "No usable feature and target rows are available.",
        }

    X = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    returns = pd.to_numeric(frame["target_next_return"], errors="coerce")
    current_close = pd.to_numeric(frame["close"], errors="coerce")
    target_close = pd.to_numeric(frame["target_next_close"], errors="coerce")

    valid = (
        returns.notna()
        & current_close.notna()
        & target_close.notna()
        & np.isfinite(returns)
        & np.isfinite(current_close)
        & np.isfinite(target_close)
        & (current_close > 0)
        & (target_close > 0)
        & X.notna().all(axis=1)
    )

    X = X.loc[valid].reset_index(drop=True)
    returns = returns.loc[valid].reset_index(drop=True)
    current_close = current_close.loc[valid].reset_index(drop=True)
    target_close = target_close.loc[valid].reset_index(drop=True)

    n = len(X)
    initial_train_rows = max(MIN_ROWS, n // 2)

    if (
        initial_train_rows >= n
        or n - initial_train_rows < folds * min_test_rows
    ):
        return {
            "status": "INSUFFICIENT_HISTORY",
            "message": (
                f"Not enough usable rows: {n}. Need at least {MIN_ROWS} "
                "training rows and enough observations for every test fold."
            ),
            "usable_rows": n,
            "folds_requested": folds,
            "minimum_test_rows_per_fold": min_test_rows,
        }

    boundaries = np.linspace(
        initial_train_rows, n, folds + 1, dtype=int
    )

    actual_all = []
    predicted_all = []
    baseline_all = []
    actual_returns_all = []
    predicted_returns_all = []
    fold_results = []

    for fold in range(folds):
        start = int(boundaries[fold])
        end = int(boundaries[fold + 1])

        if end - start < min_test_rows:
            return {
                "status": "INSUFFICIENT_HISTORY",
                "message": f"Fold {fold + 1} has too few test rows.",
            }

        model = _make_model()
        model.fit(X.iloc[:start], returns.iloc[:start])

        predicted_returns = np.asarray(
            model.predict(X.iloc[start:end]), dtype=float
        )

        if not np.isfinite(predicted_returns).all():
            return {
                "status": "INVALID_FORECAST",
                "message": f"Non-finite predictions in fold {fold + 1}.",
            }

        current = current_close.iloc[start:end].to_numpy(dtype=float)
        actual = target_close.iloc[start:end].to_numpy(dtype=float)
        predicted = current * (1.0 + predicted_returns)
        baseline = current.copy()

        if not np.isfinite(predicted).all() or (predicted <= 0).any():
            return {
                "status": "INVALID_FORECAST",
                "message": f"Invalid predicted prices in fold {fold + 1}.",
            }

        model_metrics = _metrics(actual, predicted)
        baseline_metrics = _metrics(actual, baseline)

        fold_results.append({
            "fold": fold + 1,
            "train_rows": start,
            "test_rows": end - start,
            "model_mae": model_metrics["mae"],
            "model_rmse": model_metrics["rmse"],
            "baseline_mae": baseline_metrics["mae"],
            "baseline_rmse": baseline_metrics["rmse"],
            "model_beats_baseline_mae": (
                model_metrics["mae"] < baseline_metrics["mae"]
            ),
            "model_beats_baseline_rmse": (
                model_metrics["rmse"] < baseline_metrics["rmse"]
            ),
        })

        actual_all.extend(actual)
        predicted_all.extend(predicted)
        baseline_all.extend(baseline)
        actual_returns_all.extend(
            returns.iloc[start:end].to_numpy(dtype=float)
        )
        predicted_returns_all.extend(predicted_returns)

    model_metrics = _metrics(actual_all, predicted_all)
    baseline_metrics = _metrics(actual_all, baseline_all)

    return {
        "status": "OK",
        "forecast_horizon": "NEXT_OBSERVATION",
        "validation_method": "EXPANDING_WINDOW",
        "folds_completed": len(fold_results),
        "usable_rows": n,
        "initial_training_rows": initial_train_rows,
        "evaluated_rows": len(actual_all),
        "model_mae": model_metrics["mae"],
        "model_rmse": model_metrics["rmse"],
        "baseline_name": "LAST_CLOSE",
        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],
        "model_beats_baseline_mae": (
            model_metrics["mae"] < baseline_metrics["mae"]
        ),
        "model_beats_baseline_rmse": (
            model_metrics["rmse"] < baseline_metrics["rmse"]
        ),
        "directional_accuracy_pct": _directional_accuracy(
            actual_returns_all, predicted_returns_all
        ),
        "folds": fold_results,
        "warning": (
            "Historical validation does not guarantee future performance. "
            "This evaluates next-observation predictions only, not "
            "one-month or multi-year forecasts. Not financial advice."
        ),
    }
