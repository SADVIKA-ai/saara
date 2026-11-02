import unittest

import numpy as np
import pandas as pd

from src.features import features, FEATURES
from src.predict import decision, predict


class CoreTests(unittest.TestCase):

    def test_feature_columns(self):
        rng = np.random.default_rng(3)
        n = 220
        close = 100 * np.cumprod(
            1 + rng.normal(0, 0.01, n)
        )

        data = pd.DataFrame({
            "Date": pd.bdate_range("2020-01-01", periods=n),
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(100, 10000, n),
        })

        result = features(data)

        self.assertTrue(set(FEATURES).issubset(result.columns))
        self.assertEqual(len(result), n)

    def test_signal_thresholds(self):
        self.assertEqual(decision(0.02), "BUY")
        self.assertEqual(decision(-0.02), "SELL")
        self.assertEqual(decision(0.0), "HOLD")

    def test_predict_rejects_insufficient_history(self):
        rng = np.random.default_rng(42)
        n = 60
        close = 100 * np.cumprod(
            1 + rng.normal(0, 0.01, n)
        )

        data = pd.DataFrame({
            "Date": pd.bdate_range("2024-01-01", periods=n),
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(100, 10000, n),
        })

        result = predict(data)

        self.assertNotEqual(result.get("status"), "OK")

    def test_prediction_validation_fields(self):
        rng = np.random.default_rng(123)
        n = 350

        close = 100 * np.cumprod(
            1 + rng.normal(0.0002, 0.01, n)
        )

        data = pd.DataFrame({
            "Date": pd.bdate_range("2023-01-01", periods=n),
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1000, 100000, n),
        })

        result = predict(data)

        self.assertEqual(result.get("status"), "OK")
        self.assertIn("mae_holdout", result)
        self.assertIn("rmse_holdout", result)
        self.assertIn("baseline_mae_holdout", result)
        self.assertIn("baseline_rmse_holdout", result)
        self.assertIn("model_beats_baseline_mae", result)
        self.assertIn("model_beats_baseline_rmse", result)
        self.assertEqual(
            result.get("forecast_horizon"),
            "NEXT_OBSERVATION",
        )

        self.assertGreaterEqual(result["mae_holdout"], 0)
        self.assertGreaterEqual(result["rmse_holdout"], 0)
        self.assertGreaterEqual(result["baseline_mae_holdout"], 0)
        self.assertGreaterEqual(result["baseline_rmse_holdout"], 0)

        self.assertEqual(
            result["model_beats_baseline_mae"],
            result["mae_holdout"] < result["baseline_mae_holdout"],
        )
        self.assertEqual(
            result["model_beats_baseline_rmse"],
            result["rmse_holdout"] < result["baseline_rmse_holdout"],
        )

    def test_ema_9_21_features(self):
        import numpy as np
        import pandas as pd
        from src.features import features

        close = pd.Series(
            np.linspace(100.0, 150.0, 160),
            dtype=float
        )
        raw = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=len(close), freq="D"),
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.arange(len(close), dtype=float) + 1000.0,
        })

        result = features(raw)

        self.assertIn("ema_9_ratio", result.columns)
        self.assertIn("ema_21_ratio", result.columns)

        # features() normalizes OHLCV column names to lowercase.
        expected_ema_9 = result["close"].ewm(
            span=9, adjust=False
        ).mean()
        expected_ema_21 = result["close"].ewm(
            span=21, adjust=False
        ).mean()

        np.testing.assert_allclose(
            result["ema_9_ratio"],
            result["close"] / expected_ema_9 - 1.0,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            result["ema_21_ratio"],
            result["close"] / expected_ema_21 - 1.0,
            equal_nan=True,
        )

    def test_training_frame_includes_ema_features(self):
        import numpy as np
        import pandas as pd
        from src.features import training_frame, FEATURES

        close = pd.Series(
            np.linspace(100.0, 180.0, 180),
            dtype=float
        )
        raw = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=len(close), freq="D"),
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.arange(len(close), dtype=float) + 1000.0,
        })

        frame = training_frame(raw)
        self.assertIn("ema_9_ratio", FEATURES)
        self.assertIn("ema_21_ratio", FEATURES)
        self.assertIn("ema_9_ratio", frame.columns)
        self.assertIn("ema_21_ratio", frame.columns)
        self.assertGreater(len(frame), 0)

if __name__ == "__main__":
    unittest.main()
