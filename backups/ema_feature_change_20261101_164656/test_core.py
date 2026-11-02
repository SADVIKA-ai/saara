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


if __name__ == "__main__":
    unittest.main()
