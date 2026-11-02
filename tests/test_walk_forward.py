
import unittest

import numpy as np
import pandas as pd

from src.walk_forward import walk_forward_validate


class WalkForwardTests(unittest.TestCase):

    @staticmethod
    def sample_history(n=350):
        rng = np.random.default_rng(123)

        close = 100 * np.cumprod(
            1 + rng.normal(0.0002, 0.01, n)
        )

        return pd.DataFrame({
            "Date": pd.bdate_range("2023-01-01", periods=n),
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1000, 100000, n),
        })

    def test_walk_forward_returns_five_folds(self):
        result = walk_forward_validate(
            self.sample_history(),
            folds=5,
            min_test_rows=10,
        )

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["folds_completed"], 5)
        self.assertEqual(result["validation_method"], "EXPANDING_WINDOW")
        self.assertEqual(len(result["folds"]), 5)
        self.assertGreater(result["evaluated_rows"], 0)

        for key in (
            "model_mae",
            "model_rmse",
            "baseline_mae",
            "baseline_rmse",
        ):
            self.assertTrue(np.isfinite(result[key]))
            self.assertGreaterEqual(result[key], 0)

    def test_walk_forward_rejects_short_history(self):
        result = walk_forward_validate(
            self.sample_history(150),
            folds=5,
            min_test_rows=10,
        )

        self.assertEqual(result["status"], "INSUFFICIENT_HISTORY")


if __name__ == "__main__":
    unittest.main()
