"""Regression metric evaluation."""

import numpy as np
import pytest

from movie_rating.experiment_supervised import evaluate_model



def test_regression_clips_predictions_before_scoring():
    class FixedRegressor:
        def predict(self, X):
            return np.array([-1.0, 6.0, 3.0])

    result = evaluate_model(FixedRegressor(), np.zeros((3, 1)), [0.5, 5.0, 2.0])
    np.testing.assert_array_equal(result["predictions"], [0.5, 5.0, 3.0])
    assert result["metrics"]["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert result["metrics"]["mae"] == pytest.approx(1 / 3)
