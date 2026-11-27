"""RMSE and MAE in the original rating units; lower values are better."""

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def _validated_arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Actual and predicted ratings must be one-dimensional arrays.")
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Actual and predicted ratings must be nonempty and equally sized.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted ratings must contain only finite numbers.")
    return actual, predicted


def rmse(actual, predicted) -> float:
    """Return root mean squared error without clipping or rounding predictions."""
    actual, predicted = _validated_arrays(actual, predicted)
    return float(root_mean_squared_error(actual, predicted))


def mae(actual, predicted) -> float:
    """Return mean absolute error without clipping or rounding predictions."""
    actual, predicted = _validated_arrays(actual, predicted)
    return float(mean_absolute_error(actual, predicted))


def evaluate_predictions(actual, predicted) -> dict[str, float]:
    """Evaluate predictions for the same rating rows, in the same order."""
    actual, predicted = _validated_arrays(actual, predicted)
    return {
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
    }
