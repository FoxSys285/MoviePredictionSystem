"""Common interface and parameter validation for experiment algorithms."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError


class MovieAlgorithm:
    """Accept rating rows for fitting and user/movie pairs for prediction.

    Each algorithm implements its own features or ID encoding. Predictions are
    returned on the original 0.5-5 star scale.
    """

    supports_early_stopping = False
    iteration_parameter: str | None = None

    def __init__(self, **parameters):
        self.parameters = parameters

    def configuration(self, defaults: dict) -> dict:
        """Resolve defaults and fail on misspelled/unsupported grid parameters."""
        supplied = dict(self.parameters)
        if "random_state" in supplied:
            if "seed" in supplied and supplied["seed"] != supplied["random_state"]:
                raise ValueError("seed and random_state must agree.")
            supplied["seed"] = supplied.pop("random_state")
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f"Unsupported {type(self).__name__} parameters: {sorted(unknown)}")
        return {**defaults, **supplied}

    def require_fitted(self, attribute: str) -> None:
        if not hasattr(self, attribute):
            raise NotFittedError(f"Fit {type(self).__name__} before predicting.")

    def emit_progress(self, message: str) -> None:
        callback = getattr(self, "report", None)
        if callback is not None:
            callback(message)

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame | None = None) -> Self:
        """Learn from training ratings and optional movie metadata."""
        raise NotImplementedError(f"Implement {type(self).__name__}.fit() first.")

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        """Return one rating per userId/movieId pair in the input row order."""
        raise NotImplementedError(f"Implement {type(self).__name__}.predict() first.")

    def fit_with_validation(
        self, ratings: pd.DataFrame, validation: pd.DataFrame,
        movies: pd.DataFrame | None = None,
    ) -> Self:
        """Early-stop using Validation and expose best_iterations_ as a count.

        Called only after Train-only CV. XGBoost/SVD/NCF implement this hook; final
        85% fitting receives the fixed count through iteration_parameter.
        """
        raise NotImplementedError(f"Implement {type(self).__name__}.fit_with_validation().")


def positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def finite_number(name, value, *, minimum=0.0, strict=False):
    if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} must be {'greater than' if strict else 'at least'} {minimum}.")
    return float(value)


def limit_blas_threads(count):
    """Use the fast Unicode DLL repair before enumerating BLAS thread pools."""
    from threadpoolctl import threadpool_limits

    from movie_rating.windows_dlls import enable_unicode_dll_paths

    enable_unicode_dll_paths()
    return threadpool_limits(limits=count, user_api="blas")
