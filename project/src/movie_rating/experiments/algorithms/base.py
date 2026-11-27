"""Common interface to implement in each baseline.py and improved.py."""

from typing import Self

import numpy as np
import pandas as pd


class MovieAlgorithm:
    """Accept rating rows for fitting and user/movie pairs for prediction.

    Each algorithm implements its own features or ID encoding. Algorithm files
    currently inherit these explicit placeholders until their implementation step.
    """

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame | None = None) -> Self:
        """Learn from training ratings and optional movie metadata."""
        raise NotImplementedError(f"Implement {type(self).__name__}.fit() first.")

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        """Return one rating per userId/movieId pair in the input row order."""
        raise NotImplementedError(f"Implement {type(self).__name__}.predict() first.")
