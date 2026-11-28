"""Global Mean reference: one constant learned only from the fitting ratings."""

import numpy as np
import pandas as pd


class GlobalMeanBaseline:
    def fit(self, ratings: pd.DataFrame):
        if "rating" not in ratings or ratings.empty:
            raise ValueError("Global Mean needs nonempty training ratings.")
        values = ratings.rating.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Global Mean needs finite training ratings.")
        self.mean_ = float(values.mean())
        return self

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise ValueError("Fit Global Mean on Train before predicting.")
        return np.full(len(pairs), self.mean_, dtype=np.float64)
