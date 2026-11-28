"""StandardScaler followed by ordinary least-squares Linear Regression."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from movie_rating.experiments.algorithms.base import (
    MovieAlgorithm,
    limit_blas_threads,
    positive_integer,
)
from movie_rating.experiments.features import TabularFeatures
from movie_rating.memory import feature_buffer


class LinearRegressionBaseline(MovieAlgorithm):
    """Fit smoothed history features; IDs are never numeric regressors."""

    rich_features = False
    defaults = {"fit_intercept": True, "smoothing": 10.0, "n_jobs": 4, "seed": 42}

    def _estimator(self, options):
        return LinearRegression(fit_intercept=options["fit_intercept"], copy_X=False,
                                n_jobs=options["n_jobs"])

    def fit(self, ratings, movies=None):
        self.options_ = self.configuration(self.defaults)
        positive_integer("n_jobs", self.options_["n_jobs"])
        self.features_ = TabularFeatures(
            rich=self.rich_features, smoothing=self.options_["smoothing"]).fit(ratings, movies)
        self.scaler_ = StandardScaler(copy=False)
        self.estimator_ = self._estimator(self.options_)
        with feature_buffer(len(ratings), len(self.features_.feature_names_),
                            self.features_.iter_batches(ratings, training=True)) as values:
            with limit_blas_threads(self.options_["n_jobs"]):
                scaled = self.scaler_.fit_transform(values)
                self.estimator_.fit(scaled, ratings.rating.to_numpy())
        return self

    def predict(self, pairs):
        self.require_fitted("estimator_")
        predictions = []
        with limit_blas_threads(self.options_["n_jobs"]):
            for values in self.features_.iter_batches(pairs):
                predictions.append(self.estimator_.predict(self.scaler_.transform(values)))
        output = np.concatenate(predictions) if predictions else np.empty(0)
        return np.clip(output, 0.5, 5.0)
