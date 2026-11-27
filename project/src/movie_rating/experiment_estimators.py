"""Cloneable regression estimators for leakage-safe GridSearchCV."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from movie_rating.memory import feature_buffer
from movie_rating.models.factorization import SVD
from movie_rating.models.neural import NeuralCF
from movie_rating.models.regression import FeatureRegression, XGBoostRegression
from movie_rating.experiment_reduction_features import MovieFeatureRepresentation


def _training_frame(X, y):
    if not isinstance(X, pd.DataFrame) or not {"userId", "movieId"}.issubset(X.columns):
        raise ValueError("Movie estimators require a DataFrame with userId and movieId.")
    frame = X.copy()
    frame["rating"] = np.asarray(y, dtype=float)
    return frame


class _MovieFeatures:
    def _report(self, message):
        if self.report is not None:
            self.report(message)

    def _fit_tabular(self, frame, y, estimator):
        family = self.feature_family
        if self.reduction is not None:
            expected = {"linear": "pca"}.get(self.algorithm)
            if self.reduction != expected:
                raise ValueError("Movie feature PCA is supported for Linear Regression only.")
            family = "movie_features"
        self.feature_family_ = family
        if family not in {"rating_aggregates", "movie_features"}:
            raise ValueError(f"Unsupported feature family: {family}")
        if family == "movie_features" and self.algorithm != "linear":
            raise ValueError("Movie-table PCA comparisons are used by Linear Regression only.")
        # Fit all rating aggregates inside the current CV training fold.
        self.builder_ = FeatureRegression()
        self.builder_.set_movie_metadata(self.movies)
        self.builder_.fit_statistics(frame)
        steps = []
        if self.algorithm == "linear":
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        self.model_ = Pipeline(steps)
        dimensions = self.builder_.feature_count
        if family == "movie_features":
            self.representation_ = MovieFeatureRepresentation().fit(
                frame, self.movies, components=self.components if self.reduction else None,
            )
            dimensions = 11 + (self.components if self.reduction else
                               len(self.representation_.feature_names_))
        with feature_buffer(len(frame), dimensions,
                            self._feature_batches(frame, training=True)) as features:
            # Leave the current row's rating out of its own aggregate features.
            self.model_.fit(features, y)

    def _feature_batches(self, frame, training=False):
        offset = 0
        for values in self.builder_.iter_features(frame, training=training):
        if self.feature_family_ == "movie_features":
                movie_features = self.representation_.transform(frame.iloc[offset:offset + len(values)])
                # Keep user/item aggregate controls identical across the PCA pair.
                values = np.column_stack((values[:, :11], movie_features)).astype(np.float32)
            offset += len(values)
            yield values

    def _tabular_predict(self, X):
        check_is_fitted(self, "model_")
        output = []
        for features in self._feature_batches(X, training=False):
            output.append(self.model_.predict(features))
        if not output:
            return np.empty(0)
        return np.concatenate(output)


class MovieRegressor(RegressorMixin, _MovieFeatures, BaseEstimator):
    """Adapt the rating models to sklearn's fit(X, y) API."""

    def __init__(self, algorithm="linear", movies=None, parameters=None, reduction=None,
                 components=8, seed=42, report=None, feature_family="rating_aggregates"):
        self.algorithm = algorithm
        self.movies = movies
        self.parameters = parameters
        self.reduction = reduction
        self.components = components
        self.seed = seed
        self.report = report
        self.feature_family = feature_family

    def fit(self, X, y):
        frame = _training_frame(X, y)
        params = dict(self.parameters or {})
        self.n_features_in_ = X.shape[1]
        if hasattr(self.report, "start_fit"):
            self.report.start_fit(len(frame))
        self._report(f"{self.algorithm}: fitting {len(frame):,} training ratings; params={params}")
        if self.algorithm in {"svd", "ncf"}:
            if self.reduction is not None:
                raise ValueError("SVD/NCF use native user/movie factors, not PCA on identifiers.")
            constructor = SVD if self.algorithm == "svd" else NeuralCF
            self.model_ = constructor(seed=self.seed, **params)
            self.model_.fit(frame, report=self.report,
                            progress=lambda done, total: self._report(
                                f"{self.algorithm}: epoch {done}/{total}"))
        elif self.algorithm == "xgboost" and self.reduction is None:
            # Retain the project's bounded feature/external-memory implementation.
            self.model_ = XGBoostRegression(seed=self.seed, **params)
            self.model_.set_movie_metadata(self.movies)
            self.model_.fit(frame, report=self.report)
        elif self.algorithm in {"linear", "xgboost"}:
            if self.algorithm == "linear":
                estimator = LinearRegression(**params)
            else:
                from xgboost import XGBRegressor

                estimator = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                         random_state=self.seed, **params)
            self._fit_tabular(frame, np.asarray(y), estimator)
        else:
            raise ValueError(f"Unsupported regression algorithm: {self.algorithm}")
        self._report(f"{self.algorithm}: fit completed")
        if hasattr(self.report, "finish_fit"):
            self.report.finish_fit()
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        if hasattr(self, "builder_"):
            values = self._tabular_predict(X)
        else:
            values = self.model_.predict(X.userId.to_numpy(), X.movieId.to_numpy())
        return np.clip(values, 0.5, 5.0)
