"""Cloneable movie estimators for leakage-safe GridSearchCV across all six models."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from movie_rating.memory import feature_buffer
from movie_rating.models.factorization import SVD
from movie_rating.models.neural import NeuralCF
from movie_rating.models.regression import FeatureRegression, XGBoostRegression


def _training_frame(X, y, regression):
    if not isinstance(X, pd.DataFrame) or not {"userId", "movieId"}.issubset(X.columns):
        raise ValueError("Movie estimators require a DataFrame with userId and movieId.")
    frame = X.copy()
    if regression:
        frame["rating"] = np.asarray(y, dtype=float)
    elif "rating" not in frame:
        raise ValueError("Classification feature fitting requires original training ratings.")
    return frame


class InteractionReduction(TransformerMixin, BaseEstimator):
    """Reduce preference/genre interaction columns while retaining base features."""

    def __init__(self, method="pca", components=8, base_columns=11, seed=42):
        self.method = method
        self.components = components
        self.base_columns = base_columns
        self.seed = seed

    def fit(self, X, y=None):
        self.reducer_ = (PCA(n_components=self.components, svd_solver="full")
                         if self.method == "pca" else
                         TruncatedSVD(n_components=self.components, random_state=self.seed))
        self.reducer_.fit(X[:, self.base_columns:])
        return self

    def transform(self, X):
        return np.column_stack((X[:, :self.base_columns],
                                self.reducer_.transform(X[:, self.base_columns:])))


class _MovieFeatures:
    def _report(self, message):
        if self.report is not None:
            self.report(message)

    def _fit_tabular(self, frame, y, estimator):
        # Fit all rating aggregates inside the current CV training fold.
        self.builder_ = FeatureRegression()
        self.builder_.set_movie_metadata(self.movies)
        self.builder_.fit_statistics(frame)
        steps = []
        if self.algorithm in {"linear", "logistic"}:
            steps.append(("scaler", StandardScaler()))
        if self.reduction is not None:
            steps.append(("reducer", InteractionReduction(
                method=self.reduction, components=self.components,
                base_columns=11 + len(self.builder_.genre_names), seed=self.seed,
            )))
        steps.append(("model", estimator))
        self.model_ = Pipeline(steps)
        with feature_buffer(len(frame), self.builder_.feature_count,
                            self.builder_.iter_features(frame, training=True)) as features:
            # Leave the current row's rating out of its own aggregate features.
            self.model_.fit(features, y)

    def _tabular_predict(self, X, probabilities=False):
        check_is_fitted(self, "model_")
        output = []
        method = self.model_.predict_proba if probabilities else self.model_.predict
        for features in self.builder_.iter_features(X, training=False):
            output.append(method(features))
        if not output:
            return np.empty((0, len(self.classes_))) if probabilities else np.empty(0)
        return np.concatenate(output)


class MovieRegressor(RegressorMixin, _MovieFeatures, BaseEstimator):
    """Adapt linear, XGBoost, SVD, and NCF to sklearn's fit(X, y) API."""

    def __init__(self, algorithm="linear", movies=None, parameters=None, reduction=None,
                 components=8, seed=42, report=None):
        self.algorithm = algorithm
        self.movies = movies
        self.parameters = parameters
        self.reduction = reduction
        self.components = components
        self.seed = seed
        self.report = report

    def fit(self, X, y):
        frame = _training_frame(X, y, regression=True)
        params = dict(self.parameters or {})
        self.n_features_in_ = X.shape[1]
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
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        if hasattr(self, "builder_"):
            values = self._tabular_predict(X)
        else:
            values = self.model_.predict(X.userId.to_numpy(), X.movieId.to_numpy())
        return np.clip(values, 0.5, 5.0)


class MovieClassifier(ClassifierMixin, _MovieFeatures, BaseEstimator):
    """Adapt logistic regression and random forest; only logistic is scaled."""

    def __init__(self, algorithm="logistic", movies=None, parameters=None, reduction=None,
                 components=8, seed=42, report=None):
        self.algorithm = algorithm
        self.movies = movies
        self.parameters = parameters
        self.reduction = reduction
        self.components = components
        self.seed = seed
        self.report = report

    def fit(self, X, y):
        frame = _training_frame(X, y, regression=False)
        params = dict(self.parameters or {})
        self.n_features_in_ = X.shape[1]
        self._report(f"{self.algorithm}: fitting {len(frame):,} training ratings; params={params}")
        if self.algorithm == "logistic":
            estimator = LogisticRegression(random_state=self.seed, **params)
        elif self.algorithm == "random_forest":
            # Integer bootstrap limits must not exceed the current fold size.
            if isinstance(params.get("max_samples"), int):
                params["max_samples"] = min(params["max_samples"], len(frame))
            estimator = RandomForestClassifier(random_state=self.seed, **params)
        else:
            raise ValueError(f"Unsupported classification algorithm: {self.algorithm}")
        self._fit_tabular(frame, np.asarray(y), estimator)
        self.classes_ = self.model_.classes_
        self._report(f"{self.algorithm}: fit completed")
        return self

    def predict(self, X):
        return self._tabular_predict(X)

    def predict_proba(self, X):
        return self._tabular_predict(X, probabilities=True)
