"""Histogram XGBoost on Train-only history features, without StandardScaler."""

from contextlib import ExitStack

import numpy as np

from movie_rating.experiments.algorithms.base import (
    MovieAlgorithm,
    positive_integer,
)
from movie_rating.experiments.data import validate_ratings
from movie_rating.experiments.features import TabularFeatures
from movie_rating.memory import feature_buffer


def _rating_rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - np.clip(predicted, 0.5, 5.0)) ** 2)))


class XGBoostBaseline(MovieAlgorithm):
    """Refit from scratch for outer Validation early stopping."""

    supports_early_stopping = True
    iteration_parameter = "n_estimators"
    rich_features = False
    defaults = {
        "n_estimators": 150, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0,
        "reg_alpha": 0.0, "min_child_weight": 1.0, "max_bin": 256,
        "n_jobs": 4, "seed": 42, "patience": 10, "smoothing": 10.0,
    }

    def fit(self, ratings, movies=None):
        return self._fit(ratings, movies)

    def fit_with_validation(self, ratings, validation, movies=None):
        validate_ratings(validation)
        return self._fit(ratings, movies, validation)

    def _fit(self, ratings, movies, validation=None):
        from xgboost import XGBRegressor
        from xgboost.callback import EarlyStopping

        options = self.configuration(self.defaults)
        positive_integer("n_estimators", options["n_estimators"])
        positive_integer("patience", options["patience"])
        positive_integer("n_jobs", options["n_jobs"])
        self.features_ = TabularFeatures(
            rich=self.rich_features, smoothing=options["smoothing"]).fit(ratings, movies)
        model_params = {key: value for key, value in options.items()
                        if key not in {"seed", "smoothing", "patience"}}
        callbacks = ([EarlyStopping(rounds=options["patience"], save_best=True)]
                     if validation is not None else None)
        self.estimator_ = XGBRegressor(
            **model_params, random_state=options["seed"], objective="reg:squarederror",
            tree_method="hist", eval_metric=_rating_rmse, callbacks=callbacks,
        )
        with ExitStack() as stack:
            features = stack.enter_context(feature_buffer(
                len(ratings), len(self.features_.feature_names_),
                self.features_.iter_batches(ratings, training=True)))
            eval_set = None
            if validation is not None:
                validation_features = stack.enter_context(feature_buffer(
                    len(validation), len(self.features_.feature_names_),
                    self.features_.iter_batches(validation)))
                eval_set = [(validation_features, validation.rating.to_numpy())]
            self.estimator_.fit(features, ratings.rating.to_numpy(),
                                eval_set=eval_set, verbose=False)
        self.best_iterations_ = (int(self.estimator_.best_iteration) + 1
                                 if validation is not None else options["n_estimators"])
        self.learning_curve_ = []
        if validation is not None:
            history = self.estimator_.evals_result()["validation_0"]["_rating_rmse"]
            self.learning_curve_ = [{"iteration": i + 1, "validation_rmse": float(value)}
                                    for i, value in enumerate(history)]
        return self

    def predict(self, pairs):
        self.require_fitted("estimator_")
        values = [self.estimator_.predict(batch) for batch in self.features_.iter_batches(pairs)]
        return np.clip(np.concatenate(values) if values else np.empty(0), 0.5, 5.0)
