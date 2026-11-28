"""Biased matrix factorization (Funk SVD), trained on observed ratings by SGD."""

import numpy as np
from numba import njit

from movie_rating.experiments.algorithms.base import (
    MovieAlgorithm,
    finite_number,
    positive_integer,
)
from movie_rating.experiments.data import validate_ratings
from movie_rating.experiments.evaluation import rmse
from movie_rating.experiments.features import RatingIndex, fit_biases
from movie_rating.memory import BATCH_ROWS


@njit(cache=True)
def _sgd_epoch(users, items, values, order, p, q, bu, bi, mean, learning_rate, regularization):
    squared_error = 0.0
    for row in order:
        u, i = users[row], items[row]
        prediction = mean + bu[u] + bi[i]
        for factor in range(p.shape[1]):
            prediction += p[u, factor] * q[i, factor]
        error = values[row] - prediction
        squared_error += error * error
        bu[u] += learning_rate * (error - regularization * bu[u])
        bi[i] += learning_rate * (error - regularization * bi[i])
        for factor in range(p.shape[1]):
            old_p, old_q = p[u, factor], q[i, factor]
            p[u, factor] += learning_rate * (error * old_q - regularization * old_p)
            q[i, factor] += learning_rate * (error * old_p - regularization * old_q)
    return squared_error / len(values)


class SVDBaseline(MovieAlgorithm):
    """Global mean + user bias + movie bias + user/movie latent dot product."""

    supports_early_stopping = True
    iteration_parameter = "epochs"
    defaults = {
        "factors": 32, "epochs": 20, "learning_rate": 0.005, "regularization": 0.02,
        "seed": 42, "patience": 3, "min_delta": 0.0001, "learning_rate_decay": 1.0,
        "bias_epochs": 0, "bias_regularization": 10.0,
    }

    def fit(self, ratings, movies=None):
        return self._fit(ratings)

    def fit_with_validation(self, ratings, validation, movies=None):
        validate_ratings(validation)
        return self._fit(ratings, validation)

    def _fit(self, ratings, validation=None):
        options = self.configuration(self.defaults)
        factors = positive_integer("factors", options["factors"])
        epochs = positive_integer("epochs", options["epochs"])
        patience = positive_integer("patience", options["patience"])
        learning_rate = finite_number("learning_rate", options["learning_rate"], strict=True)
        regularization = finite_number("regularization", options["regularization"])
        min_delta = finite_number("min_delta", options["min_delta"])
        decay = finite_number("learning_rate_decay", options["learning_rate_decay"], strict=True)
        if decay > 1 or not isinstance(options["bias_epochs"], int) or options["bias_epochs"] < 0:
            raise ValueError("Require decay <= 1 and nonnegative integer bias_epochs.")
        bias_reg = finite_number("bias_regularization", options["bias_regularization"], strict=True)
        self.index_ = RatingIndex()
        users, items, values = self.index_.fit(ratings)
        rng = np.random.default_rng(options["seed"])
        self.user_factors_ = rng.normal(0, 0.05, (len(self.index_.user_ids_), factors)).astype(
            np.float32)
        self.item_factors_ = rng.normal(0, 0.05, (len(self.index_.movie_ids_), factors)).astype(
            np.float32)
        self.user_bias_, self.item_bias_ = fit_biases(
            users, items, values, self.index_.mean_, len(self.index_.user_ids_),
            len(self.index_.movie_ids_), epochs=options["bias_epochs"], regularization=bias_reg)
        self.learning_curve_, self.best_iterations_ = [], epochs
        best, stale, checkpoint = float("inf"), 0, None
        for epoch in range(epochs):
            loss = _sgd_epoch(
                users, items, values, rng.permutation(len(values)), self.user_factors_,
                self.item_factors_, self.user_bias_, self.item_bias_, self.index_.mean_,
                learning_rate * decay**epoch, regularization,
            )
            if not np.isfinite(loss):
                raise ValueError("SVD diverged; reduce learning_rate.")
            row = {"epoch": epoch + 1, "online_train_rmse": float(np.sqrt(loss))}
            if validation is not None:
                score = rmse(validation.rating, self.predict(validation))
                row["validation_rmse"] = score
                if score < best - min_delta:
                    best, stale = score, 0
                    self.best_iterations_ = epoch + 1
                    checkpoint = tuple(array.copy() for array in (
                        self.user_factors_, self.item_factors_, self.user_bias_, self.item_bias_))
                else:
                    stale += 1
            self.learning_curve_.append(row)
            self.emit_progress(f"SVD epoch {epoch + 1}/{epochs}")
            if validation is not None and stale >= patience:
                break
        if checkpoint is not None:
            (self.user_factors_, self.item_factors_, self.user_bias_, self.item_bias_) = checkpoint
        for array in (self.user_factors_, self.item_factors_, self.user_bias_, self.item_bias_):
            if not np.isfinite(array).all():
                raise ValueError("SVD factors contain nonfinite values; reduce learning_rate.")
        return self

    def predict(self, pairs):
        self.require_fitted("index_")
        users, items = self.index_.transform(pairs)
        output = np.full(len(pairs), self.index_.mean_)
        known_u, known_i = users >= 0, items >= 0
        output[known_u] += self.user_bias_[users[known_u]]
        output[known_i] += self.item_bias_[items[known_i]]
        both = np.flatnonzero(known_u & known_i)
        for start in range(0, len(both), BATCH_ROWS):
            rows = both[start:start + BATCH_ROWS]
            output[rows] += np.einsum("ij,ij->i", self.user_factors_[users[rows]],
                                     self.item_factors_[items[rows]])
        return np.clip(output, 0.5, 5.0)
