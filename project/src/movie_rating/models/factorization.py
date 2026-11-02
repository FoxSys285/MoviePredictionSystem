import numpy as np
from numba import njit

from movie_rating.models.baseline import BiasBaseline


@njit(cache=True)
def _sgd_epoch(users, movies, values, order, p, q, bu, bi, mean, lr, reg, nonnegative):
    for row in order:
        u, i = users[row], movies[row]
        estimate = np.dot(p[u], q[i])
        if not nonnegative:
            estimate += mean + bu[u] + bi[i]
        error = values[row] - estimate
        if not nonnegative:
            bu[u] += lr * (error - reg * bu[u])
            bi[i] += lr * (error - reg * bi[i])
        for factor in range(p.shape[1]):
            old_p, old_q = p[u, factor], q[i, factor]
            next_p = old_p + lr * (error * old_q - reg * old_p)
            next_q = old_q + lr * (error * old_p - reg * old_q)
            p[u, factor] = max(0.0, next_p) if nonnegative else next_p
            q[i, factor] = max(0.0, next_q) if nonnegative else next_q


class MatrixFactorization(BiasBaseline):
    nonnegative = False

    def __init__(self, factors=32, epochs=15, learning_rate=0.005, regularization=0.02, seed=42):
        super().__init__(epochs=epochs, regularization=regularization, seed=seed)
        if factors < 1 or learning_rate <= 0:
            raise ValueError("factors and learning_rate must be positive.")
        self.factors, self.learning_rate, self.seed = factors, learning_rate, seed

    def fit(self, ratings):
        # Train-only bias model also supplies meaningful cold-start predictions.
        super().fit(ratings)
        users, movies, values = self.encode_training(ratings)
        rng = np.random.default_rng(self.seed)
        if self.nonnegative:
            scale = np.sqrt(self.global_mean / self.factors)
            self.p = rng.uniform(
                0.5 * scale, 1.5 * scale, (len(self.user_ids), self.factors)
            ).astype(np.float32)
            self.q = rng.uniform(
                0.5 * scale, 1.5 * scale, (len(self.movie_ids), self.factors)
            ).astype(np.float32)
        else:
            self.p = rng.normal(0, 0.1, (len(self.user_ids), self.factors)).astype(np.float32)
            self.q = rng.normal(0, 0.1, (len(self.movie_ids), self.factors)).astype(np.float32)
        for _ in range(self.epochs):
            _sgd_epoch(
                users,
                movies,
                values,
                rng.permutation(len(values)),
                self.p,
                self.q,
                self.user_bias,
                self.movie_bias,
                self.global_mean,
                self.learning_rate,
                self.regularization,
                self.nonnegative,
            )
        if not (np.isfinite(self.p).all() and np.isfinite(self.q).all()):
            raise ValueError("Factorization diverged; lower learning_rate.")
        return self

    def predict(self, user_ids, movie_ids):
        users, movies = self.encode_queries(user_ids, movie_ids)
        output = super().predict(user_ids, movie_ids)
        known = (users >= 0) & (movies >= 0)
        # Bound temporary factor products when evaluating millions of ratings.
        indices = np.flatnonzero(known)
        for start in range(0, len(indices), 100_000):
            rows = indices[start : start + 100_000]
            u, i = users[rows], movies[rows]
            scores = np.einsum("ij,ij->i", self.p[u], self.q[i])
            if not self.nonnegative:
                scores += self.global_mean + self.user_bias[u] + self.movie_bias[i]
            output[rows] = scores
        return self.clip(output)


class SVD(MatrixFactorization):
    """Biased explicit-feedback matrix factorization trained with SGD (Funk SVD)."""


class NMF(MatrixFactorization):
    """Observed-rating squared loss with nonnegative factors via projected SGD."""

    nonnegative = True
