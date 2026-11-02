import numpy as np

from movie_rating.models.base import RatingModel


class BiasBaseline(RatingModel):
    """Regularized global mean + user bias + movie bias."""

    def __init__(self, epochs=10, regularization=10.0, seed=42):
        if epochs < 1 or regularization <= 0:
            raise ValueError("epochs and regularization must be positive.")
        self.epochs = epochs
        self.regularization = regularization

    def fit(self, ratings):
        users, movies, values = self.encode_training(ratings)
        self.user_bias = np.zeros(len(self.user_ids))
        self.movie_bias = np.zeros(len(self.movie_ids))
        user_counts = np.bincount(users, minlength=len(self.user_ids))
        movie_counts = np.bincount(movies, minlength=len(self.movie_ids))
        for _ in range(self.epochs):
            residual = values - self.global_mean - self.movie_bias[movies]
            self.user_bias = np.bincount(users, weights=residual) / (
                user_counts + self.regularization
            )
            residual = values - self.global_mean - self.user_bias[users]
            self.movie_bias = np.bincount(movies, weights=residual) / (
                movie_counts + self.regularization
            )
        return self

    def predict(self, user_ids, movie_ids):
        users, movies = self.encode_queries(user_ids, movie_ids)
        values = np.full(len(users), self.global_mean)
        known_u, known_i = users >= 0, movies >= 0
        values[known_u] += self.user_bias[users[known_u]]
        values[known_i] += self.movie_bias[movies[known_i]]
        return self.clip(values)
