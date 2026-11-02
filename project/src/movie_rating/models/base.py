from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RatingModel(ABC):
    @abstractmethod
    def fit(self, ratings: pd.DataFrame): ...

    @abstractmethod
    def predict(self, user_ids, movie_ids) -> np.ndarray: ...

    def encode_training(self, ratings: pd.DataFrame):
        self.user_ids, users = np.unique(ratings.userId.to_numpy(), return_inverse=True)
        self.movie_ids, movies = np.unique(ratings.movieId.to_numpy(), return_inverse=True)
        values = ratings.rating.to_numpy(dtype=np.float32)
        self.global_mean = float(values.mean())
        return users, movies, values

    def encode_queries(self, user_ids, movie_ids):
        users = np.atleast_1d(np.asarray(user_ids, dtype=np.int64))
        movies = np.atleast_1d(np.asarray(movie_ids, dtype=np.int64))
        if users.ndim != 1 or movies.ndim != 1 or len(users) != len(movies):
            raise ValueError("user_ids and movie_ids must be equally sized one-dimensional arrays.")
        u = np.searchsorted(self.user_ids, users)
        i = np.searchsorted(self.movie_ids, movies)
        valid_u = self.user_ids[np.minimum(u, len(self.user_ids) - 1)] == users
        valid_i = self.movie_ids[np.minimum(i, len(self.movie_ids) - 1)] == movies
        return np.where(valid_u, u, -1), np.where(valid_i, i, -1)

    @staticmethod
    def clip(values):
        return np.clip(values, 0.5, 5.0).astype(np.float32)
