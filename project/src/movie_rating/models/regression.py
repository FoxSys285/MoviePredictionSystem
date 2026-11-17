"""Explicit-rating regression with train-only statistics and genre preferences."""

import numpy as np
from scipy import sparse

from movie_rating.models.base import RatingModel

_FEATURE_BATCH_ROWS = 250_000


class FeatureRegression(RatingModel):
    use_identity = False

    def set_movie_metadata(self, movies):
        # Only static genre metadata is used; catalogue ratings never enter features.
        self.metadata = dict(zip(movies.movieId, movies.genres))

    def _genre_matrix(self):
        metadata = getattr(self, "metadata", {})
        genres = sorted(
            {
                genre
                for movie in self.movie_ids
                for genre in metadata.get(movie, "").split("|")
                if genre and genre != "(no genres listed)"
            }
        )
        return np.asarray(
            [
                [float(genre in metadata.get(movie, "").split("|")) for genre in genres]
                for movie in self.movie_ids
            ],
            dtype=np.float32,
        ).reshape(len(self.movie_ids), len(genres))

    def fit(self, ratings):
        users, items, values = self.encode_training(ratings)
        self.user_count = np.bincount(users, minlength=len(self.user_ids)).astype(np.float32)
        self.item_count = np.bincount(items, minlength=len(self.movie_ids)).astype(np.float32)
        self.user_sum = np.bincount(
            users, weights=values, minlength=len(self.user_ids)
        ).astype(np.float32)
        self.item_sum = np.bincount(
            items, weights=values, minlength=len(self.movie_ids)
        ).astype(np.float32)
        self.genres = self._genre_matrix()
        self.preference_sum = np.zeros(
            (len(self.user_ids), self.genres.shape[1]), dtype=np.float32
        )
        centered_values = values - np.float32(self.global_mean)
        for column in range(self.genres.shape[1]):
            self.preference_sum[:, column] = np.bincount(
                users,
                weights=centered_values * self.genres[items, column],
                minlength=len(self.user_ids),
            )
        # Exclude each row's target from its aggregate features during fitting.
        features = self._features(users, items, exclude=values)
        self.estimator = self._estimator()
        self.estimator.fit(features, values - self.global_mean)
        return self

    def _features(self, users, items, exclude=None, profile=None):
        if self.use_identity or len(users) <= _FEATURE_BATCH_ROWS:
            return self._feature_batch(users, items, exclude, profile)
        features = np.empty(
            (len(users), 6 + 2 * self.genres.shape[1]), dtype=np.float32
        )
        exclude_values = None if exclude is None else np.asarray(exclude, dtype=np.float32)
        for start in range(0, len(users), _FEATURE_BATCH_ROWS):
            stop = min(start + _FEATURE_BATCH_ROWS, len(users))
            features[start:stop] = self._feature_batch(
                users[start:stop],
                items[start:stop],
                None if exclude_values is None else exclude_values[start:stop],
                profile,
            )
        return features

    def _feature_batch(self, users, items, exclude=None, profile=None):
        known_u, known_i = users >= 0, items >= 0
        u, i = np.maximum(users, 0), np.maximum(items, 0)
        uc = np.where(known_u, self.user_count[u], np.float32(0))
        us = np.where(known_u, self.user_sum[u], np.float32(0))
        ic = np.where(known_i, self.item_count[i], np.float32(0))
        its = np.where(known_i, self.item_sum[i], np.float32(0))
        preferences = self.preference_sum[u] * known_u[:, None]
        if profile is not None:
            uc = np.full(len(items), profile[0], dtype=np.float32)
            us = np.full(len(items), profile[1], dtype=np.float32)
            preferences = np.broadcast_to(
                np.asarray(profile[2], dtype=np.float32), preferences.shape
            ).copy()
        item_genres = self.genres[i] * known_i[:, None]
        if exclude is not None:
            exclude = np.asarray(exclude, dtype=np.float32)
            mean = np.float32(self.global_mean)
            uc, ic = uc - 1, ic - 1
            us, its = us - exclude, its - exclude
            preferences = preferences - (exclude - mean)[:, None] * item_genres
        else:
            mean = np.float32(self.global_mean)
        user_mean = (us + 5 * mean) / (uc + 5) - mean
        item_mean = (its + 5 * mean) / (ic + 5) - mean
        preferences = preferences / (uc[:, None] + 5)
        dense = np.column_stack(
            [
                user_mean,
                item_mean,
                np.log1p(uc) / 10,
                np.log1p(ic) / 10,
                uc == 0,
                ic == 0,
                item_genres,
                preferences * item_genres,
            ]
        ).astype(np.float32, copy=False)
        if not self.use_identity:
            return dense
        rows = np.arange(len(users))
        identities = sparse.csr_matrix(
            (
                np.ones(known_u.sum() + known_i.sum(), dtype=np.float32),
                (
                    np.concatenate([rows[known_u], rows[known_i]]),
                    np.concatenate([users[known_u], len(self.user_ids) + items[known_i]]),
                ),
            ),
            shape=(len(users), len(self.user_ids) + len(self.movie_ids)),
        )
        return sparse.hstack([identities, sparse.csr_matrix(dense)], format="csr")

    def predict(self, user_ids, movie_ids):
        users, items = self.encode_queries(user_ids, movie_ids)
        if not len(users):
            return np.array([], dtype=np.float32)
        result = np.empty(len(users), dtype=np.float32)
        for start in range(0, len(users), _FEATURE_BATCH_ROWS):
            stop = min(start + _FEATURE_BATCH_ROWS, len(users))
            features = self._feature_batch(users[start:stop], items[start:stop])
            result[start:stop] = self.global_mean + self.estimator.predict(features)
        result[(users < 0) & (items < 0)] = self.global_mean
        return self.clip(result)

    def predict_profile(self, history):
        item_index = {int(movie): index for index, movie in enumerate(self.movie_ids)}
        available = [row for row in history if row["movie_id"] in item_index]
        if not available:
            return None
        indices = np.array([item_index[row["movie_id"]] for row in available])
        values = np.array([row["rating"] for row in available])
        preferences = ((values - self.global_mean)[:, None] * self.genres[indices]).sum(axis=0)
        features = self._features(
            np.full(len(self.movie_ids), -1),
            np.arange(len(self.movie_ids)),
            profile=(len(values), values.sum(), preferences),
        )
        return self.clip(self.global_mean + self.estimator.predict(features))


class RidgeRegression(FeatureRegression):
    """Sparse user/item one-hot + aggregate/genre features with L2 regularization."""

    use_identity = True

    def __init__(self, alpha=10.0, seed=42):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha, self.seed = alpha, seed

    def _estimator(self):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=self.alpha, solver="lsqr", tol=1e-5)


class XGBoostRegression(FeatureRegression):
    """Histogram gradient boosting; numeric IDs are never treated as distances."""

    def __init__(self, n_estimators=160, max_depth=6, learning_rate=0.05, n_jobs=4, seed=42):
        if min(n_estimators, max_depth, learning_rate, n_jobs) <= 0:
            raise ValueError("Boosting parameters must be positive.")
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.learning_rate, self.n_jobs, self.seed = learning_rate, n_jobs, seed

    def _estimator(self):
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            objective="reg:squarederror",
            tree_method="hist",
            reg_lambda=10.0,
        )
