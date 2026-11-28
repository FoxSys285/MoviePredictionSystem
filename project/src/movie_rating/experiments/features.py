"""Train-only ID maps and leave-one-out features for tabular rating regressors."""

import numpy as np
import pandas as pd

from movie_rating.experiments.data import validate_ratings
from movie_rating.memory import BATCH_ROWS


def fit_biases(users, items, values, mean, n_users, n_items, *, epochs=0, regularization=10.0):
    """Initialize user/item offsets using only observed fitting ratings."""
    user_bias = np.zeros(n_users, dtype=np.float32)
    item_bias = np.zeros(n_items, dtype=np.float32)
    uc = np.bincount(users, minlength=n_users)
    ic = np.bincount(items, minlength=n_items)
    for _ in range(epochs):
        user_bias = (np.bincount(users, weights=values - mean - item_bias[items],
                                 minlength=n_users) / (uc + regularization)).astype(np.float32)
        item_bias = (np.bincount(items, weights=values - mean - user_bias[users],
                                 minlength=n_items) / (ic + regularization)).astype(np.float32)
    return user_bias, item_bias


class RatingIndex:
    """IDs are categories; unknown query IDs map to -1, never to a known user."""

    def fit(self, ratings):
        validate_ratings(ratings)
        self.user_ids_, users = np.unique(ratings.userId, return_inverse=True)
        self.movie_ids_, items = np.unique(ratings.movieId, return_inverse=True)
        values = ratings.rating.to_numpy(dtype=np.float32)
        self.mean_ = float(values.mean(dtype=np.float64))
        return users.astype(np.int32), items.astype(np.int32), values

    def transform(self, pairs):
        if not hasattr(self, "user_ids_"):
            raise ValueError("Fit the rating index before predicting.")
        if not isinstance(pairs, pd.DataFrame) or not {"userId", "movieId"}.issubset(pairs):
            raise ValueError("Prediction rows require userId and movieId columns.")
        if pairs.empty:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        output = []
        for column, vocabulary in (("userId", self.user_ids_), ("movieId", self.movie_ids_)):
            values = pairs[column].to_numpy()
            if (not pd.api.types.is_numeric_dtype(values) or not np.isfinite(values).all()
                    or (values <= 0).any() or (values != np.floor(values)).any()):
                raise ValueError(f"{column} must contain positive integer IDs.")
            codes = np.searchsorted(vocabulary, values)
            known = vocabulary[np.minimum(codes, len(vocabulary) - 1)] == values
            output.append(np.where(known, codes, -1).astype(np.int64))
        return tuple(output)


class TabularFeatures:
    """Smoothed user/movie means, activity, and optional content interactions.

    Training rows exclude their own rating from all target-derived statistics,
    including the global prior. Prediction uses only the fitted Train statistics.
    """

    def __init__(self, *, rich=False, smoothing=10.0):
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be positive and finite.")
        self.rich, self.smoothing = rich, smoothing

    def fit(self, ratings, movies=None):
        self.index_ = RatingIndex()
        users, items, values = self.index_.fit(ratings)
        self.total_ = float(values.sum(dtype=np.float64))
        self.rows_ = len(values)
        self.feature_names_ = ["user_mean", "movie_mean", "user_log_count", "movie_log_count",
                               "unknown_user", "unknown_movie"]
        for name, codes, size in (("user", users, len(self.index_.user_ids_)),
                                  ("movie", items, len(self.index_.movie_ids_))):
            setattr(self, f"{name}_count_", np.bincount(codes, minlength=size))
            setattr(self, f"{name}_sum_", np.bincount(codes, weights=values, minlength=size))
        self.genre_names_ = []
        if self.rich:
            self._fit_metadata(movies)
            self.genre_count_ = np.zeros((len(self.index_.user_ids_), len(self.genre_names_)))
            self.genre_sum_ = np.zeros_like(self.genre_count_)
            for column in range(len(self.genre_names_)):
                mask = self.item_genres_[items, column]
                self.genre_count_[:, column] = np.bincount(
                    users, weights=mask, minlength=len(self.index_.user_ids_))
                self.genre_sum_[:, column] = np.bincount(
                    users, weights=mask * values, minlength=len(self.index_.user_ids_))
            self.feature_names_ += [f"genre:{name}" for name in self.genre_names_]
            self.feature_names_ += [f"preference:{name}" for name in self.genre_names_]
            self.feature_names_ += ["release_year", "missing_release_year"]
        return self

    def _fit_metadata(self, movies):
        if movies is None:
            movies = pd.DataFrame({"movieId": self.index_.movie_ids_})
        if "movieId" not in movies or movies.movieId.duplicated().any():
            raise ValueError("Movie metadata must have unique movieId values.")
        metadata = movies.set_index("movieId").copy()
        if "genres" not in metadata:
            metadata["genres"] = ""
        if "year" in metadata:
            years = pd.to_numeric(metadata.year, errors="coerce")
        elif "title" in metadata:
            years = pd.to_numeric(
                metadata.title.fillna("").str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
        else:
            years = pd.Series(np.nan, index=metadata.index)
        known = metadata.reindex(self.index_.movie_ids_)
        self.genre_names_ = sorted({genre for value in known.genres.fillna("")
                                   for genre in value.split("|")
                                   if genre and genre != "(no genres listed)"})
        training_years = years.reindex(self.index_.movie_ids_).dropna()
        fill = float(training_years.median()) if len(training_years) else 0.0
        self.metadata_ = metadata
        self.years_ = years.fillna(fill)
        self.missing_years_ = years.isna()
        self.year_fill_ = fill
        self.item_genres_, _, _ = self._movie_features(self.index_.movie_ids_)

    def _movie_features(self, movie_ids):
        ids, inverse = np.unique(movie_ids, return_inverse=True)
        metadata = self.metadata_.reindex(ids)
        genres = metadata.genres.fillna("").str.split("|")
        matrix = np.asarray([[float(genre in values) for genre in self.genre_names_]
                             for values in genres], dtype=np.float32)
        matrix = matrix.reshape(len(ids), len(self.genre_names_))
        years = self.years_.reindex(ids).fillna(self.year_fill_).to_numpy(dtype=np.float32)
        missing = self.missing_years_.reindex(ids, fill_value=True).to_numpy(dtype=np.float32)
        return matrix[inverse], years[inverse], missing[inverse]

    def iter_batches(self, frame, *, training=False):
        for start in range(0, len(frame), BATCH_ROWS):
            yield self.transform(frame.iloc[start:start + BATCH_ROWS], training=training)

    def transform(self, frame, *, training=False):
        users, items = self.index_.transform(frame)
        known_u, known_i = users >= 0, items >= 0
        u, i = np.maximum(users, 0), np.maximum(items, 0)
        uc = np.where(known_u, self.user_count_[u], 0).astype(np.float64)
        ic = np.where(known_i, self.movie_count_[i], 0).astype(np.float64)
        us = np.where(known_u, self.user_sum_[u], 0)
        its = np.where(known_i, self.movie_sum_[i], 0)
        prior = np.full(len(frame), self.index_.mean_)
        if training:
            target = frame.rating.to_numpy(dtype=np.float32).astype(np.float64)
            uc, ic, us, its = uc - 1, ic - 1, us - target, its - target
            prior = ((self.total_ - target) / (self.rows_ - 1)
                     if self.rows_ > 1 else np.full(len(frame), 2.75))
        smooth = self.smoothing
        user_mean = (us + smooth * prior) / (uc + smooth)
        movie_mean = (its + smooth * prior) / (ic + smooth)
        columns = [user_mean, movie_mean, np.log1p(uc), np.log1p(ic), uc == 0, ic == 0]
        if self.rich:
            genres, years, missing = self._movie_features(frame.movieId.to_numpy())
            gc = self.genre_count_[u] * known_u[:, None]
            gs = self.genre_sum_[u] * known_u[:, None]
            if training:
                gc, gs = gc - genres, gs - genres * target[:, None]
            preference = (gs + smooth * prior[:, None]) / (gc + smooth)
            columns += [genres, (preference - user_mean[:, None]) * genres, years, missing]
        return np.column_stack(columns).astype(np.float32)
