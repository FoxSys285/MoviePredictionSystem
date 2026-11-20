"""Compact train-only aggregates and bounded feature batches for explicit ratings."""

import numpy as np
from scipy import sparse

from movie_rating.memory import BATCH_ROWS
from movie_rating.models.base import RatingModel


class RatingFeatures(RatingModel):
    use_identity = False

    def set_movie_metadata(self, movies):
        self.metadata = dict(zip(movies.movieId, movies.genres.fillna("")))

    def fit_statistics(self, ratings):
        """Build aggregates only; classification must not fit a throwaway regressor."""
        users, items, values = self.encode_training(ratings)
        self.metadata = getattr(self, "metadata", {})
        self.global_positive = float(np.mean(values >= 3.5))
        self.global_variance = float(np.var(values, dtype=np.float64))
        self.genre_names = sorted(
            {
                g
                for value in getattr(self, "metadata", {}).values()
                for g in value.split("|")
                if g and g != "(no genres listed)"
            }
        )
        self.genres = np.asarray(
            [
                [float(g in self.metadata.get(movie, "").split("|")) for g in self.genre_names]
                for movie in self.movie_ids
            ],
            dtype=np.float32,
        ).reshape(len(self.movie_ids), len(self.genre_names))
        for prefix, codes, size in (
            ("user", users, len(self.user_ids)),
            ("item", items, len(self.movie_ids)),
        ):
            for name, weights in (
                ("count", None),
                ("sum", values),
                ("squares", values.astype(np.float64) ** 2),
                ("positive", values >= 3.5),
            ):
                setattr(
                    self,
                    f"{prefix}_{name}",
                    np.bincount(codes, weights=weights, minlength=size).astype(np.float32),
                )
        self.preference_sum = self._preferences(users, items, values)
        # Only the latest 20 training ratings of each user enter recent preferences.
        if "timestamp" in ratings:
            order = np.lexsort((ratings.movieId.to_numpy(), ratings.timestamp.to_numpy(), users))
            ordered_users = users[order]
            starts = np.r_[0, np.flatnonzero(ordered_users[1:] != ordered_users[:-1]) + 1]
            ends = np.r_[starts[1:], len(order)]
            positions = ends[:, None] - 1 - np.arange(20)
            valid = positions >= starts[:, None]
            recent = order[positions[valid]]
            cut_rows = order[np.maximum(starts, ends - 20)]
            self.recent_time = ratings.timestamp.to_numpy()[cut_rows]
            self.recent_movie = ratings.movieId.to_numpy()[cut_rows]
            del order, ordered_users, positions, valid
        else:
            recent = np.arange(len(ratings))
            self.recent_time = self.recent_movie = None
        ru, ri, rv = users[recent], items[recent], values[recent]
        self.recent_count = np.bincount(ru, minlength=len(self.user_ids)).astype(np.float32)
        self.recent_sum = np.bincount(ru, weights=rv, minlength=len(self.user_ids)).astype(
            np.float32
        )
        self.recent_preference = self._preferences(ru, ri, rv)
        return self

    def _preferences(self, users, items, values):
        output = np.zeros((len(self.user_ids), len(self.genre_names)), dtype=np.float32)
        centered = values - np.float32(self.global_mean)
        for column in range(len(self.genre_names)):
            output[:, column] = np.bincount(
                users, weights=centered * self.genres[items, column], minlength=len(self.user_ids)
            )
        return output

    @property
    def feature_count(self):
        return 11 + 3 * len(self.genre_names)

    def iter_features(self, ratings, training=False, batch_rows=BATCH_ROWS):
        for start in range(0, len(ratings), batch_rows):
            batch = ratings.iloc[start : start + batch_rows]
            users, items = self.encode_queries(batch.userId, batch.movieId)
            yield self._feature_batch(
                users,
                items,
                exclude=batch.rating.to_numpy() if training else None,
                timestamps=batch.timestamp.to_numpy() if "timestamp" in batch else None,
                movie_ids=batch.movieId.to_numpy(),
            )

    def _features(self, users, items, exclude=None, profile=None):
        if self.use_identity or len(users) <= BATCH_ROWS:
            return self._feature_batch(users, items, exclude, profile)
        output = np.empty((len(users), self.feature_count), dtype=np.float32)
        for start in range(0, len(users), BATCH_ROWS):
            stop = start + BATCH_ROWS
            output[start:stop] = self._feature_batch(
                users[start:stop],
                items[start:stop],
                None if exclude is None else exclude[start:stop],
                profile,
            )
        return output

    def _feature_batch(
        self, users, items, exclude=None, profile=None, timestamps=None, movie_ids=None
    ):
        known_u, known_i = users >= 0, items >= 0
        u, i = np.maximum(users, 0), np.maximum(items, 0)
        uc, ic = np.where(known_u, self.user_count[u], 0), np.where(known_i, self.item_count[i], 0)
        us, its = np.where(known_u, self.user_sum[u], 0), np.where(known_i, self.item_sum[i], 0)
        uq = np.where(known_u, self.user_squares[u], 0)
        iq = np.where(known_i, self.item_squares[i], 0)
        up = np.where(known_u, self.user_positive[u], 0)
        ip = np.where(known_i, self.item_positive[i], 0)
        preferences = self.preference_sum[u] * known_u[:, None]
        rc = np.where(known_u, self.recent_count[u], 0)
        rs = np.where(known_u, self.recent_sum[u], 0)
        rp = self.recent_preference[u] * known_u[:, None]
        item_genres = self.genres[i] * known_i[:, None]
        # Static genre metadata is usable even for movies absent from training ratings.
        if movie_ids is not None and len(self.genre_names) and (~known_i).any():
            item_genres[~known_i] = np.asarray(
                [
                    [float(g in self.metadata.get(movie, "").split("|")) for g in self.genre_names]
                    for movie in movie_ids[~known_i]
                ],
                dtype=np.float32,
            ).reshape(-1, len(self.genre_names))
        if profile is not None:
            uc = np.full(len(items), profile["count"], dtype=np.float32)
            us = np.full(len(items), profile["sum"], dtype=np.float32)
            uq = np.full(len(items), profile["squares"], dtype=np.float32)
            up = np.full(len(items), profile["positive"], dtype=np.float32)
            preferences = np.broadcast_to(profile["preferences"], preferences.shape).copy()
            rc, rs, rp = uc.copy(), us.copy(), preferences.copy()
        mean = np.float32(self.global_mean)
        if exclude is not None:
            target = np.asarray(exclude, dtype=np.float32)
            uc, ic, us, its = uc - 1, ic - 1, us - target, its - target
            uq, iq = uq - target**2, iq - target**2
            positive = target >= 3.5
            up, ip = up - positive, ip - positive
            contribution = (target - mean)[:, None] * item_genres
            preferences -= contribution
            recent = np.ones(len(users), dtype=bool)
            if timestamps is not None and self.recent_time is not None:
                recent = (timestamps > self.recent_time[u]) | (
                    (timestamps == self.recent_time[u]) & (movie_ids >= self.recent_movie[u])
                )
            rc, rs = rc - recent, rs - recent * target
            rp -= contribution * recent[:, None]
        user_mean = (us + 5 * mean) / (uc + 5)
        item_mean = (its + 5 * mean) / (ic + 5)
        prior_square = np.float32(self.global_variance + self.global_mean**2)
        dense = np.column_stack(
            [
                user_mean - mean,
                item_mean - mean,
                np.log1p(uc) / 10,
                np.log1p(ic) / 10,
                uc == 0,
                ic == 0,
                (up + 5 * self.global_positive) / (uc + 5),
                (ip + 5 * self.global_positive) / (ic + 5),
                np.sqrt(np.maximum(0, (uq + 5 * prior_square) / (uc + 5) - user_mean**2)),
                np.sqrt(np.maximum(0, (iq + 5 * prior_square) / (ic + 5) - item_mean**2)),
                (rs + 5 * mean) / (rc + 5) - mean,
                item_genres,
                preferences / (uc[:, None] + 5) * item_genres,
                rp / (rc[:, None] + 5) * item_genres,
            ]
        ).astype(np.float32, copy=False)
        if not self.use_identity:
            return dense
        rows = np.arange(len(users))
        identities = sparse.csr_matrix(
            (
                np.ones(known_u.sum() + known_i.sum(), dtype=np.float32),
                (
                    np.r_[rows[known_u], rows[known_i]],
                    np.r_[users[known_u], len(self.user_ids) + items[known_i]],
                ),
            ),
            shape=(len(users), len(self.user_ids) + len(self.movie_ids)),
        )
        return sparse.hstack([identities, sparse.csr_matrix(dense)], format="csr")
