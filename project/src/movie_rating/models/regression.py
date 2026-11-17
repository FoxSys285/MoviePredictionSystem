"""Explicit-rating regressors with bounded feature batches."""

import shutil

import numpy as np

from movie_rating.memory import feature_buffer, workspace
from movie_rating.models.tabular import RatingFeatures

_FEATURE_BATCH_ROWS = 65_536
_EXTERNAL_MEMORY_ROWS = 250_000


class StreamingLinearRegression:
    """OLS from small sufficient statistics; never factors an N-by-P dense matrix."""

    def fit_batches(self, batches):
        count, sx, sy, xx, xy = 0, None, 0.0, None, None
        for features, target in batches:
            x, y = np.asarray(features, dtype=np.float64), np.asarray(target, dtype=np.float64)
            if sx is None:
                sx = np.zeros(x.shape[1])
                xx = np.zeros((x.shape[1], x.shape[1]))
                xy = np.zeros(x.shape[1])
            count += len(x)
            sx += x.sum(axis=0)
            sy += y.sum()
            xx += x.T @ x
            xy += x.T @ y
        center, target_mean = sx / count, sy / count
        covariance = xx / count - np.outer(center, center)
        scale = np.sqrt(np.maximum(np.diag(covariance), 0))
        scale[scale < 1e-8] = 1
        correlation = covariance / scale[:, None] / scale[None, :]
        rhs = (xy / count - center * target_mean) / scale
        self.coef_ = np.linalg.lstsq(correlation, rhs, rcond=1e-7)[0] / scale
        self.intercept_ = target_mean - center @ self.coef_
        return self

    def predict(self, features):
        return features @ self.coef_ + self.intercept_


class FeatureRegression(RatingFeatures):
    def fit(self, ratings):
        self.fit_statistics(ratings)
        self.estimator = self._estimator()
        if hasattr(self.estimator, "fit_batches"):

            def batches():
                offset = 0
                for features in self.iter_features(ratings, training=True):
                    target = ratings.rating.iloc[offset : offset + len(features)].to_numpy()
                    yield features, target - self.global_mean
                    offset += len(features)

            self.estimator.fit_batches(batches())
        elif self.use_identity:
            users, items = self.encode_queries(ratings.userId, ratings.movieId)
            features = self._feature_batch(
                users,
                items,
                exclude=ratings.rating.to_numpy(),
                timestamps=ratings.timestamp.to_numpy() if "timestamp" in ratings else None,
                movie_ids=ratings.movieId.to_numpy(),
            )
            self.estimator.fit(features, ratings.rating.to_numpy() - self.global_mean)
        else:
            with feature_buffer(
                len(ratings), self.feature_count, self.iter_features(ratings, training=True)
            ) as features:
                self.estimator.fit(features, ratings.rating.to_numpy() - self.global_mean)
        return self

    def predict(self, user_ids, movie_ids):
        user_ids, movie_ids = np.asarray(user_ids), np.asarray(movie_ids)
        if user_ids.ndim != 1 or movie_ids.ndim != 1 or len(user_ids) != len(movie_ids):
            raise ValueError("user_ids and movie_ids must be equally sized one-dimensional arrays.")
        output = np.empty(len(user_ids), dtype=np.float32)
        for start in range(0, len(user_ids), _FEATURE_BATCH_ROWS):
            stop = start + _FEATURE_BATCH_ROWS
            users, items = self.encode_queries(user_ids[start:stop], movie_ids[start:stop])
            features = self._feature_batch(users, items, movie_ids=movie_ids[start:stop])
            values = self.global_mean + self.estimator.predict(features)
            values[(users < 0) & (items < 0)] = self.global_mean
            output[start:stop] = values
        return self.clip(output)

    def predict_profile(self, history):
        index = {int(movie): position for position, movie in enumerate(self.movie_ids)}
        available = [row for row in history if row["movie_id"] in index]
        if not available:
            return None
        items = np.array([index[row["movie_id"]] for row in available])
        values = np.asarray([row["rating"] for row in available], dtype=np.float32)
        preferences = ((values - self.global_mean)[:, None] * self.genres[items]).sum(axis=0)
        profile = {
            "count": len(values),
            "sum": values.sum(),
            "squares": (values**2).sum(),
            "positive": (values >= 3.5).sum(),
            "preferences": preferences,
        }
        output = np.empty(len(self.movie_ids), dtype=np.float32)
        for start in range(0, len(output), _FEATURE_BATCH_ROWS):
            items = np.arange(start, min(start + _FEATURE_BATCH_ROWS, len(output)))
            features = self._feature_batch(np.full(len(items), -1), items, profile=profile)
            output[items] = self.global_mean + self.estimator.predict(features)
        return self.clip(output)


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


class _BoosterPredictor:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, features):
        return self.booster.inplace_predict(features)


class XGBoostRegression(FeatureRegression):
    """Histogram boosting with disk-backed quantile pages for large training sets."""

    def __init__(self, n_estimators=160, max_depth=6, learning_rate=0.05, n_jobs=2, seed=42):
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

    def fit(self, ratings):
        if len(ratings) < _EXTERNAL_MEMORY_ROWS:
            return super().fit(ratings)
        import xgboost as xgb

        self.fit_statistics(ratings)
        builder = self

        class Iterator(xgb.DataIter):
            def __init__(self, cache):
                super().__init__(cache_prefix=str(cache))
                self.reset()

            def reset(self):
                self.batches = builder.iter_features(ratings, training=True, batch_rows=250_000)
                self.offset = 0

            def next(self, input_data):
                features = next(self.batches, None)
                if features is None:
                    return False
                target = ratings.rating.iloc[self.offset : self.offset + len(features)].to_numpy()
                input_data(data=features, label=target - builder.global_mean)
                self.offset += len(features)
                return True

        with workspace("xgboost-", native_ascii=True) as directory:
            required = len(ratings) * self.feature_count * 4 + 512 * 2**20
            if shutil.disk_usage(directory).free < required:
                raise OSError("Ổ đĩa cache không đủ chỗ cho XGBoost external-memory.")
            iterator = Iterator(directory / "cache")
            matrix_type = getattr(xgb, "ExtMemQuantileDMatrix", xgb.QuantileDMatrix)
            if matrix_type is xgb.QuantileDMatrix:
                iterator.cache_prefix = None
            data = matrix_type(iterator, nthread=self.n_jobs)
            booster = xgb.train(
                self._estimator().get_xgb_params(), data, num_boost_round=self.n_estimators
            )
            self.estimator = _BoosterPredictor(booster)
            del data, iterator
        return self
