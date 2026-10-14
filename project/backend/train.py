from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import polars as pl
import torch
from scipy.sparse import load_npz
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from backend.recommender import (
    MODEL_DIR,
    MatrixFactorization,
    NeuralCollaborativeFiltering,
    ROOT,
)


LOGGER = logging.getLogger("movielens_training")
DATA_DIR = ROOT / "data" / "processed" / "movielens25m"
RESULT_DIR = ROOT / "results"
TIME_BUCKETS = 64
FEATURE_CHUNK = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rating prediction models on processed MovieLens 25M data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear-rows", type=int, default=1_500_000)
    parser.add_argument("--xgboost-rows", type=int, default=600_000)
    parser.add_argument("--svd-rows", type=int, default=1_500_000)
    parser.add_argument("--ncf-rows", type=int, default=600_000)
    parser.add_argument("--svd-epochs", type=int, default=5)
    parser.add_argument("--ncf-epochs", type=int, default=6)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--ranking-users", type=int, default=20_000)
    parser.add_argument("--ranking-negatives", type=int, default=100)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    return parser.parse_args()


def make_lookup(raw_ids: np.ndarray) -> np.ndarray:
    lookup = np.full(int(raw_ids.max()) + 1, -1, dtype=np.int32)
    lookup[raw_ids] = np.arange(len(raw_ids), dtype=np.int32)
    return lookup


def read_split(path: Path, user_lookup: np.ndarray, movie_lookup: np.ndarray) -> dict[str, np.ndarray]:
    frame = pl.read_parquet(path).select("userId", "movieId", "rating", "timestamp")
    raw_users = frame.get_column("userId").to_numpy().astype(np.int32, copy=False)
    raw_movies = frame.get_column("movieId").to_numpy().astype(np.int32, copy=False)
    return {
        "users": user_lookup[raw_users],
        "movies": movie_lookup[raw_movies],
        "ratings": frame.get_column("rating").to_numpy().astype(np.float32, copy=False),
        "timestamps": frame.get_column("timestamp").to_numpy().astype(np.int64, copy=False),
    }


class FeatureContext:
    def __init__(self, train: dict[str, np.ndarray], user_ids: np.ndarray, movie_ids: np.ndarray):
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.user_count = len(user_ids)
        self.movie_count = len(movie_ids)
        self.user_lookup = make_lookup(user_ids)
        self.movie_lookup = make_lookup(movie_ids)
        self.user_indices = train["users"]
        self.movie_indices = train["movies"]
        self.ratings = train["ratings"]
        self.timestamps = train["timestamps"]
        self.global_mean = float(self.ratings.mean())
        self.timestamp_min = int(self.timestamps.min())
        self.timestamp_max = int(self.timestamps.max())

        genre_rows = pl.read_parquet(DATA_DIR / "movie_genres.parquet").select("movieId", "genre")
        self.genres = sorted(genre_rows.get_column("genre").drop_nulls().unique().to_list())
        genre_to_idx = {genre: index for index, genre in enumerate(self.genres)}
        self.genre_matrix = np.zeros((self.movie_count, len(self.genres)), dtype=np.uint8)
        raw_genres = genre_rows.to_dicts()
        for row in raw_genres:
            raw_movie_id = int(row["movieId"])
            if raw_movie_id < len(self.movie_lookup):
                movie_idx = int(self.movie_lookup[raw_movie_id])
                if movie_idx >= 0 and row["genre"] in genre_to_idx:
                    self.genre_matrix[movie_idx, genre_to_idx[row["genre"]]] = 1

        self.user_counts = np.bincount(self.user_indices, minlength=self.user_count).astype(np.float32)
        self.user_sums = np.bincount(self.user_indices, weights=self.ratings, minlength=self.user_count).astype(np.float64)
        self.movie_counts = np.bincount(self.movie_indices, minlength=self.movie_count).astype(np.float32)
        self.movie_sums = np.bincount(self.movie_indices, weights=self.ratings, minlength=self.movie_count).astype(np.float64)
        self.user_means = (self.user_sums / np.maximum(self.user_counts, 1)).astype(np.float32)
        self.movie_means = (self.movie_sums / np.maximum(self.movie_counts, 1)).astype(np.float32)
        self.user_latest_timestamps = np.zeros(self.user_count, dtype=np.int64)
        np.maximum.at(self.user_latest_timestamps, self.user_indices, self.timestamps)

        genre_count = len(self.genres)
        self.user_genre_counts = np.zeros((self.user_count, genre_count), dtype=np.float32)
        self.user_genre_sums = np.zeros((self.user_count, genre_count), dtype=np.float64)
        for genre_idx in range(genre_count):
            has_genre = self.genre_matrix[self.movie_indices, genre_idx].astype(bool)
            users = self.user_indices[has_genre]
            ratings = self.ratings[has_genre]
            self.user_genre_counts[:, genre_idx] = np.bincount(users, minlength=self.user_count)
            self.user_genre_sums[:, genre_idx] = np.bincount(
                users, weights=ratings, minlength=self.user_count
            )
        self.user_genre_means = (
            self.user_genre_sums / np.maximum(self.user_genre_counts, 1)
        ).astype(np.float32)
        no_genre_history = self.user_genre_counts == 0
        self.user_genre_means[no_genre_history] = np.broadcast_to(
            self.user_means[:, None], self.user_genre_means.shape
        )[no_genre_history]

    def _normalized_time(self, timestamps: np.ndarray) -> np.ndarray:
        return np.clip(
            (timestamps.astype(np.float32) - self.timestamp_min)
            / max(self.timestamp_max - self.timestamp_min, 1),
            0,
            1,
        )

    def time_buckets(self, timestamps: np.ndarray) -> np.ndarray:
        return np.minimum((self._normalized_time(timestamps) * (TIME_BUCKETS - 1)).astype(np.int64), TIME_BUCKETS - 1)

    def make_features(
        self,
        users: np.ndarray,
        movies: np.ndarray,
        ratings: np.ndarray,
        timestamps: np.ndarray,
        training_rows: bool,
    ) -> np.ndarray:
        count = len(users)
        user_denominator = self.user_counts[users].astype(np.float64)
        movie_denominator = self.movie_counts[movies].astype(np.float64)
        user_numerator = self.user_sums[users].copy()
        movie_numerator = self.movie_sums[movies].copy()
        if training_rows:
            user_denominator -= 1
            movie_denominator -= 1
            user_numerator -= ratings
            movie_numerator -= ratings
        user_mean = np.where(
            user_denominator > 0,
            user_numerator / np.maximum(user_denominator, 1),
            self.global_mean,
        ).astype(np.float32)
        movie_mean = np.where(
            movie_denominator > 0,
            movie_numerator / np.maximum(movie_denominator, 1),
            self.global_mean,
        ).astype(np.float32)
        user_counts = np.maximum(user_denominator, 0).astype(np.float32)
        movie_counts = np.maximum(movie_denominator, 0).astype(np.float32)
        genre_features = self.genre_matrix[movies].astype(np.float32)
        preferences = self.user_genre_means[users].copy()
        if training_rows:
            genre_counts = self.user_genre_counts[users]
            genre_sums = self.user_genre_sums[users]
            for genre_idx in range(len(self.genres)):
                included = genre_features[:, genre_idx] > 0
                enough = included & (genre_counts[:, genre_idx] > 1)
                singleton = included & ~enough
                preferences[enough, genre_idx] = (
                    (genre_sums[enough, genre_idx] - ratings[enough])
                    / (genre_counts[enough, genre_idx] - 1)
                )
                preferences[singleton, genre_idx] = user_mean[singleton]
        time_feature = self._normalized_time(timestamps)
        features = np.concatenate(
            (
                np.column_stack(
                    (
                        user_mean,
                        movie_mean,
                        np.log1p(user_counts),
                        np.log1p(movie_counts),
                        time_feature,
                    )
                ),
                genre_features,
                preferences,
            ),
            axis=1,
        )
        return features.astype(np.float32, copy=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "user_ids": self.user_ids,
            "movie_ids": self.movie_ids,
            "user_lookup": self.user_lookup,
            "movie_lookup": self.movie_lookup,
            "user_means": self.user_means,
            "user_counts": self.user_counts,
            "user_latest_timestamps": self.user_latest_timestamps,
            "movie_means": self.movie_means,
            "movie_counts": self.movie_counts,
            "user_genre_means": self.user_genre_means,
            "user_genre_counts": self.user_genre_counts,
            "user_genre_sums": self.user_genre_sums,
            "genre_matrix": self.genre_matrix,
            "genres": self.genres,
            "global_mean": self.global_mean,
            "timestamp_min": self.timestamp_min,
            "timestamp_max": self.timestamp_max,
            "feature_count": 5 + len(self.genres) * 2,
        }


def sample_indices(size: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    return rng.choice(size, size=limit, replace=False)


def fit_factor_model(
    model: torch.nn.Module,
    users: np.ndarray,
    movies: np.ndarray,
    ratings: np.ndarray,
    time_buckets: np.ndarray,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model.to(device)
    user_tensor = torch.from_numpy(users.astype(np.int64, copy=False))
    movie_tensor = torch.from_numpy(movies.astype(np.int64, copy=False))
    rating_tensor = torch.from_numpy(ratings.astype(np.float32, copy=False))
    time_tensor = torch.from_numpy(time_buckets.astype(np.int64, copy=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    loss_fn = torch.nn.MSELoss()
    batch_size = 8192
    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(user_tensor))
        total_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(user_tensor[batch], movie_tensor[batch], time_tensor[batch])
            loss = loss_fn(prediction, rating_tensor[batch])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
        LOGGER.info("%s epoch %d/%d RMSE=%.4f", type(model).__name__, epoch + 1, epochs, (total_loss / seen) ** 0.5)
    model.eval()
    return time.perf_counter() - started


def predict_factors(
    model: torch.nn.Module,
    users: np.ndarray,
    movies: np.ndarray,
    time_buckets: np.ndarray,
    chunk_size: int = FEATURE_CHUNK,
) -> np.ndarray:
    output = np.empty(len(users), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(users), chunk_size):
            end = min(start + chunk_size, len(users))
            batch_users = torch.from_numpy(users[start:end].astype(np.int64, copy=False))
            batch_movies = torch.from_numpy(movies[start:end].astype(np.int64, copy=False))
            batch_times = torch.from_numpy(time_buckets[start:end].astype(np.int64, copy=False))
            output[start:end] = model(batch_users, batch_movies, batch_times).cpu().numpy()
    return output


def metric_row(actual: np.ndarray, predicted: np.ndarray, elapsed: float) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "inference_ms_per_1000": float(elapsed * 1000 / max(len(actual), 1) * 1000),
    }


def predict_batches(
    predictor: Callable[[dict[str, np.ndarray], slice], np.ndarray],
    split: dict[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    predictions = np.empty(len(split["ratings"]), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, len(predictions), FEATURE_CHUNK):
        end = min(start + FEATURE_CHUNK, len(predictions))
        predictions[start:end] = predictor(split, slice(start, end))
    return predictions, time.perf_counter() - started


def build_ranking_candidates(
    split: dict[str, np.ndarray],
    train_matrix: Any,
    movie_count: int,
    rng: np.random.Generator,
    user_limit: int,
    negative_count: int,
    relevance_threshold: float,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int, int, np.ndarray]], np.ndarray, int]:
    order = np.argsort(split["users"], kind="stable")
    sorted_users = split["users"][order]
    group_users, group_starts, group_counts = np.unique(
        sorted_users, return_index=True, return_counts=True
    )
    eligible: list[tuple[int, int, int, int]] = []
    for user_idx, start, count in zip(group_users, group_starts, group_counts, strict=True):
        test_positions = order[start : start + count]
        positive_count = int(np.count_nonzero(split["ratings"][test_positions] >= relevance_threshold))
        if positive_count:
            eligible.append((int(user_idx), int(start), int(count), positive_count))
    if not eligible:
        raise ValueError("No users have a relevant rating in this ranking-evaluation split.")
    if user_limit > 0 and len(eligible) > user_limit:
        selected = np.sort(rng.choice(len(eligible), size=user_limit, replace=False))
        eligible = [eligible[int(index)] for index in selected]

    candidate_user_parts: list[np.ndarray] = []
    candidate_movie_parts: list[np.ndarray] = []
    relevance_parts: list[np.ndarray] = []
    groups: list[tuple[int, int, int, np.ndarray]] = []
    available_mask = np.ones(movie_count, dtype=bool)
    cursor = 0
    for user_idx, start, count, positive_count in eligible:
        test_positions = order[start : start + count]
        test_movies = split["movies"][test_positions]
        test_ratings = split["ratings"][test_positions]
        row_start = int(train_matrix.indptr[user_idx])
        row_end = int(train_matrix.indptr[user_idx + 1])
        train_movies = train_matrix.indices[row_start:row_end]

        available_mask.fill(True)
        available_mask[train_movies] = False
        available_mask[test_movies] = False
        negative_pool = np.flatnonzero(available_mask)
        sampled_count = min(negative_count, len(negative_pool))
        negatives = rng.choice(negative_pool, size=sampled_count, replace=False).astype(np.int32, copy=False)

        candidates = np.concatenate((test_movies.astype(np.int32, copy=False), negatives))
        relevant = np.concatenate(
            (test_ratings >= relevance_threshold, np.zeros(sampled_count, dtype=bool))
        )
        candidate_user_parts.append(np.full(len(candidates), user_idx, dtype=np.int32))
        candidate_movie_parts.append(candidates)
        relevance_parts.append(relevant)
        group_end = cursor + len(candidates)
        groups.append((cursor, group_end, positive_count, candidates))
        cursor = group_end

    candidate_users = np.concatenate(candidate_user_parts)
    candidate_movies = np.concatenate(candidate_movie_parts)
    candidate_split = {
        "users": candidate_users,
        "movies": candidate_movies,
        "ratings": np.zeros(len(candidate_users), dtype=np.float32),
        "timestamps": np.maximum(1, np.zeros(len(candidate_users), dtype=np.int64)),
    }
    return candidate_split, groups, np.concatenate(relevance_parts), len(eligible)


def ranking_metrics(
    scores: np.ndarray,
    groups: list[tuple[int, int, int, np.ndarray]],
    relevant: np.ndarray,
    movie_count: int,
    k: int = 10,
) -> dict[str, float | int]:
    precision_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    recommended = np.zeros(movie_count, dtype=bool)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    for start, end, positive_count, candidate_ids in groups:
        ranked = np.argsort(-scores[start:end], kind="stable")[:k]
        ranked_relevant = relevant[start:end][ranked]
        hits = int(ranked_relevant.sum())
        precision_values.append(hits / k)
        recall_values.append(hits / positive_count)
        dcg = float(discounts[: len(ranked_relevant)][ranked_relevant].sum())
        ideal_count = min(positive_count, k)
        idcg = float(discounts[:ideal_count].sum())
        ndcg_values.append(dcg / idcg if idcg else 0.0)
        recommended[candidate_ids[ranked]] = True

    users = len(groups)
    return {
        "precision_at_10": float(np.mean(precision_values)),
        "recall_at_10": float(np.mean(recall_values)),
        "ndcg_at_10": float(np.mean(ndcg_values)),
        "item_coverage": float(recommended.sum() / movie_count),
        "users_evaluated": users,
    }


def evaluate_ranking_split(
    split_name: str,
    split: dict[str, np.ndarray],
    train_matrix: Any,
    latest_timestamps: np.ndarray,
    context: FeatureContext,
    predictors: dict[str, Callable[[dict[str, np.ndarray], slice], np.ndarray]],
    metrics: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    user_limit: int,
    negative_count: int,
    relevance_threshold: float,
) -> None:
    candidate_split, groups, relevant, evaluated_users = build_ranking_candidates(
        split,
        train_matrix,
        context.movie_count,
        rng,
        user_limit,
        negative_count,
        relevance_threshold,
    )
    candidate_split["timestamps"] = (latest_timestamps[candidate_split["users"]] + 1).astype(
        np.int64, copy=False
    )
    LOGGER.info(
        "Evaluating Top-10 on %s: %s users, %s candidate user-item pairs, %s sampled unrated candidates/user",
        split_name,
        f"{evaluated_users:,}",
        f"{len(candidate_split['users']):,}",
        negative_count,
    )
    for model_key, predictor in predictors.items():
        predictions, _ = predict_batches(predictor, candidate_split)
        predictions = np.clip(predictions, 0.5, 5.0)
        result = ranking_metrics(predictions, groups, relevant, context.movie_count)
        result.update(
            {
                "negative_candidates_per_user": negative_count,
                "relevance_threshold": relevance_threshold,
                "candidate_protocol": "all held-out ratings + fixed sampled unrated movies",
            }
        )
        metrics[model_key][split_name]["ranking"] = result
        LOGGER.info(
            "%s %s NDCG@10=%.4f Recall@10=%.4f Precision@10=%.4f",
            model_key,
            split_name,
            result["ndcg_at_10"],
            result["recall_at_10"],
            result["precision_at_10"],
        )

    baseline_predictions = context.movie_means[candidate_split["movies"]]
    baseline = ranking_metrics(baseline_predictions, groups, relevant, context.movie_count)
    baseline.update(
        {
            "negative_candidates_per_user": negative_count,
            "relevance_threshold": relevance_threshold,
            "candidate_protocol": "all held-out ratings + fixed sampled unrated movies",
        }
    )
    metrics["baseline"][split_name]["ranking"] = baseline


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(args.torch_threads, 1))
    torch.manual_seed(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    user_ids = (
        pl.read_parquet(DATA_DIR / "user_id_map.parquet")
        .sort("user_idx")
        .get_column("userId")
        .to_numpy()
        .astype(np.int32, copy=False)
    )
    movie_ids = (
        pl.read_parquet(DATA_DIR / "movie_id_map.parquet")
        .sort("movie_idx")
        .get_column("movieId")
        .to_numpy()
        .astype(np.int32, copy=False)
    )
    user_lookup = make_lookup(user_ids)
    movie_lookup = make_lookup(movie_ids)
    LOGGER.info("Loading train and holdout splits")
    train = read_split(DATA_DIR / "ratings_train.parquet", user_lookup, movie_lookup)
    validation = read_split(DATA_DIR / "ratings_validation.parquet", user_lookup, movie_lookup)
    test = read_split(DATA_DIR / "ratings_test.parquet", user_lookup, movie_lookup)
    train_matrix = load_npz(DATA_DIR / "user_item_train.npz").tocsr()
    validation_matrix = load_npz(DATA_DIR / "user_item_validation.npz").tocsr()
    test_history_matrix = (train_matrix + validation_matrix).tocsr()
    context = FeatureContext(train, user_ids, movie_ids)
    test_latest_timestamps = context.user_latest_timestamps.copy()
    np.maximum.at(test_latest_timestamps, validation["users"], validation["timestamps"])
    catalog = json.loads((ROOT / "src" / "data" / "movielens-catalog.json").read_text(encoding="utf-8"))["catalog"]

    model_metrics: dict[str, Any] = {}
    sample = sample_indices(len(train["ratings"]), args.linear_rows, rng)
    LOGGER.info("Fitting linear Ridge on %s sampled train ratings", f"{len(sample):,}")
    x_linear = context.make_features(
        train["users"][sample], train["movies"][sample], train["ratings"][sample], train["timestamps"][sample], True
    )
    linear = Ridge(alpha=20.0, solver="lsqr", tol=1e-3, max_iter=300)
    started = time.perf_counter()
    linear.fit(x_linear, train["ratings"][sample])
    linear_seconds = time.perf_counter() - started
    joblib.dump(linear, MODEL_DIR / "linear.joblib", compress=3)
    del x_linear

    xgb_sample = sample_indices(len(train["ratings"]), args.xgboost_rows, rng)
    LOGGER.info("Fitting XGBoost on %s sampled train ratings", f"{len(xgb_sample):,}")
    x_xgb = context.make_features(
        train["users"][xgb_sample], train["movies"][xgb_sample], train["ratings"][xgb_sample], train["timestamps"][xgb_sample], True
    )
    xgboost_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=140,
        max_depth=8,
        learning_rate=0.07,
        subsample=0.82,
        colsample_bytree=0.9,
        reg_lambda=8.0,
        min_child_weight=8,
        tree_method="hist",
        max_bin=128,
        n_jobs=min(args.torch_threads, 8),
        random_state=args.seed,
    )
    started = time.perf_counter()
    xgboost_model.fit(x_xgb, train["ratings"][xgb_sample], verbose=False)
    xgb_seconds = time.perf_counter() - started
    xgboost_model.save_model(MODEL_DIR / "xgboost.json")
    del x_xgb

    factor_sample = sample_indices(len(train["ratings"]), args.svd_rows, rng)
    LOGGER.info("Fitting SVD-style matrix factorization on %s sampled train ratings", f"{len(factor_sample):,}")
    factorization = MatrixFactorization(len(user_ids), len(movie_ids))
    svd_seconds = fit_factor_model(
        factorization,
        train["users"][factor_sample],
        train["movies"][factor_sample],
        train["ratings"][factor_sample],
        context.time_buckets(train["timestamps"][factor_sample]),
        args.svd_epochs,
        0.025,
        args.seed,
    )
    torch.save(factorization.state_dict(), MODEL_DIR / "svd.pt")

    ncf_sample = sample_indices(len(train["ratings"]), args.ncf_rows, rng)
    LOGGER.info("Fitting Neural Collaborative Filtering on %s sampled train ratings", f"{len(ncf_sample):,}")
    ncf = NeuralCollaborativeFiltering(len(user_ids), len(movie_ids))
    ncf_seconds = fit_factor_model(
        ncf,
        train["users"][ncf_sample],
        train["movies"][ncf_sample],
        train["ratings"][ncf_sample],
        context.time_buckets(train["timestamps"][ncf_sample]),
        args.ncf_epochs,
        0.001,
        args.seed + 1,
    )
    torch.save(ncf.state_dict(), MODEL_DIR / "ncf.pt")

    def regression_predictor(model: Any) -> Callable[[dict[str, np.ndarray], slice], np.ndarray]:
        def predict(split: dict[str, np.ndarray], span: slice) -> np.ndarray:
            features = context.make_features(
                split["users"][span],
                split["movies"][span],
                split["ratings"][span],
                split["timestamps"][span],
                False,
            )
            result = model.predict(features).astype(np.float32)
            del features
            return result
        return predict

    def factor_predictor(model: torch.nn.Module) -> Callable[[dict[str, np.ndarray], slice], np.ndarray]:
        def predict(split: dict[str, np.ndarray], span: slice) -> np.ndarray:
            return predict_factors(
                model,
                split["users"][span],
                split["movies"][span],
                context.time_buckets(split["timestamps"][span]),
            )
        return predict

    model_predictors: dict[str, Callable[[dict[str, np.ndarray], slice], np.ndarray]] = {
        "linear": regression_predictor(linear),
        "xgboost": regression_predictor(xgboost_model),
        "svd": factor_predictor(factorization),
        "ncf": factor_predictor(ncf),
    }
    training_seconds = {
        "linear": linear_seconds,
        "xgboost": xgb_seconds,
        "svd": svd_seconds,
        "ncf": ncf_seconds,
    }

    for split_name, split in (("validation", validation), ("test", test)):
        LOGGER.info("Scoring %s split (%s rows)", split_name, f"{len(split['ratings']):,}")
        for key, predictor in model_predictors.items():
            prediction, elapsed = predict_batches(predictor, split)
            score = metric_row(split["ratings"], prediction, elapsed)
            model_metrics.setdefault(key, {"training_seconds": training_seconds[key], "sampled_train_rows": {
                "linear": len(sample), "xgboost": len(xgb_sample), "svd": len(factor_sample), "ncf": len(ncf_sample)
            }[key]})[split_name] = score
            LOGGER.info("%s %s RMSE=%.4f MAE=%.4f", key, split_name, score["rmse"], score["mae"])
            del prediction
        baseline_predictions = context.movie_means[split["movies"]]
        model_metrics.setdefault("baseline", {"training_seconds": 0.0, "sampled_train_rows": len(train["ratings"])})[split_name] = metric_row(
            split["ratings"], baseline_predictions, 0.0
        )

    for split_name, split in (("validation", validation), ("test", test)):
        history_matrix = train_matrix if split_name == "validation" else test_history_matrix
        latest_timestamps = context.user_latest_timestamps if split_name == "validation" else test_latest_timestamps
        evaluate_ranking_split(
            split_name,
            split,
            history_matrix,
            latest_timestamps,
            context,
            model_predictors,
            model_metrics,
            rng,
            args.ranking_users,
            args.ranking_negatives,
            args.relevance_threshold,
        )

    metadata = context.metadata()
    metadata["seed"] = args.seed
    metadata["training_rows"] = len(train["ratings"])
    metadata["sampled_rows"] = {
        "linear": len(sample), "xgboost": len(xgb_sample), "svd": len(factor_sample), "ncf": len(ncf_sample)
    }
    metadata["rating_limits"] = [0.5, 5.0]
    joblib.dump(metadata, MODEL_DIR / "metadata.joblib", compress=3)
    joblib.dump(catalog, MODEL_DIR / "catalog.joblib", compress=3)
    joblib.dump(model_metrics, MODEL_DIR / "metrics.joblib", compress=3)
    (RESULT_DIR / "model_metrics.json").write_text(
        json.dumps(model_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (RESULT_DIR / "model_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        columns = [
            "model", "training_seconds", "sampled_train_rows", "split", "rmse", "mae",
            "inference_ms_per_1000", "precision_at_10", "recall_at_10", "ndcg_at_10",
            "item_coverage", "ranking_users", "negative_candidates_per_user", "relevance_threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model_key, data in model_metrics.items():
            for split_name in ("validation", "test"):
                score = data[split_name]
                ranking = score.get("ranking", {})
                writer.writerow({
                    "model": model_key,
                    "training_seconds": data["training_seconds"],
                    "sampled_train_rows": data["sampled_train_rows"],
                    "split": split_name,
                    "rmse": score["rmse"],
                    "mae": score["mae"],
                    "inference_ms_per_1000": score["inference_ms_per_1000"],
                    "precision_at_10": ranking.get("precision_at_10"),
                    "recall_at_10": ranking.get("recall_at_10"),
                    "ndcg_at_10": ranking.get("ndcg_at_10"),
                    "item_coverage": ranking.get("item_coverage"),
                    "ranking_users": ranking.get("users_evaluated"),
                    "negative_candidates_per_user": ranking.get("negative_candidates_per_user"),
                    "relevance_threshold": ranking.get("relevance_threshold"),
                })
    LOGGER.info("Training complete. Artifacts: %s; metrics: %s", MODEL_DIR, RESULT_DIR / "model_metrics.csv")


if __name__ == "__main__":
    main()
