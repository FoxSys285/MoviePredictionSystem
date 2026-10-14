from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "models"
RATING_MIN = 0.5
RATING_MAX = 5.0


class MatrixFactorization(nn.Module):
    def __init__(self, users: int, movies: int, time_buckets: int = 64, factors: int = 32):
        super().__init__()
        self.user_embedding = nn.Embedding(users, factors)
        self.movie_embedding = nn.Embedding(movies, factors)
        self.user_bias = nn.Embedding(users, 1)
        self.movie_bias = nn.Embedding(movies, 1)
        self.time_bias = nn.Embedding(time_buckets, 1)
        self.global_bias = nn.Parameter(torch.tensor(3.5))
        nn.init.normal_(self.user_embedding.weight, std=0.06)
        nn.init.normal_(self.movie_embedding.weight, std=0.06)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        nn.init.zeros_(self.time_bias.weight)

    def forward(self, users: torch.Tensor, movies: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        user = self.user_embedding(users)
        movie = self.movie_embedding(movies)
        return (
            (user * movie).sum(dim=1)
            + self.user_bias(users).squeeze(1)
            + self.movie_bias(movies).squeeze(1)
            + self.time_bias(times).squeeze(1)
            + self.global_bias
        )


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, users: int, movies: int, time_buckets: int = 64, factors: int = 16):
        super().__init__()
        self.user_embedding = nn.Embedding(users, factors)
        self.movie_embedding = nn.Embedding(movies, factors)
        self.time_embedding = nn.Embedding(time_buckets, 8)
        self.network = nn.Sequential(
            nn.Linear(factors * 2 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.user_embedding.weight, std=0.06)
        nn.init.normal_(self.movie_embedding.weight, std=0.06)
        nn.init.normal_(self.time_embedding.weight, std=0.03)

    def forward(self, users: torch.Tensor, movies: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (self.user_embedding(users), self.movie_embedding(movies), self.time_embedding(times)),
            dim=1,
        )
        return self.network(features).squeeze(1)


@dataclass
class ProfileFeatures:
    user_idx: int
    mean_rating: float
    rating_count: int
    genre_preferences: np.ndarray
    latest_timestamp: int


class Recommender:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.metadata: dict[str, Any] = joblib.load(model_dir / "metadata.joblib")
        self.linear = joblib.load(model_dir / "linear.joblib")
        self.xgboost = XGBRegressor()
        self.xgboost.load_model(model_dir / "xgboost.json")
        self.factorization = MatrixFactorization(
            len(self.metadata["user_ids"]), len(self.metadata["movie_ids"])
        )
        self.factorization.load_state_dict(
            torch.load(model_dir / "svd.pt", map_location="cpu", weights_only=True)
        )
        self.ncf = NeuralCollaborativeFiltering(
            len(self.metadata["user_ids"]), len(self.metadata["movie_ids"])
        )
        self.ncf.load_state_dict(
            torch.load(model_dir / "ncf.pt", map_location="cpu", weights_only=True)
        )
        self.factorization.eval()
        self.ncf.eval()
        self.global_mean = float(self.metadata["global_mean"])
        self.genres: list[str] = self.metadata["genres"]
        self.catalog = joblib.load(model_dir / "catalog.joblib")
        self.movie_by_id = {int(movie["movieId"]): movie for movie in self.catalog}
        self.metrics = joblib.load(model_dir / "metrics.joblib")

    def _lookup(self, values: np.ndarray, raw_ids: np.ndarray) -> np.ndarray:
        valid = (raw_ids >= 0) & (raw_ids < len(values))
        result = np.full(raw_ids.shape, -1, dtype=np.int32)
        result[valid] = values[raw_ids[valid]]
        return result

    def _time_normalized(self, timestamps: np.ndarray) -> np.ndarray:
        low = float(self.metadata["timestamp_min"])
        high = float(self.metadata["timestamp_max"])
        return np.clip((timestamps.astype(np.float32) - low) / max(high - low, 1.0), 0.0, 1.0)

    def _time_buckets(self, timestamps: np.ndarray) -> np.ndarray:
        return np.minimum((self._time_normalized(timestamps) * 63).astype(np.int64), 63)

    def _profile_features(
        self,
        user_id: int | None,
        history: list[dict[str, Any]],
    ) -> ProfileFeatures:
        user_lookup = self.metadata["user_lookup"]
        user_idx = int(user_lookup[user_id]) if user_id is not None and 0 <= user_id < len(user_lookup) else -1
        if user_idx >= 0:
            default_mean = float(self.metadata["user_means"][user_idx])
            default_count = int(self.metadata["user_counts"][user_idx])
            default_genres = self.metadata["user_genre_means"][user_idx].astype(np.float32, copy=True)
            default_timestamp = int(self.metadata["user_latest_timestamps"][user_idx])
        else:
            default_mean = self.global_mean
            default_count = 0
            default_genres = np.full(len(self.genres), self.global_mean, dtype=np.float32)
            default_timestamp = int(self.metadata["timestamp_max"])

        if not history:
            return ProfileFeatures(user_idx, default_mean, default_count, default_genres, default_timestamp)

        movie_lookup = self.metadata["movie_lookup"]
        history_rows: list[tuple[int, float, int]] = []
        for row in history:
            movie_id = int(row["movieId"])
            movie_idx = int(movie_lookup[movie_id]) if 0 <= movie_id < len(movie_lookup) else -1
            if movie_idx >= 0:
                history_rows.append((movie_idx, float(row["rating"]), int(row["timestamp"])))
        if not history_rows:
            return ProfileFeatures(user_idx, default_mean, default_count, default_genres, default_timestamp)

        timestamps = np.asarray([row[2] for row in history_rows], dtype=np.int64)
        min_time = int(timestamps.min())
        max_time = int(timestamps.max())
        weights = 0.75 + 0.25 * (timestamps - min_time) / max(max_time - min_time, 1)
        ratings = np.asarray([row[1] for row in history_rows], dtype=np.float32)
        mean_rating = float(np.average(ratings, weights=weights))
        genre_sums = np.zeros(len(self.genres), dtype=np.float64)
        genre_weights = np.zeros(len(self.genres), dtype=np.float64)
        for (movie_idx, rating, _), weight in zip(history_rows, weights, strict=True):
            selected_genres = self.metadata["genre_matrix"][movie_idx].astype(bool)
            genre_sums[selected_genres] += rating * weight
            genre_weights[selected_genres] += weight
        observed = genre_weights > 0
        preferences = np.full(len(self.genres), mean_rating, dtype=np.float32)
        preferences[observed] = (genre_sums[observed] / genre_weights[observed]).astype(np.float32)
        return ProfileFeatures(
            user_idx,
            mean_rating,
            len(history_rows),
            preferences,
            int(timestamps.max()),
        )

    def _candidate_features(
        self,
        candidate_ids: np.ndarray,
        profile: ProfileFeatures,
    ) -> tuple[np.ndarray, np.ndarray]:
        movie_lookup = self.metadata["movie_lookup"]
        movie_indices = self._lookup(movie_lookup, candidate_ids.astype(np.int32))
        safe_indices = np.maximum(movie_indices, 0)
        movie_means = self.metadata["movie_means"][safe_indices]
        movie_counts = self.metadata["movie_counts"][safe_indices]
        genres = self.metadata["genre_matrix"][safe_indices].astype(np.float32)
        user_preferences = np.repeat(profile.genre_preferences[None, :], len(candidate_ids), axis=0)
        user_mean = np.full(len(candidate_ids), profile.mean_rating, dtype=np.float32)
        user_count = np.full(len(candidate_ids), profile.rating_count, dtype=np.float32)
        timestamp = np.full(len(candidate_ids), profile.latest_timestamp + 1, dtype=np.int64)
        features = np.column_stack(
            (
                user_mean,
                movie_means,
                np.log1p(user_count),
                np.log1p(movie_counts),
                self._time_normalized(timestamp),
                genres,
                user_preferences,
            )
        ).astype(np.float32, copy=False)
        return features, movie_indices

    def _factor_predictions(
        self,
        model: nn.Module,
        user_idx: int,
        movie_indices: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        with torch.inference_mode():
            user_tensor = torch.full((len(movie_indices),), user_idx, dtype=torch.long)
            movie_tensor = torch.from_numpy(movie_indices.astype(np.int64, copy=False))
            time_tensor = torch.from_numpy(self._time_buckets(timestamps))
            predictions = model(user_tensor, movie_tensor, time_tensor).cpu().numpy()
        return predictions.astype(np.float32)

    def _fold_in_predictions(
        self,
        model_key: str,
        profile: ProfileFeatures,
        history: list[dict[str, Any]],
        movie_indices: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray | None:
        movie_lookup = self.metadata["movie_lookup"]
        valid_rows = [
            row for row in history
            if 0 <= int(row["movieId"]) < len(movie_lookup)
            and int(movie_lookup[int(row["movieId"])]) >= 0
        ]
        if not valid_rows:
            return None

        history_movies = np.asarray(
            [int(movie_lookup[int(row["movieId"])]) for row in valid_rows], dtype=np.int64
        )
        history_ratings = torch.tensor(
            [float(row["rating"]) for row in valid_rows], dtype=torch.float32
        )
        history_times = torch.from_numpy(
            self._time_buckets(np.asarray([int(row["timestamp"]) for row in valid_rows], dtype=np.int64))
        )
        candidate_movies = torch.from_numpy(movie_indices.astype(np.int64, copy=False))
        candidate_times = torch.from_numpy(self._time_buckets(timestamps))
        known_user = profile.user_idx >= 0

        if model_key == "svd":
            model = self.factorization
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if known_user:
                initial_vector = model.user_embedding.weight[profile.user_idx].detach().clone()
                initial_bias = model.user_bias.weight[profile.user_idx].detach().clone().squeeze()
            else:
                initial_vector = torch.zeros(model.user_embedding.embedding_dim)
                initial_bias = torch.tensor(0.0)
            user_vector = torch.nn.Parameter(initial_vector.clone())
            user_bias = torch.nn.Parameter(initial_bias.clone())
            optimizer = torch.optim.Adam([user_vector, user_bias], lr=0.035)
            history_item_vectors = model.movie_embedding(torch.from_numpy(history_movies))
            history_item_bias = model.movie_bias(torch.from_numpy(history_movies)).squeeze(1)
            history_time_bias = model.time_bias(history_times).squeeze(1)
            for _ in range(70):
                history_predictions = (
                    (history_item_vectors * user_vector).sum(dim=1)
                    + user_bias
                    + history_item_bias
                    + history_time_bias
                    + model.global_bias
                )
                loss = torch.nn.functional.mse_loss(history_predictions, history_ratings)
                loss = loss + 0.08 * (user_vector - initial_vector).square().mean() + 0.04 * user_bias.square()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.inference_mode():
                predictions = (
                    (model.movie_embedding(candidate_movies) * user_vector).sum(dim=1)
                    + user_bias
                    + model.movie_bias(candidate_movies).squeeze(1)
                    + model.time_bias(candidate_times).squeeze(1)
                    + model.global_bias
                )
        else:
            model = self.ncf
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if known_user:
                initial_vector = model.user_embedding.weight[profile.user_idx].detach().clone()
            else:
                initial_vector = torch.zeros(model.user_embedding.embedding_dim)
            user_vector = torch.nn.Parameter(initial_vector.clone())
            optimizer = torch.optim.Adam([user_vector], lr=0.012)
            history_item_vectors = model.movie_embedding(torch.from_numpy(history_movies))
            history_time_vectors = model.time_embedding(history_times)
            for _ in range(70):
                user_vectors = user_vector.unsqueeze(0).expand(len(history_movies), -1)
                history_features = torch.cat((user_vectors, history_item_vectors, history_time_vectors), dim=1)
                history_predictions = model.network(history_features).squeeze(1)
                loss = torch.nn.functional.mse_loss(history_predictions, history_ratings)
                loss = loss + 0.08 * (user_vector - initial_vector).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.inference_mode():
                user_vectors = user_vector.unsqueeze(0).expand(len(candidate_movies), -1)
                candidate_features = torch.cat(
                    (user_vectors, model.movie_embedding(candidate_movies), model.time_embedding(candidate_times)),
                    dim=1,
                )
                predictions = model.network(candidate_features).squeeze(1)

        return predictions.cpu().numpy().astype(np.float32)

    def predict_candidates(
        self,
        model_key: str,
        user_id: int | None,
        history: list[dict[str, Any]],
        movie_ids: list[int],
    ) -> tuple[list[dict[str, Any]], float]:
        import time

        started = time.perf_counter()
        profile = self._profile_features(user_id, history)
        watched = {int(row["movieId"]) for row in history}
        candidates = np.asarray(
            [movie_id for movie_id in movie_ids if movie_id not in watched and movie_id in self.movie_by_id],
            dtype=np.int32,
        )
        if not len(candidates):
            return [], (time.perf_counter() - started) * 1000

        features, movie_indices = self._candidate_features(candidates, profile)
        if model_key == "linear":
            predictions = self.linear.predict(features).astype(np.float32)
        elif model_key == "xgboost":
            predictions = self.xgboost.predict(features).astype(np.float32)
        elif model_key in {"svd", "ncf"}:
            timestamps = np.full(len(candidates), profile.latest_timestamp + 1, dtype=np.int64)
            predictions = self._fold_in_predictions(
                model_key, profile, history, movie_indices, timestamps
            )
            if predictions is None:
                if profile.user_idx >= 0:
                    factor_model = self.factorization if model_key == "svd" else self.ncf
                    predictions = self._factor_predictions(
                        factor_model, profile.user_idx, movie_indices, timestamps
                    )
                else:
                    predictions = self.metadata["movie_means"][np.maximum(movie_indices, 0)].astype(np.float32)
            genre_masks = self.metadata["genre_matrix"][np.maximum(movie_indices, 0)].astype(bool)
            genre_blend = min(0.35, 0.12 + 0.23 * min(profile.rating_count / 50.0, 1.0))
            for index, genre_mask in enumerate(genre_masks):
                if genre_mask.any():
                    personal_genre_score = float(profile.genre_preferences[genre_mask].mean())
                    predictions[index] = (1.0 - genre_blend) * predictions[index] + genre_blend * personal_genre_score
        else:
            raise ValueError(f"Unknown model: {model_key}")

        predictions = np.clip(predictions, RATING_MIN, RATING_MAX)
        top_indices = np.argsort(-predictions, kind="stable")[:10]
        results = []
        for index in top_indices:
            movie = self.movie_by_id[int(candidates[index])]
            results.append(
                {
                    "movieId": int(movie["movieId"]),
                    "title": movie["title"],
                    "year": movie["year"],
                    "genres": movie["genres"],
                    "predictedRating": round(float(predictions[index]), 3),
                    "averageRating": float(movie["averageRating"]),
                    "ratingCount": int(movie["ratingCount"]),
                }
            )
        return results, (time.perf_counter() - started) * 1000

    def recommend(
        self,
        model_key: str,
        user_id: int | None,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        recommendations, latency_ms = self.predict_candidates(
            model_key, user_id, history, list(self.movie_by_id)
        )
        return {
            "model": model_key,
            "recommendations": recommendations,
            "latencyMs": round(latency_ms, 3),
            "ratingCount": len(history),
            "isModelPrediction": True,
        }
