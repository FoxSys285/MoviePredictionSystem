"""Account recommendations, using explicit profile ratings without MovieLens ID aliasing."""

import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from movie_rating.services.catalog import CatalogService, movie_records
from movie_rating.services.demo_profiles import dataset_source_key
from movie_rating.services.prediction import PredictionService
from movie_rating.settings import ALGORITHMS, Settings
from movie_rating.training import current_run


class RecommendationService:
    def __init__(self, settings: Settings, catalog: CatalogService):
        self.settings = settings
        self.catalog = catalog
        self.prediction = PredictionService(settings)
        self.genre_fingerprint = None
        self.features = None

    def _genre_features(self, movies):
        if self.genre_fingerprint != self.catalog.fingerprint:
            genres = sorted(
                {
                    genre
                    for value in movies.genres
                    for genre in value.split("|")
                    if genre != "(no genres listed)"
                }
            )
            self.features = np.array(
                [[float(genre in value.split("|")) for genre in genres] for value in movies.genres],
                dtype=np.float32,
            ).reshape(len(movies), len(genres))
            norms = np.linalg.norm(self.features, axis=1, keepdims=True)
            self.features /= np.maximum(norms, 1)
            self.genre_fingerprint = self.catalog.fingerprint
        return self.features

    @staticmethod
    def profile_scores(model, history: list[dict], algorithm: str):
        """Fold in only the real account's ratings; freeze all trained item parameters."""
        if hasattr(model, "predict_profile"):
            scores = model.predict_profile(history)
            if scores is None:
                return None
            if not np.isfinite(scores).all():
                raise ValueError("Model returned nonfinite profile scores.")
            return pd.Series(scores, index=model.movie_ids)
        item_index = {int(movie): index for index, movie in enumerate(model.movie_ids)}
        available = [row for row in history if row["movie_id"] in item_index]
        if not available:
            return None
        indices = np.array([item_index[row["movie_id"]] for row in available])
        ratings = np.array([row["rating"] for row in available], dtype=np.float64)
        regularization = max(float(model.regularization), 1e-6)
        user_bias = float(
            np.sum(ratings - model.global_mean - model.movie_bias[indices])
            / (len(ratings) + regularization)
        )
        base = model.global_mean + user_bias + model.movie_bias
        if algorithm == "baseline":
            scores = base
        elif algorithm == "item_knn":
            observed = np.zeros(len(model.movie_ids))
            residual = observed.copy()
            observed[indices] = 1
            residual[indices] = ratings - base[indices]
            denominator = np.asarray(model.similarity @ observed).ravel()
            correction = np.asarray(model.similarity @ residual).ravel()
            scores = base + np.divide(
                correction, denominator, out=np.zeros_like(correction), where=denominator > 1e-8
            )
        elif algorithm == "svd":
            design = np.column_stack([np.ones(len(indices)), model.q[indices]])
            target = ratings - model.global_mean - model.movie_bias[indices]
            factors = np.linalg.solve(
                design.T @ design + regularization * np.eye(design.shape[1]), design.T @ target
            )
            scores = model.global_mean + model.movie_bias + factors[0] + model.q @ factors[1:]
        elif algorithm == "nmf":
            design = np.vstack([model.q[indices], np.sqrt(regularization) * np.eye(model.factors)])
            factors, _ = nnls(design, np.concatenate([ratings, np.zeros(model.factors)]))
            scores = model.q @ factors
        else:
            return None
        if not np.isfinite(scores).all():
            raise ValueError("Model returned nonfinite profile scores.")
        return pd.Series(np.clip(scores, 0.5, 5), index=model.movie_ids)

    def recommend(
        self,
        history: list[dict],
        limit: int = 20,
        algorithm: str = "auto",
        dataset_profile: dict | None = None,
    ):
        catalog = self.catalog.load()
        movies = catalog.movies.copy()
        mean = catalog.summary["mean_rating"]
        movies["prior"] = (movies.mean_rating.fillna(mean) * movies.rating_count + 25 * mean) / (
            movies.rating_count + 25
        )
        seen = {row["movie_id"] for row in history}
        mode, selected, run_id = "popular", None, None
        strategy = "community_popularity"
        model_status = "not_trained"
        available_algorithms = []
        if (self.settings.artifacts_dir / "latest.json").is_file():
            run = current_run(self.settings)
            report = json.loads((run / "validation.json").read_text(encoding="utf-8"))
            available_algorithms = [
                name for name in ALGORITHMS if (run / f"{name}.joblib").is_file()
            ]
            model_status = "ready"
            name = (
                report.get("best_recommendation_algorithm", report["best_algorithm"])
                if algorithm == "auto"
                else algorithm
            )
            if name not in available_algorithms:
                raise ValueError("Mô hình được chọn chưa được huấn luyện.")
            if history:
                model = self.prediction.load_model(str(run), name)
                known_user = (
                    dataset_profile is not None
                    and dataset_profile["source_key"] == dataset_source_key(self.settings)
                    and not dataset_profile["profile_modified"]
                    and dataset_profile["dataset_user_id"] in model.user_ids
                )
                if known_user:
                    uid = dataset_profile["dataset_user_id"]
                    scores = pd.Series(
                        model.predict(np.full(len(model.movie_ids), uid), model.movie_ids),
                        index=model.movie_ids,
                    )
                    strategy = "trained_dataset_user"
                else:
                    scores = self.profile_scores(model, history, name)
                    strategy = "profile_fold_in" if scores is not None else "community_popularity"
                if scores is not None:
                    movies["predicted_rating"] = movies.movieId.map(scores)
                    movies = movies.loc[movies.predicted_rating.notna()].copy()
                    movies["affinity"] = movies.predicted_rating
                    mode, selected, run_id = "model", name, report["run_id"]
                else:
                    model_status = "no_known_history_items"
        elif algorithm != "auto":
            raise ValueError("Mô hình được chọn chưa được huấn luyện.")
        if mode != "model" and history:
            features = self._genre_features(catalog.movies)
            index = {int(movie): position for position, movie in enumerate(catalog.movies.movieId)}
            profile = np.zeros(features.shape[1], dtype=np.float64)
            for row in history:
                if row["movie_id"] in index:
                    profile += (row["rating"] - 3) * features[index[row["movie_id"]]]
            norm = np.linalg.norm(profile)
            if norm > 1e-8:
                movies["affinity"] = features @ (profile / norm)
                mode = "content"
                strategy = "genre_profile"
        movies = movies.loc[~movies.movieId.isin(seen)]
        order = (
            ["affinity", "movieId"]
            if mode == "model"
            else (
                ["affinity", "prior", "rating_count", "movieId"]
                if "affinity" in movies
                else ["prior", "rating_count", "movieId"]
            )
        )
        ranked = movies.sort_values(order, ascending=[False] * (len(order) - 1) + [True]).head(
            limit
        )
        items = movie_records(ranked)
        for item, (_, row) in zip(items, ranked.iterrows()):
            item["predicted_rating"] = float(row.predicted_rating) if mode == "model" else None
        return {
            "items": items,
            "limit": limit,
            "mode": mode,
            "algorithm": selected,
            "run_id": run_id,
            "history_count": len(history),
            "model_status": model_status,
            "available_algorithms": available_algorithms,
            "requested_algorithm": algorithm,
            "cold_start": strategy != "trained_dataset_user",
            "strategy": strategy,
            "profile_type": "dataset" if dataset_profile else "personal",
            "dataset_user_id": dataset_profile["dataset_user_id"] if dataset_profile else None,
        }
