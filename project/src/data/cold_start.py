"""Content/popularity fallback for a new user with no trained user embedding.

The fallback uses movie genres and rating statistics learned from TRAIN only.
It is a simple demo strategy, not a measured replacement for SVD/NCF.

Example: ``python -m src.data.cold_start --ratings 1:5 296:4.5 --top-n 10``
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.data.common import load_config, project_path, read_json


def _check_rating(value: float) -> float:
    rating = float(value)
    if not (0.5 <= rating <= 5.0 and rating * 2 == round(rating * 2)):
        raise ValueError(f"Rating must be in 0.5-star increments from 0.5 to 5.0: {value}")
    return rating


class ColdStartRecommender:
    """Rank movies using a smoothed train mean plus selected-genre preference."""

    def __init__(self) -> None:
        config = load_config()
        movies_path = project_path(config["dataset_dir"]) / "movies.csv"
        stats_path = project_path(config["processed_dir"]) / "movie_stats.parquet"
        if not stats_path.is_file():
            raise FileNotFoundError(f"Run prepare and split first: {stats_path}")
        movies = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
        stats = pd.read_parquet(stats_path)
        self.movies = movies.merge(stats, on="movieId", how="left", validate="one_to_one")
        self.movies["n_ratings"] = self.movies["n_ratings"].fillna(0).astype(int)
        artifact = project_path(config["artifacts_dir"]) / "mean_baseline.json"
        if artifact.exists():
            self.global_mean = float(read_json(artifact)["global_mean"])
        else:
            weights = stats["n_ratings"].to_numpy(dtype=float)
            self.global_mean = float(np.average(stats["mean_rating"], weights=weights))
        self.movies["mean_rating"] = self.movies["mean_rating"].fillna(self.global_mean)
        self._by_id = self.movies.set_index("movieId")

    def recommend(self, ratings: Mapping[int, float], top_n: int = 10) -> dict:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        selected = {int(movie_id): _check_rating(rating) for movie_id, rating in ratings.items()}
        ignored = [movie_id for movie_id in selected if movie_id not in self._by_id.index]
        genre_sum: dict[str, float] = defaultdict(float)
        genre_count: dict[str, int] = defaultdict(int)
        for movie_id, rating in selected.items():
            if movie_id in ignored:
                continue
            genres = str(self._by_id.at[movie_id, "genres"]).split("|")
            for genre in genres:
                if genre and genre != "(no genres listed)":
                    genre_sum[genre] += rating - 3.0
                    genre_count[genre] += 1
        genre_preference = {
            genre: total / genre_count[genre] for genre, total in genre_sum.items()
        }

        candidates = self.movies.loc[~self.movies["movieId"].isin(selected)].copy()
        genre_score = candidates["genres"].fillna("").map(
            lambda cell: np.mean([genre_preference.get(g, 0.0) for g in cell.split("|")])
            if cell and cell != "(no genres listed)" else 0.0
        )
        n = candidates["n_ratings"].to_numpy(dtype=float)
        mean = candidates["mean_rating"].to_numpy(dtype=float)
        smoothed_mean = self.global_mean + (n / (n + 20.0)) * (mean - self.global_mean)
        max_n = max(1.0, float(n.max()))
        popularity = np.log1p(n) / np.log1p(max_n)
        candidates["score"] = np.clip(smoothed_mean + 0.25 * genre_score.to_numpy() + 0.05 * popularity, 0.5, 5.0)
        candidates = candidates.sort_values(
            ["score", "n_ratings", "movieId"], ascending=[False, False, True], kind="stable"
        ).head(top_n)
        strategy = "genre_popularity_cold_start" if genre_preference else "popularity_cold_start"
        return {
            "strategy": strategy,
            "ignored_movie_ids": ignored,
            "recommendations": [
                {"movieId": int(row.movieId), "title": str(row.title), "score": round(float(row.score), 4)}
                for row in candidates.itertuples(index=False)
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", nargs="*", default=[], metavar="MOVIE_ID:RATING")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    ratings = {}
    for token in args.ratings:
        movie_id, rating = token.split(":", maxsplit=1)
        ratings[int(movie_id)] = float(rating)
    import json

    print(json.dumps(ColdStartRecommender().recommend(ratings, args.top_n), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
