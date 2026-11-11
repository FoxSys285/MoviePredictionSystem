"""Cached prepared MovieLens data; no dense user-item matrix."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


def dataset_statistics(ratings, movies):
    users, items = ratings.userId.nunique(), ratings.movieId.nunique()
    genres = {
        g for s in movies.genres.fillna("") for g in s.split("|") if g and g != "(no genres listed)"
    }
    return {
        "ratings": len(ratings),
        "users": int(users),
        "movies": int(items),
        "catalog_movies": len(movies),
        "genres": len(genres),
        "rating_min": float(ratings.rating.min()),
        "rating_max": float(ratings.rating.max()),
        "rating_mean": float(ratings.rating.mean()),
        "sparsity": 1 - len(ratings) / max(1, users * items),
    }


def split_ratings(frame):
    """Exact rounded 70/15/15 IID rating protocol, shared by every group."""
    if len(frame) < 40:
        raise ValueError("Cần ít nhất 40 ratings để chia dữ liệu và chạy 5-Fold.")
    indices = np.arange(len(frame))
    heldout_size = max(1, round(len(frame) * 0.15))
    train_val, test = train_test_split(indices, test_size=heldout_size, random_state=42)
    train, validation = train_test_split(train_val, test_size=heldout_size, random_state=42)
    return {
        name: frame.iloc[idx].reset_index(drop=True)
        for name, idx in [("train", train), ("validation", validation), ("test", test)]
    }


class ExperimentData:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.signature = None
        self.full_statistics = None
        self.sample_key = None
        self.sample = None

    def paths(self):
        return [self.directory / f"{s}.parquet" for s in ("train", "validation", "test")]

    def current_signature(self):
        paths = [*self.paths(), self.directory / "movies.parquet", self.directory / "manifest.json"]
        return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.exists())

    def catalog_statistics(self):
        signature = self.current_signature()
        if self.signature != signature:
            # Only three compact numeric columns, once per prepared data revision.
            ratings = pd.concat(
                [pd.read_parquet(p, columns=["userId", "movieId", "rating"]) for p in self.paths()],
                ignore_index=True,
            )
            movies = pd.read_parquet(self.directory / "movies.parquet")
            self.full_statistics = dataset_statistics(ratings, movies)
            self.signature = signature
        return self.full_statistics

    def load(self, request):
        key = (self.current_signature(), request.max_users)
        if key == self.sample_key:
            return self.sample
        users = np.unique(
            np.concatenate(
                [pq.read_table(p, columns=["userId"]).column(0).to_numpy() for p in self.paths()]
            )
        )
        selected = np.sort(
            np.random.default_rng(42).choice(
                users, min(request.max_users, len(users)), replace=False
            )
        )
        ratings = pd.concat(
            [
                pd.read_parquet(p, filters=[("userId", "in", selected.tolist())])
                for p in self.paths()
            ],
            ignore_index=True,
        )
        ratings = ratings.sort_values(["userId", "movieId"], kind="stable").reset_index(drop=True)
        movies = pd.read_parquet(self.directory / "movies.parquet")
        splits = split_ratings(ratings)
        manifest = self.directory / "manifest.json"
        provenance = {
            "source": "MovieLens: complete histories of sampled prepared users",
            "protocol": "iid_rating_70_15_15_v3",
            "data_seed": 42,
            "sampled_ratings": len(ratings),
            "sampled_users": len(selected),
            "prepared": self.catalog_statistics(),
            "sample": dataset_statistics(ratings, movies),
            "sample_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(ratings, index=False).values.tobytes()
            ).hexdigest(),
            "data_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
            if manifest.exists()
            else None,
            "splits": {
                name: {
                    "ratings": len(f),
                    "users": int(f.userId.nunique()),
                    "movies": int(f.movieId.nunique()),
                }
                for name, f in splits.items()
            },
        }
        self.sample_key, self.sample = key, (splits, movies, provenance)
        return self.sample
