"""Cached prepared MovieLens data; no dense user-item matrix."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from movie_rating.data.temporal import PERSONALIZED_PROTOCOL, user_temporal_split
from movie_rating.memory import check_memory


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


def parquet_fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def split_ratings(frame):
    """Exact rounded 70/15/15 IID rating protocol, shared by every group."""
    if len(frame) < 40:
        raise ValueError("Cần ít nhất 40 ratings để chia dữ liệu và chạy 3-Fold.")
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

    def catalog_statistics(self, movies=None):
        signature = self.current_signature()
        if self.signature != signature:
            # Exact counts without concatenating 25M rows just to render the catalog.
            users, items = set(), set()
            count, total, low, high = 0, 0.0, float("inf"), float("-inf")
            for path in self.paths():
                for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=250_000, columns=["userId", "movieId", "rating"]
                ):
                    frame = batch.to_pandas()
                    users.update(frame.userId.unique())
                    items.update(frame.movieId.unique())
                    values = frame.rating.to_numpy()
                    count += len(values)
                    total += values.sum(dtype=np.float64)
                    low, high = min(low, float(values.min())), max(high, float(values.max()))
            if movies is None:
                movies = pd.read_parquet(self.directory / "movies.parquet")
            genres = {
                g
                for value in movies.genres.fillna("")
                for g in value.split("|")
                if g and g != "(no genres listed)"
            }
            self.full_statistics = {
                "ratings": count,
                "users": len(users),
                "movies": len(items),
                "catalog_movies": len(movies),
                "genres": len(genres),
                "rating_min": low,
                "rating_max": high,
                "rating_mean": total / count,
                "sparsity": 1 - count / max(1, len(users) * len(items)),
            }
            self.signature = signature
        return self.full_statistics

    def personalized_directory(self):
        """Migrate legacy splits once to a separate cache; original artifacts stay intact."""
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") == PERSONALIZED_PROTOCOL:
            return self.directory
        digest = hashlib.sha256(repr(self.current_signature()).encode()).hexdigest()[:12]
        destination = self.directory.parent / f"data-personalized-v2-{digest}"
        if (destination / "manifest.json").exists():
            return destination
        check_memory()
        staging_root = Path(os.environ.get("MOVIE_SCRATCH_DIR", self.directory.parent))
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = tempfile.mkdtemp(prefix=".personalized-", dir=staging_root)
        staging = Path(staging)
        try:
            frames = [pd.read_parquet(p) for p in self.paths()]
            ratings = pd.concat(frames, ignore_index=True)
            del frames
            splits, exclusions = user_temporal_split(ratings)
            del ratings
            manifest = {
                **manifest,
                "protocol": PERSONALIZED_PROTOCOL,
                "source_prepared_signature": digest,
                "exclusions": exclusions,
                "splits": {name: {"ratings": len(frame)} for name, frame in splits.items()},
            }
            for name in ("train", "validation", "test"):
                frame = splits.pop(name)
                frame.to_parquet(staging / f"{name}.parquet", index=False)
                del frame
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(staging, destination)
        finally:
            # Only the explicitly created staging directory can be removed.
            if staging.exists() and staging.resolve().parent == staging_root.resolve():
                shutil.rmtree(staging)
        return destination

    def load(self, request):
        supervised = request.group in ("rating", "classification")
        key = (self.current_signature(), request.use_all_data, request.max_users, supervised)
        if key == self.sample_key:
            return self.sample
        movies = pd.read_parquet(self.directory / "movies.parquet")
        exclusions = None
        if request.use_all_data:
            directory = self.personalized_directory() if supervised else self.directory
            splits = {
                name: pd.read_parquet(directory / f"{name}.parquet")
                for name in ("train", "validation", "test")
            }
            manifest_data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            exclusions = manifest_data.get("exclusions")
            # Statistics include only eligible ratings, not excluded histories.
            sample_statistics = ExperimentData(directory).catalog_statistics(movies)
            ratings_count = sample_statistics["ratings"]
            source = "All ratings in the prepared MovieLens dataset"
            protocol = manifest_data.get("protocol", "global_temporal_70_15_15_v1")
            selected_users = sample_statistics["users"]
            sample_sha256 = parquet_fingerprint(
                [directory / f"{s}.parquet" for s in ("train", "validation", "test")]
            )
        else:
            user_set = set()
            for path in self.paths():
                for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=250_000, columns=["userId"]
                ):
                    user_set.update(np.unique(batch.column(0).to_numpy()))
            users = np.asarray(sorted(user_set))
            del user_set
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
            ratings = ratings.sort_values(["userId", "movieId"], kind="stable").reset_index(
                drop=True
            )
            if supervised:
                splits, exclusions = user_temporal_split(ratings)
            else:
                splits = split_ratings(ratings)
            source = "MovieLens: complete histories of sampled prepared users"
            protocol = PERSONALIZED_PROTOCOL if supervised else "iid_rating_70_15_15_v3"
            if supervised and exclusions["excluded_ratings"]:
                ratings = pd.concat(splits.values(), ignore_index=True)
            selected_users = int(ratings.userId.nunique())
            ratings_count = len(ratings)
            sample_statistics = dataset_statistics(ratings, movies)
            sample_sha256 = hashlib.sha256(
                pd.util.hash_pandas_object(ratings, index=False).values.tobytes()
            ).hexdigest()
        manifest = self.directory / "manifest.json"
        provenance = {
            "exclusions": exclusions,
            "source": source,
            "protocol": protocol,
            "data_seed": 42,
            "sampled_ratings": ratings_count,
            "sampled_users": selected_users,
            "prepared": self.catalog_statistics(),
            "sample": sample_statistics,
            "sample_sha256": sample_sha256,
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
