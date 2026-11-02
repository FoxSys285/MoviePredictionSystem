import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from movie_rating.settings import Settings

RATING_DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}


def load_ratings(settings: Settings) -> pd.DataFrame:
    source = settings.dataset_dir / "ratings.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}. Add MovieLens files or change data.directory.")
    frames = []
    for chunk in pd.read_csv(
        source,
        usecols=list(RATING_DTYPES),
        dtype=RATING_DTYPES,
        chunksize=settings.data.chunksize,
    ):
        if chunk.isna().any().any():
            raise ValueError("ratings.csv contains missing values.")
        if ((chunk.userId <= 0) | (chunk.movieId <= 0) | (chunk.timestamp < 0)).any():
            raise ValueError("IDs must be positive and timestamps must be nonnegative.")
        if not chunk.rating.isin(np.arange(0.5, 5.1, 0.5)).all():
            raise ValueError("MovieLens ratings must be 0.5, 1.0, ..., 5.0.")
        # An integer hash avoids retaining only the first users/rows in the CSV.
        hashed = (chunk.userId.astype("int64") ^ settings.experiment.seed) * 2654435761
        selected = (hashed % 2**32) / 2**32 < settings.data.user_sample_fraction
        frames.append(chunk.loc[selected])
    ratings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(ratings) < 10:
        raise ValueError("Too few sampled ratings; increase user_sample_fraction.")
    if ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate user/movie ratings found; inspect the input before training.")
    return ratings


def chronological_split(ratings: pd.DataFrame, train_fraction: float, validation_fraction: float):
    ordered = ratings.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(ordered)
    if n < 3 or not (0 < train_fraction < train_fraction + validation_fraction < 1):
        raise ValueError("Invalid split fractions or insufficient ratings.")
    # Keep equal timestamps together at the boundary.
    train_time = int(ordered.iloc[max(0, int(n * train_fraction) - 1)].timestamp)
    val_time = int(
        ordered.iloc[max(0, int(n * (train_fraction + validation_fraction)) - 1)].timestamp
    )
    train = ordered.loc[ordered.timestamp <= train_time].copy()
    validation = ordered.loc[(ordered.timestamp > train_time) & (ordered.timestamp <= val_time)]
    test = ordered.loc[ordered.timestamp > val_time]
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("Timestamp boundaries produce an empty split; use more varied data.")
    return {"train": train, "validation": validation.copy(), "test": test.copy()}


def prepare(settings: Settings) -> dict:
    destination = settings.artifacts_dir / "data"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "Prepared data already exists. Use a new artifacts_directory for another experiment."
        )
    ratings = load_ratings(settings)
    splits = chronological_split(
        ratings, settings.data.train_fraction, settings.data.validation_fraction
    )
    movies_path = settings.dataset_dir / "movies.csv"
    if not movies_path.is_file():
        raise FileNotFoundError(f"Missing {movies_path}.")
    movies = pd.read_csv(movies_path, dtype={"movieId": "int32"})
    if not {"movieId", "title", "genres"}.issubset(movies.columns):
        raise ValueError("movies.csv must contain movieId, title, genres.")
    if movies.isna().any().any() or movies.movieId.duplicated().any():
        raise ValueError("movies.csv contains missing values or duplicate IDs.")
    if not ratings.movieId.isin(movies.movieId).all():
        raise ValueError("Some ratings refer to IDs missing from movies.csv.")
    source = settings.dataset_dir / "ratings.csv"
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "settings": settings.model_dump(),
        "splits": {
            name: {
                "ratings": len(frame),
                "users": int(frame.userId.nunique()),
                "movies": int(frame.movieId.nunique()),
                "min_timestamp": int(frame.timestamp.min()),
                "max_timestamp": int(frame.timestamp.max()),
            }
            for name, frame in splits.items()
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(destination / f"{name}.parquet", index=False)
    movies.to_parquet(destination / "movies.parquet", index=False)
    # Catalogue popularity uses training ratings only, preserving held-out data.
    splits["train"].groupby("movieId").rating.agg(
        mean_rating="mean", rating_count="count"
    ).reset_index().to_parquet(destination / "movie_stats.parquet", index=False)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
