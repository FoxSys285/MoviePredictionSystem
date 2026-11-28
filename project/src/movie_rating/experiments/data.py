"""Shared per-user 70/15/15 splits and five expanding time folds on Train."""

from pathlib import Path

import numpy as np
import pandas as pd

from movie_rating.data.temporal import split_by_time

CV_FOLDS = 5
SPLIT_NAMES = ("train", "validation", "test")
RATING_DTYPES = {
    "userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64",
}


def validate_ratings(ratings: pd.DataFrame) -> None:
    """Reject invalid rows before splitting or fitting."""
    if not isinstance(ratings, pd.DataFrame) or not set(RATING_DTYPES).issubset(ratings):
        raise ValueError("Ratings require userId, movieId, rating, and timestamp columns.")
    if ratings.empty:
        raise ValueError("Ratings must not be empty.")
    for column in RATING_DTYPES:
        values = ratings[column].to_numpy()
        if not pd.api.types.is_numeric_dtype(values) or not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite numeric values.")
    for column in ("userId", "movieId"):
        values = ratings[column].to_numpy()
        if (values <= 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain positive integer IDs.")
    if not ratings.rating.between(0.5, 5.0).all():
        raise ValueError("Ratings must be within the 0.5-5 star scale.")


def split_data(ratings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split each user's history 70/15/15, keeping equal timestamps together.

    With 100 distinct timestamps the counts are exactly 70/15/15. Short histories
    and users without three separable time periods are reported as exclusions in
    each frame's attrs['split_report']; tied timestamps can change actual ratios.
    """
    validate_ratings(ratings)
    if ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("Each user/movie rating pair must be unique before splitting.")
    splits = split_by_time(ratings, train_fraction=0.70, validation_fraction=0.15)
    counts = {name: len(frame) for name, frame in splits.items()}
    total = sum(counts.values())
    for frame in splits.values():
        frame.attrs["split_report"].update({
            "target_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "ratings": counts,
            "actual_ratios": {name: count / total for name, count in counts.items()},
        })
    return splits


def load_data(directory: str | Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load ratings and static movie metadata, then create the shared outer splits."""
    directory = Path(directory)
    ratings = pd.read_csv(
        directory / "ratings.csv", usecols=list(RATING_DTYPES), dtype=RATING_DTYPES,
    )
    movies = pd.read_csv(directory / "movies.csv", dtype={"movieId": "int32"})
    return split_data(ratings), movies


def make_cv_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Five past-to-future folds per user, made only from the outer Train split.

    Use six chronological blocks. Each fold expands the fit history and holds
    out its next block. Equal timestamps remain together; users with no held-out
    block in a fold are omitted from that fold's validation rows.
    """
    validate_ratings(train)
    order = np.lexsort((train.movieId.to_numpy(), train.timestamp.to_numpy(),
                        train.userId.to_numpy()))
    users = train.userId.to_numpy()[order]
    times = train.timestamp.to_numpy()[order]
    starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    eligible = np.repeat(counts >= CV_FOLDS + 1, counts)
    folds = []
    for fold in range(CV_FOLDS):
        cut_index = np.maximum(0, counts * (fold + 1) // (CV_FOLDS + 1) - 1)
        end_index = np.maximum(0, counts * (fold + 2) // (CV_FOLDS + 1) - 1)
        cut = np.repeat(times[starts + cut_index], counts)
        end = np.repeat(times[starts + end_index], counts)
        fit = (times <= cut) & eligible
        held_out = (times > cut) & (times <= end) & eligible
        if not fit.any() or not held_out.any():
            raise ValueError(f"Train has insufficient distinct history for CV fold {fold + 1}.")
        folds.append((order[fit], order[held_out]))
    return folds


def validate_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Require disjoint user/movie pairs and strictly ordered user histories."""
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError("Provide exactly train, validation, and test splits.")
    summaries = {}
    for name in SPLIT_NAMES:
        frame = splits[name]
        validate_ratings(frame)
        if frame.duplicated(["userId", "movieId"]).any():
            raise ValueError(f"Duplicate user/movie pairs in {name}.")
        summaries[name] = frame.groupby("userId").timestamp.agg(["min", "max"])
    train, validation, test = (summaries[name] for name in SPLIT_NAMES)
    if not train.index.equals(validation.index) or not train.index.equals(test.index):
        raise ValueError("Every user must have Train, Validation, and Test history.")
    if not ((train["max"] < validation["min"]) & (validation["max"] < test["min"])).all():
        raise ValueError("User histories must follow Train < Validation < Test in time.")
    # Timestamp checks do not detect the same movie rated again in another split.
    pairs = [pd.MultiIndex.from_frame(splits[name][["userId", "movieId"]])
             for name in SPLIT_NAMES]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        if pairs[left].intersection(pairs[right]).size:
            raise ValueError("A user/movie pair occurs in multiple outer splits.")
