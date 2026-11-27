"""User-wise temporal boundaries, including ties, without Python groups of frames."""

import numpy as np
import pandas as pd

PERSONALIZED_PROTOCOL = "per_user_temporal_70_15_15_v2"


def _ordered_groups(frame):
    if "timestamp" not in frame:
        raise ValueError("Dữ liệu cần timestamp để chia theo thời gian.")
    order = np.lexsort(
        (frame.movieId.to_numpy(), frame.timestamp.to_numpy(), frame.userId.to_numpy())
    )
    users = frame.userId.to_numpy()[order]
    starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    return order, starts, counts


def user_temporal_split(frame, train_fraction=0.7, validation_fraction=0.15, minimum_ratings=10):
    """Return splits and exclusion counts. Equal timestamps never cross a boundary."""
    if not 0 < train_fraction < train_fraction + validation_fraction < 1:
        raise ValueError("Invalid temporal split fractions.")
    if not len(frame):
        raise ValueError("Không có rating để chia theo user.")
    order, starts, counts = _ordered_groups(frame)
    times = frame.timestamp.to_numpy()[order]
    first = times[starts + np.maximum(0, (counts * train_fraction).astype(int) - 1)]
    second = times[
        starts + np.maximum(0, (counts * (train_fraction + validation_fraction)).astype(int) - 1)
    ]
    eligible = (counts >= minimum_ratings) & (first < second)
    eligible &= second < times[starts + counts - 1]
    if not eligible.any():
        raise ValueError("Không đủ user có lịch sử và ba khoảng thời gian tách biệt.")
    codes = np.full(len(frame), 2, dtype=np.uint8)
    codes[times <= np.repeat(second, counts)] = 1
    codes[times <= np.repeat(first, counts)] = 0
    codes[~np.repeat(eligible, counts)] = 255
    report = {
        "input_users": len(counts),
        "input_ratings": len(frame),
        "excluded_users": int((~eligible).sum()),
        "excluded_ratings": int(counts[~eligible].sum()),
        "minimum_ratings": minimum_ratings,
        "reason": "short history or inseparable timestamp boundaries",
    }
    splits = {
        name: frame.iloc[order[codes == code]].reset_index(drop=True)
        for code, name in enumerate(("train", "validation", "test"))
    }
    return splits, report


def split_by_time(
    ratings: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    minimum_ratings: int = 3,
) -> dict[str, pd.DataFrame]:
    """Split each user's history chronologically, approximately 70/15/15 by default.

    Round cumulative boundaries: 10 ratings with distinct timestamps yield 7/2/1.
    Keep equal timestamps in the same split, so actual proportions may differ.
    Exclude users with short histories or inseparable timestamp boundaries.
    Store exclusion statistics in attrs["split_report"] on each output frame.
    """
    required = {"userId", "movieId", "timestamp"}
    if not required.issubset(ratings.columns):
        raise ValueError("Ratings must contain userId, movieId, and timestamp columns.")
    if ratings.empty:
        raise ValueError("No ratings are available to split.")
    if not 0 < train_fraction < train_fraction + validation_fraction < 1:
        raise ValueError("Train and validation fractions must be positive and sum to less than 1.")
    if not isinstance(minimum_ratings, int) or minimum_ratings < 3:
        raise ValueError("minimum_ratings must be an integer greater than or equal to 3.")
    if ratings[list(required)].isna().any().any():
        raise ValueError("userId, movieId, and timestamp must not contain missing values.")
    if not pd.api.types.is_numeric_dtype(ratings["timestamp"]):
        raise ValueError("timestamp must contain numeric Unix timestamps in seconds.")
    if not np.isfinite(ratings["timestamp"].to_numpy()).all():
        raise ValueError("timestamp must contain only finite values.")

    order = np.lexsort((ratings.movieId.to_numpy(), ratings.timestamp.to_numpy(),
                        ratings.userId.to_numpy()))
    users = ratings.userId.to_numpy()[order]
    times = ratings.timestamp.to_numpy()[order]
    starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(order)])

    # Reserve at least one rating per split before handling equal timestamps.
    train_counts = np.floor(counts * train_fraction + 0.5).astype(int)
    train_counts = np.minimum(np.maximum(train_counts, 1), np.maximum(counts - 2, 1))
    train_valid_counts = np.floor(counts * (train_fraction + validation_fraction) + 0.5).astype(int)
    train_valid_counts = np.minimum(
        np.maximum(train_valid_counts, train_counts + 1), np.maximum(counts - 1, 1)
    )
    train_cut = times[starts + np.minimum(train_counts, counts) - 1]
    valid_cut = times[starts + np.minimum(train_valid_counts, counts) - 1]
    eligible = ((counts >= minimum_ratings) & (train_cut < valid_cut)
                & (valid_cut < times[starts + counts - 1]))
    if not eligible.any():
        raise ValueError("No eligible users have enough history for three distinct time periods.")

    codes = np.full(len(order), 2, dtype=np.uint8)
    codes[times <= np.repeat(valid_cut, counts)] = 1
    codes[times <= np.repeat(train_cut, counts)] = 0
    codes[~np.repeat(eligible, counts)] = 255
    report = {
        "protocol": "per_user_temporal",
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "minimum_ratings": minimum_ratings,
        "input_users": int(len(counts)),
        "included_users": int(eligible.sum()),
        "excluded_users": int((~eligible).sum()),
        "excluded_ratings": int(counts[~eligible].sum()),
        "reason": "short history or inseparable timestamp boundaries",
    }
    splits = {}
    for code, name in enumerate(("train", "validation", "test")):
        part = ratings.iloc[order[codes == code]].reset_index(drop=True)
        part.attrs["split_report"] = dict(report)
        splits[name] = part
    return splits
