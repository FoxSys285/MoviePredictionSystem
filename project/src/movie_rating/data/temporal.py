"""User-wise temporal boundaries, including ties, without Python groups of frames."""

import numpy as np

PERSONALIZED_PROTOCOL = "per_user_temporal_70_15_15_v2"


def _ordered_groups(frame):
    if "timestamp" not in frame:
        raise ValueError("Dữ liệu thực nghiệm cần timestamp để chia theo thời gian.")
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


def temporal_folds(frame, n_splits=3):
    """Expanding history per user: past only, shared users, held-out timestamp blocks."""
    order, starts, counts = _ordered_groups(frame)
    times = frame.timestamp.to_numpy()[order]
    ends = starts + counts
    for fold in range(n_splits):
        cut = times[starts + np.maximum(0, (counts * (fold + 1) / (n_splits + 1)).astype(int) - 1)]
        upper = times[
            starts + np.maximum(0, (counts * (fold + 2) / (n_splits + 1)).astype(int) - 1)
        ]
        if fold == n_splits - 1:
            upper = times[ends - 1]
        # Keep expanding training histories even when a user's next block is all ties.
        # Such a user simply contributes no held-out rows in this particular fold.
        eligible = counts >= n_splits + 1
        valid = np.repeat(eligible, counts)
        fit_mask = (times <= np.repeat(cut, counts)) & valid
        held_mask = (times <= np.repeat(upper, counts)) & ~fit_mask & valid
        if not fit_mask.any() or not held_mask.any():
            raise ValueError("Không đủ lịch sử/timestamp khác nhau cho 3 lượt CV thời gian.")
        yield order[fit_mask], order[held_mask]
