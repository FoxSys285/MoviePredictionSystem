"""Algorithms and evaluation metrics for regression and classification experiments."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

# Regression: Linear Regression, XGBoost, Funk SVD, and Neural CF.
from sklearn.linear_model import LinearRegression

from movie_rating.models.factorization import SVD
from movie_rating.models.neural import NeuralCF
from movie_rating.models.regression import StreamingLinearRegression, XGBoostRegression

# Classification: predict whether a user likes a movie.
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Dimensionality reduction and feature scaling before training.
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Regression metrics: MAE, MSE, RMSE, and R2.
# Classification metrics: accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)


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


def build_scaled_model(algorithm: str, **model_params) -> Pipeline:
    """Build a feature-scaling pipeline for linear or logistic regression.

    Fit the returned pipeline on training features and targets only. Use predict
    (or predict_proba for logistic regression) on validation/test features; the
    fitted scaler is reused without learning from held-out data. Cross-validation
    must fit a fresh pipeline in each fold. Scale feature columns, not targets
    or raw user/movie identifiers.
    """
    if algorithm == "linear":
        model = LinearRegression(**model_params)
    elif algorithm == "logistic":
        params = {"max_iter": 1000, "random_state": 42, **model_params}
        model = LogisticRegression(**params)
    else:
        raise ValueError("Scaled pipelines support only 'linear' and 'logistic'.")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])
