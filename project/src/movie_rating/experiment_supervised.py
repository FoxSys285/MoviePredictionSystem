"""Algorithms and evaluation metrics for regression and classification experiments."""

from __future__ import annotations

from time import perf_counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Regression: Linear Regression, XGBoost, Funk SVD, and Neural CF.
from sklearn.linear_model import LinearRegression

from movie_rating.models.factorization import SVD
from movie_rating.models.neural import NeuralCF
from movie_rating.models.regression import StreamingLinearRegression, XGBoostRegression
from movie_rating.experiment_tracking import (
    DEFAULT_TRACKING_DIRECTORY, save_evaluation, save_search_run,
)

# Classification: predict dislike, neutral, or like from a user's rating.
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
    make_scorer,
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


def validate_ratings(ratings) -> np.ndarray:
    """Return finite, nonempty one-dimensional ratings in the range [0.5, 5.0]."""
    values = np.asarray(ratings, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Ratings must be a nonempty one-dimensional array.")
    if not np.isfinite(values).all() or ((values < 0.5) | (values > 5.0)).any():
        raise ValueError("Ratings must be finite values between 0.5 and 5.0.")
    return values


def rating_labels(ratings) -> np.ndarray:
    """Encode ratings: 0 = dislike (<2.5), 1 = neutral ([2.5, 3.5]), 2 = like (>3.5)."""
    values = validate_ratings(ratings)
    return np.where(values < 2.5, 0, np.where(values <= 3.5, 1, 2)).astype(np.int64)


def _rating_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, np.clip(y_pred, 0.5, 5.0))


def _rating_mae(y_true, y_pred):
    return mean_absolute_error(y_true, np.clip(y_pred, 0.5, 5.0))


def grid_search_model(
    model,
    X_train,
    y_train,
    param_grid: dict | list[dict],
    task: str = "regression",
    n_jobs: int = 1,
    log_directory: str | Path | None = DEFAULT_TRACKING_DIRECTORY,
    run_metadata: dict | None = None,
) -> GridSearchCV:
    """Tune parameters with five-fold CV on the outer training set only.

    Pass an unfitted estimator or pipeline. Pipeline parameter names use the
    step prefix, for example {"model__C": [0.1, 1.0, 10.0]} for logistic
    regression. An empty grid evaluates the model's current configuration.
    Scaling and any learned feature generation must be fitted inside each fold.

    With cv=5, sklearn uses KFold for regression and StratifiedKFold for
    classification, both without shuffling. These are not forward-only temporal
    folds. Outer validation and test sets remain separate from this search.

    Select parameters using mean CV RMSE (minimize) or macro F1 (maximize), then refit
    the best configuration on all training rows. Access best_estimator_,
    best_params_, best_score_, and cv_results_ on the returned search object.
    RMSE/MAE scores in cv_results_ are negative because sklearn maximizes scores.
    Classification targets must be labels from rating_labels(): 0, 1, and 2.
    Regression predictions are clipped to [0.5, 5.0] before CV scoring.
    Save results to CSV by default; set log_directory=None to disable tracking.
    Supply dataset/split/feature identifiers and seeds through run_metadata.
    """
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'.")
    if not hasattr(X_train, "shape"):
        X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if len(X_train.shape) != 2 or y_train.ndim != 1:
        raise ValueError("Features must be two-dimensional and targets one-dimensional.")
    if X_train.shape[0] != len(y_train) or len(y_train) < 5:
        raise ValueError("Features and targets must have matching lengths of at least 5.")
    if not np.isfinite(y_train).all():
        raise ValueError("Targets must contain only finite values.")

    if task == "regression":
        validate_ratings(y_train)
        scoring = {
            "rmse": make_scorer(_rating_rmse, greater_is_better=False),
            "mae": make_scorer(_rating_mae, greater_is_better=False),
        }
        refit_metric = "rmse"
    else:
        if not np.isin(y_train, [0, 1, 2]).all():
            raise ValueError("Classification targets must be labels 0, 1, and 2 from rating_labels().")
        labels, counts = np.unique(y_train, return_counts=True)
        if len(labels) != 3 or counts.min() < 5:
            raise ValueError("Five-fold classification requires at least 5 samples in each of 3 labels.")
        scoring = {
            "accuracy": "accuracy",
            "f1": make_scorer(f1_score, average="macro", labels=[0, 1, 2], zero_division=0),
            "auc_roc": "roc_auc_ovr",
        }
        refit_metric = "f1"

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=5,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=False,
    )
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:12]
    started = perf_counter()
    try:
        search.fit(X_train, y_train)
    except Exception as error:
        if log_directory is not None:
            try:
                save_search_run(
                    search, directory=log_directory, run_id=run_id,
                    started_at=started_at.isoformat(), task=task, train_rows=len(y_train),
                    fit_seconds=perf_counter() - started, metadata=run_metadata, error=error,
                )
            except Exception as log_error:
                error.add_note(f"Could not save failed run results: {log_error}")
        raise
    if log_directory is not None:
        search.run_directory_ = save_search_run(
            search, directory=log_directory, run_id=run_id,
            started_at=started_at.isoformat(), task=task, train_rows=len(y_train),
            fit_seconds=perf_counter() - started, metadata=run_metadata,
        )
        search.run_id_ = run_id
    return search


def evaluate_model(model, X, ratings, task: str = "regression", split: str = "validation") -> dict:
    """Evaluate a fitted model on held-out ratings without fitting any component.

    Always pass original ratings, including for classification. Return predictions
    and metrics; automatically log metrics when model is a tracked GridSearchCV.
    Multi-class AUC uses probabilities, macro averaging, and one-vs-rest. If the
    held-out set lacks a class, AUC is undefined and reported as None with a reason.
    """
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'.")
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    truth = validate_ratings(ratings)
    if not hasattr(X, "shape"):
        X = np.asarray(X)
    if len(X.shape) != 2 or X.shape[0] != len(truth):
        raise ValueError("Features must be two-dimensional and match the number of ratings.")
    started = perf_counter()
    predicted = np.asarray(model.predict(X))
    if predicted.shape != truth.shape or not np.isfinite(predicted).all():
        raise ValueError("Predictions must be finite and match the number of ratings.")
    result = {"task": task, "split": split, "rows": len(truth)}
    if task == "regression":
        predicted = np.clip(predicted, 0.5, 5.0)
        metrics = {"rmse": float(root_mean_squared_error(truth, predicted)),
                   "mae": float(mean_absolute_error(truth, predicted))}
    else:
        truth = rating_labels(truth)
        if not np.isin(predicted, [0, 1, 2]).all():
            raise ValueError("Predicted classes must be 0, 1, or 2.")
        classes = np.asarray(model.classes_)
        if not np.array_equal(np.sort(classes), [0, 1, 2]):
            raise ValueError("The classifier must be trained on all three classes.")
        probabilities = np.asarray(model.predict_proba(X))[:, np.argsort(classes)]
        metrics = {
            "accuracy": float(accuracy_score(truth, predicted)),
            "f1": float(f1_score(truth, predicted, average="macro", labels=[0, 1, 2],
                                  zero_division=0)),
            "auc_roc": None,
        }
        result["auc_roc_reason"] = ""
        if len(np.unique(truth)) == 3:
            metrics["auc_roc"] = float(roc_auc_score(
                truth, probabilities, labels=[0, 1, 2], multi_class="ovr", average="macro",
            ))
        else:
            result["auc_roc_reason"] = "AUC-ROC is undefined because a held-out class is missing."
        result["probabilities"] = probabilities
        result["class_names"] = {0: "dislike", 1: "neutral", 2: "like"}
    result.update({"metrics": metrics, "predictions": predicted,
                   "inference_seconds": perf_counter() - started})
    if hasattr(model, "run_directory_"):
        save_evaluation(model.run_directory_, result)
    return result
