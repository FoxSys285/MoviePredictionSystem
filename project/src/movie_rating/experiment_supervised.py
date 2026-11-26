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


def candidates(algorithm, request):
    """Return parameter dictionaries for every supervised algorithm.

    Fixed requests still run five-fold GridSearchCV with one configuration.
    Tuning requests compare a bounded set of additional configurations.
    """
    p = request.parameters
    base = {
        "linear": {"fit_intercept": True},
        "xgboost": {"n_estimators": p.n_estimators, "max_depth": p.max_depth,
                    "learning_rate": p.xgb_learning_rate, "n_jobs": p.xgb_n_jobs},
        "svd": {"factors": p.factors, "epochs": request.epochs,
                "learning_rate": p.learning_rate, "regularization": p.regularization},
        "ncf": {"factors": p.factors, "epochs": request.epochs,
                "hidden_size": p.hidden_size, "learning_rate": p.ncf_learning_rate,
                "regularization": p.regularization, "dropout": p.dropout,
                "batch_size": p.batch_size, "patience": request.patience},
        "logistic": {"C": p.c, "max_iter": p.max_iter},
        "random_forest": {"n_estimators": p.forest_estimators, "max_depth": p.forest_depth,
                          "min_samples_leaf": p.min_samples_leaf,
                          "max_samples": p.forest_max_samples, "n_jobs": 4},
    }
    if algorithm not in base:
        raise ValueError(f"Unsupported supervised algorithm: {algorithm}")
    options = [base[algorithm]]
    if request.tune:
        alternatives = {
            "linear": {"fit_intercept": False},
            "xgboost": {"max_depth": max(1, p.max_depth - 2) if p.max_depth > 1 else 3},
            "svd": {"factors": max(2, p.factors // 2) if p.factors > 2 else 4},
            "ncf": {"factors": max(2, p.factors // 2) if p.factors > 2 else 4},
            "logistic": {"C": p.c * 0.1},
            "random_forest": {"max_depth": max(1, p.forest_depth - 2) if p.forest_depth > 1 else 3},
        }
        alternative = {**options[0], **alternatives[algorithm]}
        if alternative != options[0]:
            options.append(alternative)
    return options


def _supervised_variants(algorithm, request):
    """Keep factor IDs native; optionally reduce tabular features inside each fold."""
    if algorithm in {"svd", "ncf"}:
        return [(None, None)]
    if request.preset_id is not None:
        from movie_rating.experiment_config import preset_reduction_plan

        plan = preset_reduction_plan(request.group, algorithm, request.reduction_choice)
        return [(plan["method"], plan["components"])] if plan else [(None, None)]
    variants = [(None, None)] if request.reduction_mode in {"off", "compare"} else []
    if request.reduction_mode != "off":
        variants.extend((method, k) for method in request.reduction_methods for k in request.components)
    return variants


def _cv_report(search):
    """Present the selected configuration's fold scores in original metric units."""
    results, best = search.cv_results_, search.best_index_
    metrics = list(search.scoring)
    summary = {}
    folds = [{"fold": fold + 1} for fold in range(5)]
    for metric in metrics:
        sign = -1 if metric in {"rmse", "mae"} else 1
        summary[metric] = {
            "mean": float(sign * results[f"mean_test_{metric}"][best]),
            "std": float(results[f"std_test_{metric}"][best]),
        }
        for fold in range(5):
            folds[fold][metric] = float(sign * results[f"split{fold}_test_{metric}"][best])
    return {"metrics": summary, "folds": folds}


def supervised(splits, movies, request, update, progress_store=None):
    """Run the shared GridSearchCV/evaluation/CSV workflow for all six algorithms.

    Outer splits come from ExperimentData's per-user chronological splitter.
    Parameter selection uses only five-fold CV on Train. All models are evaluated
    on the same untouched Validation/Test sets; no refit on Validation is done.
    Completed algorithm/representation units can be reused when resuming a job.
    """
    import hashlib

    from movie_rating.experiment_estimators import MovieClassifier, MovieRegressor

    if request.group not in {"rating", "classification"}:
        raise ValueError("The supervised workflow supports rating and classification groups.")
    task = "regression" if request.group == "rating" else "classification"
    allowed = ({"linear", "xgboost", "svd", "ncf"} if task == "regression" else
               {"logistic", "random_forest"})
    if not set(request.algorithms).issubset(allowed):
        raise ValueError("The selected algorithms do not belong to this supervised task.")
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"The {name} split must not be empty.")
        validate_ratings(frame.rating)
    # Verify that held-out ratings occur strictly after each user's fit history.
    train_max = train.groupby("userId").timestamp.max()
    valid_min = validation.groupby("userId").timestamp.min()
    valid_max = validation.groupby("userId").timestamp.max()
    test_min = test.groupby("userId").timestamp.min()
    if not (set(train_max.index) == set(valid_min.index) == set(test_min.index)):
        raise ValueError("Each user must have ratings in Train, Validation, and Test.")
    if not (train_max < valid_min.reindex(train_max.index)).all() or not (
        valid_max < test_min.reindex(valid_max.index)
    ).all():
        raise ValueError("Outer splits must follow time order within every user.")

    y_train = train.rating.to_numpy() if task == "regression" else rating_labels(train.rating)
    source_hash = hashlib.sha256()
    for name in ("train", "validation", "test"):
        source_hash.update(name.encode())
        source_hash.update(pd.util.hash_pandas_object(
            splits[name][["userId", "movieId", "rating", "timestamp"]], index=False,
        ).to_numpy().tobytes())
    source_hash.update(pd.util.hash_pandas_object(
        movies[["movieId", "genres"]], index=False,
    ).to_numpy().tobytes())
    fingerprint = source_hash.hexdigest()
    result = {"rows": [], "charts": [], "cv": {}, "best_configs": {},
              "cv_folds": 5, "cv_protocol": "grid_search_cv_5",
              "notes": [
                  "All supervised algorithms use GridSearchCV(cv=5) on Train only.",
                  "Only Linear Regression and Logistic Regression use StandardScaler.",
                  "Outer splits are chronological per user; inner CV is not forward-only.",
                  "Fixed requests evaluate one configuration; enable tuning to compare parameters.",
                  "SVD/NCF keep native user/movie factors; reduction applies to tabular models.",
              ]}
    units = [(a, method, k) for a in request.algorithms
             for method, k in _supervised_variants(a, request)]
    for index, (algorithm, reduction, components) in enumerate(units):
        variant_id = f"{algorithm}__{reduction}_{components}" if reduction else algorithm
        saved = progress_store.get_unit(variant_id, algorithm) if progress_store else {}
        unit = saved.get("final")
        if unit is None:
            def report(message):
                update(None, f"{variant_id} | {message}")

            update(10 + int(80 * index / len(units)),
                   f"{variant_id}: GridSearchCV, 5 folds on Train")
            constructor = MovieRegressor if task == "regression" else MovieClassifier
            model = constructor(algorithm=algorithm, movies=movies, reduction=reduction,
                                components=components or 8, seed=request.seed, report=report)
            log_directory = (progress_store.path.parent / "model-comparison"
                             if progress_store else DEFAULT_TRACKING_DIRECTORY)
            search = grid_search_model(
                model, train, y_train, {"parameters": candidates(algorithm, request)},
                task=task, log_directory=log_directory,
                run_metadata={"algorithm": algorithm, "variant_id": variant_id,
                              "dataset_split_sha256": fingerprint,
                              "split": "per_user_chronological", "seed": request.seed,
                              "features": "fold_train_only_rating_aggregates_and_genres",
                              "reduction": reduction, "components": components},
            )
            unit_rows = []
            for split_name, frame in (("validation", validation), ("test", test)):
                evaluated = evaluate_model(search, frame, frame.rating, task=task, split=split_name)
                row = {
                    "algorithm": algorithm, "base_algorithm": algorithm,
                    "variant_id": variant_id, "name": variant_id, "split": split_name,
                    "fit_scope": "train", "refit_ratings": len(train),
                    "fit_seconds": search.refit_time_,
                    "search_seconds": float(search.cv_results_["mean_fit_time"].sum() * 5),
                    "inference_seconds": evaluated["inference_seconds"],
                    "run_id": search.run_id_, "tracking_directory": str(search.run_directory_),
                    "representation": "reduced_features" if reduction else "raw",
                    "reduction_method": reduction, "components": components,
                    **evaluated["metrics"],
                }
                if task == "classification":
                    row["auc_roc_reason"] = evaluated["auc_roc_reason"]
                unit_rows.append(row)
                report(f"{split_name}: {evaluated['metrics']}")
            unit = {"rows": unit_rows, "cv": _cv_report(search),
                    "best_config": search.best_params_["parameters"]}
            if progress_store:
                progress_store.save_unit(variant_id, algorithm, {"final": unit})
        else:
            update(None, f"{variant_id}: restored completed experiment results")
        result["rows"].extend(unit["rows"])
        result["cv"][variant_id] = unit["cv"]
        result["best_configs"][variant_id] = unit["best_config"]

    valid_rows = [row for row in result["rows"] if row["split"] == "validation"]
    chosen = (min(valid_rows, key=lambda row: row["rmse"]) if task == "regression" else
              max(valid_rows, key=lambda row: row["f1"]))
    result["selected_variant_id"] = chosen["variant_id"]
    result["selection"] = "Parameters selected by mean CV; model comparison uses Validation, never Test."
    if task == "classification":
        result["classification_protocol"] = {
            "labels": {"0": "rating < 2.5", "1": "2.5 <= rating <= 3.5", "2": "rating > 3.5"},
            "f1_average": "macro", "auc_roc": "macro_one_vs_rest",
        }
        result["class_balance"] = {}
        for name, frame in splits.items():
            labels = rating_labels(frame.rating)
            result["class_balance"][name] = {
                "negative": int((labels == 0).sum()), "neutral": int((labels == 1).sum()),
                "positive": int((labels == 2).sum()),
                "positive_fraction": float((labels == 2).mean()),
            }
    update(95, "All selected models evaluated; CV and held-out results saved to CSV")
    return result
