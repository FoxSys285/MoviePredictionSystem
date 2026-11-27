"""Regression algorithms and evaluation for rating prediction experiments."""

from __future__ import annotations

from time import perf_counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid

from sklearn.linear_model import LinearRegression

from movie_rating.models.factorization import SVD
from movie_rating.models.neural import NeuralCF
from movie_rating.models.regression import StreamingLinearRegression, XGBoostRegression
from movie_rating.experiment_tracking import (
    DEFAULT_TRACKING_DIRECTORY, save_evaluation, save_search_run,
)

# Feature scaling before linear regression.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    root_mean_squared_error,
)


def build_scaled_model(algorithm: str = "linear", **model_params) -> Pipeline:
    """Build a feature-scaling pipeline for Linear Regression.

    Fit the returned pipeline on training features and targets only. Use predict
    on validation/test features; the fitted scaler is reused without learning
    from held-out data. Cross-validation
    must fit a fresh pipeline in each fold. Scale feature columns, not targets
    or raw user/movie identifiers.
    """
    if algorithm != "linear":
        raise ValueError("Feature scaling is configured for Linear Regression only.")
    model = LinearRegression(**model_params)
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


def _rating_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, np.clip(y_pred, 0.5, 5.0))


def _rating_mae(y_true, y_pred):
    return mean_absolute_error(y_true, np.clip(y_pred, 0.5, 5.0))


def grid_search_model(
    model,
    X_train,
    y_train,
    param_grid: dict | list[dict],
    n_jobs: int = 1,
    log_directory: str | Path | None = DEFAULT_TRACKING_DIRECTORY,
    run_metadata: dict | None = None,
) -> GridSearchCV:
    """Tune parameters with five-fold CV on the outer training set only.

    Pass an unfitted estimator or pipeline. An empty grid evaluates the model's
    current configuration.
    Scaling and any learned feature generation must be fitted inside each fold.

    With cv=5, sklearn uses KFold without shuffling. These are not forward-only
    temporal folds. Outer validation and test sets remain separate from this search.

    Select parameters using mean CV RMSE (minimize), then refit the best
    configuration on all training rows. Access best_estimator_,
    best_params_, best_score_, and cv_results_ on the returned search object.
    RMSE/MAE scores in cv_results_ are negative because sklearn maximizes scores.
    Regression predictions are clipped to [0.5, 5.0] before CV scoring.
    Save results to CSV by default; set log_directory=None to disable tracking.
    Supply dataset/split/feature identifiers and seeds through run_metadata.
    """
    if not hasattr(X_train, "shape"):
        X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if len(X_train.shape) != 2 or y_train.ndim != 1:
        raise ValueError("Features must be two-dimensional and targets one-dimensional.")
    if X_train.shape[0] != len(y_train) or len(y_train) < 5:
        raise ValueError("Features and targets must have matching lengths of at least 5.")
    if not np.isfinite(y_train).all():
        raise ValueError("Targets must contain only finite values.")

    validate_ratings(y_train)
    scoring = {
        "rmse": make_scorer(_rating_rmse, greater_is_better=False),
        "mae": make_scorer(_rating_mae, greater_is_better=False),
    }
    refit_metric = "rmse"

    # Shared fold context also prefixes nested XGBoost cache/epoch messages.
    report = model.get_params(deep=False).get("report")
    if callable(report) and n_jobs in (None, 1):
        from movie_rating.experiment_logs import FoldProgressReporter

        model = clone(model).set_params(report=FoldProgressReporter(report, len(ParameterGrid(param_grid))))

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
                    started_at=started_at.isoformat(), task="regression", train_rows=len(y_train),
                    fit_seconds=perf_counter() - started, metadata=run_metadata, error=error,
                )
            except Exception as log_error:
                error.add_note(f"Could not save failed run results: {log_error}")
        raise
    if log_directory is not None:
        search.run_directory_ = save_search_run(
            search, directory=log_directory, run_id=run_id,
            started_at=started_at.isoformat(), task="regression", train_rows=len(y_train),
            fit_seconds=perf_counter() - started, metadata=run_metadata,
        )
        search.run_id_ = run_id
    return search


def evaluate_model(model, X, ratings, split: str = "validation") -> dict:
    """Evaluate a fitted model on held-out ratings without fitting any component.

    Return predictions and metrics; automatically log metrics when model is a
    tracked GridSearchCV.
    """
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
    predicted = np.clip(predicted, 0.5, 5.0)
    result = {"task": "regression", "split": split, "rows": len(truth)}
    metrics = {"rmse": float(root_mean_squared_error(truth, predicted)),
               "mae": float(mean_absolute_error(truth, predicted))}
    result.update({"metrics": metrics, "predictions": predicted,
                   "inference_seconds": perf_counter() - started})
    if hasattr(model, "run_directory_"):
        save_evaluation(model.run_directory_, result)
    return result


def candidates(algorithm, request):
    """Return bounded parameter dictionaries for regression algorithms.

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
        }
        alternative = {**options[0], **alternatives[algorithm]}
        if alternative != options[0]:
            options.append(alternative)
    return options


def _supervised_variants(algorithm, request):
    """Compare movie metadata PCA for Linear Regression when selected."""
    method = "pca" if algorithm == "linear" else None
    if method is None or request.reduction_mode == "off":
        return [(None, None)]
    if request.preset_id is not None:
        from movie_rating.experiment_config import preset_reduction_plan

        plan = preset_reduction_plan(request.group, algorithm, request.reduction_choice)
        return [(None, None), (plan["method"], plan["components"])] if plan else [(None, None)]
    if request.reduction_mode == "off" or method not in request.reduction_methods:
        return [(None, None)]
    return [(None, None), *((method, k) for k in request.components)]


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
    """Run GridSearchCV, evaluation, and CSV tracking for rating regressors.

    Outer splits come from ExperimentData's per-user chronological splitter.
    Parameter selection uses only five-fold CV on Train. All models are evaluated
    on the same untouched Validation/Test sets; no refit on Validation is done.
    Completed algorithm/representation units can be reused when resuming a job.
    """
    import hashlib

    from movie_rating.experiment_estimators import MovieRegressor

    if request.group != "rating":
        raise ValueError("The regression workflow only supports rating prediction.")
    allowed = {"linear", "xgboost", "svd", "ncf"}
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

    y_train = train.rating.to_numpy()
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
                  "All regression algorithms use GridSearchCV(cv=5) on Train only.",
                  "Linear Regression uses StandardScaler.",
                  "Outer splits are chronological per user; inner CV is not forward-only.",
                  "Fixed requests evaluate one configuration; enable tuning to compare parameters.",
                  "PCA compresses movie metadata only for Linear Regression comparisons.",
                  "Each reduction is compared with an unreduced baseline from the same feature family.",
              ]}
    from movie_rating.experiment_baseline import global_mean_baseline

    result["baseline"] = global_mean_baseline(splits)
    update(None, f"Global Mean baseline | Train mean={result['baseline']['prediction']:.6f} | {result['baseline']['rows']}")
    units = [(a, method, k) for a in request.algorithms
             for method, k in _supervised_variants(a, request)]
    comparison_families = {
        algorithm: "movie_features"
        for algorithm, method, _ in units if method is not None
    }
    for index, (algorithm, reduction, components) in enumerate(units):
        variant_id = f"{algorithm}__{reduction}_{components}" if reduction else algorithm
        saved = progress_store.get_unit(variant_id, algorithm) if progress_store else {}
        unit = saved.get("final")
        if unit is None:
            def report(message):
                update(None, f"{variant_id} | {message}")

            update(10 + int(80 * index / len(units)),
                   f"{variant_id}: GridSearchCV, 5 folds on Train")
            feature_family = comparison_families.get(algorithm, "rating_aggregates")
            model = MovieRegressor(algorithm=algorithm, movies=movies, reduction=reduction,
                                   components=components or 8, seed=request.seed, report=report,
                                   feature_family=feature_family)
            log_directory = (progress_store.path.parent / "model-comparison"
                             if progress_store else DEFAULT_TRACKING_DIRECTORY)
            search = grid_search_model(
                model, train, y_train, {"parameters": candidates(algorithm, request)},
                log_directory=log_directory,
                run_metadata={"algorithm": algorithm, "variant_id": variant_id,
                              "dataset_split_sha256": fingerprint,
                              "split": "per_user_chronological", "seed": request.seed,
                              "features": feature_family,
                              "comparison_baseline": algorithm if algorithm in comparison_families else None,
                              "reduction": reduction, "components": components},
            )
            unit_rows = []
            for split_name, frame in (("validation", validation), ("test", test)):
                evaluated = evaluate_model(search, frame, frame.rating, split=split_name)
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
                    "feature_family": feature_family,
                    "comparison_baseline": algorithm if algorithm in comparison_families else None,
                    **evaluated["metrics"],
                }
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
    chosen = min(valid_rows, key=lambda row: row["rmse"])
    result["selected_variant_id"] = chosen["variant_id"]
    result["selection"] = "Parameters selected by mean CV; model comparison uses Validation, never Test."
    result["reduction_comparisons"] = []
    for row in result["rows"]:
        if row["reduction_method"] is None:
            continue
        baseline = next(item for item in result["rows"]
                        if item["variant_id"] == row["base_algorithm"] and item["split"] == row["split"])
        metrics = ("rmse", "mae")
        result["reduction_comparisons"].append({
            "baseline_variant_id": baseline["variant_id"], "variant_id": row["variant_id"],
            "split": row["split"], "feature_family": row["feature_family"],
            "delta_reduced_minus_baseline": {
                metric: (row[metric] - baseline[metric]
                         if row[metric] is not None and baseline[metric] is not None else None)
                for metric in metrics
            },
        })
    update(95, "All selected models evaluated; CV and held-out results saved to CSV")
    return result
