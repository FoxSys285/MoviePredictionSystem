"""CSV result tracking for comparable supervised experiment runs."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from movie_rating.experiment_logs import _append_lock

DEFAULT_TRACKING_DIRECTORY = Path(__file__).resolve().parents[2] / "artifacts" / "model-comparison"
METRICS = ("rmse", "mae", "accuracy", "precision", "recall", "f1", "auc_roc")
RUN_COLUMNS = (
    "run_id", "started_at_utc", "status", "task", "model", "stage", "train_rows",
    "cv_folds", "candidate_count", "selection_metric", "best_params", "fit_seconds",
    "refit_seconds", "metadata", "error",
    *(f"cv_{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")),
)


def save_search_run(search, *, directory, run_id, started_at, task, train_rows,
                    fit_seconds, metadata=None, error=None):
    """Save candidate/fold scores and append one completed or failed run summary.

    Error metrics are converted to positive values in exported CSV files.
    Metadata should identify the dataset, split, feature set, and random seed so
    readers can compare runs that used the same experimental conditions.
    The shared summary is locked across processes; each run has its own folder.
    """
    directory = Path(directory)
    run_directory = directory / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    configuration = {
        "run_id": run_id,
        "task": task,
        "cv": search.cv,
        "refit_metric": search.refit,
        "param_grid": search.param_grid,
        "estimator_params": search.estimator.get_params(deep=True),
        "metadata": metadata or {},
        "evaluation_protocol": (
            {"rating_range": [0.5, 5.0], "clip_predictions": True,
             "metrics": ["rmse", "mae"]}
            if task == "regression" else
            {"labels": {"0": "rating < 2.5", "1": "2.5 <= rating <= 3.5",
                        "2": "rating > 3.5"},
             "f1_average": "macro", "auc_roc_average": "macro", "auc_roc_mode": "ovr"}
        ),
    }
    (run_directory / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    estimator = search.estimator
    if hasattr(estimator, "named_steps"):
        estimator = estimator.steps[-1][1]
    row = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "status": "failed" if error is not None else "completed",
        "task": task,
        "model": type(estimator).__name__,
        "stage": "train_cross_validation",
        "train_rows": train_rows,
        "cv_folds": search.cv,
        "selection_metric": search.refit,
        "fit_seconds": fit_seconds,
        "metadata": json.dumps(metadata or {}, sort_keys=True, default=str),
        "error": str(error) if error is not None else "",
    }
    if error is None:
        results = pd.DataFrame(search.cv_results_)
        for column in results.columns:
            if ("_test_rmse" in column or "_test_mae" in column) and not column.startswith(
                ("std_", "rank_")
            ):
                results[column] = -results[column]
        results.insert(0, "run_id", run_id)
        results.insert(1, "candidate_index", np.arange(len(results)))
        results.insert(2, "selected", np.arange(len(results)) == search.best_index_)
        results.to_csv(run_directory / "cv_results.csv", index=False, encoding="utf-8-sig")
        row.update({
            "candidate_count": len(results),
            "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
            "refit_seconds": search.refit_time_,
        })
        for metric in METRICS:
            mean_key = f"mean_test_{metric}"
            if mean_key in results:
                row[f"cv_{metric}_mean"] = results.loc[search.best_index_, mean_key]
                row[f"cv_{metric}_std"] = results.loc[search.best_index_, f"std_test_{metric}"]

    # A per-run summary remains available even if appending the shared CSV fails.
    with (run_directory / "run.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    path = directory / "runs.csv"
    with _append_lock(path):
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RUN_COLUMNS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
    return run_directory


def save_evaluation(run_directory, result):
    """Append held-out metrics linked to the same run as its CV results."""
    run_directory = Path(run_directory)
    row = {
        "run_id": run_directory.name,
        "task": result["task"],
        "split": result["split"],
        "rows": result["rows"],
        "inference_seconds": result["inference_seconds"],
        **result["metrics"],
        "auc_roc_reason": result.get("auc_roc_reason", ""),
    }
    columns = ("run_id", "task", "split", "rows", "inference_seconds",
               "rmse", "mae", "accuracy", "f1", "auc_roc", "auc_roc_reason")
    for path in (run_directory / "evaluations.csv", run_directory.parent / "evaluations.csv"):
        with _append_lock(path):
            needs_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
