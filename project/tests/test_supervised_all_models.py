"""End-to-end smoke coverage of all six models in the rebuilt workflow."""

import numpy as np
import pandas as pd
import pytest
import json
from sklearn.base import clone, is_classifier

import movie_rating.experiment_supervised as supervised_module
from movie_rating.experiment_config import ExperimentRequest
from movie_rating.experiment_estimators import MovieClassifier, MovieRegressor
from movie_rating.experiment_pipeline import execute


def movie_data():
    ratings = pd.DataFrame(
        [(u, m, [1.0, 3.0, 5.0][(u + m) % 3], 1000 * u + m)
         for u in range(1, 7) for m in range(1, 16)],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.DataFrame({"movieId": range(1, 16),
                           "genres": ["Drama" if m % 2 else "Comedy" for m in range(1, 16)]})
    return supervised_module.split_by_time(ratings), movies


@pytest.mark.parametrize("group,algorithms", [
    ("rating", ["linear", "xgboost", "svd", "ncf"]),
    ("classification", ["logistic", "random_forest"]),
])
def test_every_model_runs_through_dispatch_and_logs(group, algorithms, tmp_path, monkeypatch):
    monkeypatch.setattr(supervised_module, "DEFAULT_TRACKING_DIRECTORY", tmp_path)
    splits, movies = movie_data()
    request = ExperimentRequest(
        group=group, algorithms=algorithms, use_all_data=False, epochs=2, tune=True,
        parameters={"factors": 2, "hidden_size": 4, "batch_size": 32,
                    "n_estimators": 10, "forest_estimators": 10,
                    "max_iter": 100, "min_samples_leaf": 1},
    )
    messages = []
    result = execute(splits, movies, request, lambda progress, message: messages.append(message))
    assert {row["algorithm"] for row in result["rows"]} == set(algorithms)
    assert len(result["rows"]) == 2 * len(algorithms)
    assert result["schema_version"] == 4
    assert result["cv_folds"] == 5
    assert len(result["cv"]) == len(algorithms)
    for row in result["rows"]:
        assert row["fit_scope"] == "train"
        assert row["refit_ratings"] == len(splits["train"])
        assert np.isfinite(row["rmse"] if group == "rating" else row["f1"])
    for data in result["cv"].values():
        assert len(data["folds"]) == 5
    runs = pd.read_csv(tmp_path / "runs.csv")
    assert set(runs.model) == set(algorithms)
    assert set(runs.status) == {"completed"}
    assert (runs.candidate_count == 2).all()
    evaluation = pd.read_csv(tmp_path / "evaluations.csv")
    assert len(evaluation) == 2 * len(algorithms)
    assert set(evaluation.split) == {"validation", "test"}
    assert any("GridSearchCV" in message for message in messages)
    if group == "classification":
        assert "neutral" in result["class_balance"]["train"]


@pytest.mark.parametrize("algorithm", ["linear", "xgboost", "logistic", "random_forest"])
def test_scaler_only_on_linear_and_logistic_and_features_fit_on_fold(algorithm):
    splits, movies = movie_data()
    classification = algorithm in {"logistic", "random_forest"}
    constructor = MovieClassifier if classification else MovieRegressor
    params = ({"n_estimators": 10} if algorithm in {"xgboost", "random_forest"} else {})
    adapter = constructor(algorithm=algorithm, movies=movies, parameters=params)
    assert is_classifier(adapter) == classification
    frame = splits["train"].iloc[:20]
    y = supervised_module.rating_labels(frame.rating) if classification else frame.rating
    fitted = clone(adapter).fit(frame, y)
    if hasattr(fitted, "builder_"):
        assert ("scaler" in fitted.model_.named_steps) == (algorithm in {"linear", "logistic"})
        assert fitted.builder_.global_mean == pytest.approx(frame.rating.mean())
    if classification:
        original = splits["test"]
        altered = original.assign(rating=0.5)
        np.testing.assert_allclose(fitted.predict_proba(original), fitted.predict_proba(altered))


@pytest.mark.parametrize("method", ["pca", "truncated_svd"])
def test_reduction_is_inside_the_fitted_pipeline(method):
    splits, movies = movie_data()
    adapter = MovieClassifier(algorithm="logistic", movies=movies,
                              reduction=method, components=2)
    frame = splits["train"]
    adapter.fit(frame, supervised_module.rating_labels(frame.rating))
    assert "scaler" in adapter.model_.named_steps
    reducer = adapter.model_.named_steps["reducer"]
    assert reducer.reducer_.components_.shape == (2, 4)
    assert adapter.predict_proba(splits["test"]).shape == (len(splits["test"]), 3)


@pytest.mark.parametrize("group", ["rating", "classification"])
def test_lab_service_persists_complete_results_for_each_group(configured, group):
    from movie_rating.data.prepare import prepare
    from movie_rating.lab import LabService

    prepare(configured)
    service = LabService(configured)
    run_id = f"smoke-{group}"
    folder = service.directory / run_id
    folder.mkdir(parents=True)
    job = {"id": run_id, "group": group, "schema_version": 4,
           "status": "queued", "progress": 0, "message": ""}
    request = ExperimentRequest(
        group=group, use_all_data=False, max_users=10, epochs=2,
        parameters={"factors": 2, "hidden_size": 4, "batch_size": 32,
                    "n_estimators": 10, "forest_estimators": 10, "max_iter": 100},
    )
    try:
        service._run_inline(job, request)
        assert job["status"] == "completed", job["message"]
        saved = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        assert len(saved["rows"]) == 2 * len(request.algorithms)
        assert saved["provenance"]["protocol"].startswith("per_user_temporal")
        assert (folder / "model-comparison" / "runs.csv").exists()
        assert (folder / "progress.json").exists()
    finally:
        service.executor.shutdown(wait=True)
