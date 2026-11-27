"""End-to-end smoke coverage of the rating regression workflow."""

import numpy as np
import pandas as pd
import pytest
import json
from sklearn.base import clone

import movie_rating.experiment_supervised as supervised_module
from movie_rating.experiment_config import ExperimentRequest
from movie_rating.data.temporal import user_temporal_split
from movie_rating.experiment_estimators import MovieRegressor
from movie_rating.experiment_pipeline import execute


def movie_data():
    ratings = pd.DataFrame(
        [(u, m, [1.0, 3.0, 5.0][(u + m) % 3], 1000 * u + m)
         for u in range(1, 7) for m in range(1, 16)],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.DataFrame({"movieId": range(1, 16),
                           "genres": ["Drama" if m % 2 else "Comedy" for m in range(1, 16)]})
    return user_temporal_split(ratings)[0], movies


def test_every_regressor_runs_through_dispatch_and_logs(tmp_path, monkeypatch):
    algorithms = ["linear", "xgboost", "svd", "ncf"]
    monkeypatch.setattr(supervised_module, "DEFAULT_TRACKING_DIRECTORY", tmp_path)
    splits, movies = movie_data()
    request = ExperimentRequest(
        group="rating", algorithms=algorithms, use_all_data=False, epochs=2, tune=True,
        parameters={"factors": 2, "hidden_size": 4, "batch_size": 32,
                    "n_estimators": 10},
    )
    messages = []
    result = execute(splits, movies, request, lambda progress, message: messages.append(message))
    assert {row["algorithm"] for row in result["rows"]} == set(algorithms)
    assert len(result["rows"]) == 2 * len(algorithms)
    assert result["schema_version"] == 4
    assert result["cv_folds"] == 5
    baseline = result["baseline"]
    assert baseline["prediction"] == pytest.approx(splits["train"].rating.mean())
    test_baseline = next(row for row in baseline["rows"] if row["split"] == "test")
    errors = splits["test"].rating.to_numpy() - baseline["prediction"]
    assert test_baseline["mae"] == pytest.approx(np.abs(errors).mean())
    assert test_baseline["rmse"] == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert len(result["cv"]) == len(algorithms)
    for row in result["rows"]:
        assert row["fit_scope"] == "train"
        assert row["refit_ratings"] == len(splits["train"])
        assert np.isfinite(row["rmse"])
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
    for algorithm in algorithms:
        starts = [message for message in messages
                  if message.startswith(f"{algorithm} | Configuration")
                  and " | Starting training | " in message]
        assert len(starts) == 10
        for candidate in range(1, 3):
            for fold in range(1, 6):
                assert f"Configuration {candidate}/2 | Fold {fold}/5" in starts[(candidate - 1) * 5 + fold - 1]
        assert any(message.startswith(f"{algorithm} | Refit best configuration on all Train rows")
                   and " | Starting training | " in message for message in messages)
        assert any(f"{algorithm} | Configuration 1/2 | Fold 1/5 | {algorithm}: fitting" in message
                   for message in messages)


@pytest.mark.parametrize("algorithm", ["linear", "xgboost"])
def test_scaler_only_on_linear_and_features_fit_on_fold(algorithm):
    splits, movies = movie_data()
    params = {"n_estimators": 10} if algorithm == "xgboost" else {}
    adapter = MovieRegressor(algorithm=algorithm, movies=movies, parameters=params)
    frame = splits["train"].iloc[:20]
    fitted = clone(adapter).fit(frame, frame.rating)
    if hasattr(fitted, "builder_"):
        assert ("scaler" in fitted.model_.named_steps) == (algorithm == "linear")
        assert fitted.builder_.global_mean == pytest.approx(frame.rating.mean())


def test_reduction_is_fitted_on_train_movie_metadata():
    splits, movies = movie_data()
    adapter = MovieRegressor(algorithm="linear", movies=movies, reduction="pca", components=2)
    frame = splits["train"]
    adapter.fit(frame, frame.rating)
    assert "scaler" in adapter.model_.named_steps
    representation = adapter.representation_
    assert representation.reducer_.components_.shape == (2, 4)
    assert representation.scaler_.n_samples_seen_ == frame.movieId.nunique()
    assert adapter.predict(splits["test"]).shape == (len(splits["test"]),)


def test_lab_service_persists_complete_regression_results(configured):
    from movie_rating.data.prepare import prepare
    from movie_rating.lab import LabService

    prepare(configured)
    service = LabService(configured)
    run_id = "smoke-rating"
    folder = service.directory / run_id
    folder.mkdir(parents=True)
    job = {"id": run_id, "group": "rating", "schema_version": 4,
           "status": "queued", "progress": 0, "message": ""}
    request = ExperimentRequest(
        group="rating", algorithms=["linear", "xgboost"], use_all_data=False, max_users=10, epochs=2,
        parameters={"factors": 2, "hidden_size": 4, "batch_size": 32,
                    "n_estimators": 10},
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
