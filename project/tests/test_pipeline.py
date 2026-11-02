import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from movie_rating.api.app import create_app
from movie_rating.data.prepare import chronological_split, load_ratings, prepare
from movie_rating.evaluation import regression_metrics
from movie_rating.models import MODEL_REGISTRY
from movie_rating.training import current_run, evaluate_test, train


def test_temporal_split_keeps_equal_timestamps_together(ratings):
    ratings.loc[95:97, "timestamp"] = ratings.loc[95, "timestamp"]
    splits = chronological_split(ratings, 0.8, 0.1)
    assert sum(map(len, splits.values())) == len(ratings)
    assert splits["train"].timestamp.max() < splits["validation"].timestamp.min()
    assert splits["validation"].timestamp.max() < splits["test"].timestamp.min()


def test_sampling_preserves_complete_user_histories(configured, ratings):
    configured.data.user_sample_fraction = 0.6
    sample = load_ratings(configured)
    assert 0 < sample.userId.nunique() < ratings.userId.nunique()
    assert (sample.groupby("userId").size() == 12).all()


@pytest.mark.parametrize("name", MODEL_REGISTRY)
def test_models_handle_unknown_ids_and_serialize(name, ratings, tmp_path):
    params = {"epochs": 2, "factors": 3} if name in ("svd", "nmf") else {}
    model = MODEL_REGISTRY[name](**params).fit(ratings)
    users, movies = [1, 999, 1, 999], [1, 1, 999, 999]
    prediction = model.predict(users, movies)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0.5) & (prediction <= 5)).all()
    assert prediction[-1] == pytest.approx(ratings.rating.mean())
    assert len(model.predict([], [])) == 0
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(users, movies), prediction)
    if name == "nmf":
        assert (model.p >= 0).all() and (model.q >= 0).all()


def test_metrics():
    metrics = regression_metrics([1, 3], [2, 3])
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))
    assert metrics["mae"] == 0.5


@pytest.mark.parametrize("name", MODEL_REGISTRY)
def test_models_learn_user_preferences(name, ratings):
    ratings = ratings.copy()
    ratings["rating"] = np.where(ratings.userId <= 5, 1.0, 4.5)
    params = {"epochs": 40, "factors": 4, "learning_rate": 0.01} if name in ("svd", "nmf") else {}
    model = MODEL_REGISTRY[name](**params).fit(ratings)
    scores = model.predict(ratings.userId, ratings.movieId)
    naive = np.full(len(ratings), ratings.rating.mean())
    assert regression_metrics(ratings.rating, scores)["rmse"] < (
        regression_metrics(ratings.rating, naive)["rmse"] * 0.8
    )


def test_api_without_data(configured):
    configured.data.directory = str(configured.dataset_dir / "missing")
    with TestClient(create_app(configured)) as client:
        assert client.get("/api/v1/health").json()["models_ready"] is False
        assert client.get("/api/v1/movies").status_code == 503
        assert (
            client.post("/api/v1/predictions", json={"user_id": 1, "movie_id": 1}).status_code
            == 503
        )
        assert (
            client.post("/api/v1/predictions", json={"user_id": -1, "movie_id": 1}).status_code
            == 422
        )


def test_end_to_end(configured):
    manifest = prepare(configured)
    assert manifest["splits"]["train"]["ratings"] == 96
    with pytest.raises(FileExistsError):
        prepare(configured)
    validation = train(configured)
    assert len(validation["results"]) == 4
    assert validation["best_algorithm"] == validation["results"][0]["algorithm"]
    run_dir = current_run(configured)
    fitted = joblib.load(run_dir / "svd.joblib")
    assert 12 not in fitted.movie_ids  # Movie 12 appears only in the held-out test set.
    assert not (run_dir / "test.json").exists()
    test = evaluate_test(configured)
    assert test["best_algorithm"] == validation["best_algorithm"]
    assert len(test["results"]) == 4
    assert evaluate_test(configured) == test
    with TestClient(create_app(configured)) as client:
        assert client.get("/api/v1/health").json()["models_ready"]
        catalog = client.get("/api/v1/movies", params={"q": "Film 1", "limit": 2}).json()
        assert catalog["total"] == 4 and len(catalog["items"]) == 2
        response = client.post("/api/v1/predictions", json={"user_id": 1, "movie_id": 1})
        assert response.status_code == 200
        assert 0.5 <= response.json()["predicted_rating"] <= 5
        assert response.json()["algorithm"] == validation["best_algorithm"]
        assert client.get("/api/v1/experiments/latest").json()["test"] is not None
    saved = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    assert saved["data_manifest_sha256"]
