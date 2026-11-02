import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from movie_rating.api.app import create_app
from movie_rating.models import MODEL_REGISTRY
from movie_rating.services.accounts import AccountStore


def register(client, email="viewer@example.com"):
    response = client.post("/api/v1/auth/register", json={
        "name": "Movie Viewer", "email": email, "password": "test-password-123",
    })
    assert response.status_code == 201
    return response.json()


def test_rating_lifecycle_isolation_validation_and_persistence(configured):
    app = create_app(configured)
    with TestClient(app) as first, TestClient(app) as second:
        assert first.get("/api/v1/me/ratings").status_code == 401
        assert first.put("/api/v1/me/ratings/1", json={"rating": 4}).status_code == 401
        register(first)
        assert first.get("/api/v1/me/ratings").json()["total"] == 0
        assert first.get("/api/v1/me/ratings/1").json()["rating"] is None
        assert first.put("/api/v1/me/ratings/1", json={"rating": 4.5}).status_code == 200
        history = first.get("/api/v1/me/ratings").json()
        assert history["total"] == 1 and history["items"][0]["user_rating"] == 4.5
        created = first.get("/api/v1/me/ratings/1").json()["created_at"]
        assert first.put("/api/v1/me/ratings/1", json={"rating": 2.5}).status_code == 200
        assert first.get("/api/v1/me/ratings").json()["total"] == 1
        assert first.get("/api/v1/me/ratings/1").json()["created_at"] == created
        for invalid in (0, 5.5, 3.2):
            assert first.put("/api/v1/me/ratings/1", json={"rating": invalid}).status_code == 422
        assert first.put("/api/v1/me/ratings/999", json={"rating": 4}).status_code == 404
        assert first.put("/api/v1/me/ratings/1", json={"rating": 4},
                         headers={"Origin": "https://other.example"}).status_code == 403
        register(second, "other@example.com")
        assert second.get("/api/v1/me/ratings").json()["total"] == 0
        assert second.get("/api/v1/me/ratings/1").json()["rating"] is None
        second.delete("/api/v1/me/ratings/1")
        assert first.get("/api/v1/me/ratings/1").json()["rating"] == 2.5
        first.post("/api/v1/auth/logout")
        assert first.get("/api/v1/me/ratings").status_code == 401
        first.post("/api/v1/auth/login", json={"email": "viewer@example.com", "password": "test-password-123"})
        assert first.get("/api/v1/me/ratings").json()["items"][0]["user_rating"] == 2.5
        assert first.delete("/api/v1/me/ratings/1").status_code == 200
        assert first.get("/api/v1/me/ratings").json()["total"] == 0


def test_legacy_cookie_is_migrated_to_personal_endpoints(configured):
    store = AccountStore(configured.artifacts_dir / "application.sqlite3")
    user = store.register("Legacy Viewer", "legacy@example.com", "test-password-123")
    token = store.new_session(user["id"])
    with TestClient(create_app(configured)) as client:
        client.cookies.set("movie_session", token, domain="testserver.local", path="/api/v1/auth")
        assert client.get("/api/v1/me/ratings").status_code == 401
        assert client.get("/api/v1/auth/me").status_code == 200
        assert client.get("/api/v1/me/ratings").status_code == 200
        client.post("/api/v1/auth/logout")
        assert client.get("/api/v1/me/ratings").status_code == 401


def test_content_recommendations_follow_own_ratings_and_exclude_seen(configured):
    path = configured.dataset_dir / "movies.csv"
    movies = pd.read_csv(path)
    movies.loc[movies.movieId.isin([1, 2]), "genres"] = "Action"
    movies.loc[movies.movieId.isin([3, 4]), "genres"] = "Comedy"
    movies.to_csv(path, index=False)
    with TestClient(create_app(configured)) as client:
        guest = client.get("/api/v1/recommendations").json()
        assert guest["mode"] == "popular" and guest["authenticated"] is False
        assert all(item["predicted_rating"] is None for item in guest["items"])
        register(client)
        client.put("/api/v1/me/ratings/1", json={"rating": 5})
        result = client.get("/api/v1/recommendations").json()
        assert result["mode"] == "content" and result["history_count"] == 1
        assert result["items"][0]["movieId"] == 2
        assert 1 not in {item["movieId"] for item in result["items"]}
        assert result["algorithm"] is None and result["model_status"] == "not_trained"
        client.put("/api/v1/me/ratings/1", json={"rating": 1})
        result = client.get("/api/v1/recommendations").json()
        assert result["items"][0]["movieId"] != 2
        assert client.get("/api/v1/recommendations?limit=21").status_code == 422
        assert client.get("/api/v1/recommendations?algorithm=svd").status_code == 503


def test_discovery_uses_real_genres_and_limits_each_shelf(configured):
    with TestClient(create_app(configured)) as client:
        result = client.get("/api/v1/discovery/genres").json()
        assert len(result["items"]) == 1
        assert result["items"][0]["name"] == "Drama"
        assert result["items"][0]["movie_count"] == 12
        assert len(result["items"][0]["movies"]) == 10
        assert client.get("/api/v1/discovery/genres?movies_per_genre=11").status_code == 422


def test_recommendation_cap_and_history_pagination(configured):
    path = configured.dataset_dir / "movies.csv"
    movies = pd.read_csv(path)
    extra = pd.DataFrame({"movieId": range(13, 33),
                          "title": [f"Film {number}" for number in range(13, 33)],
                          "genres": ["Drama"] * 20})
    pd.concat([movies, extra], ignore_index=True).to_csv(path, index=False)
    with TestClient(create_app(configured)) as client:
        assert len(client.get("/api/v1/recommendations").json()["items"]) == 20
        register(client)
        for movie_id in range(1, 22):
            assert client.put(f"/api/v1/me/ratings/{movie_id}", json={"rating": 4}).status_code == 200
        first = client.get("/api/v1/me/ratings?limit=20").json()
        second = client.get("/api/v1/me/ratings?offset=20&limit=20").json()
        assert first["total"] == second["total"] == 21
        assert len(first["items"]) == 20 and len(second["items"]) == 1
        assert not {item["movieId"] for item in first["items"]} & {item["movieId"] for item in second["items"]}
        result = client.get("/api/v1/recommendations").json()
        assert len(result["items"]) == 11
        assert all(item["movieId"] > 21 for item in result["items"])


@pytest.mark.parametrize("algorithm", MODEL_REGISTRY)
def test_trained_models_predict_for_new_account_profile(configured, ratings, algorithm):
    params = {"epochs": 2, "factors": 3} if algorithm in ("svd", "nmf") else {}
    model = MODEL_REGISTRY[algorithm](**params).fit(ratings)
    before = model.user_ids.copy()
    run = configured.artifacts_dir / "runs/profile-test"
    run.mkdir(parents=True)
    joblib.dump(model, run / f"{algorithm}.joblib")
    (run / "validation.json").write_text(json.dumps({"best_algorithm": algorithm, "run_id": "profile-test"}))
    (configured.artifacts_dir / "latest.json").write_text(json.dumps({"run_id": "profile-test"}))
    with TestClient(create_app(configured)) as client:
        register(client)
        client.put("/api/v1/me/ratings/1", json={"rating": 5})
        client.put("/api/v1/me/ratings/2", json={"rating": 1})
        result = client.get(f"/api/v1/recommendations?algorithm={algorithm}").json()
        assert result["mode"] == "model" and result["algorithm"] == algorithm
        assert result["run_id"] == "profile-test"
        assert len(result["items"]) == 10
        assert not {1, 2} & {item["movieId"] for item in result["items"]}
        assert all(0.5 <= item["predicted_rating"] <= 5 for item in result["items"])
        assert np.array_equal(before, model.user_ids)
