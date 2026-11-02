import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from movie_rating.api.app import create_app
from movie_rating.models import MODEL_REGISTRY


def login_demo(client, account):
    response = client.post("/api/v1/auth/login", json={
        "email": account["email"], "password": account["password"],
    })
    assert response.status_code == 200
    return response.json()


def personal_account(client):
    response = client.post("/api/v1/auth/register", json={
        "name": "New viewer", "email": "new@example.com", "password": "test-password-123",
    })
    assert response.status_code == 201
    return response.json()


def test_demo_import_is_complete_idempotent_and_separate_from_new_accounts(configured, ratings):
    # Interleaved users cross many chunks; retain the latest duplicate's rating and timestamp.
    path = configured.dataset_dir / "ratings.csv"
    duplicate = pd.DataFrame([(1, 1, 5.0, 2_000_000)], columns=ratings.columns)
    pd.concat([ratings, duplicate]).to_csv(path, index=False)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with TestClient(create_app(configured)) as client:
        new_user = personal_account(client)
        assert new_user["id"] == 1  # Deliberate collision with MovieLens userId=1.
        assert new_user["profile_type"] == "personal" and new_user["dataset_user_id"] is None
        assert client.get("/api/v1/me/ratings").json()["total"] == 0
        assert client.get("/api/v1/recommendations").json()["strategy"] == "community_popularity"
        response = client.get("/api/v1/auth/demo-accounts")
        assert response.status_code == 200
        demos = response.json()["accounts"]
        assert len(demos) == 5
        assert [row["dataset_user_id"] for row in demos] == [1, 2, 3, 4, 5]
        assert all(row["imported_rating_count"] == 12 for row in demos)
        assert "password_hash" not in response.text and "salt" not in response.text
        user = login_demo(client, demos[0])
        assert user["id"] != user["dataset_user_id"]
        assert user["profile_type"] == "dataset" and not user["profile_modified"]
        assert "password" not in json.dumps(user)
        history = client.get("/api/v1/me/ratings").json()
        assert history["total"] == 12
        first = history["items"][0]
        assert first["movieId"] == 1 and first["user_rating"] == 5
        assert first["updated_at"] == 2_000_000 * 1000
        client.put("/api/v1/me/ratings/1", json={"rating": 0.5})
        client.delete("/api/v1/me/ratings/2")
        assert client.get("/api/v1/auth/me").json()["profile_modified"]
        assert client.get("/api/v1/auth/demo-accounts").json()["accounts"] == demos
        assert client.get("/api/v1/me/ratings").json()["total"] == 11
        assert client.get("/api/v1/me/ratings/1").json()["rating"] == 0.5
    # Recreating the application does not reset edited/deleted imported data.
    with TestClient(create_app(configured)) as client:
        assert client.get("/api/v1/auth/demo-accounts").json()["accounts"] == demos
        login_demo(client, demos[0])
        assert client.get("/api/v1/me/ratings/2").json()["rating"] is None
        assert client.get("/api/v1/me/ratings/1").json()["rating"] == 0.5
        login_demo(client, demos[1])
        assert client.get("/api/v1/me/ratings").json()["total"] == 12
        client.post("/api/v1/auth/login", json={"email": "new@example.com", "password": "test-password-123"})
        assert client.get("/api/v1/me/ratings").json()["total"] == 0
        assert client.post("/api/v1/auth/register", json={"name": "Impersonator",
            "email": "fake@demo.movie.local", "password": "test-password-123"}).status_code == 409
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum


@pytest.mark.parametrize("algorithm", MODEL_REGISTRY)
def test_known_dataset_user_prediction_and_profile_updates(configured, ratings, algorithm):
    # Demo user 1 has three ratings, while the trained model knows all twelve items.
    history = ratings.loc[(ratings.userId != 1) | (ratings.movieId <= 3)]
    history.to_csv(configured.dataset_dir / "ratings.csv", index=False)
    params = {"epochs": 2, "factors": 3} if algorithm in ("svd", "nmf") else {}
    model = MODEL_REGISTRY[algorithm](**params).fit(ratings)
    run = configured.artifacts_dir / "runs/demo-test"
    run.mkdir(parents=True)
    joblib.dump(model, run / f"{algorithm}.joblib")
    (run / "validation.json").write_text(json.dumps({"best_algorithm": algorithm, "run_id": "demo-test"}))
    (configured.artifacts_dir / "latest.json").write_text(json.dumps({"run_id": "demo-test"}))
    with TestClient(create_app(configured)) as client:
        personal_account(client)
        # A new app account never inherits a trained profile just because its numeric ID matches.
        cold = client.get("/api/v1/recommendations").json()
        assert cold["mode"] == "popular" and cold["history_count"] == 0
        account = client.get("/api/v1/auth/demo-accounts").json()["accounts"][0]
        user = login_demo(client, account)
        assert user["id"] != user["dataset_user_id"]
        result = client.get(f"/api/v1/recommendations?algorithm={algorithm}").json()
        assert result["strategy"] == "trained_dataset_user"
        assert result["dataset_user_id"] == 1 and result["history_count"] == 3
        assert len(result["items"]) == 9
        ids = [row["movieId"] for row in result["items"]]
        assert not {1, 2, 3} & set(ids)
        np.testing.assert_allclose([row["predicted_rating"] for row in result["items"]],
                                   model.predict(np.ones(len(ids)), ids))
        client.put("/api/v1/me/ratings/4", json={"rating": 5})
        updated = client.get(f"/api/v1/recommendations?algorithm={algorithm}").json()
        assert updated["strategy"] == "profile_fold_in"
        assert updated["history_count"] == 4 and len(updated["items"]) == 8
        assert 4 not in {row["movieId"] for row in updated["items"]}


def test_dataset_user_not_in_trained_sample_uses_fold_in(configured, ratings):
    model = MODEL_REGISTRY["svd"](epochs=2, factors=3).fit(ratings.loc[ratings.userId != 1])
    run = configured.artifacts_dir / "runs/demo-test"
    run.mkdir(parents=True)
    joblib.dump(model, run / "svd.joblib")
    (run / "validation.json").write_text(json.dumps({"best_algorithm": "svd", "run_id": "demo-test"}))
    (configured.artifacts_dir / "latest.json").write_text(json.dumps({"run_id": "demo-test"}))
    with TestClient(create_app(configured)) as client:
        account = client.get("/api/v1/auth/demo-accounts").json()["accounts"][0]
        login_demo(client, account)
        assert client.get("/api/v1/recommendations").json()["strategy"] == "profile_fold_in"


def test_small_or_missing_dataset(configured, ratings):
    ratings.loc[ratings.userId.isin([8, 10])].to_csv(configured.dataset_dir / "ratings.csv", index=False)
    with TestClient(create_app(configured)) as client:
        accounts = client.get("/api/v1/auth/demo-accounts").json()["accounts"]
        assert [row["dataset_user_id"] for row in accounts] == [8, 10]
    (configured.dataset_dir / "ratings.csv").unlink()
    with TestClient(create_app(configured)) as client:
        assert client.get("/api/v1/auth/demo-accounts").status_code == 503
        personal_account(client)  # Demo availability does not block ordinary sign-up.
