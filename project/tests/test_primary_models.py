import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from movie_rating.api.app import create_app
from movie_rating.evaluation import evaluate_ranking
from movie_rating.models import MODEL_REGISTRY
from movie_rating.services.recommendations import RecommendationService
from movie_rating.settings import PRIMARY_ALGORITHMS


@pytest.mark.parametrize("algorithm", PRIMARY_ALGORITHMS)
def test_new_profile_changes_predictions_without_mutating_trained_model(algorithm, ratings):
    ratings = ratings.copy()
    ratings["rating"] = np.where(ratings.userId <= 5, 1.0, 4.5)
    params = {"epochs": 2} if algorithm in ("svd", "ncf") else {}
    model = MODEL_REGISTRY[algorithm](**params).fit(ratings)
    state = joblib.hash(model)
    # Use a supported history length comparable to the fitted users. A tree model
    # can legitimately map two one-rating profiles to the same leaf.
    movies = model.movie_ids[:10]
    high = RecommendationService.profile_scores(
        model,
        [{"movie_id": int(movie), "rating": 5} for movie in movies],
        algorithm,
    ).to_numpy()
    low = RecommendationService.profile_scores(
        model,
        [{"movie_id": int(movie), "rating": 1} for movie in movies],
        algorithm,
    ).to_numpy()
    assert (
        RecommendationService.profile_scores(
            model,
            [{"movie_id": 9999, "rating": 5}],
            algorithm,
        )
        is None
    )
    assert np.isfinite(high).all() and np.isfinite(low).all()
    assert not np.allclose(high, low), "Real account ratings must affect the predictions"
    assert joblib.hash(model) == state, "Profile updates must not change shared model weights"


def test_cold_account_switches_models_without_fabricated_personal_scores(configured, ratings):
    run = configured.artifacts_dir / "runs/cold-start"
    run.mkdir(parents=True)
    for algorithm in PRIMARY_ALGORITHMS:
        params = {"epochs": 2} if algorithm in ("svd", "ncf") else {}
        model = MODEL_REGISTRY[algorithm](**params).fit(ratings)
        joblib.dump(model, run / f"{algorithm}.joblib")
    (run / "validation.json").write_text(
        json.dumps(
            {
                "best_algorithm": "ridge",
                "best_recommendation_algorithm": "svd",
                "run_id": "cold-start",
            }
        )
    )
    (configured.artifacts_dir / "latest.json").write_text(json.dumps({"run_id": "cold-start"}))
    with TestClient(create_app(configured)) as client:
        client.post(
            "/api/v1/auth/register",
            json={
                "name": "New viewer",
                "email": "cold@example.com",
                "password": "test-password-123",
            },
        )
        starter_ids = None
        for algorithm in PRIMARY_ALGORITHMS:
            response = client.get("/api/v1/recommendations", params={"algorithm": algorithm})
            assert response.status_code == 200
            result = response.json()
            assert result["cold_start"] and result["history_count"] == 0
            assert result["requested_algorithm"] == algorithm and result["algorithm"] is None
            assert result["mode"] == "popular"
            assert result["available_algorithms"] == list(PRIMARY_ALGORITHMS)
            assert all(row["predicted_rating"] is None for row in result["items"])
            ids = [row["movieId"] for row in result["items"]]
            if starter_ids is not None:
                assert ids == starter_ids
            starter_ids = ids
        client.put("/api/v1/me/ratings/1", json={"rating": 5})
        for algorithm in PRIMARY_ALGORITHMS:
            result = client.get("/api/v1/recommendations", params={"algorithm": algorithm}).json()
            assert result["mode"] == "model" and result["algorithm"] == algorithm
            assert result["strategy"] == "profile_fold_in"
            assert 1 not in {row["movieId"] for row in result["items"]}
        assert client.get("/api/v1/recommendations").json()["algorithm"] == "svd"
        assert client.get("/api/v1/recommendations?algorithm=invalid").status_code == 422


def test_ranking_uses_heldout_positives_and_excludes_history():
    class FixedModel:
        user_ids = np.array([1])
        movie_ids = np.array([1, 2, 3, 4])

        def predict(self, users, items):
            return np.array([0, 5, 3, 4, 2])[items]

    history = pd.DataFrame({"userId": [1], "movieId": [1]})
    heldout = pd.DataFrame({"userId": [1, 1, 1], "movieId": [2, 3, 4], "rating": [5, 1, 4]})
    result = evaluate_ranking(FixedModel(), heldout, history, k=2)
    assert result["precision_at_2"] == 0.5
    assert result["recall_at_2"] == 0.5
    assert result["ndcg_at_2"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert result["ranking_users"] == 1
    assert result["ranking_mean_candidates"] == 3
    heldout["rating"] = 1
    assert evaluate_ranking(FixedModel(), heldout, history)["ndcg_at_10"] is None
