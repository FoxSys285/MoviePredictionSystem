import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from movie_rating.api.app import create_app
from movie_rating.experiment_analysis import cold_user_cohort, evaluate_cold_history


def history_rows():
    rows = pd.DataFrame([
        {"userId": user, "movieId": movie, "rating": 4.0, "timestamp": movie}
        for user in (1, 2, 3)
        for movie in range(1, 31)
    ])
    # The 21st rating ties with the support boundary and must not enter query.
    rows.loc[rows.movieId == 21, "timestamp"] = 20
    return rows


def test_cold_cohort_excludes_train_users_cold_items_and_timestamp_ties():
    cohort, eligible = cold_user_cohort(history_rows(), [1], range(1, 31), seed=42)
    assert eligible == 2
    assert [user for user, _, _ in cohort] == [2, 3]
    for _, support, query in cohort:
        assert len(support) == 20 and len(query) == 9
        assert support.timestamp.max() < query.timestamp.min()
        assert set(support.movieId).isdisjoint(query.movieId)
    assert cold_user_cohort(history_rows(), [1], range(1, 24))[0] == []
    first = cold_user_cohort(history_rows(), [1], range(1, 31), max_users=1, seed=13)[0]
    again = cold_user_cohort(history_rows(), [1], range(1, 31), max_users=1, seed=13)[0]
    assert first[0][0] == again[0][0]


def test_history_sizes_share_query_and_only_adapt_on_support():
    class ProfileModel:
        movie_ids = np.arange(1, 31)

        def __init__(self):
            self.supports = []
            self.zero_queries = []

        def predict(self, users, movies):
            self.zero_queries.append(list(movies))
            return np.full(len(movies), 3.0)

        def predict_profile(self, history):
            self.supports.append(history)
            return np.full(len(self.movie_ids), 4.0)

    cohort, _ = cold_user_cohort(history_rows(), [1], range(1, 31))
    model = ProfileModel()
    result = evaluate_cold_history(model, "ridge", cohort)
    assert [point["support_ratings"] for point in result["points"]] == [0, 5, 10, 20]
    assert {point["query_ratings"] for point in result["points"]} == {18}
    assert {point["users"] for point in result["points"]} == {2}
    assert result["points"][0]["rmse"] == 1
    assert result["points"][-1]["rmse"] == 0
    assert [len(history) for history in model.supports] == [5, 5, 10, 10, 20, 20]
    assert all(max(row["movie_id"] for row in history) <= 20 for history in model.supports)
    assert all(min(movies) == 22 for movies in model.zero_queries)


def test_empty_history_cohort_is_missing_measurement_not_zero():
    result = evaluate_cold_history(None, "ridge", [])
    assert all(point["rmse"] is None and point["users"] == 0 for point in result["points"])


def test_api_serves_optional_analysis_from_the_current_run(configured):
    run = configured.artifacts_dir / "runs/example"
    run.mkdir(parents=True)
    (configured.artifacts_dir / "latest.json").write_text(json.dumps({"run_id": "example"}))
    (run / "validation.json").write_text(json.dumps({"run_id": "example", "results": []}))
    with TestClient(create_app(configured)) as client:
        assert client.get("/api/v1/experiments/latest").json()["analysis"] is None
        analysis = {"run_id": "example", "cold_history": {}}
        (run / "analysis.json").write_text(json.dumps(analysis))
        assert client.get("/api/v1/experiments/latest").json()["analysis"] == analysis


def test_analysis_rejects_changed_data_manifest(configured):
    from movie_rating.experiment_analysis import analyze_experiments

    run = configured.artifacts_dir / "runs/example"
    data = configured.artifacts_dir / "data"
    run.mkdir(parents=True)
    data.mkdir()
    (configured.artifacts_dir / "latest.json").write_text(json.dumps({"run_id": "example"}))
    (run / "validation.json").write_text(json.dumps({"data_manifest_sha256": "wrong"}))
    (data / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Prepared data changed"):
        analyze_experiments(configured)
