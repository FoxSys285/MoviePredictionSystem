import json

import numpy as np
import pandas as pd
import pytest

from movie_rating.data.prepare import prepare
from movie_rating.stability import (
    paired_bootstrap,
    run_stability,
    summarize,
    time_blocks,
    user_errors,
)
from movie_rating.training import current_run, evaluate_test, train


def test_time_blocks_keep_timestamps_and_every_row():
    rows = pd.DataFrame({"timestamp": np.repeat(np.arange(30), 3)})
    blocks = time_blocks(rows)
    assert len(blocks) == 6
    assert sum(map(len, blocks)) == len(rows)
    assert all(a.timestamp.max() < b.timestamp.min() for a, b in zip(blocks, blocks[1:]))
    with pytest.raises(ValueError):
        time_blocks(pd.DataFrame({"timestamp": [1] * 30}))


def test_bootstrap_recomputes_rating_weighted_rmse_and_is_paired():
    ratings = pd.DataFrame({"userId": [1, 1, 1, 2], "rating": [1.0, 1.0, 1.0, 1.0]})
    a = user_errors(ratings, [1, 1, 1, 3])
    b = user_errors(ratings, [2, 2, 2, 2])
    result = paired_bootstrap(a, b, "rmse", repetitions=200)
    assert result["delta"] == 0  # sqrt((0 + 4) / 4) - sqrt(4 / 4)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result == paired_bootstrap(a, b, "rmse", repetitions=200)
    identical = paired_bootstrap(a, a, "mae", repetitions=200)
    assert identical["ci_low"] == identical["ci_high"] == identical["delta"] == 0
    assert identical["conclusion"] == "uncertain"
    with pytest.raises(ValueError, match="identical user"):
        paired_bootstrap(a, b.iloc[:1], "rmse")


def test_bootstrap_directions_and_insufficient_users():
    a = pd.DataFrame({"userId": [1, 2], "ndcg_at_10": [0.5, 0.6]})
    b = pd.DataFrame({"userId": [1, 2], "ndcg_at_10": [0.1, 0.2]})
    assert paired_bootstrap(a, b, "ndcg_at_10")["conclusion"] == "a_better"
    assert paired_bootstrap(b, a, "ndcg_at_10")["conclusion"] == "b_better"
    assert paired_bootstrap(a.iloc[:1], b.iloc[:1], "ndcg_at_10")["delta"] is None


def test_summary_uses_sample_std_and_keeps_missing_measurements():
    result = summarize(
        [
            {"algorithm": "ridge", "validation": {"rmse": 1.0, "ndcg_at_10": None}},
            {"algorithm": "ridge", "validation": {"rmse": 3.0, "ndcg_at_10": None}},
        ],
        "validation",
    )[0]["metrics"]
    assert result["rmse"]["mean"] == 2
    assert result["rmse"]["std"] == pytest.approx(np.sqrt(2))
    assert result["ndcg_at_10"]["mean"] is None


def test_stability_end_to_end_preserves_original_run_and_resumes(configured):
    configured.experiment.algorithms = ["ridge", "ncf"]
    prepare(configured)
    train(configured)
    evaluate_test(configured)
    run = current_run(configured)
    original = {name: (run / name).read_bytes() for name in ("validation.json", "test.json")}
    pointer = (configured.artifacts_dir / "latest.json").read_bytes()
    report = run_stability(configured)
    assert report["status"] == "complete"
    assert len(report["seed_runs"]) == len(report["temporal_runs"]) == 10
    assert len(report["bootstrap"]["test"]) == 3
    assert report["seed_summary"]["test"][0]["metrics"]["rmse"]["n"] == 5
    assert all(
        row["train"]["max_timestamp"] < row["heldout"]["min_timestamp"]
        for row in report["temporal_runs"]
    )
    for name, content in original.items():
        assert (run / name).read_bytes() == content
    assert (configured.artifacts_dir / "latest.json").read_bytes() == pointer
    first = report["seed_runs"][0]
    assert run_stability(configured)["seed_runs"][0] == first
    plan = json.loads((run / "stability/plan.json").read_text())
    plan["ranking_seed"] = 999
    (run / "stability/plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan changed"):
        run_stability(configured)


def test_stability_api_and_export_allowlist(configured):
    from fastapi.testclient import TestClient

    from movie_rating.api.app import create_app

    run = configured.artifacts_dir / "runs/api-test"
    (run / "stability").mkdir(parents=True)
    (configured.artifacts_dir / "latest.json").write_text(json.dumps({"run_id": "api-test"}))
    (run / "validation.json").write_text(json.dumps({"run_id": "api-test"}))
    with TestClient(create_app(configured)) as client:
        assert client.get("/api/v1/experiments/latest").json()["stability"] is None
        assert client.get("/api/v1/experiments/exports/seeds").status_code == 404
        (run / "stability.json").write_text(
            json.dumps({"run_id": "api-test", "status": "complete"})
        )
        assert client.get("/api/v1/experiments/latest").json()["stability"]["status"] == "complete"
        for name, relative in (
            ("stability", "stability.json"),
            ("seeds", "stability/seed-runs.csv"),
            ("temporal", "stability/temporal-runs.csv"),
            ("bootstrap", "stability/bootstrap.csv"),
        ):
            if name != "stability":
                (run / relative).write_text("algorithm,rmse\nridge,1.0\n")
            response = client.get(f"/api/v1/experiments/exports/{name}")
            assert response.status_code == 200
            assert "attachment" in response.headers["content-disposition"]
        assert client.get("/api/v1/experiments/exports/unknown").status_code == 422
