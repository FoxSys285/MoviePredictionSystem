import json
import time

import numpy as np
import pandas as pd
import pytest
from mlxtend.frequent_patterns import apriori, association_rules

from movie_rating.data.prepare import prepare
from movie_rating.experiment_config import ExperimentRequest
from movie_rating.experiment_data import ExperimentData
from movie_rating.experiment_unsupervised import association
from movie_rating.lab import Cancelled, LabService


def test_only_association_forces_full_prepared_train():
    for group in ("rating", "classification", "clustering", "pca", "reinforcement"):
        assert not ExperimentRequest(group=group, use_all_data=False).use_all_data
    assert ExperimentRequest(group="association", use_all_data=False, max_users=10).use_all_data


def test_full_train_load_preserves_splits_and_source_even_after_other_group_cache(tmp_path):
    train = pd.DataFrame(
        [(u, m, 4.5, u * 100 + m) for u in range(1, 25) for m in range(1, 13)],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    validation = train.assign(movieId=lambda f: f.movieId + 100)
    test = train.assign(movieId=lambda f: f.movieId + 200)
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    movies = pd.DataFrame(
        {"movieId": range(1, 213), "title": [f"Film {m}" for m in range(1, 213)], "genres": "Drama"}
    )
    movies.to_parquet(tmp_path / "movies.parquet", index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"settings": {"data": {"user_sample_fraction": 1}}})
    )
    data = ExperimentData(tmp_path)
    data.load(ExperimentRequest(group="pca", max_users=10))
    splits, _, provenance = data.load(
        ExperimentRequest(group="association", use_all_data=False, max_users=10)
    )
    pd.testing.assert_frame_equal(splits["train"], train)
    pd.testing.assert_frame_equal(splits["validation"], validation)
    pd.testing.assert_frame_equal(splits["test"], test)
    assert splits["train"].userId.nunique() == 24  # max_users=10 must not truncate Train.
    source = provenance["association_source"]
    assert source["train_file"] == str((tmp_path / "train.parquet").resolve())
    assert source["prepared_user_sample_fraction"] == 1
    assert source["heldout_files"] == [
        str((tmp_path / f"{s}.parquet").resolve()) for s in ("validation", "test")
    ]


@pytest.mark.parametrize("max_length", [2, 3])
def test_bounded_baskets_preserve_original_metrics_empty_users_and_catalog(max_length):
    frame = pd.DataFrame(
        [
            (u, m, 4.5 if u % 7 and (u + m) % 4 else 2.0)
            for u in range(1, 301)
            for m in range(1, 15)
        ],
        columns=["userId", "movieId", "rating"],
    )
    movies = pd.DataFrame({"movieId": range(1, 15), "title": [f"Film {m}" for m in range(1, 15)]})
    request = ExperimentRequest(group="association", max_items=10, max_rule_length=max_length)
    messages = []
    result = association(
        {"train": frame}, movies, request, lambda _, message: messages.append(message)
    )
    positive = frame.loc[frame.rating >= 4]
    items = (
        positive.groupby("movieId")
        .size()
        .sort_values(ascending=False, kind="stable")
        .head(10)
        .index
    )
    original = (
        pd.crosstab(positive.userId, positive.movieId)
        .reindex(index=np.sort(frame.userId.unique()), columns=items, fill_value=0)
        .fillna(0)
        .astype(bool)
    )
    expected = association_rules(
        apriori(original, min_support=0.08, max_len=max_length, use_colnames=True),
        num_itemsets=len(original),
        metric="confidence",
        min_threshold=0.3,
    )
    assert result["rows"][0]["rule_count"] == len(expected)
    for key in ("support", "confidence", "lift"):
        assert result["rows"][0][key] == pytest.approx(expected[key].mean())
    lookup = {
        (frozenset(r["antecedent_ids"]), frozenset(r["consequent_ids"])): r for r in result["rules"]
    }
    for r in expected.itertuples():
        actual = lookup[(r.antecedents, r.consequents)]
        for key in ("support", "confidence", "lift"):
            assert actual[key] == pytest.approx(getattr(r, key))
    summary = result["association_data"]
    assert summary["train_users"] == 300 and summary["candidate_movies"] == 10
    assert summary["empty_transactions"] == int((~original.any(axis=1)).sum())
    assert summary["transaction_bytes"] == 300 * 10
    assert "Transactions" in "\n".join(messages) and "min_confidence" in "\n".join(messages)


def test_no_positive_ratings_still_keeps_all_train_users_and_logs_no_rules():
    frame = pd.DataFrame({"userId": range(1, 31), "movieId": 1, "rating": 2.0})
    movies = pd.DataFrame({"movieId": [1], "title": ["Film 1"]})
    result = association(
        {"train": frame}, movies, ExperimentRequest(group="association"), lambda *_: None
    )
    assert result["rules"] == []
    assert result["rows"][0]["support"] is None
    assert result["association_data"]["empty_transactions"] == 30


def test_real_association_job_persists_full_source_and_reopenable_stage_logs(configured):
    prepare(configured)
    service = LabService(configured)
    job = service.start(ExperimentRequest(group="association", use_all_data=False, max_users=10))
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            finished = service.get(job["id"])
            if finished["status"] not in ("queued", "running") and service.active is None:
                break
            time.sleep(0.02)
        assert finished["status"] == "completed", finished["message"]
        train = pd.read_parquet(configured.artifacts_dir / "data/train.parquet")
        assert finished["request"]["use_all_data"] is True
        assert finished["result"]["association_data"]["train_ratings"] == len(train)
        reopened = LabService(configured)
        page = reopened.logs(job["id"], limit=500)
        messages = "\n".join(e["message"] for e in page["entries"])
        for text in (
            "train.parquet",
            "movies.parquet",
            "Transactions",
            "min_support",
            "min_confidence",
            "Validation/Test",
        ):
            assert text in messages
        assert page["entries"][-1]["level"] == "success"
        assert page["entries"][-1]["progress"] == 100
        reopened.executor.shutdown(wait=True)
    finally:
        service.executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "error,status,level",
    [(RuntimeError("mining failed"), "failed", "error"), (Cancelled(), "cancelled", "warning")],
)
def test_association_failed_and_cancelled_jobs_leave_logs_without_results(
    configured, monkeypatch, error, status, level
):
    from movie_rating import experiment_pipeline

    prepare(configured)

    def fail(*args):
        raise error

    monkeypatch.setattr(experiment_pipeline, "execute", fail)
    service = LabService(configured)
    try:
        job = service.start(ExperimentRequest(group="association"))
        service.executor.shutdown(wait=True)
        finished = service.get(job["id"])
        assert finished["status"] == status
        assert "result" not in finished
        assert any(e["level"] == level for e in service.logs(job["id"], limit=500)["entries"])
    finally:
        service.executor.shutdown(wait=True)
