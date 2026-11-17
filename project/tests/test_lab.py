import io
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook
from pydantic import ValidationError
from sklearn.metrics import roc_auc_score

from movie_rating.api.app import create_app
from movie_rating.data.prepare import prepare, user_chronological_split
from movie_rating.experiment_data import split_ratings
from movie_rating.experiment_pipeline import execute
from movie_rating.experiment_supervised import classification_scores, supervised
from movie_rating.experiment_unsupervised import genre_profiles
from movie_rating.lab import CATALOG, LabRequest, LabService, native


@pytest.fixture
def lab_data():
    rows = [
        (u, m, 0.5 + 0.5 * ((u + m) % 10), 1000 + u * 100 + m)
        for u in range(1, 25)
        for m in range(1, 33)
    ]
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    movies = pd.DataFrame(
        {
            "movieId": range(1, 33),
            "title": [f"Movie {m}" for m in range(1, 33)],
            "genres": ["Action|Comedy" if m % 2 else "Drama|Romance" for m in range(1, 33)],
        }
    )
    return user_chronological_split(frame), movies


def test_catalog_covers_six_groups_and_required_algorithms():
    groups = {g["id"]: g for g in CATALOG["groups"]}
    assert set(groups) == {
        "rating",
        "classification",
        "clustering",
        "association",
        "pca",
        "reinforcement",
    }
    assert {a["id"] for a in groups["rating"]["algorithms"]} == {"linear", "xgboost", "svd", "ncf"}
    assert {a["id"] for a in groups["reinforcement"]["algorithms"]} == {
        "q_learning",
        "random",
        "popular",
    }
    assert "auc_roc" in {m["key"] for m in groups["classification"]["metrics"]}
    for g in groups.values():
        assert g["question"] and all(m["definition"] for m in g["metrics"])


@pytest.mark.parametrize(
    "group", ["rating", "classification", "clustering", "association", "pca", "reinforcement"]
)
def test_six_groups_fit_real_models_and_serialize_finite_results(group, lab_data):
    splits, movies = lab_data
    request = LabRequest(
        group=group, max_users=24, epochs=2, tune=False, components=[1, 2], episodes=10
    )
    result = execute(splits, movies, request, lambda *_: None)
    assert json.dumps(native(result), allow_nan=False)
    assert result["schema_version"] == 3 and result["rows"] and result["charts"]
    assert result["selection"] and result["best_configs"] and result["notes"]
    if group in ("rating", "classification"):
        assert set(result["cv"]) == set(request.algorithms)
        for data in result["cv"].values():
            assert len(data["folds"]) == 3
            for key, stats in data["metrics"].items():
                assert stats["n"] == 3
                assert stats["mean"] == pytest.approx(np.mean([r[key] for r in data["folds"]]))
                assert stats["std"] == pytest.approx(
                    np.std([r[key] for r in data["folds"]], ddof=1)
                )
        assert all(
            r["refit_ratings"] == len(splits["train"]) + len(splits["validation"])
            for r in result["rows"]
            if r["split"] == "test"
        )
    if group == "rating":
        curve = next(c for c in result["charts"] if c["kind"] == "line")
        assert [s["label"] for s in curve["series"]] == ["Train Loss", "Validation Loss"]
    if group == "reinforcement":
        assert all(-1 <= r["average_reward"] <= 1 for r in result["rows"])
        assert all(0 <= r["observed_feedback_fraction"] <= 1 for r in result["rows"])
    if group == "clustering":
        assert len(result["search"]) == 9
        assert all("noise_fraction" in r for r in result["rows"])


def test_auc_uses_probability_and_classification_boundary():
    actual = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.7, 0.4, 0.8])
    scores = classification_scores(actual, probability)
    assert scores["auc_roc"] == pytest.approx(roc_auc_score(actual, probability))
    assert scores["confusion_matrix"] == [[1, 1], [1, 1]]
    assert scores["f1"] == 0.5
    assert classification_scores([1, 1], np.array([0.8, 0.9]))["auc_roc"] is None


def test_tuning_is_opt_in_for_new_experiments():
    assert LabRequest(group="rating").tune is False
    assert LabRequest(group="classification", tune=True).tune is True


def test_full_data_rating_and_classification_defaults_are_bounded():
    from movie_rating.experiment_supervised import candidates

    rating = LabRequest(group="rating")
    classification = LabRequest(group="classification")
    assert rating.use_all_data and rating.epochs == 15
    assert candidates("xgboost", rating)[0] == {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.05,
    }
    assert candidates("logistic", classification)[0] == {"C": 1.0, "max_iter": 600}
    assert candidates("random_forest", classification)[0] == {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_leaf": 5,
    }


def test_linear_feature_statistics_stay_float32_and_build_in_batches(lab_data, monkeypatch):
    from movie_rating.lab import LinearRating
    from movie_rating.models import regression

    monkeypatch.setattr(regression, "_FEATURE_BATCH_ROWS", 64)
    splits, movies = lab_data
    model = LinearRating()
    model.set_movie_metadata(movies)
    model.fit(splits["train"])

    assert model.user_count.dtype == np.float32
    assert model.item_count.dtype == np.float32
    assert model.user_sum.dtype == np.float32
    assert model.item_sum.dtype == np.float32
    assert model.preference_sum.dtype == np.float32
    users, items = model.encode_queries(
        splits["validation"].userId.to_numpy(),
        splits["validation"].movieId.to_numpy(),
    )
    features = model._features(users, items)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features, model._feature_batch(users, items))
    predictions = model.predict(
        splits["validation"].userId.to_numpy(),
        splits["validation"].movieId.to_numpy(),
    )
    assert np.isfinite(predictions).all()


def test_full_data_mode_uses_prepared_temporal_splits(configured):
    prepare(configured)
    service = LabService(configured)
    splits, _, provenance = service._load(
        LabRequest(group="rating", max_users=10, use_all_data=True)
    )
    manifest = json.loads((configured.artifacts_dir / "data" / "manifest.json").read_text())
    assert provenance["protocol"] == "per_user_temporal_70_15_15_v2"
    assert provenance["sampled_ratings"] == sum(
        part["ratings"] for part in manifest["splits"].values()
    )
    assert provenance["sampled_users"] == len(
        set().union(*(set(frame.userId) for frame in splits.values()))
    )
    assert set(splits["train"].userId) == set(splits["test"].userId)
    for user in splits["train"].userId.unique():
        assert splits["train"].query("userId == @user").timestamp.max() < (
            splits["validation"].query("userId == @user").timestamp.min()
        )


def test_sampled_data_mode_respects_user_limit(configured):
    prepare(configured)
    service = LabService(configured)
    _, _, provenance = service._load(LabRequest(group="rating", max_users=10, use_all_data=False))
    assert provenance["protocol"] == "per_user_temporal_70_15_15_v2"
    assert provenance["sampled_users"] == 10


def test_svd_reports_epoch_progress(lab_data):
    from movie_rating.models.factorization import SVD

    splits, _ = lab_data
    progress = []
    SVD(factors=2, epochs=2).fit(
        splits["train"], progress=lambda epoch, total: progress.append((epoch, total))
    )
    assert progress == [(1, 2), (2, 2)]


def test_user_genre_features_use_rating_mean_not_genre_frequency():
    frame = pd.DataFrame(
        {"userId": [1, 1, 1, 2, 2], "movieId": [1, 2, 3, 1, 3], "rating": [1.0, 5.0, 3.0, 4.0, 2.0]}
    )
    movies = pd.DataFrame({"movieId": [1, 2, 3], "genres": ["A", "A", "B"]})
    x, users, genres = genre_profiles(frame, movies, 2)
    assert genres == ["A", "B"] and list(users) == [1, 2]
    np.testing.assert_allclose(x, [[3.0, 3.0], [4.0, 2.0]])


def test_shared_split_is_reproducible_disjoint_and_exactly_70_15_15(lab_data):
    parts, _ = lab_data
    frame = pd.concat(list(parts.values()), ignore_index=True).iloc[:200]
    a, b = split_ratings(frame), split_ratings(frame)
    assert [len(a[s]) for s in a] == [140, 30, 30]
    identities = []
    for s in a:
        pd.testing.assert_frame_equal(a[s], b[s])
        identities.append(set(zip(a[s].userId, a[s].movieId)))
    assert not identities[0] & identities[1]
    assert not identities[0] & identities[2]
    assert not identities[1] & identities[2]


def test_cv_features_never_receive_heldout_rows_and_test_cannot_change_selection(
    lab_data, monkeypatch
):
    import movie_rating.experiment_supervised as module

    splits, movies = lab_data
    calls = []
    original = module.feature_fit

    def checked(train, targets, metadata, algorithm, config, report=None):
        fit_ids = set(zip(train.userId, train.movieId))
        for frame in targets:
            assert not fit_ids & set(zip(frame.userId, frame.movieId))
        calls.append(len(train))
        return original(train, targets, metadata, algorithm, config, report=report)

    monkeypatch.setattr(module, "feature_fit", checked)
    request = LabRequest(group="classification", algorithms=["logistic"], tune=True)
    first = supervised(splits, movies, request, lambda *_: None)
    changed = {s: f.copy() for s, f in splits.items()}
    changed["test"]["rating"] = 0.5
    second = supervised(changed, movies, request, lambda *_: None)
    assert len(calls) == 2 * (3 + 3 + 1)  # 3 candidates, selected-config CV, final refit
    assert first["best_configs"] == second["best_configs"]
    assert first["cv"]["logistic"]["metrics"]["f1"] == second["cv"]["logistic"]["metrics"]["f1"]
    assert any(n == len(splits["train"]) + len(splits["validation"]) for n in calls)


@pytest.mark.parametrize(
    "payload",
    [
        dict(group="unknown"),
        dict(group="rating", seed=7),
        dict(group="rating", algorithms=[]),
        dict(group="rating", algorithms=["logistic"]),
        dict(group="rating", algorithms=["svd", "svd"]),
        dict(group="clustering", eps_values=[-1]),
        dict(group="reinforcement", discount_factor=1.1),
        dict(group="association", min_confidence=0),
    ],
)
def test_invalid_experiment_configuration_is_rejected(payload):
    with pytest.raises(ValidationError):
        LabRequest(**payload)


def wait_for_job(service, run_id):
    for _ in range(500):
        job = service.get(run_id)
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.02)
    pytest.fail("Experiment did not finish")


def test_async_api_history_provenance_csv_and_excel(configured):
    prepare(configured)
    root = "/api/v1/experiment-lab"
    with TestClient(create_app(configured)) as client:
        catalog = client.get(root + "/catalog").json()
        assert catalog["data_ready"] and catalog["statistics"]["ratings"] == 120
        response = client.post(
            root + "/runs",
            json={
                "group": "classification",
                "max_users": 10,
                "algorithms": ["logistic"],
                "tune": False,
            },
        )
        assert response.status_code == 202
        run_id = response.json()["id"]
        job = wait_for_job(LabService(configured), run_id)
        assert job["status"] == "completed", job
        assert job["result"]["provenance"]["protocol"] == "per_user_temporal_70_15_15_v2"
        assert job["result"]["runtime"]["code_sha256"]
        csv = client.get(root + f"/runs/{run_id}/export/csv")
        assert csv.status_code == 200 and "cv_f1_std" in csv.text
        excel = client.get(root + f"/runs/{run_id}/export/xlsx")
        assert excel.status_code == 200
        workbook = load_workbook(io.BytesIO(excel.content))
        assert {"Metrics", "CV folds", "Protocol"} <= set(workbook.sheetnames)
        assert client.get(root + f"/runs/{run_id}/export/json").status_code == 200
        assert client.get(root + f"/runs/{run_id}/export/exe").status_code == 422
        assert client.get(root + "/runs?group=classification").json()[0]["id"] == run_id
        assert client.get(root + "/runs/not-a-run").status_code == 404
        assert client.post(root + "/runs", json={"group": "rating", "seed": 1}).status_code == 422
    assert LabService(configured).get(run_id)["status"] == "completed"


def test_missing_data_and_duplicate_active_jobs(configured):
    service = LabService(configured)
    with pytest.raises(FileNotFoundError):
        service.start(LabRequest(group="rating"))
    prepare(configured)
    service.active = "already-running"
    with pytest.raises(RuntimeError):
        service.start(LabRequest(group="rating"))


def test_cancellation_does_not_publish_partial_results(configured, monkeypatch):
    prepare(configured)
    service = LabService(configured)
    entered, release = threading.Event(), threading.Event()
    original = service._load

    def blocked(request):
        entered.set()
        release.wait(3)
        return original(request)

    monkeypatch.setattr(service, "_load", blocked)
    job = service.start(LabRequest(group="classification", max_users=10))
    assert entered.wait(3)
    service.cancel(job["id"])
    release.set()
    finished = wait_for_job(service, job["id"])
    assert finished["status"] == "cancelled" and "result" not in finished
    assert service.active is None


def test_preprocessing_cache_is_shared_across_groups(configured):
    prepare(configured)
    service = LabService(configured)
    a = service._load(LabRequest(group="classification", max_users=10))
    b = service._load(LabRequest(group="rating", max_users=10))
    assert a is b
    assert a[2]["sample_sha256"] == b[2]["sample_sha256"]


def test_atomic_report_write_retries_windows_sharing_lock(tmp_path, monkeypatch):
    import movie_rating.lab as lab

    path = tmp_path / "job.json"
    original = lab.os.replace
    attempts = []

    def busy_once(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("Temporarily being read")
        return original(source, destination)

    monkeypatch.setattr(lab.os, "replace", busy_once)
    monkeypatch.setattr(lab.time, "sleep", lambda _: None)
    lab.save_json(path, {"status": "completed"})
    assert json.loads(path.read_text()) == {"status": "completed"}
    assert len(attempts) == 3 and not list(tmp_path.glob("*.tmp"))


def test_existing_per_user_temporal_split_preserves_time_ties_and_short_histories():
    rows = [(u, i, 4.0, i // 2) for u in (1, 2) for i in range(24)]
    rows += [(3, i, 3.0, i) for i in range(5)]
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    splits = user_chronological_split(frame)
    assert sum(map(len, splits.values())) == 48
    for u in (1, 2):
        histories = [
            splits[s].loc[lambda f: f.userId == u] for s in ("train", "validation", "test")
        ]
        assert histories[0].timestamp.max() < histories[1].timestamp.min()
        assert histories[1].timestamp.max() < histories[2].timestamp.min()
    assert all(set(f.userId) == {1, 2} for f in splits.values())


def test_offline_rewards_are_observed_and_unknown_feedback_is_zero():
    from movie_rating.experiment_reinforcement import evaluate

    for policy in ("q_learning", "random", "popular"):
        full = evaluate(policy, {}, {1: {a: 1.0 for a in range(5)}}, list(range(5)), episodes=10)
        assert full["cumulative_reward"] == 50 and full["average_reward"] == 1
        assert full["observed_feedback_fraction"] == 1
        empty = evaluate(policy, {}, {1: {}}, list(range(5)), episodes=10)
        assert empty["cumulative_reward"] == 0 and empty["observed_feedback_fraction"] == 0


def test_failed_run_is_saved_without_results_and_releases_worker(configured, monkeypatch):
    import movie_rating.experiment_pipeline as pipeline

    prepare(configured)
    service = LabService(configured)

    def fail(*args):
        raise ValueError("Expected training failure")

    monkeypatch.setattr(pipeline, "execute", fail)
    job = service.start(LabRequest(group="rating", max_users=10))
    failed = wait_for_job(service, job["id"])
    assert failed["status"] == "failed" and "Expected training failure" in failed["message"]
    assert "result" not in failed and service.active is None


def test_legacy_history_remains_readable(configured):
    from movie_rating.lab import save_json

    service = LabService(configured)
    folder = service.directory / "20260101T000000-aabbccdd"
    folder.mkdir(parents=True)
    save_json(
        folder / "job.json",
        {"id": folder.name, "schema_version": 2, "status": "completed", "group": "rating"},
    )
    save_json(
        folder / "result.json",
        {"schema_version": 2, "rows": [{"algorithm": "linear", "rmse": 1.2}]},
    )
    assert service.list_runs()[0]["schema_version"] == 2
    assert service.get(folder.name)["result"]["rows"][0]["rmse"] == 1.2


def test_ncf_stops_when_validation_does_not_improve(lab_data, monkeypatch):
    from movie_rating.models.neural import NeuralCF

    splits, _ = lab_data
    model = NeuralCF(epochs=8, patience=1, factors=2, hidden_size=4, seed=42)
    monkeypatch.setattr(model, "predict", lambda users, movies: np.full(len(users), 3.0))
    progress = []
    model.fit(
        splits["train"],
        validation=splits["validation"],
        progress=lambda epoch, total: progress.append((epoch, total)),
    )
    assert model.best_epoch == 1
    assert len(model.learning_curve) == 2
    assert all("train_rating_mse" in row for row in model.learning_curve)
    assert progress == [(1, 8), (2, 8)]


def test_status_reader_retries_transient_windows_sharing_lock(tmp_path, monkeypatch):
    from pathlib import Path

    import movie_rating.lab as lab

    path = tmp_path / "job.json"
    path.write_text('{"status":"completed"}', encoding="utf-8")
    original = Path.read_text
    attempts = []

    def busy_once(current, *args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("Temporary Windows sharing violation")
        return original(current, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", busy_once)
    monkeypatch.setattr(lab.time, "sleep", lambda _: None)
    assert lab.read_json(path) == {"status": "completed"}
    assert len(attempts) == 3
