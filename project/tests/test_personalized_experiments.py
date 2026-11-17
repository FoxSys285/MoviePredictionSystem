import json
import pickle
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from movie_rating.data.prepare import prepare
from movie_rating.data.temporal import PERSONALIZED_PROTOCOL, temporal_folds, user_temporal_split
from movie_rating.experiment_config import ExperimentRequest
from movie_rating.experiment_data import ExperimentData
from movie_rating.experiment_supervised import candidates, feature_fit
from movie_rating.lab import LabService, LinearRating
from movie_rating.memory import feature_buffer, serialized_size
from movie_rating.models.regression import StreamingLinearRegression, XGBoostRegression


def test_temporal_holdout_and_cv_are_past_only_with_timestamp_ties():
    frame = pd.DataFrame(
        [(u, m, 1 + (m % 5), m // 2) for u in range(1, 5) for m in range(1, 41)],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    frame.loc[frame.userId == 4, "timestamp"] = np.repeat(np.arange(5), [7, 14, 7, 6, 6])
    frame = pd.concat(
        [frame, pd.DataFrame([(5, 1, 4, 1), (6, 1, 4, 1)] * 2, columns=frame.columns)],
        ignore_index=True,
    )
    splits, report = user_temporal_split(frame.sample(frac=1, random_state=9))
    assert report["excluded_users"] == 2 and report["excluded_ratings"] == 4
    for user in range(1, 5):
        groups = [f.loc[f.userId == user] for f in splits.values()]
        assert groups[0].timestamp.max() < groups[1].timestamp.min()
        assert groups[1].timestamp.max() < groups[2].timestamp.min()
    previous = set()
    for fit_idx, held_idx in temporal_folds(splits["train"]):
        fit, held = splits["train"].iloc[fit_idx], splits["train"].iloc[held_idx]
        assert not set(fit_idx) & set(held_idx)
        assert previous <= set(fit_idx)
        previous = set(fit_idx)
        assert set(held.userId) <= set(fit.userId)
        for user in held.userId.unique():
            assert fit.loc[fit.userId == user].timestamp.max() < (
                held.loc[held.userId == user].timestamp.min()
            )


def test_prepare_and_legacy_migration_record_protocol_without_overwriting(configured):
    configured.data.split_strategy = "per_user_temporal"
    configured.data.train_fraction, configured.data.validation_fraction = 0.7, 0.15
    manifest = prepare(configured)
    assert manifest["protocol"] == PERSONALIZED_PROTOCOL
    assert manifest["exclusions"]["excluded_ratings"] == 0
    path = configured.artifacts_dir / "data" / "manifest.json"
    original = path.read_bytes()
    legacy = json.loads(original)
    legacy.pop("protocol")
    path.write_text(json.dumps(legacy), encoding="utf-8")
    source = path.read_bytes()
    data = ExperimentData(path.parent)
    directory = data.personalized_directory()
    assert directory != path.parent and path.read_bytes() == source
    assert data.personalized_directory() == directory
    assert (
        json.loads((directory / "manifest.json").read_text())["protocol"] == PERSONALIZED_PROTOCOL
    )


def test_new_feature_targets_are_excluded_and_recent_history_is_bounded():
    frame = pd.DataFrame(
        {
            "userId": [1] * 30,
            "movieId": range(1, 31),
            "rating": [1.0] * 10 + [4.0] * 20,
            "timestamp": range(30),
        }
    )
    movies = pd.DataFrame({"movieId": range(1, 31), "genres": ["Drama"] * 30})
    model = LinearRating()
    model.set_movie_metadata(movies)
    model.fit_statistics(frame)
    assert model.recent_count[0] == 20 and model.recent_sum[0] == 80
    features = np.concatenate(list(model.iter_features(frame, training=True)))
    prior = model.global_positive
    assert features[0, 6] == pytest.approx((20 + 5 * prior) / 34)
    assert features[-1, 6] == pytest.approx((19 + 5 * prior) / 34)
    assert features[-1, 7] == pytest.approx(prior)  # singleton movie target excluded
    assert features[0, 10] == pytest.approx((80 + 5 * model.global_mean) / 25 - model.global_mean)
    assert features[-1, 10] == pytest.approx((76 + 5 * model.global_mean) / 24 - model.global_mean)
    assert features.dtype == np.float32 and np.isfinite(features).all()


def test_streaming_ols_matches_dense_predictions_with_collinear_columns():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1000, 5)).astype(np.float32)
    x = np.column_stack([x, x[:, 0] * 2, np.ones(1000)]).astype(np.float32)
    y = x[:, 1] * 3 - x[:, 2] + rng.normal(size=1000) * 0.05
    streaming = StreamingLinearRegression().fit_batches(
        (x[start : start + 73], y[start : start + 73]) for start in range(0, len(x), 73)
    )
    dense = LinearRegression().fit(x, y)
    np.testing.assert_allclose(streaming.predict(x), dense.predict(x), atol=1e-5)


def test_classifier_does_not_fit_an_auxiliary_regressor(configured, monkeypatch):
    from movie_rating.data.prepare import load_ratings

    def forbidden(*args):
        raise AssertionError("classification must not fit a regression model")

    monkeypatch.setattr(LinearRating, "fit", forbidden)
    train = load_ratings(configured)
    movies = pd.read_csv(configured.dataset_dir / "movies.csv")
    builder, model, _ = feature_fit(train, [], movies, "logistic", {"C": 1.0, "max_iter": 100})
    assert builder.user_count.sum() == len(train) and model.classes_.tolist() == [0, 1]


def test_disk_backed_features_and_streaming_model_size_clean_up(tmp_path, monkeypatch):
    import movie_rating.memory as memory

    monkeypatch.setenv("MOVIE_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setattr(memory, "MEMMAP_THRESHOLD_BYTES", 1)
    with feature_buffer(100, 5, (np.full((20, 5), i, dtype=np.float32) for i in range(5))) as x:
        assert isinstance(x, np.memmap)
        np.testing.assert_array_equal(x[80:], 4)
    assert not list(tmp_path.iterdir())
    artifact = {"values": np.arange(100, dtype=np.float32)}
    assert serialized_size(artifact) == len(pickle.dumps(artifact))


def test_xgboost_external_memory_predicts_and_releases_cache(configured, tmp_path, monkeypatch):
    from movie_rating.data.prepare import load_ratings
    from movie_rating.models import regression

    unicode_scratch = tmp_path / "bộ nhớ tạm"
    monkeypatch.setenv("MOVIE_SCRATCH_DIR", str(unicode_scratch))
    monkeypatch.setattr(regression, "_EXTERNAL_MEMORY_ROWS", 1)
    frame = load_ratings(configured)
    model = XGBoostRegression(n_estimators=10, max_depth=2, n_jobs=1).fit(frame)
    predictions = model.predict(frame.userId, frame.movieId)
    assert np.isfinite(predictions).all()
    assert not list(unicode_scratch.iterdir())
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(frame.userId, frame.movieId), predictions)


def test_tuning_includes_both_simpler_and_stronger_candidates():
    request = ExperimentRequest(group="rating", tune=True)
    assert {c["factors"] for c in candidates("svd", request)} == {16, 32, 64}
    configs = candidates("xgboost", request)
    assert {c["max_depth"] for c in configs} == {3, 6, 8}
    request = ExperimentRequest(group="classification", tune=True)
    assert {c["C"] for c in candidates("logistic", request)} == {0.1, 1.0, 10.0}


def _finished(service, run_id):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        job = service.get(run_id)
        if job["status"] in ("completed", "failed", "cancelled") and service.active is None:
            return job
        time.sleep(0.05)
    raise AssertionError("Worker did not finish")


@pytest.mark.parametrize("force_limit", [False, True])
def test_large_job_is_isolated_and_memory_limit_stops_worker(configured, monkeypatch, force_limit):
    import movie_rating.experiment_worker as worker

    prepare(configured)
    scratch_parent = configured.artifacts_dir / "worker-scratch"
    monkeypatch.setenv("MOVIE_SCRATCH_DIR", str(scratch_parent))
    service = LabService(configured)
    stats = service.data.catalog_statistics()
    monkeypatch.setattr(service.data, "catalog_statistics", lambda: {**stats, "ratings": 1_000_000})
    if force_limit:
        monkeypatch.setattr(
            worker.psutil,
            "Process",
            lambda pid: SimpleNamespace(memory_full_info=lambda: SimpleNamespace(uss=2**40)),
        )
    job = service.start(
        ExperimentRequest(group="rating", algorithms=["linear"], memory_budget_mb=2048)
    )
    result = _finished(service, job["id"])
    if force_limit:
        assert result["status"] == "failed" and "RAM" in result["message"]
        assert "result" not in result
    else:
        assert result["status"] == "completed", result["message"]
        assert result["result"]["runtime"]["isolated_process"]
    assert not (service.directory / job["id"] / "scratch").exists()
    assert not (scratch_parent / job["id"]).exists()
