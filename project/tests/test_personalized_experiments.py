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
from movie_rating.experiment_supervised import candidates, feature_fit, fit_rating, supervised
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


@pytest.mark.parametrize("use_all_data", [False, True])
def test_reinforcement_loader_uses_per_user_temporal_splits(configured, use_all_data):
    from movie_rating.experiment_config import ExperimentRequest

    prepare(configured)
    source_manifest = configured.artifacts_dir / "data" / "manifest.json"
    original_manifest = source_manifest.read_bytes()
    data = ExperimentData(configured.artifacts_dir / "data")
    splits, _, provenance = data.load(
        ExperimentRequest(group="reinforcement", use_all_data=use_all_data, max_users=10)
    )
    assert provenance["protocol"] == PERSONALIZED_PROTOCOL
    assert provenance["exclusions"] is not None
    assert source_manifest.read_bytes() == original_manifest
    for user in set(splits["train"].userId) & set(splits["validation"].userId):
        train_user = splits["train"].loc[lambda frame: frame.userId == user]
        valid_user = splits["validation"].loc[lambda frame: frame.userId == user]
        assert train_user.timestamp.max() < valid_user.timestamp.min()
    for user in set(splits["validation"].userId) & set(splits["test"].userId):
        valid_user = splits["validation"].loc[lambda frame: frame.userId == user]
        test_user = splits["test"].loc[lambda frame: frame.userId == user]
        assert valid_user.timestamp.max() < test_user.timestamp.min()


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


def _reduction_rating_data():
    train_rows, validation_rows, test_rows = [], [], []
    for user in range(1, 21):
        for movie in range(1, 33):
            rating = 0.5 + 0.5 * ((user * 7 + movie * 3 + (user % 4) * (movie % 5)) % 10)
            row = (user, movie, rating, user * 1000 + movie)
            (train_rows if movie <= 24 else validation_rows if movie <= 28 else test_rows).append(row)
    columns = ["userId", "movieId", "rating", "timestamp"]
    movies = pd.DataFrame({
        "movieId": range(1, 33), "title": [f"Movie {i}" for i in range(1, 33)],
        "genres": ["Action" if i % 4 == 0 else "Comedy" if i % 4 == 1
                   else "Drama" if i % 4 == 2 else "Romance" for i in range(1, 33)],
    })
    return {
        "train": pd.DataFrame(train_rows, columns=columns),
        "validation": pd.DataFrame(validation_rows, columns=columns),
        "test": pd.DataFrame(test_rows, columns=columns),
    }, movies


def test_supervised_compare_runs_train_only_reducer_per_fold_and_ignores_test_labels():
    from movie_rating.experiment_config import ExperimentRequest

    splits, movies = _reduction_rating_data()
    request = ExperimentRequest(
        group="rating", algorithms=["linear"], reduction_mode="compare",
        reduction_methods=["pca"], components=[1], tune=False,
    )
    result = supervised(splits, movies, request, lambda *_: None)
    assert {row["variant_id"] for row in result["rows"]} == {"linear_raw", "linear_pca_1"}
    assert result["best_configs"]["linear_pca_1"]["reduction"]["fit_scope"] == "train_plus_validation"
    assert result["cv"]["linear_pca_1"]["fit_scope"] == "train_fold_specific"
    folds = result["cv"]["linear_pca_1"]["folds"]
    assert {fold["reduction"]["fit_scope"] for fold in folds} == {
        "train_fold_1", "train_fold_2", "train_fold_3"
    }
    changed = {name: frame.copy() for name, frame in splits.items()}
    changed["test"]["rating"] = 5.0 - changed["test"]["rating"]
    after = supervised(changed, movies, request, lambda *_: None)
    first_config = result["best_configs"]["linear_pca_1"]
    second_config = after["best_configs"]["linear_pca_1"]
    assert first_config.keys() == second_config.keys()
    assert first_config["reduction"]["fingerprint"] == second_config["reduction"]["fingerprint"]
    assert {k: v for k, v in first_config.items() if k != "reduction"} == {
        k: v for k, v in second_config.items() if k != "reduction"
    }
    assert result["reduction"]["selected_variant_id"] == after["reduction"]["selected_variant_id"]


@pytest.mark.parametrize("algorithm,config", [
    ("logistic", {"C": 1.0, "max_iter": 100}),
    ("random_forest", {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 1}),
])
@pytest.mark.parametrize("method", ["pca", "truncated_svd"])
def test_reduced_classifiers_transform_validation_with_train_scaler_and_reducer(algorithm, config, method):
    splits, movies = _reduction_rating_data()
    builder, model, _ = feature_fit(
        splits["train"], [splits["validation"]], movies, algorithm, config,
        reduction_variant={"method": method, "components": 1},
        fit_scope="train",
    )
    features = np.concatenate(list(builder.iter_features(splits["validation"])))
    probability = model.predict_proba(features)
    assert model.reduction_metadata["fit_scope"] == "train"
    assert model.reduction_metadata["output_dimensions"] == 1
    assert np.isfinite(probability).all() and len(probability) == len(splits["validation"])


@pytest.mark.parametrize("algorithm", ["linear", "xgboost"])
def test_reduced_rating_adapters_use_compressed_interaction_features(algorithm):
    from movie_rating.experiment_config import ExperimentRequest

    splits, movies = _reduction_rating_data()
    request = ExperimentRequest(group="rating", algorithms=[algorithm], epochs=2)
    config = (
        {} if algorithm == "linear"
        else {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.05}
    )
    model = fit_rating(
        algorithm, config, request, splits["train"], movies,
        reduction_variant={"method": "pca", "components": 1}, fit_scope="train",
    )
    prediction = model.predict(splits["validation"].userId, splits["validation"].movieId)
    assert model.reduction_metadata["output_dimensions"] == 1
    assert model.reduction_metadata["fit_rows"] == len(splits["train"])
    assert prediction.shape == (len(splits["validation"]),)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("algorithm", ["svd", "ncf"])
def test_svd_and_ncf_hybrids_use_reduced_features_and_survive_serialization(algorithm):
    from movie_rating.experiment_config import ExperimentRequest, ModelParameters

    splits, movies = _reduction_rating_data()
    request = ExperimentRequest(
        group="rating", algorithms=[algorithm], epochs=2, patience=1,
        parameters=ModelParameters(factors=4, hidden_size=8, batch_size=32),
    )
    config = (
        {"factors": 4, "learning_rate": 0.005, "regularization": 0.02}
        if algorithm == "svd"
        else {"factors": 4, "hidden_size": 8, "learning_rate": 0.003,
              "regularization": 0.02, "dropout": 0.0, "batch_size": 32}
    )
    model = fit_rating(
        algorithm, config, request, splits["train"], movies,
        validation=splits["validation"], reduction_variant={"method": "pca", "components": 1},
        fit_scope="train",
    )
    users, items = splits["validation"].userId, splits["validation"].movieId
    predicted = model.predict(users, items)
    baseline = model.base_model.predict(users, items)
    assert model.reduction_metadata["fit_scope"] == "train"
    assert model.reduction_metadata["output_dimensions"] == 1
    assert model.side_model is not None
    assert predicted.shape == baseline.shape and np.isfinite(predicted).all()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(users, items), predicted, atol=1e-6)

    if algorithm == "svd":
        original = model.reducer.components_.copy()
        model.reducer.components_[0] *= -1
        flipped = model.predict(users, items)
        model.reducer.components_[:] = original
        assert not np.allclose(flipped, predicted)


@pytest.mark.parametrize("mode", ["enabled", "compare"])
def test_ncf_reduction_pipeline_execute_has_distinct_hybrid_curve_and_locked_epochs(mode, monkeypatch):
    from movie_rating import experiment_supervised
    from movie_rating.experiment_config import ExperimentRequest, ModelParameters
    from movie_rating.experiment_pipeline import execute

    splits, movies = _reduction_rating_data()
    request = ExperimentRequest(
        group="rating", algorithms=["ncf"], reduction_mode=mode,
        reduction_methods=["pca"], components=[1], epochs=3, patience=1,
        parameters=ModelParameters(factors=4, hidden_size=8, batch_size=32),
    )
    fit_calls = []
    original_fit_rating = experiment_supervised.fit_rating

    def tracked_fit_rating(algorithm, *args, **kwargs):
        if algorithm == "ncf" and kwargs.get("reduction_variant") is not None:
            fit_calls.append({
                "epochs": kwargs.get("epochs"),
                "side_epochs": kwargs.get("side_epochs"),
                "fit_scope": kwargs.get("fit_scope", "train"),
            })
        return original_fit_rating(algorithm, *args, **kwargs)

    monkeypatch.setattr(experiment_supervised, "fit_rating", tracked_fit_rating)
    result = execute(splits, movies, request, lambda *_: None)
    reduced_id = "ncf_pca_1"
    assert result["schema_version"] == 4
    assert reduced_id in result["best_configs"]
    selected = result["best_configs"][reduced_id]
    assert 1 <= selected["epochs"] <= request.epochs
    assert 1 <= selected["side_epochs"] <= request.epochs
    assert len(fit_calls) == 5  # Validation selection, three CV folds, final refit.
    assert fit_calls[0] == {"epochs": None, "side_epochs": None, "fit_scope": "train"}
    assert [call["fit_scope"] for call in fit_calls[1:4]] == [
        "train_fold_1", "train_fold_2", "train_fold_3"
    ]
    assert fit_calls[4]["fit_scope"] == "train_plus_validation"
    assert all(call["epochs"] == selected["epochs"] for call in fit_calls[1:])
    assert all(call["side_epochs"] == selected["side_epochs"] for call in fit_calls[1:])
    assert selected["reduction"]["fit_scope"] == "train_plus_validation"
    assert result["cv"][reduced_id]["fit_scope"] == "train_fold_specific"
    assert {fold["reduction"]["fit_scope"] for fold in result["cv"][reduced_id]["folds"]} == {
        "train_fold_1", "train_fold_2", "train_fold_3"
    }
    final = next(row for row in result["rows"] if row.get("variant_id") == reduced_id and row["split"] == "test")
    assert final["refit_ratings"] == len(splits["train"]) + len(splits["validation"])
    assert np.isfinite(final["rmse"])
    curve = next(chart for chart in result["charts"] if chart["title"].startswith(
        "NCF hybrid · Correction Branch / Early Stopping · pca_1"
    ))
    assert curve["y_label"] == "Correction MSE ↓"
    assert [series["label"] for series in curve["series"]] == [
        "Train correction MSE", "Validation correction MSE"
    ]
    assert all(series["points"] for series in curve["series"])
    selected_curve = next(
        option["learning_curve"] for option in result["tuning"]
        if option.get("variant_id") == reduced_id
    )
    assert all("train_rating_mse" not in row for row in selected_curve)
    assert all("train_correction_mse" in row for row in selected_curve)
    assert any("không phải lỗi tổng" in note for note in result["notes"])
    if mode == "compare":
        raw = next(chart for chart in result["charts"] if chart["title"].startswith("NCF · Rating MSE"))
        assert raw["y_label"] == "Rating MSE ↓"
        assert [series["label"] for series in raw["series"]] == [
            "Train rating MSE", "Validation rating MSE"
        ]


def test_all_rating_algorithms_reduced_pipeline_execute_completes():
    from movie_rating.experiment_config import ExperimentRequest, ModelParameters
    from movie_rating.experiment_pipeline import execute

    splits, movies = _reduction_rating_data()
    request = ExperimentRequest(
        group="rating", algorithms=["linear", "xgboost", "svd", "ncf"],
        reduction_mode="compare", reduction_methods=["pca", "truncated_svd"], components=[1],
        epochs=2, patience=1,
        parameters=ModelParameters(
            factors=4, hidden_size=8, batch_size=32, n_estimators=10, max_depth=2
        ),
    )
    result = execute(splits, movies, request, lambda *_: None)
    assert result["schema_version"] == 4
    assert {row["algorithm"] for row in result["rows"]} == {
        "linear", "xgboost", "svd", "ncf"
    }
    assert all(row["split"] in {"validation", "test"} for row in result["rows"])
    reduced_rows = [row for row in result["rows"] if row.get("reduction")]
    assert {row["reduction"]["method"] for row in reduced_rows} == {"pca", "truncated_svd"}
    assert all(row["reduction"]["output_dimensions"] == 1 for row in reduced_rows)
    assert any(chart["title"].startswith("NCF hybrid · Correction Branch") and "pca_1" in chart["title"] for chart in result["charts"])
    assert any(chart["title"].startswith("NCF hybrid · Correction Branch") and "truncated_svd_1" in chart["title"] for chart in result["charts"])


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


@pytest.mark.parametrize("pressure", ["worker", "system"])
def test_large_job_is_isolated_without_automatic_memory_stop(configured, monkeypatch, pressure):
    import movie_rating.experiment_worker as worker

    prepare(configured)
    scratch_parent = configured.artifacts_dir / "worker-scratch"
    monkeypatch.setenv("MOVIE_SCRATCH_DIR", str(scratch_parent))
    service = LabService(configured)
    stats = service.data.catalog_statistics()
    monkeypatch.setattr(service.data, "catalog_statistics", lambda: {**stats, "ratings": 1_000_000})
    if pressure == "worker":
        monkeypatch.setattr(
            worker.psutil,
            "Process",
            lambda pid: SimpleNamespace(memory_full_info=lambda: SimpleNamespace(uss=2**40)),
        )
    else:
        memory = worker.psutil.virtual_memory()
        monkeypatch.setattr(
            worker.psutil,
            "virtual_memory",
            lambda: memory._replace(available=0),
        )
    job = service.start(
        ExperimentRequest(group="rating", algorithms=["linear"], memory_budget_mb=2048)
    )
    result = _finished(service, job["id"])
    assert result["status"] == "completed", result["message"]
    assert result["result"]["runtime"]["isolated_process"]
    assert result["result"]["runtime"]["automatic_memory_limit"] is False
    assert not (service.directory / job["id"] / "scratch").exists()
    assert not (scratch_parent / job["id"]).exists()
