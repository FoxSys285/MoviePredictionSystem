import pandas as pd
import pytest


def _small_supervised_data():
    train, validation, test = [], [], []
    for user in range(1, 11):
        for movie in range(1, 13):
            row = (user, movie, 0.5 + 0.5 * ((user * 3 + movie * 7) % 10), movie)
            (train if movie <= 8 else validation if movie <= 10 else test).append(row)
    columns = ["userId", "movieId", "rating", "timestamp"]
    movies = pd.DataFrame({
        "movieId": range(1, 13),
        "title": [f"Movie {movie}" for movie in range(1, 13)],
        "genres": ["Action" if movie % 2 else "Drama" for movie in range(1, 13)],
    })
    return {
        "train": pd.DataFrame(train, columns=columns),
        "validation": pd.DataFrame(validation, columns=columns),
        "test": pd.DataFrame(test, columns=columns),
    }, movies


def test_interrupted_supervised_run_skips_completed_validation_and_keeps_test_out_of_fits(
    tmp_path, monkeypatch
):
    from movie_rating import experiment_supervised
    from movie_rating.experiment_config import ExperimentRequest
    from movie_rating.experiment_pipeline import execute
    from movie_rating.experiment_resume import ProgressStore

    splits, movies = _small_supervised_data()
    request = ExperimentRequest(group="rating", algorithms=["linear"], reduction_mode="off")
    identity = {"request": request.model_dump(), "sample": "frozen-test-data"}
    progress = ProgressStore(tmp_path, identity)
    calls = []
    original_fit = experiment_supervised.fit_rating

    def tracked_fit(algorithm, config, request, train, movies, validation=None, **kwargs):
        calls.append((train.copy(), validation.copy() if validation is not None else None))
        return original_fit(algorithm, config, request, train, movies, validation, **kwargs)

    monkeypatch.setattr(experiment_supervised, "fit_rating", tracked_fit)

    class StopAfterSavedValidation(Exception):
        pass

    def interrupt_after_validation(_progress, message):
        if "Validation RMSE=" in message:
            raise StopAfterSavedValidation

    with pytest.raises(StopAfterSavedValidation):
        execute(splits, movies, request, interrupt_after_validation, progress_store=progress)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(calls[0][0], splits["train"])
    pd.testing.assert_frame_equal(calls[0][1], splits["validation"])
    assert ProgressStore.summary(tmp_path)["completed_work_units"] == 1

    resumed = ProgressStore(tmp_path, identity)
    result = execute(splits, movies, request, lambda *_: None, progress_store=resumed)
    assert len(calls) == 5  # One Validation fit, three CV folds and the final refit.
    assert result["cv_folds"] == 3
    assert {row["split"] for row in result["rows"]} == {"validation", "test"}
    for train, validation in calls[1:4]:
        assert validation is None
        assert set(map(tuple, train[["userId", "movieId"]].to_numpy())) <= set(
            map(tuple, splits["train"][["userId", "movieId"]].to_numpy())
        )
    pd.testing.assert_frame_equal(
        calls[4][0], pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    )
    assert ProgressStore.summary(tmp_path)["completed_work_units"] == 5


def test_resume_rejects_changed_data_or_parameters(tmp_path):
    from movie_rating.experiment_resume import ProgressStore, ResumeStateMismatch

    ProgressStore(tmp_path, {"sample": "a", "seed": 42, "parameters": {"depth": 6}})
    with pytest.raises(ResumeStateMismatch, match="Dataset, split, seed"):
        ProgressStore(tmp_path, {"sample": "b", "seed": 42, "parameters": {"depth": 6}})


def test_resume_fingerprint_covers_each_split_runtime_and_training_libraries(tmp_path):
    import torch

    from movie_rating.experiment_config import ExperimentRequest
    from movie_rating.experiment_resume import ProgressStore, resume_fingerprint

    splits, movies = _small_supervised_data()
    request = ExperimentRequest(group="rating", algorithms=["ncf"], reduction_mode="off")
    fingerprint = resume_fingerprint(
        request, splits, movies,
        {"sample_sha256": "sample", "protocol": "per_user_temporal_test", "data_seed": 42},
    )
    assert fingerprint["data"]["split_counts"] == {
        name: len(frame) for name, frame in splits.items()
    }
    assert set(fingerprint["data"]["split_sha256"]) == set(splits)
    assert fingerprint["libraries"]["threadpoolctl"]
    assert fingerprint["libraries"]["torch"]
    expected_device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    assert fingerprint["runtime"]["device"] == expected_device
    changed_splits = {name: frame.copy() for name, frame in splits.items()}
    changed_splits["train"].loc[0, "rating"] += 0.01
    changed_fingerprint = resume_fingerprint(
        request, changed_splits, movies,
        {"sample_sha256": "sample", "protocol": "per_user_temporal_test", "data_seed": 42},
    )
    assert changed_fingerprint["data"]["split_sha256"]["train"] != fingerprint["data"]["split_sha256"]["train"]
    ProgressStore(tmp_path, fingerprint)
    changed_runtime = {
        **fingerprint,
        "runtime": {**fingerprint["runtime"], "driver_version": "changed"},
    }
    with pytest.raises(ValueError, match="Dataset, split, seed"):
        ProgressStore(tmp_path, changed_runtime)


def test_ncf_epoch_checkpoint_restores_optimizer_and_rng_state(tmp_path):
    import torch

    from movie_rating.models.neural import NeuralCF

    rows = []
    for user in range(1, 11):
        for movie in range(1, 9):
            rows.append((user, movie, 0.5 + 0.5 * ((user * 7 + movie * 3) % 10), movie))
    train = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    validation_mask = train.groupby("userId").cumcount() >= 6
    validation = train.loc[validation_mask].reset_index(drop=True)
    train = train.loc[~validation_mask].reset_index(drop=True)
    checkpoint_path = tmp_path / "ncf-fit.pt"
    expected_key = "raw/ncf/validation-0"

    def make_model():
        return NeuralCF(
            factors=3, hidden_size=8, epochs=4, batch_size=16, patience=4, seed=123
        )

    uninterrupted = make_model().fit(train, validation=validation)

    class StopAfterTwoEpochs(Exception):
        pass

    def stop_after_two(epoch, _total):
        if epoch == 2:
            raise StopAfterTwoEpochs

    with pytest.raises(StopAfterTwoEpochs):
        make_model().fit(
            train,
            validation=validation,
            progress=stop_after_two,
            checkpoint_path=checkpoint_path,
            checkpoint_key=expected_key,
        )
    assert checkpoint_path.is_file()
    resumed_epochs = []
    resumed = make_model().fit(
        train,
        validation=validation,
        progress=lambda epoch, _total: resumed_epochs.append(epoch),
        checkpoint_path=checkpoint_path,
        checkpoint_key=expected_key,
    )
    assert resumed_epochs == [3, 4]
    assert resumed.learning_curve == uninterrupted.learning_curve
    for name, tensor in resumed.network.state_dict().items():
        torch.testing.assert_close(tensor, uninterrupted.network.state_dict()[name])


def test_ncf_hybrid_side_checkpoint_resumes_without_retraining_completed_base(tmp_path):
    import torch

    from movie_rating.experiment_config import ExperimentRequest, ModelParameters
    from movie_rating.experiment_supervised import fit_rating

    train, validation, movies = _gpu_hybrid_data()
    request = ExperimentRequest(
        group="rating", algorithms=["ncf"], reduction_mode="enabled",
        reduction_methods=["pca"], components=[1], epochs=2, patience=3,
        parameters=ModelParameters(factors=4, hidden_size=8, batch_size=32),
    )
    config = {
        "factors": 4, "hidden_size": 8, "learning_rate": 0.003,
        "regularization": 0.02, "dropout": 0.1, "batch_size": 32,
    }
    checkpoint_path = tmp_path / "hybrid.pt"
    checkpoint_key = "pca_1/ncf/validation-0"

    class StopAfterFirstSideEpoch(Exception):
        pass

    def stop_after_side_epoch(message):
        if message.startswith("NCF hybrid correction branch · epoch 1/"):
            raise StopAfterFirstSideEpoch

    initial_progress = []
    with pytest.raises(StopAfterFirstSideEpoch):
        fit_rating(
            "ncf", config, request, train, movies, validation,
            progress=lambda epoch, _total: initial_progress.append(epoch),
            log=stop_after_side_epoch,
            reduction_variant={"method": "pca", "components": 1},
            checkpoint_path=checkpoint_path, checkpoint_key=checkpoint_key,
        )
    assert initial_progress == [1, 2, 1]  # Two base epochs, then side epoch one.
    assert checkpoint_path.with_name("hybrid-base.pt").is_file()
    assert checkpoint_path.with_name("hybrid-side.pt").is_file()

    resumed_progress = []
    resumed = fit_rating(
        "ncf", config, request, train, movies, validation,
        progress=lambda epoch, _total: resumed_progress.append(epoch),
        reduction_variant={"method": "pca", "components": 1},
        checkpoint_path=checkpoint_path, checkpoint_key=checkpoint_key,
    )
    assert resumed_progress == [2]  # Base is restored at completion; only side epoch two runs.
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert next(resumed.base_model.network.parameters()).device.type == expected_device


def test_ncf_falls_back_to_cpu_when_cuda_is_unavailable(monkeypatch):
    import torch

    from movie_rating.models.neural import NeuralCF

    splits, _movies = _small_supervised_data()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = NeuralCF(factors=2, hidden_size=4, epochs=1, batch_size=16, patience=1)
    model.fit(splits["train"], validation=splits["validation"])
    assert model.device.type == "cpu"
    assert next(model.network.parameters()).device.type == "cpu"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is not available")
def test_ncf_and_reduced_hybrid_use_cuda_for_base_network_and_cpu_for_profile_branch():
    from movie_rating.experiment_config import ExperimentRequest, ModelParameters
    from movie_rating.experiment_supervised import fit_rating
    from movie_rating.models.neural import NeuralCF

    train, validation, movies = _gpu_hybrid_data()
    request = ExperimentRequest(
        group="rating", algorithms=["ncf"], reduction_mode="enabled",
        reduction_methods=["pca"], components=[1], epochs=2, patience=1,
        parameters=ModelParameters(factors=4, hidden_size=8, batch_size=32),
    )
    baseline = NeuralCF(factors=4, hidden_size=8, epochs=2, batch_size=32, patience=1)
    baseline.fit(train, validation=validation)
    assert baseline.device.type == "cuda"
    assert next(baseline.network.parameters()).device.type == "cuda"

    hybrid = fit_rating(
        "ncf", {
            "factors": 4, "hidden_size": 8, "learning_rate": 0.003,
            "regularization": 0.02, "dropout": 0.1, "batch_size": 32,
        }, request, train, movies, validation,
        reduction_variant={"method": "pca", "components": 1},
    )
    assert hybrid.base_model.device.type == "cuda"
    assert next(hybrid.base_model.network.parameters()).device.type == "cuda"
    assert next(hybrid.side_model.parameters()).device.type == "cpu"
    assert hybrid.reducer.output_dimensions == 1


def _gpu_hybrid_data():
    train, validation = [], []
    for user in range(1, 21):
        for movie in range(1, 33):
            row = (user, movie, 0.5 + 0.5 * ((user * 7 + movie * 3) % 10), movie)
            (train if movie <= 24 else validation).append(row)
    columns = ["userId", "movieId", "rating", "timestamp"]
    movies = pd.DataFrame({
        "movieId": range(1, 33),
        "title": [f"Movie {movie}" for movie in range(1, 33)],
        "genres": ["Action" if movie % 4 == 0 else "Comedy" if movie % 4 == 1
                   else "Drama" if movie % 4 == 2 else "Romance" for movie in range(1, 33)],
    })
    return (
        pd.DataFrame(train, columns=columns),
        pd.DataFrame(validation, columns=columns),
        movies,
    )
