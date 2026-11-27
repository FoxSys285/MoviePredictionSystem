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
