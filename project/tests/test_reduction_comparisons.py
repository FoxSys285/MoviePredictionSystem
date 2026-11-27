"""Movie-table PCA and sparse user-item SVD comparison protocols."""

import json

import numpy as np
import pandas as pd
import pytest

import movie_rating.experiment_supervised as experiments
from movie_rating.experiment_config import ExperimentRequest
from movie_rating.experiment_reduction_features import MovieFeatureRepresentation


def data():
    frame = pd.DataFrame(
        [(u, i, [1.0, 3.0, 5.0][(u + i) % 3], 1000 * u + i)
         for u in range(1, 9) for i in range(1, 21)],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.DataFrame({"movieId": range(1, 21),
                           "title": [f"Movie ({1990+i})" for i in range(1, 21)],
                           "genres": ["Drama|Action" if i % 2 else "Comedy" for i in range(1, 21)]})
    return experiments.split_by_time(frame), movies


def test_pca_uses_unique_train_movies_and_heldout_metadata_cannot_change_fit():
    splits, movies = data()
    train = splits["train"]
    representation = MovieFeatureRepresentation().fit(train, movies, components=2)
    assert representation.scaler_.n_samples_seen_ == train.movieId.nunique()
    assert representation.reducer_.n_features_in_ == 5
    original = representation.reducer_.components_.copy()
    movies.loc[~movies.movieId.isin(train.movieId), "title"] = "Future movie (2099)"
    other = MovieFeatureRepresentation().fit(train, movies, components=2)
    np.testing.assert_allclose(other.reducer_.components_, original)
    assert other.transform(splits["test"]).shape == (len(splits["test"]), 2)


@pytest.mark.parametrize("group,algorithm,method,family", [
    ("rating", "linear", "pca", "movie_features"),
])
def test_comparison_runs_baseline_and_reduced_on_identical_outer_splits(
    group, algorithm, method, family, tmp_path, monkeypatch,
):
    splits, movies = data()
    monkeypatch.setattr(experiments, "DEFAULT_TRACKING_DIRECTORY", tmp_path)
    request = ExperimentRequest(group=group, algorithms=[algorithm], use_all_data=False,
                                reduction_mode="enabled", reduction_methods=[method], components=[2])
    result = experiments.supervised(splits, movies, request, lambda *args: None)
    assert len(result["rows"]) == 4
    assert {row["reduction_method"] for row in result["rows"]} == {None, method}
    assert {row["feature_family"] for row in result["rows"]} == {family}
    assert len(result["cv"]) == 2
    assert len(result["reduction_comparisons"]) == 2
    assert {row["baseline_variant_id"] for row in result["reduction_comparisons"]} == {algorithm}
    assert (pd.read_csv(tmp_path / "evaluations.csv").rows.to_numpy()
            == [len(splits["validation"]), len(splits["test"])] * 2).all()
    configurations = [json.loads(path.read_text(encoding="utf-8"))
                      for path in tmp_path.glob("*/config.json")]
    assert len({config["metadata"]["dataset_split_sha256"] for config in configurations}) == 1
