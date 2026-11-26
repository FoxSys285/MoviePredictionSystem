import numpy as np
import pandas as pd
import pytest

from movie_rating.data.prepare import user_chronological_split
from movie_rating.experiment_config import ExperimentRequest
from movie_rating.experiment_supervised import candidates, classification_predict, feature_fit


@pytest.mark.parametrize("cap", [None, 12, 250_000])
def test_bootstrap_cap_is_fold_local_and_keeps_full_feature_statistics(ratings, cap):
    splits = user_chronological_split(ratings)
    movies = pd.DataFrame({"movieId": ratings.movieId.unique(), "genres": "Drama"})
    train = splits["train"]
    config = {
        "n_estimators": 10, "max_depth": 2, "min_samples_leaf": 1, "max_samples": cap,
    }
    messages = []
    builder, model, _ = feature_fit(
        train, [], movies, "random_forest", config, report=messages.append,
    )
    expected = len(train) if cap is None else min(cap, len(train))
    assert all(len(indices) == expected for indices in model.estimator.estimators_samples_)
    assert model.estimator._n_samples == len(train)
    assert builder.global_mean == pytest.approx(train.rating.mean())
    assert config["max_samples"] == cap
    assert any(f"bootstrap={expected:,}/{len(train):,}" in message for message in messages)
    prediction = classification_predict(builder, model, splits["validation"])
    assert len(prediction) == len(splits["validation"])
    assert np.isfinite(prediction).all()


def test_candidates_preserve_bootstrap_cap_when_tuning_and_allow_full_bootstrap():
    request = ExperimentRequest(group="classification", tune=True)
    assert {c["max_samples"] for c in candidates("random_forest", request)} == {250_000}
    request = ExperimentRequest(
        group="classification", parameters={"forest_max_samples": None},
    )
    assert candidates("random_forest", request)[0]["max_samples"] is None
