"""Metric definitions, label boundaries, and held-out result tracking."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from movie_rating.experiment_supervised import (
    build_scaled_model, evaluate_model, grid_search_model, rating_labels,
)


def test_label_boundaries_and_invalid_ratings():
    np.testing.assert_array_equal(rating_labels([0.5, 2.49, 2.5, 3.0, 3.5, 3.51, 5.0]),
                                  [0, 0, 1, 1, 1, 2, 2])
    for values in ([0.0], [5.1], [np.nan], []):
        with pytest.raises(ValueError):
            rating_labels(values)


def test_regression_clips_predictions_before_scoring():
    class FixedRegressor:
        def predict(self, X):
            return np.array([-1.0, 6.0, 3.0])

    result = evaluate_model(FixedRegressor(), np.zeros((3, 1)), [0.5, 5.0, 2.0])
    np.testing.assert_array_equal(result["predictions"], [0.5, 5.0, 3.0])
    assert result["metrics"]["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert result["metrics"]["mae"] == pytest.approx(1 / 3)


def test_three_class_search_and_heldout_metrics_are_logged(tmp_path):
    ratings = np.tile([1.0, 3.0, 5.0], 10)
    x = np.column_stack((ratings, np.arange(30) % 2))
    search = grid_search_model(
        build_scaled_model("logistic"), x, rating_labels(ratings),
        {"model__C": [0.1, 1.0]}, task="classification", log_directory=tmp_path,
    )
    scaler = search.best_estimator_.named_steps["scaler"]
    original_mean = scaler.mean_.copy()
    result = evaluate_model(search, x[:9], ratings[:9], task="classification", split="test")
    truth = rating_labels(ratings[:9])
    metrics = result["metrics"]
    assert metrics["accuracy"] == accuracy_score(truth, result["predictions"])
    assert metrics["f1"] == f1_score(truth, result["predictions"], average="macro")
    assert metrics["auc_roc"] == roc_auc_score(
        truth, result["probabilities"], multi_class="ovr", average="macro",
    )
    np.testing.assert_array_equal(scaler.mean_, original_mean)
    row = pd.read_csv(tmp_path / "evaluations.csv").iloc[0]
    assert row.run_id == search.run_id_
    assert row.split == "test"
    assert row.auc_roc == pytest.approx(metrics["auc_roc"])
    missing = evaluate_model(search, x[:2], ratings[:2], task="classification")
    assert missing["metrics"]["auc_roc"] is None
    assert "missing" in missing["auc_roc_reason"]
    assert len(pd.read_csv(search.run_directory_ / "evaluations.csv")) == 2


def test_classification_requires_enough_samples_in_all_three_classes():
    with pytest.raises(ValueError, match="each of 3 labels"):
        grid_search_model(build_scaled_model("logistic"), np.zeros((10, 1)),
                          np.arange(10) % 2, {}, task="classification", log_directory=None)
