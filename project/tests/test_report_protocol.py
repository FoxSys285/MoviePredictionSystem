"""Guards for the separate temporal report pipeline (optional dependencies)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna")
pytest.importorskip("gymnasium")
path = Path(__file__).resolve().parents[1] / "scripts/report_temporal_experiments.py"
spec = importlib.util.spec_from_file_location("report_protocol_test_module", path)
report = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = report
spec.loader.exec_module(report)


def test_five_report_folds_have_strict_temporal_boundaries_and_no_overlap(ratings):
    frame = ratings.copy()
    frame["timestamp"] = np.arange(len(frame)) // 2
    folds = report.temporal_folds(frame)
    assert len(folds) == 5
    for fit, held in folds:
        assert not set(fit) & set(held)
        assert frame.iloc[fit].timestamp.max() < frame.iloc[held].timestamp.min()


def test_fit_filter_uses_only_its_argument_and_keeps_thresholds(ratings):
    filtered = report.filter_training(ratings)
    assert filtered.groupby("userId").size().min() >= report.MIN_USER
    assert filtered.groupby("movieId").size().min() >= report.MIN_ITEM
    short = pd.DataFrame([(99, 99, 4., 0)], columns=ratings.columns)
    fitted = report.filter_training(pd.concat([ratings, short], ignore_index=True))
    assert 99 not in fitted.userId.to_numpy()
    assert len(short) == 1  # Held-out input was never modified or removed globally.


def test_report_estimator_predictions_ignore_heldout_rating_values(ratings):
    movies = pd.DataFrame({"movieId": range(1, 13), "title": [str(i) for i in range(12)],
                           "genres": ["Drama|Comedy"] * 12})
    x = ratings.copy()
    reg = report.FrameRegressor(movies=movies).fit(x, x.rating)
    clf = report.FrameClassifier(movies=movies, config={"C": 1.}).fit(x, x.rating >= 3.5)
    changed = x.copy()
    changed["rating"] = .5
    np.testing.assert_allclose(reg.predict(x), reg.predict(changed))
    np.testing.assert_allclose(clf.predict_proba(x), clf.predict_proba(changed))


def test_gym_report_environment_uses_observed_rewards_and_masks_repeats():
    env = report.MovieHistoryEnv({1: {0: 1., 1: -.5}}, [0, 1, 2, 3, 4])
    state, _ = env.reset(seed=42)
    assert env.observation_space.contains(state)
    _, reward, done, _, info = env.step(0)
    assert reward == 1 and info["observed"] and not done
    with pytest.raises(ValueError):
        env.step(0)
    _, reward, _, _, info = env.step(2)
    assert reward == 0 and not info["observed"]
    for action in [1, 3, 4]:
        state, _, done, _, _ = env.step(action)
    assert done and not env.valid_actions() and env.observation_space.contains(state)
