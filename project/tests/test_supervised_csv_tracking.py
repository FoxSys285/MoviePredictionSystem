"""Result tracking tests for the rebuilt supervised experiment helpers."""

import json

import numpy as np
import pandas as pd
import pytest

from movie_rating.experiment_supervised import build_scaled_model, grid_search_model


def test_csv_tracks_candidates_folds_and_selected_config(tmp_path):
    x = np.column_stack((np.arange(30), np.sin(np.arange(30))))
    y = 2.0 + x[:, 0] * 0.05 + np.cos(x[:, 0]) * 0.1
    search = grid_search_model(
        build_scaled_model("linear"), x, y,
        {"model__fit_intercept": [True, False]},
        log_directory=tmp_path,
        run_metadata={"dataset": "fixture", "split": "per_user_temporal", "seed": 42},
    )
    runs = pd.read_csv(tmp_path / "runs.csv")
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row.status == "completed"
    assert row.model == "LinearRegression"
    assert row.stage == "train_cross_validation"
    assert row.train_rows == 30
    assert json.loads(row.metadata)["seed"] == 42
    assert json.loads(row.best_params) == search.best_params_
    assert row.cv_rmse_mean == pytest.approx(-search.best_score_)
    results = pd.read_csv(search.run_directory_ / "cv_results.csv")
    assert len(results) == 2
    assert results.selected.sum() == 1
    for fold in range(5):
        assert (results[f"split{fold}_test_rmse"] >= 0).all()
    assert results.loc[search.best_index_, "mean_test_rmse"] == pytest.approx(row.cv_rmse_mean)
    assert results.loc[search.best_index_, "rank_test_rmse"] == 1
    assert (search.run_directory_ / "run.csv").exists()
    config = json.loads((search.run_directory_ / "config.json").read_text(encoding="utf-8"))
    assert config["param_grid"] == {"model__fit_intercept": [True, False]}
    assert config["estimator_params"]["scaler__with_std"] is True


def test_regression_runs_append_to_tracking_log(tmp_path):
    x = np.arange(40).reshape(20, 2)
    y = np.linspace(0.5, 5.0, 20)
    for _ in range(2):
        grid_search_model(build_scaled_model("linear"), x, y, {}, log_directory=tmp_path)
    runs = pd.read_csv(tmp_path / "runs.csv")
    assert len(runs) == 2
    assert runs.run_id.nunique() == 2
    assert set(runs.model) == {"LinearRegression"}
    assert pd.notna(runs.iloc[0].cv_rmse_mean)
    assert pd.notna(runs.iloc[1].cv_rmse_mean)


def test_failed_fit_is_logged_and_original_error_propagates(tmp_path):
    with pytest.raises(ValueError):
        grid_search_model(
            build_scaled_model("linear"), np.arange(20).reshape(10, 2), np.linspace(0.5, 5.0, 10),
            {"model__unknown_parameter": [1]}, log_directory=tmp_path,
        )
    row = pd.read_csv(tmp_path / "runs.csv").iloc[0]
    assert row.status == "failed"
    assert "unknown_parameter" in row.error
    assert not (tmp_path / row.run_id / "cv_results.csv").exists()
