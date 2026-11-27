"""Global Mean uses every Train rating, never held-out values for fitting."""

import numpy as np
import pandas as pd
import pytest

from movie_rating.experiment_baseline import global_mean_baseline
from movie_rating.experiment_baseline import baseline_from_provenance


def splits():
    return {
        "train": pd.DataFrame({"userId": [1, 1, 1, 2], "rating": [1., 1., 1., 5.]}),
        "validation": pd.DataFrame({"rating": [2., 4.]}),
        "test": pd.DataFrame({"rating": [1., 5.]}),
    }


def test_global_mean_is_rating_weighted_and_scores_heldout_errors():
    result = global_mean_baseline(splits())
    assert result["prediction"] == 2.0  # User-weighted mean would be 3.0.
    assert result["fit_scope"] == "train"
    assert result["train_ratings"] == 4
    valid, test = result["rows"]
    assert valid["split"] == "validation"
    assert valid["mae"] == 1.0
    assert valid["rmse"] == pytest.approx(np.sqrt(2))
    assert test["split"] == "test"
    assert test["mae"] == 2.0
    assert test["rmse"] == pytest.approx(np.sqrt(5))


def test_changing_heldout_ratings_never_changes_fitted_mean():
    original = splits()
    changed = {**original, "validation": original["validation"].assign(rating=5.),
               "test": original["test"].assign(rating=5.)}
    first, second = global_mean_baseline(original), global_mean_baseline(changed)
    assert first["prediction"] == second["prediction"]
    assert first["rows"] != second["rows"]
    pd.testing.assert_frame_equal(original["train"], splits()["train"])


@pytest.mark.parametrize("split", ["train", "validation", "test"])
@pytest.mark.parametrize("ratings", [[], [np.nan], [np.inf]])
def test_rejects_empty_or_nonfinite_ratings(split, ratings):
    frames = splits()
    frames[split] = pd.DataFrame({"rating": ratings})
    with pytest.raises(ValueError, match="Global Mean requires"):
        global_mean_baseline(frames)


def saved_splits(directory):
    from movie_rating.experiment_data import parquet_fingerprint

    paths = {}
    for name, frame in splits().items():
        paths[name] = directory / f"{name}.parquet"
        frame.to_parquet(paths[name], index=False)
    return {"split_files": {name: str(path) for name, path in paths.items()},
            "sample_sha256": parquet_fingerprint(list(paths.values()))}


def test_streamed_historical_baseline_matches_in_memory(tmp_path):
    provenance = saved_splits(tmp_path)
    assert baseline_from_provenance(provenance, tmp_path) == global_mean_baseline(splits())


def test_historical_baseline_refuses_changed_data(tmp_path):
    provenance = saved_splits(tmp_path)
    pd.DataFrame({"rating": [5.]}).to_parquet(tmp_path / "train.parquet")
    with pytest.raises(ValueError, match="fingerprint"):
        baseline_from_provenance(provenance, tmp_path)


def test_missing_provenance_does_not_invent_baseline(tmp_path):
    assert baseline_from_provenance({}, tmp_path) is None


def test_saved_result_get_backfills_baseline_once(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from movie_rating.lab import LabService, save_json

    provenance = saved_splits(tmp_path)
    run_id = "20261127T000000-abcdef01"
    directory = tmp_path / "experiment-lab" / run_id
    directory.mkdir(parents=True)
    save_json(directory / "job.json", {"id": run_id, "group": "rating", "status": "completed"})
    save_json(directory / "result.json", {"rows": [], "provenance": provenance})
    service = LabService(SimpleNamespace(artifacts_dir=tmp_path))
    try:
        first = service.get(run_id)["result"]
        assert first["baseline"] == global_mean_baseline(splits())
        stored = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        assert stored["baseline"] == first["baseline"]
        monkeypatch.setattr("movie_rating.experiment_baseline.baseline_from_provenance",
                            lambda *_: pytest.fail("Baseline must be reused after backfill"))
        assert service.get(run_id)["result"]["baseline"] == first["baseline"]
    finally:
        service.executor.shutdown()
