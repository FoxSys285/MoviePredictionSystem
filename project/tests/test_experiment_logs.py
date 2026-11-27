import json
import pickle
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from movie_rating.api.app import create_app
from movie_rating.data.prepare import prepare
from movie_rating.experiment_logs import RunLogger, log_page
from movie_rating.lab import Cancelled, LabRequest, LabService, save_json
from movie_rating.models.regression import XGBoostRegression


def test_cursor_pages_unicode_without_replaying_or_dropping_partial_events(tmp_path):
    logger = RunLogger(tmp_path)
    logger.write("Bắt đầu hồi quy", progress=8)
    logger.write("Đang tạo cache quantile")
    first = log_page(tmp_path, limit=1)
    assert [row["message"] for row in first["entries"]] == ["Bắt đầu hồi quy"]
    assert first["has_more"] and first["next_cursor"] > 1
    second = log_page(tmp_path, first["next_cursor"], limit=1)
    assert [row["message"] for row in second["entries"]] == ["Đang tạo cache quantile"]
    assert second["entries"][0]["id"] == first["next_cursor"]
    cursor = second["next_cursor"]
    raw = json.dumps({"message": "Cây 1/150", "level": "info"}, ensure_ascii=False).encode()
    with logger.path.open("ab") as stream:
        stream.write(raw[:17])
    partial = log_page(tmp_path, cursor)
    assert partial["entries"] == [] and partial["next_cursor"] == cursor
    with logger.path.open("ab") as stream:
        stream.write(raw[17:] + b"\n")
    complete = log_page(tmp_path, cursor)
    assert complete["entries"][0]["message"] == "Cây 1/150"
    assert not complete["has_more"]
    replay = log_page(tmp_path, complete["next_cursor"] + 1, limit=1)
    assert replay["reset"] and replay["entries"][0]["id"] == 0


def test_worker_and_supervisor_can_append_without_corrupting_events(tmp_path):
    def write(prefix):
        logger = RunLogger(tmp_path)
        for index in range(30):
            logger.write(f"{prefix}/{index}")

    threads = [
        threading.Thread(target=write, args=(prefix,)) for prefix in ("worker", "supervisor")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    page = log_page(tmp_path, limit=100)
    assert len(page["entries"]) == 60
    assert len({row["id"] for row in page["entries"]}) == 60
    assert not page["has_more"]


@pytest.mark.parametrize("external", [False, True])
def test_xgboost_reports_every_tree_and_reporter_is_not_saved(
    ratings, tmp_path, monkeypatch, external
):
    from movie_rating.models import regression

    monkeypatch.setenv("MOVIE_SCRATCH_DIR", str(tmp_path))
    if external:
        monkeypatch.setattr(regression, "_EXTERNAL_MEMORY_ROWS", 1)
    iterations, messages = [], []
    model = XGBoostRegression(n_estimators=10, max_depth=2, n_jobs=1)
    model.fit(
        ratings,
        progress=lambda done, total: iterations.append((done, total)),
        report=messages.append,
    )
    assert iterations == [(index, 10) for index in range(1, 11)]
    assert any("Train" in message for message in messages)
    if external:
        assert any("quantile" in message for message in messages)
    prediction = model.predict(ratings.userId, ratings.movieId)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(ratings.userId, ratings.movieId), prediction)
    plain = XGBoostRegression(n_estimators=10, max_depth=2, n_jobs=1).fit(ratings)
    np.testing.assert_allclose(plain.predict(ratings.userId, ratings.movieId), prediction)


def test_external_xgboost_cancellation_preserves_error_and_removes_cache(
    ratings, tmp_path, monkeypatch
):
    from movie_rating.models import regression

    monkeypatch.setenv("MOVIE_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setattr(regression, "_EXTERNAL_MEMORY_ROWS", 1)

    def cancel(done, total):
        if done == 2:
            raise RuntimeError("stop requested")

    with pytest.raises(RuntimeError, match="stop requested"):
        XGBoostRegression(n_estimators=10, max_depth=2, n_jobs=1).fit(ratings, progress=cancel)
    assert not list(tmp_path.iterdir())


def test_real_regression_job_logs_stages_metrics_and_survives_reopening(configured):
    algorithms = ["linear", "xgboost", "svd", "ncf"]
    prepare(configured)
    service = LabService(configured)
    job = service.start(LabRequest(group="rating", algorithms=algorithms, epochs=2, parameters={"n_estimators": 10}))
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        finished = service.get(job["id"])
        if finished["status"] not in ("queued", "running") and service.active is None:
            break
        time.sleep(0.02)
    service.executor.shutdown(wait=True)
    finished = service.get(job["id"])
    assert finished["status"] == "completed", finished["message"]
    reopened = LabService(configured)
    page = reopened.logs(job["id"], limit=500)
    assert not page["has_more"]
    messages = "\n".join(row["message"] for row in page["entries"])
    for word in ("Validation", "CV", "refit", "Test"):
        assert word in messages
    assert "epoch 1/2" in messages
    assert page["entries"][-1]["level"] == "success"
    assert page["entries"][-1]["progress"] == 100
    reopened.executor.shutdown(wait=True)

@pytest.mark.parametrize(
    "error,status,level",
    [(RuntimeError("regression failed"), "failed", "error"), (Cancelled(), "cancelled", "warning")],
)
def test_regression_failure_and_cancellation_persist_logs(configured, monkeypatch, error, status, level):
    from movie_rating import experiment_pipeline

    prepare(configured)

    def stop(*args):
        raise error

    monkeypatch.setattr(experiment_pipeline, "execute", stop)
    service = LabService(configured)
    try:
        job = service.start(LabRequest(group="rating", algorithms=["linear"]))
        service.executor.shutdown(wait=True)
        finished = service.get(job["id"])
        assert finished["status"] == status, finished.get("message")
        assert "result" not in finished
        entries = service.logs(job["id"], limit=500)["entries"]
        assert any(entry["level"] == level for entry in entries)
        assert not any(entry["level"] == "success" for entry in entries)
    finally:
        service.executor.shutdown(wait=True)

def test_log_api_is_paged_validated_and_supports_old_runs(configured):
    service = LabService(configured)
    run_id = "20260101T000000-aabbccdd"
    directory = service.directory / run_id
    directory.mkdir(parents=True)
    save_json(directory / "job.json", {"id": run_id, "status": "completed", "group": "rating"})
    root = f"/api/v1/experiment-lab/runs/{run_id}/logs"
    with TestClient(create_app(configured)) as client:
        assert client.get(root).json()["entries"] == []
        assert client.get(root + "/download").status_code == 404
        logger = RunLogger(directory)
        logger.write("Cây 1/10")
        logger.write("Đã hoàn tất", "success", 100)
        page = client.get(root, params={"limit": 1}).json()
        assert len(page["entries"]) == 1 and page["has_more"]
        tail = client.get(root, params={"cursor": page["next_cursor"]}).json()
        assert tail["entries"][0]["level"] == "success"
        download = client.get(root + "/download")
        assert (
            download.status_code == 200
            and run_id + ".jsonl" in download.headers["content-disposition"]
        )
        assert len(download.content.splitlines()) == 2
        assert client.get(root, params={"cursor": -1}).status_code == 422
        assert client.get(root, params={"limit": 501}).status_code == 422
        assert client.get(root.replace(run_id, "not-a-run")).status_code == 404


def test_log_reader_bounds_reads_and_skips_corrupt_complete_lines(tmp_path):
    logger = RunLogger(tmp_path)
    logger.path.write_bytes(b"not-json\n")
    logger.write("Vẫn đọc được")
    first = log_page(tmp_path, limit=1)
    assert first["entries"] == [] and first["has_more"]
    second = log_page(tmp_path, first["next_cursor"], limit=1)
    assert second["entries"][0]["message"] == "Vẫn đọc được"
