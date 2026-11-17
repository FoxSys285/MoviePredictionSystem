"""Isolate large experiments so memory pressure cannot pin the web server."""

import multiprocessing as mp
import shutil
from datetime import UTC, datetime

import psutil


def _execute(settings, job, payload, cancelled, scratch):
    import os

    from threadpoolctl import threadpool_limits

    from movie_rating.experiment_config import ExperimentRequest
    from movie_rating.lab import LabService
    from movie_rating.settings import Settings

    os.environ["MOVIE_SCRATCH_DIR"] = str(scratch)
    service = LabService(Settings.model_validate(settings))
    service.cancel_event = cancelled
    with threadpool_limits(limits=2):
        service._run_inline(job, ExperimentRequest.model_validate(payload))


def run_isolated(service, job, request):
    from movie_rating.lab import read_json, save_json
    from movie_rating.memory import scratch_root

    folder = service.directory / job["id"]
    scratch_parent = scratch_root(native_ascii=True)
    scratch = scratch_parent / job["id"]
    scratch.mkdir(parents=True, exist_ok=True)
    memory = psutil.virtual_memory()
    reserve = max(1024 * 2**20, memory.total * 0.12)
    budget = min(
        request.memory_budget_mb * 2**20,
        memory.total * 0.5,
        memory.available - reserve - 256 * 2**20,
    )
    peak, minimum_available = 0, memory.available
    process = None
    failure, cancelled = None, False
    try:
        if budget < 768 * 2**20:
            raise MemoryError(
                "Chưa đủ RAM khả dụng để khởi chạy; hãy đóng ứng dụng đang dùng nhiều RAM."
            )
        context = mp.get_context("spawn")
        cancel_event = context.Event()
        process = context.Process(
            target=_execute,
            args=(service.settings.model_dump(), job, request.model_dump(), cancel_event, scratch),
            daemon=False,
        )
        process.start()
        monitor = psutil.Process(process.pid)
        while process.is_alive():
            process.join(timeout=0.25)
            if not process.is_alive():
                break
            if service.cancel_event.is_set():
                cancel_event.set()
                cancelled = True
                break
            memory = psutil.virtual_memory()
            minimum_available = min(minimum_available, memory.available)
            try:
                # Private resident pages; the system reserve also covers mapped file caches.
                try:
                    used = monitor.memory_full_info().uss
                except psutil.AccessDenied:
                    info = monitor.memory_info()
                    used = getattr(info, "private", info.rss)
            except psutil.NoSuchProcess:
                break
            peak = max(peak, used)
            if used > budget or memory.available < reserve:
                failure = (
                    f"Đã dừng để bảo vệ RAM: worker dùng {used / 2**20:.0f} MiB, "
                    f"giới hạn khả dụng {budget / 2**20:.0f} MiB. "
                    "Đóng ứng dụng khác hoặc giảm cấu hình rồi chạy lại."
                )
                break
        if cancelled or failure:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        latest = read_json(folder / "job.json")
        if latest["status"] not in ("completed", "failed", "cancelled"):
            latest.update(
                status="cancelled" if cancelled else "failed",
                message="Đã dừng; không công bố kết quả chạy dở."
                if cancelled
                else (
                    failure or f"Tiến trình thực nghiệm kết thúc bất thường ({process.exitcode})."
                ),
                finished_at=datetime.now(UTC).isoformat(),
            )
            (folder / "result.json").unlink(missing_ok=True)
        result_path = folder / "result.json"
        if latest["status"] == "completed" and result_path.exists():
            result = read_json(result_path)
            result.setdefault("runtime", {}).update(
                isolated_process=True,
                memory_budget_mb=round(budget / 2**20),
                worker_peak_private_mb=round(peak / 2**20, 1),
                system_min_available_mb=round(minimum_available / 2**20, 1),
            )
            save_json(result_path, result)
        save_json(folder / "job.json", latest)
    except Exception as error:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
        latest = read_json(folder / "job.json")
        latest.update(
            status="failed",
            message=f"{type(error).__name__}: {error}",
            finished_at=datetime.now(UTC).isoformat(),
        )
        (folder / "result.json").unlink(missing_ok=True)
        save_json(folder / "job.json", latest)
    finally:
        # Explicit job-owned directory, checked before any recursive removal on Windows.
        try:
            if scratch.exists() and scratch.resolve().parent == scratch_parent.resolve():
                shutil.rmtree(scratch)
        finally:
            service.data.sample = service.data.sample_key = None
            with service.lock:
                service.active = None
