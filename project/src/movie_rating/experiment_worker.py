"""Isolate large experiments so memory pressure cannot pin the web server."""

import multiprocessing as mp
import shutil
import time
from datetime import UTC, datetime

import psutil


def _execute(settings, job, payload, cancelled, scratch):
    import os

    from movie_rating.experiment_config import ExperimentRequest
    from movie_rating.lab import LabService
    from movie_rating.settings import Settings

    os.environ["MOVIE_SCRATCH_DIR"] = str(scratch)
    request = ExperimentRequest.model_validate(payload)
    service = LabService(Settings.model_validate(settings))
    service.cancel_event = cancelled
    service._run_inline(job, request)


def run_isolated(service, job, request):
    from movie_rating.experiment_logs import LOGGED_GROUPS, RunLogger
    from movie_rating.lab import read_json, save_json
    from movie_rating.memory import scratch_root

    folder = service.directory / job["id"]
    logger = RunLogger(folder) if request.group in LOGGED_GROUPS else None
    scratch_parent = scratch_root(native_ascii=True)
    scratch = scratch_parent / job["id"]
    scratch.mkdir(parents=True, exist_ok=True)
    memory = psutil.virtual_memory()
    peak, minimum_available = 0, memory.available
    process = None
    cancelled = False
    try:
        context = mp.get_context("spawn")
        cancel_event = context.Event()
        process = context.Process(
            target=_execute,
            args=(service.settings.model_dump(), job, request.model_dump(), cancel_event, scratch),
            daemon=False,
        )
        process.start()
        if logger:
            logger.write("Đã khởi chạy worker · không đặt giới hạn thời gian hoặc RAM tự động.")
        monitor = psutil.Process(process.pid)
        started = last_heartbeat = time.monotonic()
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
                # Track private resident pages for reporting; never use RAM as an auto-stop rule.
                try:
                    used = monitor.memory_full_info().uss
                except psutil.AccessDenied:
                    info = monitor.memory_info()
                    used = getattr(info, "private", info.rss)
            except psutil.NoSuchProcess:
                break
            peak = max(peak, used)
            now = time.monotonic()
            if logger and now - last_heartbeat >= 30:
                logger.write(
                    f"Worker vẫn hoạt động · đã chạy {now - started:.0f}s · "
                    f"RAM riêng {used / 2**20:.0f} MiB · "
                    f"RAM máy khả dụng {memory.available / 2**20:.0f} MiB."
                )
                last_heartbeat = now
        if cancelled and process.is_alive():
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
                else f"Tiến trình thực nghiệm kết thúc bất thường ({process.exitcode}).",
                finished_at=datetime.now(UTC).isoformat(),
            )
            if logger:
                logger.write(latest["message"], "warning" if cancelled else "error")
            (folder / "result.json").unlink(missing_ok=True)
        result_path = folder / "result.json"
        if latest["status"] == "completed" and result_path.exists():
            result = read_json(result_path)
            result.setdefault("runtime", {}).update(
                isolated_process=True,
                automatic_memory_limit=False,
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
        if logger:
            logger.write(latest["message"], "error")
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
