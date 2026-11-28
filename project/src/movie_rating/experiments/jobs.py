"""Persisted experiment jobs, subprocess launch, and incremental event logs."""

import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

import psutil

ALGORITHM_IDS = ("linear", "xgboost", "svd", "ncf")


def utc_now():
    return datetime.now(UTC).isoformat()


def valid_run_id(value):
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value))


def write_json(path, value):
    """Atomic JSON replacement with distinct temporary files for concurrent jobs."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        for attempt in range(24):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 23:
                    raise
                time.sleep(min(0.2, 0.025 * (attempt + 1)))
    finally:
        temporary.unlink(missing_ok=True)


def append_event(folder, stage, message):
    event = {"time": utc_now(), "stage": stage, "message": message}
    with (Path(folder) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_events(folder, cursor=0, limit=100):
    """Read complete JSON lines from a byte cursor; never reread the whole log."""
    path = Path(folder) / "events.jsonl"
    if not path.exists():
        return {"entries": [], "next_cursor": 0, "has_more": False, "reset": cursor != 0}
    size = path.stat().st_size
    reset = cursor > size
    entries = []
    with path.open("rb") as stream:
        stream.seek(0 if reset else cursor)
        for _ in range(limit):
            position = stream.tell()
            line = stream.readline(65_536)
            if not line or not line.endswith(b"\n"):
                stream.seek(position)
                break
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            event["cursor"] = position
            entries.append(event)
        next_cursor = stream.tell()
    return {"entries": entries, "next_cursor": next_cursor,
            "has_more": next_cursor < size, "reset": reset}


class ExperimentJobs:
    def __init__(self, settings):
        self.settings = settings
        self.root = settings.artifacts_dir / "experiments"
        self.lock = Lock()
        self.processes = {}

    def folder(self, run_id):
        if not valid_run_id(run_id):
            raise ValueError("Mã thực nghiệm không hợp lệ.")
        folder = self.root / run_id
        if not (folder / "job.json").is_file():
            raise FileNotFoundError("Không tìm thấy lượt thực nghiệm.")
        return folder

    def read_job(self, run_id):
        folder = self.folder(run_id)
        job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
        process = self.processes.get(run_id)
        if process is not None and process.poll() is not None:
            self.processes.pop(run_id, None)
        pid = job.get("pid")
        marker = folder / "worker.json"
        if not pid and marker.is_file():
            pid = json.loads(marker.read_text(encoding="utf-8")).get("pid")
        if job["status"] in {"queued", "running"} and pid and not psutil.pid_exists(pid):
            job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
            if job["status"] not in {"queued", "running"}:
                return job
            job.update(status="failed", stage="failed", completed_at=utc_now(),
                       updated_at=utc_now(),
                       message="Tiến trình thực nghiệm đã dừng trước khi hoàn tất.")
            append_event(folder, "failed", job["message"])
            write_json(folder / "job.json", job)
        return job

    def list_jobs(self):
        if not self.root.exists():
            return []
        folders = sorted((path for path in self.root.iterdir() if path.is_dir()), reverse=True)
        return [self.read_job(folder.name) for folder in folders[:50]
                if (folder / "job.json").is_file() and valid_run_id(folder.name)]

    def cached_preparation(self, max_users, seed):
        for job in self.list_jobs():
            folder = self.root / job["id"]
            manifest = folder / "provenance.json"
            if not manifest.is_file():
                continue
            data = json.loads(manifest.read_text(encoding="utf-8"))
            source = Path(data.get("source", "")).resolve()
            if (data.get("requested_users") != max_users or data.get("seed") != seed
                    or source != self.settings.dataset_dir.resolve()):
                continue
            unchanged = all(
                (self.settings.dataset_dir / name).is_file()
                and (self.settings.dataset_dir / name).stat().st_size == fingerprint["bytes"]
                and (self.settings.dataset_dir / name).stat().st_mtime_ns == fingerprint["mtime_ns"]
                for name, fingerprint in data.get("files", {}).items())
            files = [folder / "data" / f"{name}.parquet"
                     for name in ("train", "validation", "test", "movies")]
            if data.get("files") and unchanged and all(path.is_file() for path in files):
                return job["id"]
        return None

    def start(self, *, variant, algorithms, max_users=1000, seed=42, refit_85=False):
        """Allow one active worker per variant and reuse identical active requests."""
        if (variant not in {"baseline", "improved"} or not algorithms
                or len(algorithms) != len(set(algorithms)) or set(algorithms) - set(ALGORITHM_IDS)):
            raise ValueError("Chọn phiên bản và ít nhất một thuật toán hợp lệ.")
        algorithms = [name for name in ALGORITHM_IDS if name in algorithms]
        with self.lock:
            for active in self.list_jobs():
                if active["status"] not in {"queued", "running"}:
                    continue
                if active["variant"] not in {variant, "both"}:
                    continue
                if (active["variant"] == variant and active.get("algorithms") == algorithms
                        and active.get("max_users") == max_users and active.get("seed") == seed
                        and active.get("refit_85", False) == refit_85):
                    return active
                raise RuntimeError("Mục này đang có thực nghiệm chạy. Vui lòng chờ hoàn tất.")
            for name in ("ratings.csv", "movies.csv"):
                if not (self.settings.dataset_dir / name).is_file():
                    raise FileNotFoundError(
                        "Chưa có ratings.csv và movies.csv để chạy thực nghiệm.")
            prepared = self.cached_preparation(max_users, seed)
            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S-") + uuid4().hex[:8]
            folder = self.root / run_id
            folder.mkdir(parents=True, exist_ok=False)
            job = {"id": run_id, "variant": variant, "algorithms": algorithms,
                   "max_users": max_users, "seed": seed, "refit_85": refit_85,
                   "status": "queued", "stage": "queued", "progress": 0,
                   "created_at": utc_now(), "message": "Đang khởi động thực nghiệm."}
            write_json(folder / "job.json", job)
            append_event(folder, "queued", job["message"])
            command = [sys.executable, "-B", "-X", "utf8", "-m",
                       "movie_rating.experiments.run", "--variant", variant,
                       "--max-users", str(max_users), "--seed", str(seed),
                       "--run-id", run_id, "--queued", "--algorithms", *algorithms]
            if refit_85:
                command.append("--refit-85")
            if prepared:
                command += ["--prepared-from", prepared]
            env = {**os.environ, "MOVIE_DATASET_DIR": str(self.settings.dataset_dir),
                   "MOVIE_ARTIFACTS_DIR": str(self.settings.artifacts_dir),
                   "MOVIE_SCRATCH_DIR": str(folder / "scratch"), "PYTHONIOENCODING": "utf-8"}
            try:
                with (folder / "stdout.log").open("w", encoding="utf-8") as stdout:
                    with (folder / "stderr.log").open("w", encoding="utf-8") as stderr:
                        process = subprocess.Popen(
                            command, cwd=str(Path(__file__).resolve().parents[3]), env=env,
                            stdout=stdout, stderr=stderr,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                        )
                self.processes[run_id] = process
                write_json(folder / "worker.json", {"pid": process.pid})
            except OSError as error:
                job.update(status="failed", stage="failed", completed_at=utc_now(),
                           message=str(error))
                append_event(folder, "failed", str(error))
                write_json(folder / "job.json", job)
                raise
            return job
