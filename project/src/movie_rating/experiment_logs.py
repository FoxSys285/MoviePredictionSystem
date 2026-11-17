"""Append-only experiment events, paged by byte offset rather than read into RAM."""

import json
import os
import time
from contextlib import contextmanager
from datetime import UTC, datetime


@contextmanager
def _append_lock(path):
    with path.with_suffix(".lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            for attempt in range(200):
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if attempt == 199:
                        raise
                    time.sleep(0.005)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class RunLogger:
    def __init__(self, directory):
        self.path = directory / "events.jsonl"

    def write(self, message, level="info", progress=None):
        event = {
            "time": datetime.now(UTC).isoformat(),
            "level": level,
            "message": str(message)[:8000],
        }
        if progress is not None:
            event["progress"] = progress
        data = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        for attempt in range(5):
            try:
                # Windows append writes alone can overlap across processes/handles.
                with _append_lock(self.path), self.path.open("a+b", buffering=0) as stream:
                    stream.seek(0, 2)
                    if stream.tell():
                        stream.seek(-1, 2)
                        if stream.read(1) != b"\n":
                            stream.write(b"\n")
                    stream.write(data)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))


def log_page(directory, cursor=0, limit=200):
    path = directory / "events.jsonl"
    if not path.exists():
        return {"entries": [], "next_cursor": 0, "has_more": False, "reset": cursor != 0}
    size = path.stat().st_size
    reset = cursor > size
    if reset:
        cursor = 0
    entries = []
    with path.open("rb") as stream:
        stream.seek(cursor)
        for _ in range(limit):
            offset = stream.tell()
            raw = stream.readline(65536)
            if not raw or not raw.endswith(b"\n"):
                # A writer may be in the middle of appending a UTF-8 event.
                stream.seek(offset)
                break
            try:
                entry = json.loads(raw)
                if isinstance(entry, dict) and isinstance(entry.get("message"), str):
                    entries.append({**entry, "id": offset})
            except (ValueError, UnicodeDecodeError):
                continue
        cursor = stream.tell()
    return {"entries": entries, "next_cursor": cursor, "has_more": cursor < size, "reset": reset}
