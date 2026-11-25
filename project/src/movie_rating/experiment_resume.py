"""Atomic progress records for resumable supervised experiments."""

import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import torch

SCHEMA_VERSION = 1


class ResumeStateMismatch(ValueError):
    """Saved work cannot be reused with the current experiment identity."""


def _code_fingerprint():
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted([*package.rglob("*.py"), package / "lab_catalog.json"]):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _package_versions():
    values = {"python": platform.python_version()}
    for package in (
        "numpy", "pandas", "scipy", "numba", "pyarrow", "scikit-learn",
        "threadpoolctl", "torch", "xgboost",
    ):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def _runtime_identity():
    identity = {
        "platform": platform.platform(),
        "cuda_runtime": torch.version.cuda,
        "device": "cpu",
        "gpu_capability": None,
        "driver_version": None,
    }
    if not torch.cuda.is_available():
        return identity
    identity["device"] = torch.cuda.get_device_name(0)
    identity["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            identity["driver_version"] = completed.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return identity


def _frame_fingerprint(frame, chunk_rows=500_000):
    """Hash row contents and schema in bounded chunks, preserving row order."""
    digest = hashlib.sha256()
    schema = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "rows": len(frame),
    }
    digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
    for start in range(0, len(frame), chunk_rows):
        values = pd.util.hash_pandas_object(
            frame.iloc[start : start + chunk_rows], index=False
        ).to_numpy(copy=False)
        digest.update(values.tobytes())
    return digest.hexdigest()


def resume_fingerprint(request, splits, movies, provenance):
    """Identify every input that can change a supervised result."""
    return {
        "request": request.model_dump(),
        "data": {
            "sample_sha256": provenance.get("sample_sha256"),
            "data_manifest_sha256": provenance.get("data_manifest_sha256"),
            "protocol": provenance.get("protocol"),
            "data_seed": provenance.get("data_seed"),
            "split_counts": {name: len(frame) for name, frame in splits.items()},
            "split_sha256": {
                name: _frame_fingerprint(frame) for name, frame in splits.items()
            },
            "movies_sha256": _frame_fingerprint(movies),
        },
        "code_sha256": _code_fingerprint(),
        "libraries": _package_versions(),
        "runtime": _runtime_identity(),
    }


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            from movie_rating.lab import native

            json.dump(
                native(value), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


class ProgressStore:
    """Stores finished work units separately from the final report."""

    def __init__(self, run_directory, fingerprint):
        self.path = Path(run_directory) / "progress.json"
        self.checkpoint_directory = Path(run_directory) / "checkpoints"
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state.get("schema_version") != SCHEMA_VERSION:
                raise ResumeStateMismatch(
                    "Progress checkpoint schema changed; start a new experiment."
                )
            if state.get("fingerprint") != fingerprint:
                raise ResumeStateMismatch(
                    "Dataset, split, seed, parameters, code, or library versions changed; "
                    "saved work cannot be reused. Start a new experiment."
                )
            self.state = state
        else:
            self.state = {
                "schema_version": SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "units": {},
            }
            self._write()

    @staticmethod
    def unit_key(variant_id, algorithm):
        return f"{variant_id or 'raw'}::{algorithm}"

    def get_unit(self, variant_id, algorithm):
        return copy.deepcopy(self.state["units"].get(self.unit_key(variant_id, algorithm), {}))

    def save_unit(self, variant_id, algorithm, unit):
        self.state["units"][self.unit_key(variant_id, algorithm)] = copy.deepcopy(unit)
        self._write()

    def checkpoint_path(self, fit_key):
        # Keep Windows temp filenames below MAX_PATH even under pytest/temp roots.
        name = hashlib.sha256(fit_key.encode("utf-8")).hexdigest()[:32]
        return self.checkpoint_directory / f"{name}.pt"

    def clear_fit_checkpoints(self, fit_key):
        base = self.checkpoint_path(fit_key)
        for suffix in ("", "-base", "-side"):
            path = base.with_name(f"{base.stem}{suffix}{base.suffix}")
            path.unlink(missing_ok=True)

    def _write(self):
        _atomic_json(self.path, self.state)

    @staticmethod
    def summary(run_directory):
        path = Path(run_directory) / "progress.json"
        if not path.is_file():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if state.get("schema_version") != SCHEMA_VERSION:
            return None
        completed = sum(
            len(unit.get("options", []))
            + len(unit.get("folds", []))
            + int(unit.get("final") is not None)
            for unit in state.get("units", {}).values()
        )
        return {
            "resume_available": True,
            "completed_work_units": completed,
            "active_fit_checkpoint": any(Path(run_directory).glob("checkpoints/*.pt")),
        }
