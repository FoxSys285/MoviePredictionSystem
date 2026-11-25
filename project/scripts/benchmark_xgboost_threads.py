"""Measure XGBoost thread counts on one frozen per-user temporal sample."""

import argparse
import json
import os
import platform
import tempfile
import threading
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import psutil

from movie_rating.experiment_config import ExperimentRequest, ModelParameters
from movie_rating.experiment_data import ExperimentData
from movie_rating.models.regression import XGBoostRegression
from movie_rating.native_threads import limit_native_threadpools
from movie_rating.settings import load_settings


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def measure_peak_rss(call):
    process = psutil.Process()
    done = threading.Event()
    peak = [process.memory_info().rss]

    def sample():
        while not done.wait(0.025):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        return call(), peak
    finally:
        done.set()
        sampler.join(timeout=1)
        peak[0] = max(peak[0], process.memory_info().rss)


def run_trial(train, movies, n_jobs):
    request = ExperimentRequest(
        group="rating",
        algorithms=["xgboost"],
        seed=42,
        max_users=int(train.userId.nunique()),
        use_all_data=False,
        parameters=ModelParameters(
            n_estimators=150,
            max_depth=6,
            xgb_learning_rate=0.05,
            xgb_n_jobs=n_jobs,
        ),
    )
    model = XGBoostRegression(
        n_estimators=request.parameters.n_estimators,
        max_depth=request.parameters.max_depth,
        learning_rate=request.parameters.xgb_learning_rate,
        n_jobs=request.parameters.xgb_n_jobs,
        seed=request.seed,
    )
    timings = {"started": time.perf_counter()}

    def report(message):
        now = time.perf_counter()
        if message.startswith("Bắt đầu học") or message.startswith("Cache đã sẵn sàng"):
            timings["feature_cache_seconds"] = now - timings["started"]
            timings["training_started"] = now
        if message.startswith("Đã huấn luyện xong XGBoost"):
            timings["training_finished"] = now

    def fit():
        with limit_native_threadpools(n_jobs, include_xgboost=True) as inventory:
            pools = [
                {"internal_api": pool.get("internal_api"), "num_threads": pool.get("num_threads")}
                for pool in inventory
                if pool.get("user_api") in ("openmp", "blas")
            ]
            model.fit(train, progress=lambda *_: None, report=report)
            return pools

    pools, peak = measure_peak_rss(fit)
    finished = time.perf_counter()
    if "feature_cache_seconds" not in timings or "training_finished" not in timings:
        raise RuntimeError("XGBoost did not report both cache-ready and fit-complete events.")
    start = timings["started"]
    training_start = timings["training_started"]
    training_end = timings["training_finished"]
    return {
        "n_jobs": n_jobs,
        "threadpool_limits": n_jobs,
        "threadpools_at_fit_start": pools,
        "feature_cache_seconds": timings["feature_cache_seconds"],
        "fit_seconds": training_end - training_start,
        "total_seconds": finished - start,
        "peak_rss_mb": peak[0] / 2**20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=5000)
    parser.add_argument("--threads", type=int, nargs="+", choices=(2, 4, 8), default=(2, 4, 8))
    args = parser.parse_args()
    if not 10 <= args.users <= 5000:
        parser.error("--users must be in [10, 5000].")

    settings = load_settings()
    data = ExperimentData(settings.artifacts_dir / "data")
    request = ExperimentRequest(
        group="rating", algorithms=["xgboost"], max_users=args.users,
        use_all_data=False, seed=42,
    )
    splits, movies, provenance = data.load(request)
    train = splits["train"]
    results = []
    for n_jobs in args.threads:
        print(
            f"XGBoost benchmark: n_jobs={n_jobs}; train_rows={len(train):,}",
            flush=True,
        )
        results.append(run_trial(train, movies, n_jobs))

    created = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    destination = settings.artifacts_dir / "benchmarks" / (
        f"xgboost-threads-{created}-{uuid4().hex[:6]}"
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "hardware": {"processor": platform.processor(), "logical_cpus": os.cpu_count()},
        "dataset": {
            "source": provenance["source"],
            "protocol": provenance["protocol"],
            "users": provenance["sampled_users"],
            "sampled_ratings": provenance["sampled_ratings"],
            "train_ratings": len(train),
            "sample_sha256": provenance["sample_sha256"],
        },
        "parameters": {
            "seed": 42,
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_jobs": list(args.threads),
            "threadpool_limits": "matched to n_jobs for each trial",
        },
        "libraries": {"xgboost": version("xgboost"), "threadpoolctl": version("threadpoolctl")},
        "measurement": {
            "feature_cache_seconds": (
                "model statistics plus feature/quantile cache build, until training starts"
            ),
            "fit_seconds": "XGBoost tree training, from cache-ready until fit-complete event",
            "total_seconds": "entire model.fit call",
            "peak_rss_mb": "peak resident memory of the benchmark process, sampled every 25 ms",
            "scope_limit": "sampled users; not the complete prepared MovieLens dataset",
        },
        "trials": results,
    }
    report_path = destination / "results.json"
    atomic_json(report_path, report)
    printable_path = str(report_path).encode("ascii", errors="backslashreplace").decode("ascii")
    print(f"Saved benchmark: {printable_path}", flush=True)


if __name__ == "__main__":
    main()
