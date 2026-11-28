"""Run real experiments and persist progress/results for the local website."""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from movie_rating.experiments.data import CV_FOLDS, RATING_DTYPES, split_data
from movie_rating.experiments.jobs import ALGORITHM_IDS, append_event, valid_run_id, write_json
from movie_rating.settings import load_settings


def prepare_sample(directory, max_users, seed, report):
    """Uniformly sample users, then retain their complete rating histories."""
    source = directory / "ratings.csv"
    users, original_ratings = set(), 0
    for chunk in pd.read_csv(source, usecols=["userId"], dtype={"userId": "int32"},
                             chunksize=500_000):
        users.update(chunk.userId.unique())
        original_ratings += len(chunk)
        report("data", f"Scanning user IDs: {original_ratings:,} ratings")
    universe = np.asarray(sorted(users), dtype=np.int32)
    selected = (np.sort(np.random.default_rng(seed).choice(
        universe, min(max_users, len(universe)), replace=False)) if max_users else universe)
    frames = []
    for index, chunk in enumerate(pd.read_csv(source, dtype=RATING_DTYPES, chunksize=500_000)):
        subset = chunk.loc[chunk.userId.isin(selected)] if max_users else chunk
        if not subset.empty:
            frames.append(subset)
        report("data", f"Loading complete user histories: chunk {index + 1}")
    ratings = pd.concat(frames, ignore_index=True)
    movies = pd.read_csv(directory / "movies.csv", dtype={"movieId": "int32"})
    provenance = {
        "mode": "sampled_users" if len(selected) < len(universe) else "all_users",
        "seed": seed, "requested_users": max_users or None,
        "selected_users": len(selected), "source_users": len(universe),
        "source_ratings": original_ratings, "sampled_ratings": len(ratings),
        "selected_user_ids": selected.tolist(),
        "sample_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(ratings, index=False).values.tobytes()).hexdigest(),
        "source": str(directory),
        "files": {name: {"bytes": (directory / name).stat().st_size,
                         "mtime_ns": (directory / name).stat().st_mtime_ns}
                  for name in ("ratings.csv", "movies.csv")},
    }
    return split_data(ratings), movies, provenance


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-users", type=int, default=1000, help="0 uses all users")
    parser.add_argument("--variant", choices=["baseline", "improved", "both"], default="both")
    parser.add_argument("--refit-85", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id")
    parser.add_argument("--prepared-from", help="Reuse the exact saved splits of an earlier run")
    parser.add_argument("--queued", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_IDS,
                        default=list(ALGORITHM_IDS))
    args = parser.parse_args()
    if args.max_users < 0 or args.seed < 0:
        parser.error("max-users and seed must be nonnegative")
    if len(args.algorithms) != len(set(args.algorithms)):
        parser.error("algorithms must be unique")
    algorithms = [name for name in ALGORITHM_IDS if name in args.algorithms]
    settings = load_settings()
    root = settings.artifacts_dir / "experiments"
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S-") + uuid4().hex[:8]
    if not valid_run_id(run_id):
        parser.error("Invalid run ID")
    folder = root / run_id
    if args.queued:
        job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
        if (job["status"] != "queued" or job["variant"] != args.variant
                or job["algorithms"] != algorithms):
            parser.error("Queued job does not match this request")
    else:
        folder.mkdir(parents=True, exist_ok=False)
        job = {"id": run_id, "created_at": datetime.now(UTC).isoformat()}
    job.update(status="running", stage="data", progress=0, variant=args.variant,
               max_users=args.max_users, seed=args.seed, algorithms=algorithms,
               refit_85=args.refit_85, started_at=datetime.now(UTC).isoformat(),
               pid=os.getpid(), message="Preparing dataset")
    variants = ["baseline", "improved"] if args.variant == "both" else [args.variant]
    variant_index = 0
    last_write = 0.0
    fits_completed = {}
    total_fits = {}

    def report(stage, message):
        nonlocal last_write
        fraction = 0.0
        if ":" in stage:
            phase, algorithm = stage.split(":", 1)
            offset = algorithms.index(algorithm)
            key = (variant_index, algorithm)
            if phase == "search" and message == f"{algorithm}: fit completed":
                fits_completed[key] = fits_completed.get(key, 0) + 1
            search_fraction = 0.75 * fits_completed.get(key, 0) / total_fits.get(key, 1)
            fraction = (offset + {"search": search_fraction, "validation": 0.8,
                                  "finished": 1.0, "refit": 0.95}[phase]) / len(algorithms)
        elif stage == "test":
            fraction = 0.98
        progress = (10 + 88 * (variant_index + fraction) / len(variants)
                    if stage != "data" else 2)
        changed = job["stage"] != stage
        job.update(stage=stage, message=message, progress=max(job["progress"], int(progress)),
                   updated_at=datetime.now(UTC).isoformat())
        append_event(folder, stage, message)
        if changed or time.monotonic() - last_write >= 0.5:
            write_json(folder / "job.json", job)
            last_write = time.monotonic()
            print(f"[{job['updated_at']}] {job.get('current_variant', 'data')} | {message}",
                  flush=True)

    try:
        write_json(folder / "job.json", job)
        report("data", "Preparing shared Train/Validation/Test data")
        if args.prepared_from:
            if not valid_run_id(args.prepared_from):
                raise ValueError("Invalid prepared run ID")
            previous = root / args.prepared_from
            splits = {name: pd.read_parquet(previous / "data" / f"{name}.parquet")
                      for name in ("train", "validation", "test")}
            movies = pd.read_parquet(previous / "data/movies.parquet")
            provenance = json.loads((previous / "provenance.json").read_text(encoding="utf-8"))
            provenance["prepared_from"] = args.prepared_from
            report("data", f"Reusing saved splits: {args.prepared_from}")
        else:
            splits, movies, provenance = prepare_sample(
                settings.dataset_dir, args.max_users, args.seed, report)
        provenance["split_report"] = splits["train"].attrs["split_report"]
        provenance["included_users"] = int(splits["train"].userId.nunique())
        provenance["included_ratings"] = sum(len(frame) for frame in splits.values())
        data = folder / "data"
        data.mkdir()
        for name, frame in splits.items():
            frame.to_parquet(data / f"{name}.parquet", index=False)
        movies.to_parquet(data / "movies.parquet", index=False)
        job["scope"] = {key: value for key, value in provenance.items()
                        if key not in {"selected_user_ids", "files"}}
        write_json(folder / "provenance.json", provenance)
        from sklearn.model_selection import ParameterGrid

        from movie_rating.experiments.config import default_parameter_grids
        from movie_rating.experiments.runner import run_experiment

        grids = {variant: {name: grid for name, grid in default_parameter_grids(variant).items()
                           if name in algorithms} for variant in variants}
        for index, variant in enumerate(variants):
            for parameters in grids[variant].values():
                parameters["seed"] = [args.seed]
            for algorithm, grid in grids[variant].items():
                total_fits[index, algorithm] = len(ParameterGrid(grid)) * CV_FOLDS + 1
        # Freeze both variants' grids before either variant accesses Test.
        write_json(folder / "configuration.json", {"grids": grids, "refit_85": args.refit_85})
        results = {}
        for variant_index, variant in enumerate(variants):
            job["current_variant"] = variant
            results[variant] = run_experiment(
                splits, movies, grids[variant], variant=variant,
                refit_train_validation=args.refit_85, n_jobs=1, report=report,
                algorithms=algorithms,
            )
            write_json(folder / f"{variant}.json", results[variant])
        completed = datetime.now(UTC).isoformat()
        result = {"id": run_id, "created_at": job["created_at"], "completed_at": completed,
                  "scope": job["scope"], "variants": results}
        write_json(folder / "result.json", result)
        write_json(root / "latest.json", {"id": run_id})
        job.update(status="completed", stage="completed", progress=100,
                   message="Experiment completed", completed_at=completed)
        append_event(folder, "completed", "Experiment completed; results saved.")
        write_json(folder / "job.json", job)
        print(f"RESULT: {folder / 'result.json'}", flush=True)
    except Exception as error:
        job.update(status="failed", stage="failed", message=str(error),
                   updated_at=datetime.now(UTC).isoformat(),
                   completed_at=datetime.now(UTC).isoformat())
        (folder / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        append_event(folder, "failed", str(error))
        write_json(folder / "job.json", job)
        raise


if __name__ == "__main__":
    main()
