import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd

from movie_rating.evaluation import evaluate_model, evaluate_ranking
from movie_rating.models import MODEL_REGISTRY
from movie_rating.settings import Settings


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def current_run(settings: Settings) -> Path:
    pointer = settings.artifacts_dir / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError("No trained models. Run: movie-rating train")
    run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Invalid run ID in latest.json.")
    return settings.artifacts_dir / "runs" / run_id


def train(settings: Settings) -> dict:
    data_dir = settings.artifacts_dir / "data"
    if not (data_dir / "manifest.json").is_file():
        raise FileNotFoundError("No prepared data. Run: movie-rating prepare")
    train_data = pd.read_parquet(data_dir / "train.parquet")
    validation = pd.read_parquet(data_dir / "validation.parquet")
    movies = pd.read_parquet(data_dir / "movies.parquet")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    run_dir = settings.artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for algorithm in settings.experiment.algorithms:
        params = settings.models.get(algorithm, {})
        model_class = MODEL_REGISTRY[algorithm]
        # Only the legacy Numba models need JIT warm-up outside the fit timer.
        if algorithm in ("svd", "nmf", "item_knn"):
            warmup = model_class(**params, seed=settings.experiment.seed)
            tiny = train_data.iloc[: min(10, len(train_data))]
            warmup.fit(tiny)
            warmup.predict(tiny.userId, tiny.movieId)
            del warmup
        model = model_class(**params, seed=settings.experiment.seed)
        if hasattr(model, "set_movie_metadata"):
            model.set_movie_metadata(movies)
        print(f"Training {algorithm} on {len(train_data):,} ratings...", flush=True)
        started = time.perf_counter()
        if algorithm == "ncf":
            model.fit(train_data, validation=validation)
        else:
            model.fit(train_data)
        fit_seconds = time.perf_counter() - started
        measured = evaluate_model(model, validation)
        measured.update(
            evaluate_ranking(model, validation, train_data, seed=settings.experiment.seed)
        )
        model_path = run_dir / f"{algorithm}.joblib"
        joblib.dump(model, model_path, compress=3)
        results.append(
            {
                "algorithm": algorithm,
                "parameters": params,
                "fit_seconds": fit_seconds,
                "model_bytes": model_path.stat().st_size,
                **(
                    {"learning_curve": model.learning_curve, "best_epoch": model.best_epoch}
                    if algorithm == "ncf"
                    else {}
                ),
                **measured,
            }
        )
        del model
    results.sort(key=lambda row: row["rmse"])
    ranked = [row for row in results if row.get("ndcg_at_10") is not None]
    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    report = {
        "run_id": run_id,
        "selection_split": "validation",
        "best_algorithm": results[0]["algorithm"],
        "best_recommendation_algorithm": max(
            ranked,
            key=lambda row: (row["ndcg_at_10"], -row["rmse"]),
        )["algorithm"]
        if ranked
        else results[0]["algorithm"],
        "recommendation_selection_metric": "ndcg_at_10" if ranked else "rmse",
        "results": results,
        "data_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "settings": settings.model_dump(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                pkg: version(pkg)
                for pkg in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "numba",
                    "fastapi",
                    "joblib",
                    "scikit-learn",
                    "xgboost",
                    "torch",
                )
            },
        },
    }
    write_json(run_dir / "validation.json", report)
    pd.DataFrame(
        [
            {
                k: v
                for k, v in row.items()
                if k not in ("parameters", "warm", "cold", "learning_curve")
            }
            for row in results
        ]
    ).to_csv(run_dir / "comparison.csv", index=False)
    # Publish the pointer only after every requested model has finished.
    pointer_tmp = settings.artifacts_dir / f"latest-{run_id}.tmp"
    write_json(pointer_tmp, {"run_id": run_id})
    pointer_tmp.replace(settings.artifacts_dir / "latest.json")
    return report


def evaluate_test(settings: Settings) -> dict:
    run_dir = current_run(settings)
    destination = run_dir / "test.json"
    if destination.is_file():
        return json.loads(destination.read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    manifest_bytes = (settings.artifacts_dir / "data/manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != validation["data_manifest_sha256"]:
        raise ValueError(
            "Prepared data changed since training; use the matching artifacts directory."
        )
    test = pd.read_parquet(settings.artifacts_dir / "data/test.parquet")
    history = pd.concat(
        [
            pd.read_parquet(settings.artifacts_dir / f"data/{split}.parquet")
            for split in ("train", "validation")
        ],
        ignore_index=True,
    )
    results = []
    for row in validation["results"]:
        model = joblib.load(run_dir / f"{row['algorithm']}.joblib")
        model.predict(test.userId.iloc[:1], test.movieId.iloc[:1])
        results.append(
            {
                "algorithm": row["algorithm"],
                **evaluate_model(model, test),
                **evaluate_ranking(model, test, history, seed=settings.experiment.seed),
            }
        )
        del model
    report = {
        "run_id": validation["run_id"],
        "selection_split": "validation",
        "best_algorithm": validation["best_algorithm"],
        "best_recommendation_algorithm": validation.get("best_recommendation_algorithm"),
        "results": results,
    }
    write_json(destination, report)
    return report
