"""Reproducible seed/time experiments and paired user bootstrap on a frozen snapshot."""

import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from movie_rating.evaluation import evaluate_model, evaluate_ranking
from movie_rating.models import MODEL_REGISTRY
from movie_rating.training import current_run, write_json

MODEL_SEEDS = (13, 42, 73, 101, 137)
METRICS = ("rmse", "mae", "ndcg_at_10")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def time_blocks(ratings, count=6):
    """Approximately equal event counts, with entire timestamps kept together."""
    ordered = ratings.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cuts = [
        int(ordered.iloc[int(len(ordered) * index / count) - 1].timestamp)
        for index in range(1, count)
    ]
    parts = [
        part.copy()
        for _, part in ordered.groupby(
            np.searchsorted(cuts, ordered.timestamp.to_numpy(), side="left"), sort=True
        )
    ]
    if len(parts) != count or any(part.empty for part in parts):
        raise ValueError("Insufficient distinct timestamp groups for temporal folds.")
    return parts


def user_errors(ratings, predicted):
    error = ratings.rating.to_numpy(dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    if not np.isfinite(error).all():
        raise ValueError("Bootstrap requires finite predictions.")
    return (
        pd.DataFrame(
            {
                "userId": ratings.userId.to_numpy(),
                "count": 1,
                "squared_error": error**2,
                "absolute_error": np.abs(error),
            }
        )
        .groupby("userId", as_index=False)
        .sum()
    )


def paired_bootstrap(a, b, metric, repetitions=2000, seed=2026):
    """Cluster bootstrap: same sampled users and multiplicities for both estimators."""
    if metric not in METRICS or repetitions < 2:
        raise ValueError("Invalid bootstrap metric or repetition count.")
    a = a.sort_values("userId").reset_index(drop=True)
    b = b.sort_values("userId").reset_index(drop=True)
    if not np.array_equal(a.userId, b.userId):
        raise ValueError("Paired bootstrap requires identical user cohorts.")
    n = len(a)
    if n < 2:
        return {
            "metric": metric,
            "users": n,
            "delta": None,
            "ci_low": None,
            "ci_high": None,
            "conclusion": "insufficient_users",
        }
    if metric == "ndcg_at_10":
        av, bv = a[metric].to_numpy(), b[metric].to_numpy()
        delta = float(np.mean(av - bv))
    else:
        if not np.array_equal(a["count"], b["count"]):
            raise ValueError("Paired rating counts differ.")
        column = "squared_error" if metric == "rmse" else "absolute_error"
        av, bv, weights = a[column].to_numpy(), b[column].to_numpy(), a["count"].to_numpy()
        root = np.sqrt if metric == "rmse" else np.asarray
        delta = float(root(av.sum() / weights.sum()) - root(bv.sum() / weights.sum()))
    rng, differences = np.random.default_rng(seed), []
    # Batches cap allocation when the full dataset contains many users.
    for start in range(0, repetitions, 100):
        indices = rng.integers(0, n, size=(min(100, repetitions - start), n))
        if metric == "ndcg_at_10":
            values = (av[indices] - bv[indices]).mean(axis=1)
        else:
            denominator = weights[indices].sum(axis=1)
            values = root(av[indices].sum(axis=1) / denominator) - root(
                bv[indices].sum(axis=1) / denominator
            )
        differences.extend(values)
    low, high = map(float, np.quantile(differences, [0.025, 0.975]))
    a_better = high < 0 if metric != "ndcg_at_10" else low > 0
    b_better = low > 0 if metric != "ndcg_at_10" else high < 0
    return {
        "metric": metric,
        "users": n,
        "delta": delta,
        "ci_low": low,
        "ci_high": high,
        "conclusion": "a_better" if a_better else "b_better" if b_better else "uncertain",
    }


def summarize(rows, key):
    result = []
    for algorithm in sorted({row["algorithm"] for row in rows}):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        metrics = {}
        for metric in (*METRICS, "fit_seconds"):
            values = [
                row.get(key, {}).get(metric) if metric != "fit_seconds" else row.get(metric)
                for row in selected
            ]
            values = [value for value in values if value is not None]
            metrics[metric] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "n": len(values),
            }
        result.append({"algorithm": algorithm, "metrics": metrics})
    return result


def fit_model(algorithm, parameters, seed, train, validation, movies, temporal=False):
    def new_model(params):
        model = MODEL_REGISTRY[algorithm](**params, seed=seed)
        if hasattr(model, "set_movie_metadata"):
            model.set_movie_metadata(movies)
        return model

    if algorithm in ("svd", "nmf", "item_knn"):
        warmup = new_model(parameters)
        warmup.fit(train.iloc[: min(10, len(train))])
        warmup.predict(train.userId.iloc[:1], train.movieId.iloc[:1])
    start, epoch = time.perf_counter(), None
    if algorithm == "ncf" and temporal:
        # Select epoch on a strictly earlier inner holdout, then refit full fold train.
        ordered = train.sort_values("timestamp", kind="stable")
        boundary = int(ordered.iloc[max(0, int(len(ordered) * 0.9) - 1)].timestamp)
        inner_train = ordered.loc[ordered.timestamp <= boundary]
        inner_val = ordered.loc[ordered.timestamp > boundary]
        if inner_train.empty or inner_val.empty:
            raise ValueError("Temporal NCF requires a nonempty inner early-stopping split.")
        pilot = new_model(parameters).fit(inner_train, validation=inner_val)
        epoch = pilot.best_epoch
        del pilot
        model = new_model({**parameters, "epochs": epoch}).fit(train)
    elif algorithm == "ncf":
        model = new_model(parameters).fit(train, validation=validation)
        epoch = model.best_epoch
    else:
        model = new_model(parameters).fit(train)
    return model, time.perf_counter() - start, epoch


def frame_info(frame):
    return {
        "ratings": len(frame),
        "users": int(frame.userId.nunique()),
        "movies": int(frame.movieId.nunique()),
        "min_timestamp": int(frame.timestamp.min()),
        "max_timestamp": int(frame.timestamp.max()),
    }


def run_stability(settings):
    run = current_run(settings)
    source = json.loads((run / "validation.json").read_text(encoding="utf-8"))
    data_dir = settings.artifacts_dir / "data"
    fingerprint = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    if fingerprint != source["data_manifest_sha256"]:
        raise ValueError("Prepared data changed since the selected run.")
    if not (run / "test.json").is_file():
        raise FileNotFoundError("Evaluate the frozen run before its stability analysis.")
    destination = run / "stability"
    destination.mkdir(exist_ok=True)
    algorithms = [row["algorithm"] for row in source["results"]]
    plan = {
        "schema_version": 1,
        "run_id": source["run_id"],
        "model_seeds": list(MODEL_SEEDS),
        "ranking_seed": 42,
        "bootstrap_seed": 2026,
        "bootstrap_repetitions": 2000,
        "temporal_folds": 5,
        "temporal_seed": 42,
        "data_seed": json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))[
            "settings"
        ]["experiment"]["seed"],
        "data_manifest_sha256": fingerprint,
        "data_sha256": {
            name: hashlib.sha256((data_dir / f"{name}.parquet").read_bytes()).hexdigest()
            for name in ("train", "validation", "test", "movies")
        },
        "code_sha256": {
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [
                Path(__file__),
                Path(__file__).with_name("evaluation.py"),
                *sorted(Path(__file__).parent.joinpath("models").glob("*.py")),
            ]
        },
        "anchor_sha256": {
            name: hashlib.sha256((run / f"{name}.joblib").read_bytes()).hexdigest()
            for name in algorithms
        },
        "algorithms": algorithms,
        "models": source["settings"]["models"],
        "reference_rating": source["best_algorithm"],
        "reference_ranking": source.get("best_recommendation_algorithm")
        or source["best_algorithm"],
        "comparison_scope": "reference_selected_on_validation_pointwise_exploratory_intervals",
    }
    plan_path = destination / "plan.json"
    if plan_path.is_file():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Stability plan changed; keep the previous experiment intact.")
    else:
        atomic_json(plan_path, plan)
    frames = {
        name: pd.read_parquet(data_dir / f"{name}.parquet")
        for name in ("train", "validation", "test", "movies")
    }
    report = {
        **plan,
        "status": "running",
        "seed_runs": [],
        "temporal_runs": [],
        "bootstrap": {},
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
    }

    def publish():
        report["updated_at"] = datetime.now(UTC).isoformat()
        report["seed_summary"] = {
            split: summarize(report["seed_runs"], split) for split in ("validation", "test")
        }
        report["temporal_summary"] = summarize(report["temporal_runs"], "evaluation")
        atomic_json(run / "stability.json", report)

    publish()
    history_test = pd.concat([frames["train"], frames["validation"]], ignore_index=True)
    for seed in MODEL_SEEDS:
        for algorithm in algorithms:
            path = destination / f"seed-{seed}-{algorithm}.json"
            if path.is_file():
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                print(f"Seed {seed}: {algorithm}", flush=True)
                model, seconds, epoch = fit_model(
                    algorithm,
                    plan["models"].get(algorithm, {}),
                    seed,
                    frames["train"],
                    frames["validation"],
                    frames["movies"],
                )
                row = {
                    "algorithm": algorithm,
                    "model_seed": seed,
                    "fit_seconds": seconds,
                    "best_epoch": epoch,
                }
                for split in ("validation", "test"):
                    row[split] = {
                        **evaluate_model(model, frames[split]),
                        **evaluate_ranking(
                            model,
                            frames[split],
                            frames["train"] if split == "validation" else history_test,
                            seed=plan["ranking_seed"],
                        ),
                    }
                atomic_json(path, row)
                del model
            report["seed_runs"].append(row)
            publish()
    blocks = time_blocks(frames["train"])
    for fold in range(1, 6):
        fold_train, heldout = pd.concat(blocks[:fold], ignore_index=True), blocks[fold]
        for algorithm in algorithms:
            path = destination / f"fold-{fold}-{algorithm}.json"
            if path.is_file():
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                print(f"Temporal fold {fold}: {algorithm}", flush=True)
                model, seconds, epoch = fit_model(
                    algorithm,
                    plan["models"].get(algorithm, {}),
                    42,
                    fold_train,
                    None,
                    frames["movies"],
                    temporal=True,
                )
                row = {
                    "algorithm": algorithm,
                    "fold": fold,
                    "model_seed": 42,
                    "fit_seconds": seconds,
                    "best_epoch": epoch,
                    "train": frame_info(fold_train),
                    "heldout": frame_info(heldout),
                    "evaluation": {
                        **evaluate_model(model, heldout),
                        **evaluate_ranking(model, heldout, fold_train, seed=42),
                    },
                }
                atomic_json(path, row)
                del model
            report["temporal_runs"].append(row)
            publish()
    # Conditional uncertainty of the original frozen models, not the best repeated seed.
    for split in ("validation", "test"):
        errors, ranks = {}, {}
        for algorithm in algorithms:
            prediction_path = destination / f"anchor-{split}-{algorithm}.parquet"
            rank_path = destination / f"anchor-{split}-{algorithm}-ranking.json"
            print(f"Bootstrap inputs {split}: {algorithm}", flush=True)
            if not prediction_path.is_file() or not rank_path.is_file():
                model = joblib.load(run / f"{algorithm}.joblib")
                ratings = frames[split]
                predictions = model.predict(ratings.userId, ratings.movieId)
                ratings.assign(predicted=predictions).to_parquet(prediction_path, index=False)
                ranking = evaluate_ranking(
                    model,
                    ratings,
                    frames["train"] if split == "validation" else history_test,
                    seed=42,
                    details=True,
                )
                atomic_json(rank_path, ranking)
                del model
            predicted = pd.read_parquet(prediction_path)
            errors[algorithm] = user_errors(predicted, predicted.predicted)
            ranking = json.loads(rank_path.read_text(encoding="utf-8"))
            ranks[algorithm] = pd.DataFrame(
                ranking["per_user"],
                columns=["userId", "precision_at_10", "recall_at_10", "ndcg_at_10"],
            )
        comparisons = []
        # References are fixed in plan.json before any repeated test results are read.
        for metric in METRICS:
            reference = plan["reference_ranking" if metric == "ndcg_at_10" else "reference_rating"]
            inputs = ranks if metric == "ndcg_at_10" else errors
            for algorithm in algorithms:
                if algorithm == reference:
                    continue
                comparisons.append(
                    {
                        "algorithm_a": algorithm,
                        "algorithm_b": reference,
                        **paired_bootstrap(inputs[algorithm], inputs[reference], metric),
                    }
                )
        report["bootstrap"][split] = comparisons
        publish()
    report["status"] = "complete"
    pd.DataFrame(report["seed_runs"]).to_json(destination / "seed-runs.json", orient="records")
    pd.DataFrame(
        [{"split": split, **row} for split, rows in report["bootstrap"].items() for row in rows]
    ).to_csv(destination / "bootstrap.csv", index=False)
    for category, rows in (("seed", report["seed_runs"]), ("temporal", report["temporal_runs"])):
        pd.json_normalize(rows).to_csv(destination / f"{category}-runs.csv", index=False)
    publish()
    return report
