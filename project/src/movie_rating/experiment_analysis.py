"""Supplementary cold-user measurements, without retraining or changing model selection."""

import hashlib
import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from movie_rating.evaluation import regression_metrics
from movie_rating.services.recommendations import RecommendationService
from movie_rating.training import current_run, write_json

SUPPORT_SIZES = (0, 5, 10, 20)


def cold_user_cohort(heldout, train_users, trained_items, max_users=30, seed=42):
    """Fix a support pool and a strictly later query set, shared by all support sizes."""
    available = heldout.loc[
        ~heldout.userId.isin(train_users) & heldout.movieId.isin(trained_items)
    ].sort_values(["userId", "timestamp", "movieId"])
    eligible = []
    for user, rows in available.groupby("userId", sort=True):
        if len(rows) < 25:
            continue
        support = rows.iloc[:20]
        # Never split a timestamp across support/query, including tied rating sessions.
        query = rows.loc[rows.timestamp > support.timestamp.max()]
        if len(query) >= 5:
            eligible.append((int(user), support, query))
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(eligible), min(max_users, len(eligible)), replace=False))
    return [eligible[index] for index in chosen], len(eligible)


def evaluate_cold_history(model, algorithm, cohort):
    results = []
    for size in SUPPORT_SIZES:
        actual, predicted, user_rmse, user_mae = [], [], [], []
        for user, support, query in cohort:
            if size == 0:
                scores = model.predict(np.full(len(query), user), query.movieId)
            else:
                history = [
                    {"movie_id": int(row.movieId), "rating": float(row.rating)}
                    for row in support.iloc[:size].itertuples()
                ]
                profile = RecommendationService.profile_scores(model, history, algorithm)
                if profile is None:
                    raise ValueError(f"{algorithm} does not support the cold-user benchmark.")
                scores = profile.loc[query.movieId].to_numpy()
            metrics = regression_metrics(query.rating, scores)
            actual.extend(query.rating)
            predicted.extend(scores)
            user_rmse.append(metrics["rmse"])
            user_mae.append(metrics["mae"])
        results.append({
            "support_ratings": size,
            "users": len(cohort),
            "query_ratings": len(actual),
            **(regression_metrics(actual, predicted) if actual else {"rmse": None, "mae": None}),
            "user_mean_rmse": float(np.mean(user_rmse)) if user_rmse else None,
            "user_mean_mae": float(np.mean(user_mae)) if user_mae else None,
        })
    return {"algorithm": algorithm, "points": results}


def analyze_experiments(settings):
    run = current_run(settings)
    validation = json.loads((run / "validation.json").read_text(encoding="utf-8"))
    data_dir = settings.artifacts_dir / "data"
    fingerprint = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    if fingerprint != validation["data_manifest_sha256"]:
        raise ValueError("Prepared data changed since training; use the matching artifacts.")
    algorithms = [row["algorithm"] for row in validation["results"]]
    train = pd.read_parquet(data_dir / "train.parquet")
    # A common candidate universe makes support/query identical across every model.
    trained_items = set(train.movieId.unique())
    for algorithm in algorithms:
        model = joblib.load(run / f"{algorithm}.joblib")
        trained_items.intersection_update(map(int, model.movie_ids))
        del model
    report = {
        "schema_version": 1,
        "run_id": validation["run_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "data_manifest_sha256": fingerprint,
        "cold_history": {},
    }
    for split in ("validation", "test"):
        # Test analysis is available only after the original final evaluation exists.
        if split == "test" and not (run / "test.json").is_file():
            continue
        heldout = pd.read_parquet(data_dir / f"{split}.parquet")
        seed = validation["settings"]["experiment"]["seed"]
        cohort, eligible = cold_user_cohort(
            heldout, train.userId.unique(), trained_items, seed=seed
        )
        rows = []
        for algorithm in algorithms:
            print(f"Cold-history {split}: {algorithm}, {len(cohort)} users", flush=True)
            model = joblib.load(run / f"{algorithm}.joblib")
            rows.append(evaluate_cold_history(model, algorithm, cohort))
            del model
        report["cold_history"][split] = {
            "eligible_users": eligible,
            "evaluated_users": len(cohort),
            "user_ids": [user for user, _, _ in cohort],
            "max_users": 30,
            "seed": seed,
            "policy": "cold_users_first_20_support_strictly_later_query_trained_items_only",
            "aggregation": "rating_weighted",
            "results": rows,
        }
    # Preserve the original validation/test reports and the selected deployment model.
    temporary = run / "analysis.tmp"
    write_json(temporary, report)
    temporary.replace(run / "analysis.json")
    return report
