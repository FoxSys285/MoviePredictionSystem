"""Global-mean rating baseline with temporal CV and coverage reporting.

Examples (run from project/):
    python -m src.data.baseline --split validation
    python -m src.data.baseline --cv
    python -m src.data.baseline --split test --allow-test

The test command also requires report/model_selection.md, created after the group
locks its model choice. Never inspect test metrics during tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

from src.data.common import connection, load_config, parquet_sql, project_path, read_json, sql_literal, write_json


METRIC_COLUMNS = (
    "model", "task", "fold", "seed", "metric", "value", "train_time_s",
    "inference_ms", "hardware", "config",
)


def write_rows(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(config: dict, fold: str | int, metric: str, value: float,
               train_seconds: float, inference_ms: float, detail: dict) -> dict:
    return {
        "model": "mean_rating",
        "task": "rating_prediction",
        "fold": fold,
        "seed": config["random_seed"],
        "metric": metric,
        "value": round(float(value), 8),
        "train_time_s": round(train_seconds, 6),
        "inference_ms": round(inference_ms, 4),
        "hardware": platform.processor() or platform.machine(),
        "config": json.dumps(detail, ensure_ascii=False, sort_keys=True),
    }


def batch_latency_ms(con, relation: str, mean: float, where: str = "") -> float:
    condition = f"WHERE {where}" if where else ""
    start = time.perf_counter()
    samples = con.execute(
        f"SELECT {mean:.17g} AS rating_pred FROM {relation} {condition} LIMIT 100000"
    ).fetchall()
    if not samples:
        raise ValueError("Inference sample is empty")
    return (time.perf_counter() - start) * 1000


def coverage(con, relation: str, user_sql: str, movie_sql: str,
             min_user: int, min_movie: int) -> tuple[dict, list[dict]]:
    joined = (
        f"FROM {relation} t LEFT JOIN {user_sql} u USING (userId) "
        f"LEFT JOIN {movie_sql} m USING (movieId)"
    )
    conditions = {
        "all": "TRUE",
        "warm": f"u.n_ratings >= {min_user} AND m.n_ratings >= {min_movie}",
        "unseen_user": "u.userId IS NULL",
        "unseen_movie": "m.movieId IS NULL",
        "both_unseen": "u.userId IS NULL AND m.movieId IS NULL",
        "low_history_user": f"u.n_ratings IS NOT NULL AND u.n_ratings < {min_user}",
        "low_history_movie": f"m.n_ratings IS NOT NULL AND m.n_ratings < {min_movie}",
    }
    names = list(conditions)
    exprs = [f"COUNT(*) FILTER (WHERE {conditions[name]}) AS {name}" for name in names]
    values = con.execute(f"SELECT {', '.join(exprs)} {joined}").fetchone()
    counts = dict(zip(names, values))
    counts["not_warm"] = counts["all"] - counts["warm"]
    rows = [
        {"cohort": name, "rows": n, "share": round(n / counts["all"], 8)}
        for name, n in counts.items()
    ]
    return counts, rows


def evaluate_holdout(split: str, allow_test: bool = False) -> dict:
    config = load_config()
    processed = project_path(config["processed_dir"])
    results = project_path(config["results_dir"])
    if split == "test" and (not allow_test or not (project_path("report") / "model_selection.md").is_file()):
        raise PermissionError(
            "Test is locked: first create report/model_selection.md, then pass --allow-test"
        )
    train = processed / "train.parquet"
    target = processed / f"{split}.parquet"
    for required in (train, target, processed / "user_stats.parquet", processed / "movie_stats.parquet"):
        if not required.is_file():
            raise FileNotFoundError(f"Run prepare and split first: {required}")
    con = connection(config)
    train_sql, target_sql = parquet_sql(train), parquet_sql(target)
    fit_start = time.perf_counter()
    mean, train_count = con.execute(f"SELECT AVG(rating), COUNT(*) FROM {train_sql}").fetchone()
    train_seconds = time.perf_counter() - fit_start
    inference_ms = batch_latency_ms(con, target_sql, mean)

    predictions = results / f"predictions_mean_{split}.csv"
    results.mkdir(exist_ok=True)
    con.execute(
        f"COPY (SELECT row_id, 'mean_rating' AS model, {sql_literal(split)} AS split, "
        f"userId, movieId, rating AS rating_true, {mean:.17g} AS rating_pred "
        f"FROM {target_sql} ORDER BY row_id) "
        f"TO {sql_literal(predictions)} (HEADER, DELIMITER ',')"
    )
    rmse, mae, count = con.execute(
        f"SELECT SQRT(AVG(POWER(rating - {mean:.17g}, 2))), "
        f"AVG(ABS(rating - {mean:.17g})), COUNT(*) FROM {target_sql}"
    ).fetchone()
    user_sql = parquet_sql(processed / "user_stats.parquet")
    movie_sql = parquet_sql(processed / "movie_stats.parquet")
    min_user, min_movie = int(config["min_user_ratings"]), int(config["min_movie_ratings"])
    counts, coverage_rows = coverage(con, target_sql, user_sql, movie_sql, min_user, min_movie)
    write_rows(results / f"coverage_{split}.csv", ("cohort", "rows", "share"), coverage_rows)
    warm_rmse, warm_mae, warm_count = con.execute(
        f"SELECT SQRT(AVG(POWER(t.rating - {mean:.17g}, 2))), "
        f"AVG(ABS(t.rating - {mean:.17g})), COUNT(*) "
        f"FROM {target_sql} t JOIN {user_sql} u USING (userId) "
        f"JOIN {movie_sql} m USING (movieId) "
        f"WHERE u.n_ratings >= {min_user} AND m.n_ratings >= {min_movie}"
    ).fetchone()
    detail = {"fit_rows": train_count, "score_rows": count, "inference_batch_rows": min(100000, count),
              "warm_rows": warm_count, "min_user_ratings": min_user, "min_movie_ratings": min_movie}
    metrics = [
        metric_row(config, split, "rmse_all", rmse, train_seconds, inference_ms, detail),
        metric_row(config, split, "mae_all", mae, train_seconds, inference_ms, detail),
        metric_row(config, split, "rmse_warm", warm_rmse, train_seconds, inference_ms, detail),
        metric_row(config, split, "mae_warm", warm_mae, train_seconds, inference_ms, detail),
    ]
    write_rows(results / f"metrics_mean_{split}.csv", METRIC_COLUMNS, metrics)
    write_json(project_path(config["artifacts_dir"]) / "mean_baseline.json", {
        "model": "mean_rating", "global_mean": mean, "train_rows": train_count,
        "training_file": "data/processed/train.parquet", "seed": config["random_seed"],
    })
    con.close()
    return {"split": split, "rmse": rmse, "mae": mae, "rows": count,
            "warm_rows": counts["warm"], "mean": mean}


def evaluate_cv() -> dict:
    config = load_config()
    processed = project_path(config["processed_dir"])
    results = project_path(config["results_dir"])
    folds = read_json(processed / "cv_folds.json")
    con = connection(config)
    train_sql = parquet_sql(processed / "train.parquet")
    rows = []
    coverage_rows = []
    scores = {"rmse": [], "mae": [], "rmse_warm": [], "mae_warm": []}
    min_user, min_movie = int(config["min_user_ratings"]), int(config["min_movie_ratings"])
    for fold in folds:
        start = time.perf_counter()
        mean = con.execute(
            f"SELECT AVG(rating) FROM {train_sql} WHERE row_id <= ?",
            [fold["train_end_row_id"]],
        ).fetchone()[0]
        train_seconds = time.perf_counter() - start
        where = f"row_id BETWEEN {fold['validation_start_row_id']} AND {fold['validation_end_row_id']}"
        inference_ms = batch_latency_ms(con, train_sql, mean, where)
        rmse, mae, n = con.execute(
            f"SELECT SQRT(AVG(POWER(rating - {mean:.17g}, 2))), "
            f"AVG(ABS(rating - {mean:.17g})), COUNT(*) FROM {train_sql} WHERE {where}"
        ).fetchone()
        if n != fold["validation_rows"]:
            raise RuntimeError(f"Fold {fold['fold']} row count mismatch: {n}")
        fold_train = f"(SELECT * FROM {train_sql} WHERE row_id <= {fold['train_end_row_id']})"
        fold_validation = f"(SELECT * FROM {train_sql} WHERE {where})"
        user_sql = f"(SELECT userId, COUNT(*) AS n_ratings FROM {fold_train} GROUP BY userId)"
        movie_sql = f"(SELECT movieId, COUNT(*) AS n_ratings FROM {fold_train} GROUP BY movieId)"
        counts, fold_coverage = coverage(
            con, fold_validation, user_sql, movie_sql, min_user, min_movie
        )
        for coverage_row in fold_coverage:
            coverage_rows.append({"fold": fold["fold"], **coverage_row})
        warm_rmse, warm_mae, warm_n = con.execute(
            f"SELECT SQRT(AVG(POWER(t.rating - {mean:.17g}, 2))), "
            f"AVG(ABS(t.rating - {mean:.17g})), COUNT(*) "
            f"FROM {fold_validation} t JOIN {user_sql} u USING (userId) "
            f"JOIN {movie_sql} m USING (movieId) "
            f"WHERE u.n_ratings >= {min_user} AND m.n_ratings >= {min_movie}"
        ).fetchone()
        if warm_n != counts["warm"] or warm_n == 0:
            raise RuntimeError(f"Fold {fold['fold']} warm-start count mismatch/empty")
        detail = {"fit_rows": fold["train_rows"], "score_rows": n,
                  "warm_rows": warm_n, "inference_batch_rows": min(100000, n),
                  "mean": mean, "min_user_ratings": min_user,
                  "min_movie_ratings": min_movie}
        for name, value in (("rmse", rmse), ("mae", mae),
                            ("rmse_warm", warm_rmse), ("mae_warm", warm_mae)):
            scores[name].append(value)
            rows.append(metric_row(config, fold["fold"], name, value, train_seconds, inference_ms, detail))
    for name, values in scores.items():
        rows.append(metric_row(config, "summary", f"{name}_mean", statistics.mean(values), 0, 0, {"folds": len(folds)}))
        rows.append(metric_row(config, "summary", f"{name}_std", statistics.stdev(values), 0, 0, {"folds": len(folds)}))
    write_rows(results / "metrics_mean_cv.csv", METRIC_COLUMNS, rows)
    write_rows(results / "coverage_cv.csv", ("fold", "cohort", "rows", "share"), coverage_rows)
    con.close()
    return {name: {"mean": statistics.mean(values), "std": statistics.stdev(values)} for name, values in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--cv", action="store_true", help="Evaluate five folds inside train")
    parser.add_argument("--allow-test", action="store_true", help="Unlock test after model selection is recorded")
    args = parser.parse_args()
    result = evaluate_cv() if args.cv else evaluate_holdout(args.split, args.allow_test)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
