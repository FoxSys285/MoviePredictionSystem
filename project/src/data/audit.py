"""Audit chronological splits, CV fold boundaries and validation predictions.

Run from project/: ``python -m src.data.audit``. The audit reads test metadata
but does not compute test prediction metrics.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone

from src.data.common import connection, load_config, parquet_sql, project_path, read_json, sql_literal


def audit() -> list[tuple[str, bool, str]]:
    config = load_config()
    processed = project_path(config["processed_dir"])
    results = project_path(config["results_dir"])
    splits = read_json(processed / "splits.json")
    folds = read_json(processed / "cv_folds.json")
    manifest = read_json(processed / "dataset_manifest.json")
    checks: list[tuple[str, bool, str]] = []
    con = connection(config)
    stats = {}
    for name in ("train", "validation", "test"):
        path = processed / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        total, distinct_ids, min_id, max_id, min_ts, max_ts, missing, invalid = con.execute(
            f"""SELECT COUNT(*), COUNT(DISTINCT row_id), MIN(row_id), MAX(row_id),
                       MIN(timestamp), MAX(timestamp),
                       COUNT(*) FILTER (WHERE row_id IS NULL OR userId IS NULL
                           OR movieId IS NULL OR rating IS NULL OR timestamp IS NULL),
                       COUNT(*) FILTER (WHERE rating < 0.5 OR rating > 5.0
                           OR rating * 2 != FLOOR(rating * 2))
                FROM {parquet_sql(path)}"""
        ).fetchone()
        stats[name] = {"rows": total, "unique": distinct_ids, "min_id": min_id,
                       "max_id": max_id, "min_ts": min_ts, "max_ts": max_ts}
        checks.append((f"{name}: schema, rows and unique row_id",
                       total == splits["sets"][name]["rows"] == distinct_ids and missing == 0 and invalid == 0,
                       f"rows={total:,}, missing={missing}, invalid_ratings={invalid}"))
    n = sum(x["rows"] for x in stats.values())
    ordered = (
        stats["train"]["min_id"] == 0
        and stats["train"]["max_id"] + 1 == stats["validation"]["min_id"]
        and stats["validation"]["max_id"] + 1 == stats["test"]["min_id"]
        and stats["test"]["max_id"] == n - 1
        and stats["train"]["max_ts"] < stats["validation"]["min_ts"]
        and stats["validation"]["max_ts"] < stats["test"]["min_ts"]
    )
    checks.append(("Disjoint chronological ranges", ordered,
                   f"train_max={stats['train']['max_ts']}, validation_min={stats['validation']['min_ts']}, "
                   f"validation_max={stats['validation']['max_ts']}, test_min={stats['test']['min_ts']}"))
    checks.append(("Complete clean dataset", n == manifest["clean_rows"],
                   f"split_rows={n:,}, clean_rows={manifest['clean_rows']:,}"))
    clean_sql = parquet_sql(processed / "ratings_clean.parquet")
    inversions = con.execute(
        f"SELECT COUNT(*) FILTER (WHERE previous_timestamp > timestamp) "
        f"FROM (SELECT timestamp, LAG(timestamp) OVER (ORDER BY row_id) AS previous_timestamp "
        f"FROM {clean_sql})"
    ).fetchone()[0]
    checks.append(("All ratings ordered by time", inversions == 0,
                   f"timestamp_inversions={inversions}"))
    train_sql = parquet_sql(processed / "train.parquet")
    user_total = con.execute(f"SELECT SUM(n_ratings) FROM {parquet_sql(processed / 'user_stats.parquet')}").fetchone()[0]
    movie_total = con.execute(f"SELECT SUM(n_ratings) FROM {parquet_sql(processed / 'movie_stats.parquet')}").fetchone()[0]
    checks.append(("Train-only user/movie counts", user_total == movie_total == stats["train"]["rows"],
                   f"users_sum={user_total:,}, movies_sum={movie_total:,}"))

    fold_ok = len(folds) == int(config["time_cv_folds"])
    for fold in folds:
        left, right = fold["train_end_row_id"], fold["validation_end_row_id"]
        actual_left = con.execute(f"SELECT timestamp FROM {train_sql} WHERE row_id = ?", [left]).fetchone()
        actual_next = con.execute(f"SELECT timestamp FROM {train_sql} WHERE row_id = ?", [left + 1]).fetchone()
        actual_right = con.execute(f"SELECT timestamp FROM {train_sql} WHERE row_id = ?", [right]).fetchone()
        fold_ok &= bool(
            actual_left and actual_next and actual_right
            and actual_left[0] < actual_next[0]
            and actual_left[0] == fold["train_end_timestamp"]
            and actual_right[0] == fold["validation_end_timestamp"]
            and right - left == fold["validation_rows"]
        )
    fold_ok &= folds[-1]["validation_end_row_id"] == stats["train"]["max_id"]
    checks.append(("Five expanding-time folds", fold_ok, f"folds={len(folds)}"))
    warm_path = processed / "validation_warm.parquet"
    warm_rows, warm_distinct, warm_mismatch = con.execute(
        f"""SELECT COUNT(*), COUNT(DISTINCT w.row_id),
                   COUNT(*) FILTER (WHERE v.row_id IS NULL OR w.userId != v.userId
                       OR w.movieId != v.movieId OR w.rating != v.rating)
            FROM {parquet_sql(warm_path)} w
            LEFT JOIN {parquet_sql(processed / 'validation.parquet')} v USING (row_id)"""
    ).fetchone()
    checks.append(("Shared warm validation subset", warm_rows == warm_distinct == manifest["validation_warm_rows"]
                   and warm_mismatch == 0,
                   f"rows={warm_rows:,}, mismatched={warm_mismatch:,}"))
    cv_warm_path = processed / "cv_warm_ids.parquet"
    cv_warm_count, cv_warm_unique = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT row_id) FROM {parquet_sql(cv_warm_path)}"
    ).fetchone()
    cv_warm_ok = cv_warm_count == cv_warm_unique == sum(f["warm_validation_rows"] for f in folds)
    for fold in folds:
        n, outside = con.execute(
            f"SELECT COUNT(*), COUNT(*) FILTER (WHERE row_id < ? OR row_id > ?) "
            f"FROM {parquet_sql(cv_warm_path)} WHERE fold = ?",
            [fold["validation_start_row_id"], fold["validation_end_row_id"], fold["fold"]],
        ).fetchone()
        cv_warm_ok &= n == fold["warm_validation_rows"] and outside == 0
    checks.append(("Shared warm IDs for all CV folds", cv_warm_ok,
                   f"warm_cv_rows={cv_warm_count:,}, folds={len(folds)}"))

    rating_models = ("mean", "linear", "xgboost", "svd", "ncf", "knn")
    predictions = results / "predictions_mean_validation.csv"
    for model in rating_models:
        prediction_path = results / f"predictions_{model}_validation.csv"
        if not prediction_path.is_file():
            continue
        pred_sql = f"read_csv({sql_literal(prediction_path)}, header=true)"
        total, distinct_ids, min_id, max_id, mismatched = con.execute(
            f"""SELECT COUNT(*), COUNT(DISTINCT p.row_id), MIN(p.row_id), MAX(p.row_id),
                       COUNT(*) FILTER (WHERE v.row_id IS NULL OR p.userId != v.userId
                           OR p.movieId != v.movieId OR p.rating_true != v.rating)
                FROM {pred_sql} p LEFT JOIN {parquet_sql(processed / 'validation.parquet')} v
                USING (row_id)"""
        ).fetchone()
        pred_ok = (total == distinct_ids == stats["validation"]["rows"]
                   and min_id == stats["validation"]["min_id"]
                   and max_id == stats["validation"]["max_id"] and mismatched == 0)
        checks.append((f"{model} validation predictions match split", pred_ok,
                       f"rows={total:,}, distinct_row_ids={distinct_ids:,}, mismatched={mismatched:,}"))
    if predictions.is_file():
        rmse, mae = con.execute(
            f"""SELECT SQRT(AVG(POWER(rating_true - rating_pred, 2))),
                       AVG(ABS(rating_true - rating_pred))
                FROM read_csv({sql_literal(predictions)}, header=true)"""
        ).fetchone()
        metrics_path = results / "metrics_mean_validation.csv"
        with metrics_path.open(newline="", encoding="utf-8") as source:
            metrics = {row["metric"]: float(row["value"]) for row in csv.DictReader(source)}
        metric_ok = abs(rmse - metrics["rmse_all"]) < 1e-7 and abs(mae - metrics["mae_all"]) < 1e-7
        checks.append(("Mean validation metrics reproducible", metric_ok,
                       f"RMSE={rmse:.8f}, MAE={mae:.8f}"))
    else:
        checks.append(("Mean validation predictions cover split", False, "Prediction file not created"))
    con.close()

    now = datetime.now(timezone.utc).isoformat()
    lines = ["# TV1 data audit", "", f"Generated (UTC): {now}", "",
             "This audit checks split metadata and validation results. It does not use test labels for model selection.", ""]
    for title, passed, detail in checks:
        lines.append(f"- [{'x' if passed else ' '}] **{title}:** {detail}")
    lines.extend(["", f"Result: **{'PASS' if all(item[1] for item in checks) else 'FAIL'}**", ""])
    results.mkdir(exist_ok=True)
    (results / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    if not all(item[1] for item in checks):
        raise RuntimeError("Data audit failed; inspect results/data_audit.md")
    return checks


if __name__ == "__main__":
    print(f"Passed {len(audit())} TV1 audit checks")
