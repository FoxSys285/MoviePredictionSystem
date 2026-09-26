"""Create chronological 70/15/15 sets and five expanding-time CV folds.

Run after prepare: ``python -m src.data.split``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.data.common import connection, load_config, parquet_sql, project_path, read_json, sql_literal, write_json


def iso_utc(timestamp: int | None) -> str | None:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp is not None else None


def split(force: bool = False) -> dict:
    config = load_config()
    processed = project_path(config["processed_dir"])
    clean = processed / "ratings_clean.parquet"
    if not clean.is_file():
        raise FileNotFoundError(f"Run python -m src.data.prepare first: {clean}")
    outputs = {name: processed / f"{name}.parquet" for name in ("train", "validation", "test")}
    extra = [
        processed / "user_stats.parquet", processed / "movie_stats.parquet",
        processed / "train_core.parquet", processed / "validation_warm.parquet",
        processed / "cv_warm_ids.parquet", processed / "splits.json",
        processed / "cv_folds.json", processed / "dataset_manifest.json",
    ]
    existing = [p for p in list(outputs.values()) + extra if p.exists()]
    if existing and not force:
        raise FileExistsError(f"Outputs exist ({existing[0]}); use --force to regenerate")
    if force:
        for path in existing:
            path.unlink()

    con = connection(config)
    clean_sql = parquet_sql(clean)
    n = con.execute(f"SELECT COUNT(*) FROM {clean_sql}").fetchone()[0]
    ratios = config["split_ratios"]
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or any(x <= 0 for x in ratios.values()):
        raise ValueError("split_ratios must be positive and sum to 1")
    train_target = int(n * ratios["train"])
    validation_target = int(n * (ratios["train"] + ratios["validation"]))
    if not 0 < train_target < validation_target < n:
        raise ValueError("Dataset is too small for the requested split")

    # Include all rows sharing each cutoff timestamp in the earlier partition.
    cutoffs = con.execute(
        f"SELECT row_id, timestamp FROM {clean_sql} WHERE row_id IN (?, ?) ORDER BY row_id",
        [train_target - 1, validation_target - 1],
    ).fetchall()
    if len(cutoffs) != 2:
        raise RuntimeError("Chronological cutoff rows not found")
    train_ts, validation_ts = cutoffs[0][1], cutoffs[1][1]
    if train_ts >= validation_ts:
        raise ValueError("Train and validation cutoffs share a timestamp; choose another split")
    predicates = {
        "train": f"timestamp <= {train_ts}",
        "validation": f"timestamp > {train_ts} AND timestamp <= {validation_ts}",
        "test": f"timestamp > {validation_ts}",
    }
    summaries = {}
    for name, output in outputs.items():
        con.execute(
            f"COPY (SELECT row_id, userId, movieId, rating, timestamp "
            f"FROM {clean_sql} WHERE {predicates[name]} ORDER BY row_id) "
            f"TO {sql_literal(output)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        count, min_id, max_id, min_ts, max_ts = con.execute(
            f"SELECT COUNT(*), MIN(row_id), MAX(row_id), MIN(timestamp), MAX(timestamp) "
            f"FROM {parquet_sql(output)}"
        ).fetchone()
        summaries[name] = {
            "rows": count, "ratio": count / n,
            "min_row_id": min_id, "max_row_id": max_id,
            "min_timestamp": min_ts, "max_timestamp": max_ts,
            "min_time_utc": iso_utc(min_ts), "max_time_utc": iso_utc(max_ts),
            "file": str(output.relative_to(project_path("."))).replace("\\", "/"),
        }
    if sum(x["rows"] for x in summaries.values()) != n:
        raise RuntimeError("Split row counts do not sum to the clean dataset")
    if summaries["train"]["max_row_id"] + 1 != summaries["validation"]["min_row_id"]:
        raise RuntimeError("Gap or overlap between train and validation row IDs")
    if summaries["validation"]["max_row_id"] + 1 != summaries["test"]["min_row_id"]:
        raise RuntimeError("Gap or overlap between validation and test row IDs")

    train_sql = parquet_sql(outputs["train"])
    for column, filename in (
        ("userId", "user_stats.parquet"),
        ("movieId", "movie_stats.parquet"),
    ):
        con.execute(
            f"COPY (SELECT {column}, COUNT(*) AS n_ratings, AVG(rating) AS mean_rating "
            f"FROM {train_sql} GROUP BY {column}) "
            f"TO {sql_literal(processed / filename)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    min_user = int(config["min_user_ratings"])
    min_movie = int(config["min_movie_ratings"])
    user_sql = parquet_sql(processed / "user_stats.parquet")
    movie_sql = parquet_sql(processed / "movie_stats.parquet")
    core = processed / "train_core.parquet"
    con.execute(
        f"COPY (SELECT t.* FROM {train_sql} t "
        f"JOIN {user_sql} u USING (userId) JOIN {movie_sql} m USING (movieId) "
        f"WHERE u.n_ratings >= {min_user} AND m.n_ratings >= {min_movie} "
        f"ORDER BY t.row_id) "
        f"TO {sql_literal(core)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    core_rows, core_users, core_movies = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT userId), COUNT(DISTINCT movieId) FROM {parquet_sql(core)}"
    ).fetchone()
    validation_warm = processed / "validation_warm.parquet"
    validation_sql = parquet_sql(outputs["validation"])
    con.execute(
        f"COPY (SELECT t.* FROM {validation_sql} t "
        f"JOIN {user_sql} u USING (userId) JOIN {movie_sql} m USING (movieId) "
        f"WHERE u.n_ratings >= {min_user} AND m.n_ratings >= {min_movie} "
        f"ORDER BY t.row_id) "
        f"TO {sql_literal(validation_warm)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    validation_warm_rows = con.execute(
        f"SELECT COUNT(*) FROM {parquet_sql(validation_warm)}"
    ).fetchone()[0]

    folds = []
    train_n = summaries["train"]["rows"]
    k = int(config["time_cv_folds"])
    if k < 2 or train_n <= k + 1:
        raise ValueError("time_cv_folds is invalid for the training size")
    target_positions = [max(0, int(train_n * j / (k + 1)) - 1) for j in range(1, k + 1)]
    boundary_ts = [
        con.execute(f"SELECT timestamp FROM {train_sql} WHERE row_id = ?", [position]).fetchone()[0]
        for position in target_positions
    ]
    boundary_ids = [
        con.execute(f"SELECT MAX(row_id) FROM {train_sql} WHERE timestamp <= ?", [timestamp]).fetchone()[0]
        for timestamp in boundary_ts
    ]
    boundaries = boundary_ids + [summaries["train"]["max_row_id"]]
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise RuntimeError("A CV fold would be empty after preserving timestamp ties")
    for j in range(k):
        left, right = boundaries[j], boundaries[j + 1]
        folds.append({
            "fold": j + 1,
            "train_start_row_id": 0,
            "train_end_row_id": left,
            "validation_start_row_id": left + 1,
            "validation_end_row_id": right,
            "train_rows": left + 1,
            "validation_rows": right - left,
            "train_end_timestamp": boundary_ts[j],
            "validation_end_timestamp": boundary_ts[j + 1] if j + 1 < k else summaries["train"]["max_timestamp"],
        })
    warm_parts = []
    for fold in folds:
        fold_train = f"(SELECT * FROM {train_sql} WHERE row_id <= {fold['train_end_row_id']})"
        user_counts = f"(SELECT userId, COUNT(*) AS n_ratings FROM {fold_train} GROUP BY userId)"
        movie_counts = f"(SELECT movieId, COUNT(*) AS n_ratings FROM {fold_train} GROUP BY movieId)"
        warm_parts.append(
            f"SELECT {fold['fold']} AS fold, v.row_id FROM {train_sql} v "
            f"JOIN {user_counts} u USING (userId) JOIN {movie_counts} m USING (movieId) "
            f"WHERE v.row_id BETWEEN {fold['validation_start_row_id']} "
            f"AND {fold['validation_end_row_id']} "
            f"AND u.n_ratings >= {min_user} AND m.n_ratings >= {min_movie}"
        )
    cv_warm = processed / "cv_warm_ids.parquet"
    con.execute(
        f"COPY (SELECT * FROM ({' UNION ALL '.join(warm_parts)}) ORDER BY fold, row_id) "
        f"TO {sql_literal(cv_warm)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    cv_warm_counts = dict(con.execute(
        f"SELECT fold, COUNT(*) FROM {parquet_sql(cv_warm)} GROUP BY fold"
    ).fetchall())
    for fold in folds:
        fold["warm_validation_rows"] = cv_warm_counts.get(fold["fold"], 0)

    split_data = {
        "source_rows": n,
        "target_ratios": ratios,
        "timestamp_ties_kept_together": True,
        "train_cutoff_timestamp": train_ts,
        "validation_cutoff_timestamp": validation_ts,
        "sets": summaries,
    }
    manifest = {
        "clean_rows": n,
        "train_rows": train_n,
        "validation_rows": summaries["validation"]["rows"],
        "test_rows": summaries["test"]["rows"],
        "min_user_ratings": min_user,
        "min_movie_ratings": min_movie,
        "filter_rule": "one-pass counts computed on train only; full train/validation/test retained",
        "train_core_rows": core_rows,
        "train_core_users": core_users,
        "train_core_movies": core_movies,
        "train_core_share_of_train": core_rows / train_n,
        "validation_warm_rows": validation_warm_rows,
        "validation_warm_file": "data/processed/validation_warm.parquet",
        "cv_warm_ids_file": "data/processed/cv_warm_ids.parquet",
        "primary_model_training_file": "data/processed/train.parquet",
        "optional_warm_training_file": "data/processed/train_core.parquet",
        "cv_fold_file": "data/processed/cv_folds.json",
        "random_seed": config["random_seed"],
    }
    write_json(processed / "splits.json", split_data)
    write_json(processed / "cv_folds.json", folds)
    write_json(processed / "dataset_manifest.json", manifest)
    profile_path = project_path(config["results_dir"]) / "data_profile.json"
    profile = read_json(profile_path)
    profile["split"] = {name: {key: value for key, value in summary.items() if key != "file"} for name, summary in summaries.items()}
    profile["train_core"] = {"rows": core_rows, "users": core_users, "movies": core_movies}
    profile["validation_warm_rows"] = validation_warm_rows
    write_json(profile_path, profile)
    con.close()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate all split outputs")
    args = parser.parse_args()
    manifest = split(force=args.force)
    print(
        f"train={manifest['train_rows']:,} validation={manifest['validation_rows']:,} "
        f"test={manifest['test_rows']:,} core={manifest['train_core_rows']:,}"
    )


if __name__ == "__main__":
    main()
