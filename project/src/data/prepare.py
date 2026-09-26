"""Validate MovieLens 25M and write a chronologically ordered clean Parquet file.

Run from the project directory: ``python -m src.data.prepare``.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import duckdb

from src.data.common import (
    connection,
    load_config,
    project_path,
    rating_csv_sql,
    sql_literal,
    write_json,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(force: bool = False) -> dict:
    config = load_config()
    dataset = project_path(config["dataset_dir"])
    processed = project_path(config["processed_dir"])
    results = project_path(config["results_dir"])
    ratings_csv = dataset / "ratings.csv"
    movies_csv = dataset / "movies.csv"
    readme = dataset / "README.txt"
    for required in (ratings_csv, movies_csv, readme):
        if not required.is_file():
            raise FileNotFoundError(f"MovieLens input missing: {required}")
    output = processed / "ratings_clean.parquet"
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; use --force to regenerate")

    con = connection(config)
    source = rating_csv_sql(ratings_csv)
    start = time.perf_counter()
    quality = con.execute(
        f"""
        SELECT COUNT(*) AS rows,
               COUNT(*) FILTER (WHERE userId IS NULL OR movieId IS NULL
                    OR rating IS NULL OR timestamp IS NULL) AS missing_rows,
               COUNT(*) FILTER (WHERE userId <= 0 OR movieId <= 0) AS invalid_ids,
               COUNT(*) FILTER (WHERE rating < 0.5 OR rating > 5.0
                    OR rating * 2 != FLOOR(rating * 2)) AS invalid_ratings,
               COUNT(*) FILTER (WHERE timestamp <= 0) AS invalid_timestamps,
               COUNT(DISTINCT userId) AS users,
               COUNT(DISTINCT movieId) AS rated_movies,
               MIN(timestamp) AS min_timestamp,
               MAX(timestamp) AS max_timestamp,
               MIN(rating) AS min_rating,
               MAX(rating) AS max_rating
        FROM {source}
        """
    ).fetchone()
    keys = (
        "source_rows", "missing_rows", "invalid_ids", "invalid_ratings",
        "invalid_timestamps", "source_users", "rated_movies",
        "min_timestamp", "max_timestamp", "min_rating", "max_rating",
    )
    profile = dict(zip(keys, quality))
    movie_quality = con.execute(
        f"""SELECT COUNT(*), COUNT(DISTINCT movieId),
                   COUNT(*) FILTER (WHERE movieId IS NULL OR title IS NULL)
            FROM read_csv({sql_literal(movies_csv)}, header=true)"""
    ).fetchone()
    profile.update(
        movies_rows=movie_quality[0],
        distinct_movies=movie_quality[1],
        invalid_movies_rows=movie_quality[2],
    )

    valid = (
        "userId IS NOT NULL AND movieId IS NOT NULL AND rating IS NOT NULL "
        "AND timestamp IS NOT NULL AND userId > 0 AND movieId > 0 "
        "AND timestamp > 0 AND rating BETWEEN 0.5 AND 5.0 "
        "AND rating * 2 = FLOOR(rating * 2)"
    )
    valid_rows, duplicate_rows = con.execute(
        f"""SELECT COUNT(*),
                   COUNT(*) - COUNT(DISTINCT (userId, movieId, rating, timestamp))
            FROM {source} WHERE {valid}"""
    ).fetchone()
    profile["valid_before_dedup"] = valid_rows
    profile["exact_duplicate_rows"] = duplicate_rows
    base = f"SELECT userId, movieId, rating, timestamp FROM {source} WHERE {valid}"
    if duplicate_rows:
        base = f"SELECT DISTINCT * FROM ({base})"
    # row_id is a stable rank after cleaning. Identical rows have already been removed.
    con.execute(
        f"""COPY (
            SELECT ROW_NUMBER() OVER (ORDER BY timestamp, userId, movieId, rating) - 1 AS row_id,
                   userId, movieId, CAST(rating AS FLOAT) AS rating, timestamp
            FROM ({base})
            ORDER BY row_id
        ) TO {sql_literal(output)} (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )
    actual = con.execute(f"SELECT COUNT(*) FROM read_parquet({sql_literal(output)})").fetchone()[0]
    expected = valid_rows - duplicate_rows
    if actual != expected:
        raise RuntimeError(f"Clean row count {actual} differs from expected {expected}")

    profile.update(
        clean_rows=actual,
        source_files={
            "ratings.csv": {"bytes": ratings_csv.stat().st_size, "sha256": sha256_file(ratings_csv)},
            "movies.csv": {"bytes": movies_csv.stat().st_size, "sha256": sha256_file(movies_csv)},
        },
        clean_file=str(output.relative_to(project_path("."))).replace("\\", "/"),
        preparation_seconds=round(time.perf_counter() - start, 3),
        engine={"duckdb": duckdb.__version__},
    )
    results.mkdir(exist_ok=True)
    write_json(results / "data_profile.json", profile)
    con.close()
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate the clean Parquet file")
    args = parser.parse_args()
    profile = prepare(force=args.force)
    print(f"Prepared {profile['clean_rows']:,} ratings; {profile['exact_duplicate_rows']} duplicates removed")


if __name__ == "__main__":
    main()
