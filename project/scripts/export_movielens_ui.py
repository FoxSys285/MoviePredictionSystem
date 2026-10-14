"""Export a compact, train-only MovieLens catalog for the frontend demo."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]


def display_title(raw_title: str) -> tuple[str, int | None]:
    match = re.search(r"\s+\((\d{4})\)$", raw_title)
    year = int(match.group(1)) if match else None
    title = raw_title[: match.start()] if match else raw_title

    article = re.match(r"^(.*),\s+(The|A|An)$", title, flags=re.IGNORECASE)
    if article:
        title = f"{article.group(2)} {article.group(1)}"
    return title, year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "movielens25m",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "data" / "movielens-catalog.json",
    )
    args = parser.parse_args()

    report = json.loads((args.processed_dir / "processing_report.json").read_text(encoding="utf-8"))
    movie_metadata = pl.read_parquet(args.processed_dir / "movies.parquet").select(
        "movieId", "title", "genres"
    )
    train_movie_stats = (
        pl.scan_parquet(args.processed_dir / "ratings_train.parquet")
        .group_by("movieId")
        .agg(
            pl.len().alias("ratingCount"),
            pl.col("rating").mean().round(3).alias("averageRating"),
        )
        .collect()
    )
    sample_user_ratings = (
        pl.scan_parquet(args.processed_dir / "ratings_train.parquet")
        .filter(pl.col("userId").is_in([1, 2, 3, 4, 5]))
        .select("userId", "movieId", "rating", "timestamp")
        .sort(["userId", "timestamp", "movieId"])
        .collect()
        .to_dicts()
    )
    rows = (
        movie_metadata.join(train_movie_stats, on="movieId", how="inner")
        .sort("ratingCount", descending=True)
        .iter_rows(named=True)
    )

    catalog = []
    for row in rows:
        title, year = display_title(row["title"])
        genres = row["genres"].split("|") if row["genres"] else []
        genres = [genre for genre in genres if genre and genre != "(no genres listed)"]
        catalog.append(
            {
                "movieId": int(row["movieId"]),
                "title": title,
                "year": year,
                "genres": genres,
                "averageRating": round(float(row["averageRating"]), 3),
                "ratingCount": int(row["ratingCount"]),
            }
        )

    split = report["splits"]["train"]
    demo_users = []
    for user_id in range(1, 6):
        ratings = [
            {
                "movieId": int(row["movieId"]),
                "rating": float(row["rating"]),
                "timestamp": int(row["timestamp"]),
            }
            for row in sample_user_ratings
            if row["userId"] == user_id
        ]
        demo_users.append({"userId": user_id, "ratings": ratings})

    payload = {
        "dataset": report["dataset"],
        "statistics": {
            "trainRatings": int(split["rows"]),
            "users": int(split["users"]),
            "movies": int(split["movies"]),
            "averageRating": round(float(split["rating_mean"]), 3),
            "minimumRating": float(split["rating_min"]),
            "maximumRating": float(split["rating_max"]),
            "ratingStep": 0.5,
            "splitStrategy": report["split_strategy"],
            "requestedSplitRatios": report["split_ratios_requested"],
            "aprioriLikeThreshold": float(report["like_threshold_for_apriori_transactions"]),
        },
        "demoUsers": demo_users,
        "catalog": catalog,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(catalog):,} movies")


if __name__ == "__main__":
    main()
