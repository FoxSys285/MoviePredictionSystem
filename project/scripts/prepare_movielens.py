"""Prepare MovieLens 25M ratings and movie features for experiments."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, save_npz


LOGGER = logging.getLogger("prepare_movielens")
RATING_MIN = 0.5
RATING_MAX = 5.0


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Clean MovieLens 25M data, split each user's timeline, and build sparse matrices."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "Dataset",
        help="Folder containing the extracted MovieLens 25M CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed" / "movielens25m",
        help="New output folder. It must not already exist.",
    )
    parser.add_argument("--min-user-ratings", type=int, default=20)
    parser.add_argument("--min-movie-ratings", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--like-threshold", type=float, default=4.0)
    parser.add_argument(
        "--skip-genome",
        action="store_true",
        help="Skip the 15.5M-row genome-scores feature table.",
    )
    return parser.parse_args()


def collect_one(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    return lazy_frame.collect(engine="streaming")


def row_count(lazy_frame: pl.LazyFrame) -> int:
    result = collect_one(lazy_frame.select(pl.len().alias("rows")))
    return int(result.item())


def write_parquet(lazy_frame: pl.LazyFrame, path: Path) -> None:
    lazy_frame.sink_parquet(path, compression="zstd", engine="streaming")


def csv_scan(
    path: Path,
    schema_overrides: dict[str, pl.DataType],
) -> pl.LazyFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    return pl.scan_csv(
        path,
        schema_overrides=schema_overrides,
        null_values=["", "NA", "null", "NULL"],
        empty_as_null=True,
        try_parse_dates=False,
    )


def profile_ratings(lazy_frame: pl.LazyFrame) -> dict[str, Any]:
    profile = collect_one(
        lazy_frame.select(
            pl.len().alias("rows"),
            pl.col("userId").n_unique().alias("users"),
            pl.col("movieId").n_unique().alias("movies"),
            pl.col("rating").mean().alias("rating_mean"),
            pl.col("rating").std(ddof=1).alias("rating_std"),
            pl.col("rating").min().alias("rating_min"),
            pl.col("rating").max().alias("rating_max"),
            pl.col("timestamp").min().alias("timestamp_min"),
            pl.col("timestamp").max().alias("timestamp_max"),
        )
    ).row(0, named=True)
    return {key: (value.item() if isinstance(value, np.generic) else value) for key, value in profile.items()}


def clean_ratings(ratings_path: Path, work_dir: Path) -> tuple[Path, dict[str, int]]:
    LOGGER.info("Đọc ratings.csv và loại dòng thiếu/không hợp lệ...")
    raw = csv_scan(
        ratings_path,
        {
            "userId": pl.Int32,
            "movieId": pl.Int32,
            "rating": pl.Float32,
            "timestamp": pl.Int64,
        },
    )
    raw_rows = row_count(raw)
    valid = raw.filter(
        pl.col("userId").is_not_null()
        & pl.col("movieId").is_not_null()
        & pl.col("rating").is_not_null()
        & pl.col("timestamp").is_not_null()
        & (pl.col("userId") > 0)
        & (pl.col("movieId") > 0)
        & pl.col("rating").is_between(RATING_MIN, RATING_MAX)
    ).select("userId", "movieId", "rating", "timestamp")

    valid_stats = collect_one(
        valid.select(
            pl.len().alias("valid_rows"),
            pl.struct(["userId", "movieId"]).n_unique().alias("unique_user_movie_pairs"),
        )
    ).row(0, named=True)
    valid_rows = int(valid_stats["valid_rows"])
    unique_pairs = int(valid_stats["unique_user_movie_pairs"])
    duplicate_rows = valid_rows - unique_pairs
    invalid_rows = raw_rows - valid_rows
    LOGGER.info(
        "ratings.csv: %s dòng; loại %s dòng lỗi; %s cặp user–phim trùng.",
        f"{raw_rows:,}",
        f"{invalid_rows:,}",
        f"{duplicate_rows:,}",
    )

    if duplicate_rows:
        LOGGER.info("Giữ bản đánh giá mới nhất cho mỗi cặp user–phim...")
        valid = valid.sort("timestamp").unique(
            subset=["userId", "movieId"], keep="last", maintain_order=False
        )

    current_path = work_dir / "ratings_clean.parquet"
    write_parquet(valid, current_path)
    quality = {
        "source_rows": raw_rows,
        "invalid_or_missing_rows_removed": invalid_rows,
        "duplicate_user_movie_rows_removed": duplicate_rows,
        "valid_rows_before_k_core": valid_rows - duplicate_rows,
    }
    return current_path, quality


def filter_training_k_core(
    train_path: Path,
    work_dir: Path,
    output_path: Path,
    min_user_ratings: int,
    min_movie_ratings: int,
) -> dict[str, Any]:
    """Apply the k-core thresholds on train only, avoiding holdout leakage."""
    current_path = train_path
    current = pl.scan_parquet(current_path)
    previous_profile = profile_ratings(current)
    initial_train_rows = int(previous_profile["rows"])
    iterations: list[dict[str, int]] = []
    iteration = 0
    while True:
        iteration += 1
        eligible_users = (
            current.group_by("userId")
            .agg(pl.len().alias("interaction_count"))
            .filter(pl.col("interaction_count") >= min_user_ratings)
            .select("userId")
        )
        eligible_movies = (
            current.group_by("movieId")
            .agg(pl.len().alias("interaction_count"))
            .filter(pl.col("interaction_count") >= min_movie_ratings)
            .select("movieId")
        )
        next_path = work_dir / f"train_kcore_{iteration:02d}.parquet"
        filtered = current.join(eligible_users, on="userId", how="semi").join(
            eligible_movies, on="movieId", how="semi"
        )
        write_parquet(filtered, next_path)
        next_frame = pl.scan_parquet(next_path)
        next_profile = profile_ratings(next_frame)
        iterations.append(
            {
                "iteration": iteration,
                "rows": int(next_profile["rows"]),
                "users": int(next_profile["users"]),
                "movies": int(next_profile["movies"]),
            }
        )
        LOGGER.info(
            "K-core train vòng %d: %s đánh giá, %s users, %s phim.",
            iteration,
            f"{next_profile['rows']:,}",
            f"{next_profile['users']:,}",
            f"{next_profile['movies']:,}",
        )
        current_path = next_path
        if int(next_profile["rows"]) == 0:
            raise ValueError("K-core trên train đã loại toàn bộ rating; hãy giảm ngưỡng lọc.")
        if int(next_profile["rows"]) == int(previous_profile["rows"]):
            break
        previous_profile = next_profile
        current = next_frame

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(pl.scan_parquet(current_path), output_path)
    return {
        "min_user_ratings_in_train": min_user_ratings,
        "min_movie_ratings_in_train": min_movie_ratings,
        "iterations": iterations,
        "train_rows_removed": initial_train_rows - int(iterations[-1]["rows"]),
        "final_train_profile": profile_ratings(pl.scan_parquet(output_path)),
    }


def split_by_user_timeline(
    ratings_path: Path,
    work_dir: Path,
    output_dir: Path,
    train_ratio: float,
    validation_ratio: float,
    min_user_ratings: int,
    min_movie_ratings: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    LOGGER.info("Sắp xếp theo thời gian trong từng user và chia 70/15/15...")
    ratings = pl.scan_parquet(ratings_path).sort(["userId", "timestamp", "movieId"])
    indexed = ratings.with_columns(
        (pl.col("userId").cum_count().over("userId").cast(pl.Int64) - 1).alias("_user_position"),
        pl.len().over("userId").cast(pl.Int64).alias("_user_total"),
    ).with_columns(
        (pl.col("_user_total") * train_ratio).floor().cast(pl.Int64).alias("_train_end"),
        (pl.col("_user_total") * (train_ratio + validation_ratio))
        .floor()
        .cast(pl.Int64)
        .alias("_validation_end"),
    )
    labeled = indexed.with_columns(
        pl.when(pl.col("_user_position") < pl.col("_train_end"))
        .then(pl.lit("train"))
        .when(pl.col("_user_position") < pl.col("_validation_end"))
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
        .alias("split")
    ).drop("_user_position", "_user_total", "_train_end", "_validation_end")

    all_splits_path = work_dir / "ratings_with_split.parquet"
    write_parquet(labeled, all_splits_path)
    all_splits = pl.scan_parquet(all_splits_path)
    raw_paths = {
        "train": work_dir / "ratings_train_before_kcore.parquet",
        "validation": work_dir / "ratings_validation_all.parquet",
        "test": work_dir / "ratings_test_all.parquet",
    }
    for split_name, path in raw_paths.items():
        write_parquet(all_splits.filter(pl.col("split") == split_name), path)

    raw_split_counts = {name: row_count(pl.scan_parquet(path)) for name, path in raw_paths.items()}
    train_path = output_dir / "ratings_train.parquet"
    k_core_quality = filter_training_k_core(
        raw_paths["train"],
        work_dir,
        train_path,
        min_user_ratings,
        min_movie_ratings,
    )
    train_ids = pl.scan_parquet(train_path)
    train_users = train_ids.select("userId").unique()
    train_movies = train_ids.select("movieId").unique()
    filtered_counts: dict[str, Any] = {
        "raw_per_user_time_split_rows": raw_split_counts,
        "k_core_on_train_only": k_core_quality,
        "cold_start_validation_rows_removed": 0,
        "cold_start_test_rows_removed": 0,
    }
    final_paths = {"train": train_path}
    for split_name in ("validation", "test"):
        held_out = pl.scan_parquet(raw_paths[split_name])
        before = row_count(held_out)
        known_items_and_users = held_out.join(train_users, on="userId", how="semi").join(
            train_movies, on="movieId", how="semi"
        )
        after = row_count(known_items_and_users)
        filtered_counts[f"cold_start_{split_name}_rows_removed"] = before - after
        final_path = output_dir / f"ratings_{split_name}.parquet"
        write_parquet(known_items_and_users, final_path)
        final_paths[split_name] = final_path
        LOGGER.info(
            "%s: giữ %s/%s đánh giá; bỏ %s dòng có user/phim cold-start.",
            split_name,
            f"{after:,}",
            f"{before:,}",
            f"{before - after:,}",
        )
    return final_paths, filtered_counts


def add_rating_normalizations(split_paths: dict[str, Path]) -> tuple[pl.LazyFrame, Path]:
    train = pl.scan_parquet(split_paths["train"])
    user_means_path = split_paths["train"].parent / "user_train_means.parquet"
    write_parquet(
        train.group_by("userId").agg(pl.col("rating").mean().alias("user_mean_train")),
        user_means_path,
    )
    user_means = pl.scan_parquet(user_means_path)

    for split_name, path in split_paths.items():
        frame = pl.scan_parquet(path).join(user_means, on="userId", how="left")
        normalized = frame.with_columns(
            ((pl.col("rating") - RATING_MIN) / (RATING_MAX - RATING_MIN))
            .cast(pl.Float32)
            .alias("rating_norm_01"),
            (pl.col("rating") - pl.col("user_mean_train"))
            .cast(pl.Float32)
            .alias("rating_centered"),
        ).drop("user_mean_train")
        normalized_path = path.with_name(f"{path.stem}_normalized.parquet")
        write_parquet(normalized, normalized_path)
        normalized_path.replace(path)
        LOGGER.info("Đã chuẩn hóa rating cho split %s.", split_name)
    return pl.scan_parquet(split_paths["train"]), user_means_path


def build_id_maps(train: pl.LazyFrame, output_dir: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    user_map_path = output_dir / "user_id_map.parquet"
    movie_map_path = output_dir / "movie_id_map.parquet"
    user_map = train.select("userId").unique().sort("userId").with_row_index("user_idx")
    movie_map = train.select("movieId").unique().sort("movieId").with_row_index("movie_idx")
    user_map = user_map.with_columns(pl.col("user_idx").cast(pl.Int32))
    movie_map = movie_map.with_columns(pl.col("movie_idx").cast(pl.Int32))
    write_parquet(user_map, user_map_path)
    write_parquet(movie_map, movie_map_path)
    users = collect_one(pl.scan_parquet(user_map_path).sort("user_idx"))["userId"].to_numpy()
    movies = collect_one(pl.scan_parquet(movie_map_path).sort("movie_idx"))["movieId"].to_numpy()
    return user_map_path, movie_map_path, users, movies


def build_sparse_matrices(
    split_paths: dict[str, Path],
    output_dir: Path,
    users: np.ndarray,
    movies: np.ndarray,
) -> dict[str, dict[str, Any]]:
    user_lookup = np.full(int(users.max()) + 1, -1, dtype=np.int32)
    movie_lookup = np.full(int(movies.max()) + 1, -1, dtype=np.int32)
    user_lookup[users] = np.arange(len(users), dtype=np.int32)
    movie_lookup[movies] = np.arange(len(movies), dtype=np.int32)
    shape = (len(users), len(movies))
    matrix_profiles: dict[str, dict[str, Any]] = {}

    for split_name, source_path in split_paths.items():
        LOGGER.info("Tạo ma trận user–item %s...", split_name)
        frame = collect_one(
            pl.scan_parquet(source_path).select("userId", "movieId", "rating")
        )
        user_ids = frame["userId"].to_numpy()
        movie_ids = frame["movieId"].to_numpy()
        rows = user_lookup[user_ids]
        columns = movie_lookup[movie_ids]
        if np.any(rows < 0) or np.any(columns < 0):
            raise ValueError(f"Tìm thấy ID ngoài mapping train trong split {split_name}.")
        values = frame["rating"].to_numpy().astype(np.float32, copy=False)
        matrix = csr_matrix((values, (rows, columns)), shape=shape, dtype=np.float32)
        matrix.sort_indices()
        matrix_path = output_dir / f"user_item_{split_name}.npz"
        save_npz(matrix_path, matrix, compressed=True)
        matrix_profiles[split_name] = {
            "shape": [int(shape[0]), int(shape[1])],
            "nonzero_ratings": int(matrix.nnz),
            "density": float(matrix.nnz / (shape[0] * shape[1])),
            "file": matrix_path.name,
        }
        del frame, user_ids, movie_ids, rows, columns, values, matrix

    return matrix_profiles


def process_movie_metadata(
    dataset_dir: Path,
    output_dir: Path,
    movie_map_path: Path,
    like_threshold: float,
    skip_genome: bool,
) -> dict[str, str]:
    movie_ids = pl.scan_parquet(movie_map_path).select("movieId")
    movies = csv_scan(
        dataset_dir / "movies.csv",
        {"movieId": pl.Int32, "title": pl.String, "genres": pl.String},
    ).join(movie_ids, on="movieId", how="semi")
    movies_path = output_dir / "movies.parquet"
    write_parquet(movies.sort("movieId"), movies_path)
    genres_path = output_dir / "movie_genres.parquet"
    genres = (
        movies.select("movieId", pl.col("genres").str.split("|").alias("genre"))
        .explode("genre")
        .filter(pl.col("genre").is_not_null() & (pl.col("genre") != "(no genres listed)"))
    )
    write_parquet(genres, genres_path)

    train = pl.scan_parquet(output_dir / "ratings_train.parquet")
    transactions_path = output_dir / "train_like_transactions.parquet"
    transactions = (
        train.filter(pl.col("rating") >= like_threshold)
        .group_by("userId")
        .agg(pl.col("movieId").sort().alias("movieIds"))
        .filter(pl.col("movieIds").list.len() >= 2)
        .sort("userId")
    )
    write_parquet(transactions, transactions_path)

    outputs = {
        "movies": movies_path.name,
        "movie_genres": genres_path.name,
        "train_like_transactions": transactions_path.name,
    }
    tag_path = dataset_dir / "tags.csv"
    if tag_path.exists():
        tags = csv_scan(
            tag_path,
            {
                "userId": pl.Int32,
                "movieId": pl.Int32,
                "tag": pl.String,
                "timestamp": pl.Int64,
            },
        ).join(movie_ids, on="movieId", how="semi")
        tags = tags.filter(pl.col("tag").is_not_null() & (pl.col("tag").str.strip_chars() != ""))
        tags_output = output_dir / "movie_tags.parquet"
        write_parquet(tags, tags_output)
        outputs["movie_tags"] = tags_output.name

    links_path = dataset_dir / "links.csv"
    if links_path.exists():
        links = csv_scan(
            links_path,
            {"movieId": pl.Int32, "imdbId": pl.Int64, "tmdbId": pl.Int64},
        ).join(movie_ids, on="movieId", how="semi")
        links_output = output_dir / "movie_links.parquet"
        write_parquet(links, links_output)
        outputs["movie_links"] = links_output.name

    genome_tags_path = dataset_dir / "genome-tags.csv"
    if not skip_genome and genome_tags_path.exists():
        genome_tags = csv_scan(
            genome_tags_path,
            {"tagId": pl.Int32, "tag": pl.String},
        )
        genome_tags_output = output_dir / "genome_tags.parquet"
        write_parquet(genome_tags, genome_tags_output)
        outputs["genome_tags"] = genome_tags_output.name

        genome_scores_path = dataset_dir / "genome-scores.csv"
        genome_scores = csv_scan(
            genome_scores_path,
            {"movieId": pl.Int32, "tagId": pl.Int32, "relevance": pl.Float32},
        ).join(movie_ids, on="movieId", how="semi")
        genome_scores_output = output_dir / "genome_scores.parquet"
        write_parquet(genome_scores, genome_scores_output)
        outputs["genome_scores"] = genome_scores_output.name
    elif skip_genome:
        LOGGER.info("Bỏ qua genome-scores.csv theo --skip-genome.")

    return outputs


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    if args.min_user_ratings < 1 or args.min_movie_ratings < 1:
        raise SystemExit("Min user/movie ratings phải từ 1 trở lên.")
    if args.train_ratio <= 0 or args.validation_ratio <= 0 or args.train_ratio + args.validation_ratio >= 1:
        raise SystemExit("Tỷ lệ train/validation phải dương và tổng nhỏ hơn 1.")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {dataset_dir}")
    if output_dir == dataset_dir or dataset_dir in output_dir.parents:
        raise ValueError("Thư mục output không được nằm bên trong Dataset để bảo vệ dữ liệu gốc.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Thư mục output đã có dữ liệu, không ghi đè: {output_dir}. "
            "Hãy chọn --output-dir khác để giữ dữ liệu cũ an toàn."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="movielens25m-work-") as temp_name:
        work_dir = Path(temp_name)
        cleaned_path, quality = clean_ratings(dataset_dir / "ratings.csv", work_dir)
        split_paths, split_quality = split_by_user_timeline(
            cleaned_path,
            work_dir,
            output_dir,
            args.train_ratio,
            args.validation_ratio,
            args.min_user_ratings,
            args.min_movie_ratings,
        )
        train, user_means_path = add_rating_normalizations(split_paths)
        user_map_path, movie_map_path, users, movies = build_id_maps(train, output_dir)
        matrix_profiles = build_sparse_matrices(split_paths, output_dir, users, movies)
        metadata_outputs = process_movie_metadata(
            dataset_dir,
            output_dir,
            movie_map_path,
            args.like_threshold,
            args.skip_genome,
        )

    split_profiles = {
        split_name: profile_ratings(pl.scan_parquet(path))
        for split_name, path in split_paths.items()
    }
    rating_distribution = (
        collect_one(
            pl.scan_parquet(split_paths["train"])
            .group_by("rating")
            .agg(pl.len().alias("count"))
            .sort("rating")
        )
        .to_dicts()
    )
    report = {
        "dataset": "MovieLens 25M (ml-25m)",
        "source_directory": str(dataset_dir),
        "split_strategy": "Per-user chronological split; ties ordered by movieId.",
        "split_ratios_requested": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": 1.0 - args.train_ratio - args.validation_ratio,
        },
        "cold_start_policy": (
            "Split each user timeline first; apply iterative min-user/min-movie k-core to train only; "
            "remove validation/test rows for users or movies absent from train."
        ),
        "rating_normalization": {
            "rating_norm_01": "(rating - 0.5) / 4.5",
            "rating_centered": "rating minus the user's train-only mean; no validation/test leakage.",
        },
        "quality": quality,
        "split_quality": split_quality,
        "splits": split_profiles,
        "train_rating_distribution": rating_distribution,
        "matrix": matrix_profiles,
        "id_maps": {
            "users": {"rows": int(len(users)), "file": user_map_path.name},
            "movies": {"rows": int(len(movies)), "file": movie_map_path.name},
            "training_user_means": {"file": user_means_path.name},
        },
        "features": metadata_outputs,
        "like_threshold_for_apriori_transactions": args.like_threshold,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    report_path = output_dir / "processing_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Hoàn tất. Báo cáo: %s", report_path)
    for split_name, profile in split_profiles.items():
        LOGGER.info(
            "%s: %s rows, %s users, %s movies, mean rating %.4f.",
            split_name,
            f"{profile['rows']:,}",
            f"{profile['users']:,}",
            f"{profile['movies']:,}",
            profile["rating_mean"],
        )
    LOGGER.info("Ma trận train có shape %s và %s ratings.", matrix_profiles["train"]["shape"], f"{matrix_profiles['train']['nonzero_ratings']:,}")


if __name__ == "__main__":
    main()
