"""Public catalogue statistics from all CSV rows, separate from training data."""

import json
from datetime import UTC, datetime
from threading import Lock

import pandas as pd

from movie_rating.data.prepare import RATING_DTYPES
from movie_rating.settings import Settings

MOVIE_COLUMNS = [
    "movieId", "title", "genres", "year", "mean_rating", "rating_count", "imdbId", "tmdbId"
]
RATING_COLUMNS = [f"rating_{step}" for step in range(1, 11)]


def movie_records(frame: pd.DataFrame) -> list[dict]:
    records = frame[MOVIE_COLUMNS].astype(object)
    return records.where(pd.notnull(records), None).to_dict(orient="records")


class CatalogService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = Lock()
        self.fingerprint = None
        self.movies = pd.DataFrame()
        self.tags = pd.DataFrame(columns=["movieId", "tag", "count"])
        self.summary = {}

    def load(self):
        directory = self.settings.dataset_dir
        for name in ("movies.csv", "ratings.csv"):
            if not (directory / name).is_file():
                raise FileNotFoundError(
                    "Chưa tìm thấy movies.csv và ratings.csv trong thư mục dataset."
                )
        fingerprint = {
            "version": 1,
            "directory": str(directory),
            "files": {
                name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                for name in ("movies.csv", "ratings.csv", "tags.csv", "links.csv")
                if (path := directory / name).is_file()
            },
        }
        with self.lock:
            if fingerprint == self.fingerprint:
                return self
            cache = self.settings.artifacts_dir / "catalog"
            manifest = cache / "summary.json"
            if manifest.is_file():
                try:
                    summary = json.loads(manifest.read_text(encoding="utf-8"))
                    if summary.get("fingerprint") == fingerprint:
                        movies = pd.read_parquet(cache / "movies.parquet")
                        tags = pd.read_parquet(cache / "tags.parquet")
                        self.movies, self.tags, self.summary = movies, tags, summary
                        self.fingerprint = fingerprint
                        return self
                except (OSError, ValueError):
                    pass
            self._build(fingerprint)
            cache.mkdir(parents=True, exist_ok=True)
            for name, frame in (("movies", self.movies), ("tags", self.tags)):
                temporary = cache / f"{name}.tmp.parquet"
                frame.to_parquet(temporary, index=False)
                temporary.replace(cache / f"{name}.parquet")
            temporary = cache / "summary.tmp.json"
            temporary.write_text(
                json.dumps(self.summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(manifest)
            self.fingerprint = fingerprint
            return self

    def _build(self, fingerprint):
        directory = self.settings.dataset_dir
        movies = pd.read_csv(directory / "movies.csv", dtype={"movieId": "int32"})
        if movies[["movieId", "title", "genres"]].isna().any().any():
            raise ValueError("movies.csv có giá trị thiếu trong thông tin phim.")
        if movies.movieId.duplicated().any():
            raise ValueError("movies.csv có mã phim trùng lặp.")
        movies["year"] = pd.to_numeric(
            movies.title.str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce"
        ).astype("Int64")
        users = set()
        parts = []
        distributions = []
        minimum, maximum = None, None
        total, rating_sum, orphan_count = 0, 0.0, 0
        for chunk in pd.read_csv(
            directory / "ratings.csv", dtype=RATING_DTYPES,
            chunksize=self.settings.data.chunksize,
        ):
            if chunk.isna().any().any() or not chunk.rating.isin(
                [step / 2 for step in range(1, 11)]
            ).all():
                raise ValueError("ratings.csv có điểm thiếu hoặc nằm ngoài thang 0,5–5.")
            if ((chunk.userId <= 0) | (chunk.movieId <= 0) | (chunk.timestamp < 0)).any():
                raise ValueError("ratings.csv có ID hoặc thời gian không hợp lệ.")
            users.update(chunk.userId.unique().tolist())
            total += len(chunk)
            rating_sum += float(chunk.rating.astype("float64").sum())
            orphan_count += int((~chunk.movieId.isin(movies.movieId)).sum())
            lo, hi = int(chunk.timestamp.min()), int(chunk.timestamp.max())
            minimum = lo if minimum is None else min(minimum, lo)
            maximum = hi if maximum is None else max(maximum, hi)
            parts.append(chunk.groupby("movieId").rating.agg(rating_sum="sum", rating_count="count"))
            histogram = chunk.groupby(["movieId", "rating"]).size().unstack(fill_value=0)
            histogram.columns = [f"rating_{int(value * 2)}" for value in histogram.columns]
            distributions.append(histogram)
        if not total:
            raise ValueError("ratings.csv chưa có đánh giá.")
        stats = pd.concat(parts).groupby(level=0).sum()
        stats["mean_rating"] = stats.rating_sum / stats.rating_count
        histogram = pd.concat(distributions).groupby(level=0).sum()
        histogram = histogram.reindex(columns=RATING_COLUMNS, fill_value=0).fillna(0).astype("int64")
        movies = movies.merge(stats[["mean_rating", "rating_count"]], on="movieId", how="left")
        movies = movies.merge(histogram, on="movieId", how="left")
        for column in ["rating_count", *RATING_COLUMNS]:
            movies[column] = movies[column].fillna(0).astype("int64")
        links_path = directory / "links.csv"
        links_count = 0
        if links_path.is_file():
            links = pd.read_csv(links_path, dtype={"imdbId": "string", "tmdbId": "Int64"})
            links_count = len(links)
            if links.movieId.duplicated().any():
                raise ValueError("links.csv có mã phim trùng lặp.")
            movies = movies.merge(links, on="movieId", how="left")
        else:
            movies["imdbId"], movies["tmdbId"] = None, None
        tag_count, missing_tags = 0, 0
        tag_parts = []
        tags_path = directory / "tags.csv"
        if tags_path.is_file():
            for chunk in pd.read_csv(
                tags_path, usecols=["movieId", "tag"], chunksize=self.settings.data.chunksize,
                dtype={"movieId": "int32", "tag": "string"},
            ):
                tag_count += len(chunk)
                chunk["tag"] = chunk.tag.str.strip().str.lower().replace("", pd.NA)
                missing_tags += int(chunk.tag.isna().sum())
                tag_parts.append(chunk.dropna().groupby(["movieId", "tag"]).size())
        self.tags = pd.DataFrame(columns=["movieId", "tag", "count"])
        if tag_parts:
            tags = pd.concat(tag_parts).groupby(level=[0, 1]).sum().rename("count").reset_index()
            self.tags = tags.sort_values(
                ["count", "tag"], ascending=[False, True]
            ).groupby("movieId", sort=False).head(12).reset_index(drop=True)
        genres = movies.genres.str.split("|").explode()
        genre_counts = genres[genres != "(no genres listed)"].value_counts()
        years = movies.year.dropna()
        distribution = histogram.sum()
        rated_movies = int((movies.rating_count > 0).sum())
        self.movies = movies
        self.summary = {
            "name": "MovieLens",
            "fingerprint": fingerprint,
            "analyzed_at": datetime.now(UTC).isoformat(),
            "movie_count": len(movies),
            "rated_movie_count": rated_movies,
            "unrated_movie_count": len(movies) - rated_movies,
            "rating_count": total,
            "user_count": len(users),
            "mean_rating": rating_sum / total,
            "rating_scale": {"min": 0.5, "max": 5, "step": 0.5},
            "rating_source": "all_ratings",
            "tag_count": tag_count,
            "link_count": links_count,
            "genre_count": len(genre_counts),
            "genres": [{"name": name, "movie_count": int(count)}
                       for name, count in genre_counts.items()],
            "year_min": int(years.min()) if len(years) else None,
            "year_max": int(years.max()) if len(years) else None,
            "rating_start": datetime.fromtimestamp(minimum, UTC).isoformat(),
            "rating_end": datetime.fromtimestamp(maximum, UTC).isoformat(),
            "matrix_density": (total - orphan_count) / (len(users) * len(movies)),
            "rating_distribution": [{"rating": step / 2, "count": int(distribution[f"rating_{step}"])}
                                    for step in range(1, 11)],
            "quality": {
                "movies_without_year": int(movies.year.isna().sum()),
                "movies_without_genres": int((movies.genres == "(no genres listed)").sum()),
                "ratings_without_movie": orphan_count,
                "empty_tags": missing_tags,
            },
            "popular_movies": movie_records(movies.sort_values(
                ["rating_count", "movieId"], ascending=[False, True]
            ).head(6)),
        }
