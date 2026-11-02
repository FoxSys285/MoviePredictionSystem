"""Create a reproducible, schema-compatible cleaned copy of MovieLens CSV files."""

import gc
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

SCHEMAS = {
    "movies.csv": ["movieId", "title", "genres"],
    "ratings.csv": ["userId", "movieId", "rating", "timestamp"],
    "tags.csv": ["userId", "movieId", "tag", "timestamp"],
    "links.csv": ["movieId", "imdbId", "tmdbId"],
}
MAX_ID = 2**31 - 1


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_values(series: pd.Series) -> pd.Series:
    return series.str.normalize("NFC").str.replace(r"\s+", " ", regex=True).str.strip()


def integer_values(series: pd.Series, minimum: int, maximum: int):
    values = pd.to_numeric(series, errors="coerce")
    valid = (
        values.notna() & values.between(minimum, maximum) & values.mod(1).eq(0)
    ).fillna(False)
    return values, valid


def normalized_genres(value: str) -> str:
    parts = list(dict.fromkeys(part.strip() for part in value.split("|") if part.strip()))
    if len(parts) > 1:
        parts = [part for part in parts if part != "(no genres listed)"]
    return "|".join(parts) or "(no genres listed)"


def latest_records(frame: pd.DataFrame, keys: list[str]):
    """Resolve conflicts by latest timestamp, then last original row on ties."""
    repeated = frame.duplicated(keys, keep=False)
    if not repeated.any():
        return frame, 0
    winners = (
        frame.loc[repeated].sort_values("timestamp", kind="stable")
        .drop_duplicates(keys, keep="last").index
    )
    losers = frame.index[repeated].difference(winners)
    return frame.drop(index=losers), len(losers)


def base_stats():
    return {
        "input_rows": 0, "output_rows": 0,
        "removed_invalid_rows": 0, "removed_orphan_rows": 0,
        "removed_duplicate_rows": 0, "normalized_text_rows": 0,
    }


def clean_dataset(source: Path, destination: Path, chunksize: int = 500_000) -> dict:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate, non-nested directories.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing directory: {destination}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    sources = {}
    for name, columns in SCHEMAS.items():
        path = source / name
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        if list(header.columns) != columns:
            raise ValueError(f"Unexpected columns in {name}; expected {columns}.")
        sources[name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}

    staging = destination.parent / f".{destination.name}-{uuid4().hex}.staging"
    staging.mkdir(parents=True)
    report = {
        "status": "complete", "created_at": datetime.now(UTC).isoformat(),
        "source_directory": str(source), "output_directory": str(destination),
        "source_files": sources, "files": {},
        "rules": [
            "Preserve the four original CSV schemas and all valid records; no sampling.",
            "Normalize text to Unicode NFC; trim and collapse whitespace.",
            "Keep title spelling, punctuation and case; never invent a missing year.",
            "Normalize genre separators; preserve unknown genres as (no genres listed).",
            "Require positive 32-bit integer IDs, valid Unix timestamps and ratings 0.5..5 in steps of 0.5.",
            "Remove invalid required values and references to movie IDs absent from cleaned movies.csv.",
            "Deduplicate movie IDs by first valid row.",
            "Deduplicate user/movie ratings by latest timestamp; last row wins on timestamp ties.",
            "Casefold tags; remove empty/placeholder labels (na, n/a, null, none, nan).",
            "Deduplicate user/movie/normalized-tag by latest timestamp.",
            "Preserve identical tags assigned by different users.",
            "Normalize IMDb IDs to at least seven digits, retaining leading zeroes.",
            "Keep optional missing external IDs empty; clear malformed optional IDs.",
            "Deduplicate link rows by most available valid external IDs, then first original row.",
            "Retain unrated movies; do not impute ratings or remove statistical outliers.",
        ],
    }

    def read(name):
        return pd.read_csv(source / name, dtype="string", keep_default_na=False,
                           encoding="utf-8-sig")

    def write(name, frame, stats):
        stats["output_rows"] = len(frame)
        assert stats["input_rows"] == (
            stats["output_rows"] + stats["removed_invalid_rows"]
            + stats["removed_orphan_rows"] + stats["removed_duplicate_rows"]
        ), f"Row accounting failed for {name}"
        path = staging / name
        frame[SCHEMAS[name]].to_csv(
            path, index=False, encoding="utf-8", lineterminator="\n",
            chunksize=100_000, float_format="%.1f",
        )
        stats.update({"bytes": path.stat().st_size, "sha256": file_hash(path)})
        report["files"][name] = stats
        print(f"{name}: {stats['input_rows']:,} -> {len(frame):,} rows", flush=True)

    movies = read("movies.csv")
    stats = base_stats()
    stats["input_rows"] = len(movies)
    titles = text_values(movies.title)
    genres = text_values(movies.genres).map(normalized_genres)
    stats["normalized_text_rows"] = int(((titles != movies.title) | (genres != movies.genres)).sum())
    ids, valid_ids = integer_values(movies.movieId, 1, MAX_ID)
    valid = valid_ids & titles.ne("")
    stats["removed_invalid_rows"] = int((~valid).sum())
    movies = movies.loc[valid].copy()
    movies["movieId"] = ids.loc[valid].astype("int32")
    movies["title"], movies["genres"] = titles.loc[valid], genres.loc[valid]
    repeated = movies.duplicated("movieId", keep="first")
    stats["removed_duplicate_rows"] = int(repeated.sum())
    movies = movies.loc[~repeated]
    if movies.empty:
        raise ValueError("No valid movies remain; original data was not modified.")
    movie_ids = movies.movieId
    stats["movies_without_year"] = int((~movies.title.str.contains(r"\(\d{4}\)$")).sum())
    stats["movies_without_genres"] = int(movies.genres.eq("(no genres listed)").sum())
    write("movies.csv", movies, stats)
    del movies, titles, genres, ids
    gc.collect()

    now = int(datetime.now(UTC).timestamp())
    stats = base_stats()
    frames = []
    for chunk in pd.read_csv(
        source / "ratings.csv", dtype="string", keep_default_na=False,
        encoding="utf-8-sig", chunksize=chunksize,
    ):
        stats["input_rows"] += len(chunk)
        users, valid_users = integer_values(chunk.userId, 1, MAX_ID)
        ids, valid_ids = integer_values(chunk.movieId, 1, MAX_ID)
        times, valid_times = integer_values(chunk.timestamp, 0, now)
        scores = pd.to_numeric(chunk.rating, errors="coerce")
        valid_scores = scores.isin([step / 2 for step in range(1, 11)])
        valid = valid_users & valid_ids & valid_times & valid_scores
        stats["removed_invalid_rows"] += int((~valid).sum())
        orphan = valid & ~ids.isin(movie_ids)
        stats["removed_orphan_rows"] += int(orphan.sum())
        keep = valid & ~orphan
        frames.append(pd.DataFrame({
            "userId": users.loc[keep].astype("int32"),
            "movieId": ids.loc[keep].astype("int32"),
            "rating": scores.loc[keep].astype("float32"),
            "timestamp": times.loc[keep].astype("int64"),
        }))
        print(f"ratings.csv: {stats['input_rows']:,} rows checked", flush=True)
    if not frames:
        raise ValueError("ratings.csv is empty.")
    ratings = pd.concat(frames, ignore_index=True)
    del frames, chunk, users, ids, times, scores
    gc.collect()
    if ratings.empty:
        raise ValueError("No valid ratings remain; original data was not modified.")
    ratings, removed = latest_records(ratings, ["userId", "movieId"])
    stats["removed_duplicate_rows"] = removed
    stats["unique_users"] = int(ratings.userId.nunique())
    stats["rated_movies"] = int(ratings.movieId.nunique())
    stats["mean_rating"] = float(ratings.rating.astype("float64").mean())
    stats["rating_distribution"] = {
        str(float(score)): int(count) for score, count in ratings.rating.value_counts().sort_index().items()
    }
    write("ratings.csv", ratings, stats)
    del ratings
    gc.collect()

    tags = read("tags.csv")
    stats = base_stats()
    stats["input_rows"] = len(tags)
    labels = text_values(tags.tag).str.casefold()
    stats["normalized_text_rows"] = int(labels.ne(tags.tag).sum())
    stats["empty_tag_rows"] = int(labels.eq("").sum())
    placeholders = labels.isin(["na", "n/a", "null", "none", "nan"])
    stats["placeholder_tag_rows"] = int(placeholders.sum())
    users, valid_users = integer_values(tags.userId, 1, MAX_ID)
    ids, valid_ids = integer_values(tags.movieId, 1, MAX_ID)
    times, valid_times = integer_values(tags.timestamp, 0, now)
    valid = valid_users & valid_ids & valid_times & labels.ne("") & ~placeholders
    stats["removed_invalid_rows"] = int((~valid).sum())
    orphan = valid & ~ids.isin(movie_ids)
    stats["removed_orphan_rows"] = int(orphan.sum())
    keep = valid & ~orphan
    tags = pd.DataFrame({
        "userId": users.loc[keep].astype("int32"),
        "movieId": ids.loc[keep].astype("int32"),
        "tag": labels.loc[keep],
        "timestamp": times.loc[keep].astype("int64"),
    })
    tags, removed = latest_records(tags, ["userId", "movieId", "tag"])
    stats["removed_duplicate_rows"] = removed
    write("tags.csv", tags, stats)
    del tags, labels, users, ids, times
    gc.collect()

    links = read("links.csv")
    stats = base_stats()
    stats["input_rows"] = len(links)
    ids, valid = integer_values(links.movieId, 1, MAX_ID)
    stats["removed_invalid_rows"] = int((~valid).sum())
    orphan = valid & ~ids.isin(movie_ids)
    stats["removed_orphan_rows"] = int(orphan.sum())
    keep = valid & ~orphan
    links = links.loc[keep].copy()
    links["movieId"] = ids.loc[keep].astype("int32")
    changed = pd.Series(False, index=links.index)
    for column in ("imdbId", "tmdbId"):
        original = links[column].copy()
        external, present = integer_values(original.str.strip(), 1, 2**53 - 1)
        normalized = pd.Series("", index=links.index, dtype="string")
        normalized.loc[present] = external.loc[present].astype("int64").astype("string")
        if column == "imdbId":
            normalized.loc[present] = normalized.loc[present].str.zfill(7)
        stats[f"missing_{column}"] = int(original.str.strip().eq("").sum())
        stats[f"cleared_invalid_{column}"] = int((~present & original.str.strip().ne("")).sum())
        changed |= normalized.ne(original)
        links[column] = normalized
    stats["normalized_text_rows"] = int(changed.sum())
    links["_available"] = links.imdbId.ne("").astype(int) + links.tmdbId.ne("").astype(int)
    chosen = links.sort_values("_available", ascending=False, kind="stable").drop_duplicates("movieId")
    stats["removed_duplicate_rows"] = len(links) - len(chosen)
    links = chosen.sort_index().drop(columns="_available")
    write("links.csv", links, stats)

    for name, initial in sources.items():
        if file_hash(source / name) != initial["sha256"]:
            raise RuntimeError(f"Source {name} changed during cleaning. Staging retained at {staging}")
    report["original_files_unchanged"] = True
    report["total_removed_rows"] = sum(
        item["input_rows"] - item["output_rows"] for item in report["files"].values()
    )
    (staging / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = "\n".join(
        f"| {name} | {item['input_rows']:,} | {item['output_rows']:,} | "
        f"{item['removed_invalid_rows']:,} | {item['removed_orphan_rows']:,} | "
        f"{item['removed_duplicate_rows']:,} |"
        for name, item in report["files"].items()
    )
    readme = f"""# Dataset MovieLens đã làm sạch

Nguồn: `dataset/`. Giữ nguyên bốn schema CSV và mã định danh, không lấy mẫu.
Dữ liệu gốc đã được kiểm tra SHA-256 trước và sau khi chạy, không thay đổi.

| Tệp | Dòng gốc | Dòng sạch | Thiếu/sai bắt buộc | Phim không tồn tại | Trùng |
|---|---:|---:|---:|---:|---:|
{rows}

Chuẩn hóa Unicode NFC và khoảng trắng; giữ nguyên cách viết tên phim. Thể loại
chưa biết giữ `(no genres listed)`; không tự điền năm, điểm hoặc nội dung phim.
Nhãn được chuyển chữ thường bằng casefold; bỏ nhãn rỗng hoặc chỉ chứa các ký
hiệu thiếu dữ liệu `NA`, `N/A`, `null`, `none`, `nan` (không phân biệt hoa/thường).
Với cùng người dùng và phim, điểm
trùng giữ bản có timestamp mới nhất; với nhãn trùng sau chuẩn hóa cũng giữ lần
gắn nhãn mới nhất. Nhãn giống nhau của những người dùng khác nhau được giữ lại.
Khi timestamp bằng nhau, giữ dòng xuất hiện sau trong CSV gốc.

ID người dùng/phim là số nguyên dương; điểm hợp lệ 0,5–5, bước 0,5; timestamp
giữ dạng Unix giây UTC, không âm và không sau thời điểm làm sạch. Không xóa
đánh giá thấp/cao hợp lệ, phim ít đánh giá hoặc phim chưa có đánh giá.

IMDb ID giữ số 0 ở đầu. Mã IMDb/TMDB còn thiếu để trống, không điền 0 hoặc mã
phỏng đoán; mã ngoại hỏng được xóa nội dung. `links.csv` trùng movieId ưu tiên
dòng có nhiều mã ngoại hợp lệ hơn, sau đó dòng xuất hiện trước.

Chi tiết số lượng, quy tắc và SHA-256: `cleaning_report.json`.
Tạo lại từ thư mục project (đích mới phải chưa tồn tại):

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 scripts/clean_dataset.py --output ../dataset_cleaned_new
```

Để dùng dữ liệu sạch cho một lần chạy ứng dụng/học máy:

```powershell
$env:MOVIE_DATASET_DIR = '../dataset_cleaned'
```

Website mặc định vẫn dùng cấu hình hiện có. Nếu huấn luyện lại, dùng một thư mục
artifacts riêng cho dữ liệu sạch; không ghi đè các tập train/validation/test cũ.
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")
    staging.rename(destination)
    print(f"Completed. Total removed rows: {report['total_removed_rows']:,}", flush=True)
    return report
