import json
from pathlib import Path

import pandas as pd
import pytest

from movie_rating.data.clean import SCHEMAS, clean_dataset, file_hash


def write_csv(directory: Path, name: str, rows: list):
    pd.DataFrame(rows, columns=SCHEMAS[name]).to_csv(directory / name, index=False)


@pytest.fixture
def dirty_dataset(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    write_csv(source, "movies.csv", [
        [1, "  Toy  Story (1995)  ", " Animation | Children || Animation "],
        [1, "Conflicting duplicate title", "Comedy"],
        [2, "A Film Without A Year", ""],
        [3, " ", "Drama"],
        [0, "Invalid ID", "Drama"],
    ])
    write_csv(source, "ratings.csv", [
        [1, 1, 3, 20], [1, 1, 4, 10], [1, 1, 4.5, 20],
        [2, 2, 5, 30], [3, 1, 6, 40], [4, 999, 3, 40],
        ["bad", 1, 3, 40], [5, 1, "", 40], [6, 1, 3, -1],
        [7, 1, 3.2, 40], [8, 1, 3, 99_999_999_999],
    ])
    write_csv(source, "tags.csv", [
        [1, 1, "  Sci-Fi  ", 10], [1, 1, "sci-fi", 20],
        [2, 1, "SCI-FI", 30], [1, 1, "\u00a0", 40],
        [1, 1, "N/A", 40], [1, 999, "orphan", 40], [0, 1, "invalid", 40],
    ])
    write_csv(source, "links.csv", [
        [1, "0114709", "862"], [1, "", ""], [2, "bad", ""],
        [999, "123", "123"], [0, "456", "456"],
    ])
    return source


def test_cleaning_resolves_cross_chunk_conflicts_and_preserves_valid_records(dirty_dataset, tmp_path):
    destination = tmp_path / "clean"
    original = {name: file_hash(dirty_dataset / name) for name in SCHEMAS}
    report = clean_dataset(dirty_dataset, destination, chunksize=2)
    movies = pd.read_csv(destination / "movies.csv", keep_default_na=False)
    assert movies.movieId.tolist() == [1, 2]
    assert movies.title.tolist() == ["Toy Story (1995)", "A Film Without A Year"]
    assert movies.genres.tolist() == ["Animation|Children", "(no genres listed)"]
    ratings = pd.read_csv(destination / "ratings.csv")
    assert ratings.to_dict("records") == [
        {"userId": 1, "movieId": 1, "rating": 4.5, "timestamp": 20},
        {"userId": 2, "movieId": 2, "rating": 5.0, "timestamp": 30},
    ]
    tags = pd.read_csv(destination / "tags.csv")
    assert tags.to_dict("records") == [
        {"userId": 1, "movieId": 1, "tag": "sci-fi", "timestamp": 20},
        {"userId": 2, "movieId": 1, "tag": "sci-fi", "timestamp": 30},
    ]
    links = pd.read_csv(destination / "links.csv", dtype="string", keep_default_na=False)
    assert links.to_dict("records") == [
        {"movieId": "1", "imdbId": "0114709", "tmdbId": "862"},
        {"movieId": "2", "imdbId": "", "tmdbId": ""},
    ]
    for name in SCHEMAS:
        assert list(pd.read_csv(destination / name, nrows=0).columns) == SCHEMAS[name]
        assert file_hash(dirty_dataset / name) == original[name]
        assert file_hash(destination / name) == report["files"][name]["sha256"]
        stats = report["files"][name]
        assert stats["input_rows"] == sum(stats[key] for key in (
            "output_rows", "removed_invalid_rows", "removed_orphan_rows", "removed_duplicate_rows"
        ))
    assert report["files"]["ratings.csv"]["removed_duplicate_rows"] == 2
    assert report["files"]["tags.csv"]["empty_tag_rows"] == 1
    assert report["files"]["tags.csv"]["placeholder_tag_rows"] == 1
    assert report["original_files_unchanged"] is True
    assert json.loads((destination / "cleaning_report.json").read_text(encoding="utf-8")) == report


def test_cleaning_refuses_to_overwrite_or_nest_sources(dirty_dataset, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        clean_dataset(dirty_dataset, existing)
    assert marker.read_text() == "preserve"
    with pytest.raises(ValueError):
        clean_dataset(dirty_dataset, dirty_dataset / "clean")
    with pytest.raises(ValueError):
        clean_dataset(dirty_dataset, dirty_dataset)


def test_cleaning_checks_schema_before_creating_destination(dirty_dataset, tmp_path):
    (dirty_dataset / "ratings.csv").write_text("wrong,header\n1,2\n", encoding="utf-8")
    destination = tmp_path / "clean"
    with pytest.raises(ValueError, match="Unexpected columns"):
        clean_dataset(dirty_dataset, destination)
    assert not destination.exists()
