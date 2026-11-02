import pandas as pd
import pytest
from fastapi.testclient import TestClient

from movie_rating.api.app import create_app
from movie_rating.data.prepare import prepare
from movie_rating.services.catalog import CatalogService


def test_catalog_uses_all_ratings_without_preparation(configured, ratings):
    configured.data.user_sample_fraction = 0.02
    with TestClient(create_app(configured)) as client:
        assert client.get("/api/v1/health").json()["models_ready"] is False
        summary = client.get("/api/v1/dataset").json()
        assert summary["rating_count"] == len(ratings)
        assert summary["user_count"] == ratings.userId.nunique()
        assert summary["mean_rating"] == pytest.approx(ratings.rating.mean())
        assert sum(bucket["count"] for bucket in summary["rating_distribution"]) == len(ratings)
        assert "fingerprint" not in summary
        catalog = client.get("/api/v1/movies", params={"limit": 3}).json()
        assert catalog["total"] == 12 and len(catalog["items"]) == 3
        detail = client.get("/api/v1/movies/1").json()
        expected = ratings.loc[ratings.movieId == 1, "rating"]
        assert detail["rating_count"] == len(expected)
        assert detail["mean_rating"] == pytest.approx(expected.mean())
        assert sum(bucket["count"] for bucket in detail["rating_distribution"]) == len(expected)
        assert detail["tags"] == [] and detail["imdb_url"] is None
        assert detail["year"] is None
        assert all(item["movieId"] != 1 for item in detail["related"])
        assert client.get("/api/v1/movies/99999").status_code == 404


def test_tags_links_unrated_films_and_filters(configured, ratings):
    movies = pd.read_csv(configured.dataset_dir / "movies.csv")
    movies.loc[0, ["title", "genres"]] = ["Test Film (1995)", "Drama|Action"]
    movies.loc[len(movies)] = [999, "Unrated Film (2019)", "(no genres listed)"]
    movies.to_csv(configured.dataset_dir / "movies.csv", index=False)
    pd.DataFrame({"movieId": [1], "imdbId": ["0114709"], "tmdbId": [862]}).to_csv(
        configured.dataset_dir / "links.csv", index=False
    )
    pd.DataFrame({"movieId": [1, 1, 1], "tag": ["Classic", "classic", None]}).to_csv(
        configured.dataset_dir / "tags.csv", index=False
    )
    with TestClient(create_app(configured)) as client:
        detail = client.get("/api/v1/movies/1").json()
        assert detail["imdb_url"] == "https://www.imdb.com/title/tt0114709/"
        assert detail["tmdb_url"] == "https://www.themoviedb.org/movie/862"
        assert detail["tags"] == [{"tag": "classic", "count": 2}]
        assert detail["year"] == 1995
        unrated = client.get("/api/v1/movies/999").json()
        assert unrated["mean_rating"] is None and unrated["rating_count"] == 0
        assert all(bucket["count"] == 0 for bucket in unrated["rating_distribution"])
        result = client.get("/api/v1/movies", params={"genre": "Action", "year": "classic"}).json()
        assert result["total"] == 1 and result["items"][0]["movieId"] == 1
        assert client.get("/api/v1/movies", params={"q": "  Test Film  "}).json()["total"] == 1
        result = client.get("/api/v1/movies", params={"min_rating": 3}).json()
        assert all(item["mean_rating"] >= 3 for item in result["items"])
        summary = client.get("/api/v1/dataset").json()
        assert summary["quality"]["empty_tags"] == 1
        assert summary["unrated_movie_count"] == 1


def test_catalog_cache_reuse_and_invalidation(configured, ratings, monkeypatch):
    first = CatalogService(configured).load()
    original = CatalogService._build
    with monkeypatch.context() as context:
        def unexpected_build(*args):
            pytest.fail("Unchanged CSV files should reuse the catalogue cache")
        context.setattr(CatalogService, "_build", unexpected_build)
        second = CatalogService(configured).load()
        assert first.summary == second.summary
    ratings.iloc[:-1].to_csv(configured.dataset_dir / "ratings.csv", index=False)
    assert CatalogService._build is original
    refreshed = second.load()
    assert refreshed.summary["rating_count"] == len(ratings) - 1


def test_public_statistics_do_not_change_training_statistics(configured, ratings):
    prepare(configured)
    path = configured.artifacts_dir / "data/movie_stats.parquet"
    before = path.read_bytes()
    CatalogService(configured).load()
    assert path.read_bytes() == before
    train_stats = pd.read_parquet(path)
    assert train_stats.rating_count.sum() < len(ratings)
