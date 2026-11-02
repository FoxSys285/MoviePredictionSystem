import pandas as pd
import pytest

from movie_rating.settings import Settings


@pytest.fixture
def ratings():
    rows = []
    for movie in range(1, 13):
        for user in range(1, 11):
            rows.append((user, movie, 0.5 + ((user + movie) % 10) * 0.5, 1_000_000 + len(rows)))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


@pytest.fixture
def configured(tmp_path, ratings):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    ratings.to_csv(dataset / "ratings.csv", index=False)
    pd.DataFrame(
        {
            "movieId": range(1, 13),
            "title": [f"Film {i}" for i in range(1, 13)],
            "genres": ["Drama"] * 12,
        }
    ).to_csv(dataset / "movies.csv", index=False)
    return Settings.model_validate(
        {
            "data": {"directory": str(dataset), "user_sample_fraction": 1, "chunksize": 17},
            "experiment": {"artifacts_directory": str(tmp_path / "artifacts")},
            "models": {
                "svd": {"epochs": 2, "factors": 3},
                "nmf": {"epochs": 2, "factors": 3},
                "ncf": {"epochs": 2, "factors": 3},
                "xgboost": {"n_estimators": 20, "n_jobs": 1},
            },
        }
    )
