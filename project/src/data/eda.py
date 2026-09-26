"""Generate descriptive statistics and plots for MovieLens 25M.

Run after prepare: ``python -m src.data.eda``. These descriptive plots are not
used to fit features or choose model hyperparameters.
"""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.common import connection, load_config, parquet_sql, project_path, read_json, write_json


def describe(values: np.ndarray) -> dict:
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": int(values.max()),
    }


def run_eda() -> dict:
    config = load_config()
    clean = project_path(config["processed_dir"]) / "ratings_clean.parquet"
    if not clean.is_file():
        raise FileNotFoundError(f"Run prepare first: {clean}")
    figures = project_path(config["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)
    con = connection(config)
    relation = parquet_sql(clean)
    rating_counts = con.execute(
        f"SELECT rating, COUNT(*) FROM {relation} GROUP BY rating ORDER BY rating"
    ).fetchall()
    user_counts = np.asarray(
        [row[0] for row in con.execute(f"SELECT COUNT(*) FROM {relation} GROUP BY userId").fetchall()],
        dtype=np.int32,
    )
    movie_counts = np.asarray(
        [row[0] for row in con.execute(f"SELECT COUNT(*) FROM {relation} GROUP BY movieId").fetchall()],
        dtype=np.int32,
    )
    con.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(row[0]) for row in rating_counts], [row[1] / 1_000_000 for row in rating_counts], color="#3268a8")
    ax.set(xlabel="Rating (stars)", ylabel="Ratings (millions)", title="MovieLens 25M rating distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "rating_distribution.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, counts, label, color in (
        (axes[0], user_counts, "Ratings per user", "#4a8c70"),
        (axes[1], movie_counts, "Ratings per movie", "#c56f42"),
    ):
        edges = np.geomspace(1, int(counts.max()) + 1, 45)
        ax.hist(counts, bins=edges, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlabel=label, ylabel="Number of users" if label.endswith("user") else "Number of movies")
        ax.grid(alpha=0.2)
    fig.suptitle("Interaction frequency")
    fig.tight_layout()
    fig.savefig(figures / "interaction_distribution.png", dpi=170)
    plt.close(fig)

    movies = pd.read_csv(project_path(config["dataset_dir"]) / "movies.csv", usecols=["movieId", "genres"])
    genres = Counter(
        genre for cell in movies["genres"].fillna("") for genre in cell.split("|") if genre and genre != "(no genres listed)"
    )
    top_genres = genres.most_common(18)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([name for name, _ in reversed(top_genres)], [n for _, n in reversed(top_genres)], color="#725ca2")
    ax.set(xlabel="Number of movies", title="Genres in movies.csv")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "genre_distribution.png", dpi=170)
    plt.close(fig)

    profile_path = project_path(config["results_dir"]) / "data_profile.json"
    profile = read_json(profile_path)
    profile["eda"] = {
        "rating_counts": {str(rating): count for rating, count in rating_counts},
        "ratings_per_user": describe(user_counts),
        "ratings_per_rated_movie": describe(movie_counts),
        "rated_movie_count": int(movie_counts.size),
        "genres_top_18": dict(top_genres),
    }
    write_json(profile_path, profile)
    return profile["eda"]


if __name__ == "__main__":
    summary = run_eda()
    print(f"EDA complete: {len(summary['rating_counts'])} rating values, {summary['rated_movie_count']:,} rated movies")
