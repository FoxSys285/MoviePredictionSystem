import os
import re
import uuid
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from movie_rating.api.accounts import account_router
from movie_rating.api.activity import activity_router
from movie_rating.api.frontend import mount_frontend
from movie_rating.api.schemas import PredictionRequest, PredictionResponse
from movie_rating.services.catalog import RATING_COLUMNS, CatalogService, movie_records
from movie_rating.services.prediction import PredictionService
from movie_rating.settings import ALGORITHMS, PROJECT_ROOT, Settings, load_settings
from movie_rating.training import current_run


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    service = PredictionService(settings)
    catalog_service = CatalogService(settings)
    app = FastAPI(title="Movie Rating Prediction", version="0.1.0")
    server_instance = os.getenv("MOVIE_SERVER_INSTANCE") or uuid.uuid4().hex
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )
    router = APIRouter(prefix="/api/v1")

    @router.get("/health")
    def health():
        return {
            "status": "ok",
            "server_instance": server_instance,
            "process_id": os.getpid(),
            "dataset_present": all(
                (settings.dataset_dir / f).is_file() for f in ("ratings.csv", "movies.csv")
            ),
            "data_prepared": (settings.artifacts_dir / "data/manifest.json").is_file(),
            "models_ready": (settings.artifacts_dir / "latest.json").is_file(),
        }

    def get_run():
        try:
            return current_run(settings)
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from error

    @router.get("/models")
    def models():
        pointer = settings.artifacts_dir / "latest.json"
        run = get_run() if pointer.is_file() else None
        return [
            {"algorithm": name, "trained": bool(run and (run / f"{name}.joblib").is_file())}
            for name in ALGORITHMS
        ]

    def movie_catalog():
        try:
            return catalog_service.load().movies
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(503, str(error)) from error

    @router.get("/dataset")
    def dataset():
        movie_catalog()
        return {key: value for key, value in catalog_service.summary.items()
                if key != "fingerprint"}

    @router.get("/movies")
    def movies(
        q: str = Query(default="", max_length=200),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=20, ge=1, le=100),
        genre: str = Query(default="all", max_length=40),
        year: Literal["all", "classic", "2000", "2010", "2020"] = "all",
        min_rating: float = Query(default=0, ge=0, le=5),
        sort: Literal["popular", "rating", "newest", "title"] = "popular",
    ):
        catalog = movie_catalog()
        q = q.strip()
        if q:
            catalog = catalog.loc[catalog.title.str.contains(q, case=False, regex=False, na=False)]
        if genre != "all":
            catalog = catalog.loc[
                catalog.genres.str.contains(r"(?:^|\|)" + re.escape(genre) + r"(?:\||$)")
            ]
        if year == "classic":
            catalog = catalog.loc[(catalog.year > 0) & (catalog.year < 2000)]
        elif year != "all":
            catalog = catalog.loc[(catalog.year >= int(year)) & (catalog.year < int(year) + 10)]
        if min_rating > 0:
            catalog = catalog.loc[catalog.mean_rating >= min_rating]
        order = {
            "popular": "rating_count",
            "rating": "mean_rating",
            "newest": "year",
            "title": "title",
        }[sort]
        catalog = catalog.sort_values([order, "movieId"], ascending=[sort == "title", True])
        return {
            "total": len(catalog),
            "offset": offset,
            "limit": limit,
            "items": movie_records(catalog.iloc[offset : offset + limit]),
        }

    @router.get("/movies/{movie_id}")
    def movie_detail(movie_id: int):
        catalog = movie_catalog()
        found = catalog.loc[catalog.movieId == movie_id]
        if found.empty:
            raise HTTPException(404, "Không tìm thấy phim trong dataset.")
        movie = movie_records(found)[0]
        row = found.iloc[0]
        movie["rating_distribution"] = [
            {"rating": step / 2, "count": int(row[column])}
            for step, column in enumerate(RATING_COLUMNS, 1)
        ]
        movie["tags"] = catalog_service.tags.loc[
            catalog_service.tags.movieId == movie_id, ["tag", "count"]
        ].to_dict(orient="records")
        movie["imdb_url"] = (
            f"https://www.imdb.com/title/tt{str(movie['imdbId']).zfill(7)}/"
            if movie["imdbId"] else None
        )
        movie["tmdb_url"] = (
            f"https://www.themoviedb.org/movie/{movie['tmdbId']}"
            if movie["tmdbId"] else None
        )
        genres = set(row.genres.split("|")) - {"(no genres listed)"}
        related = catalog.loc[catalog.movieId != movie_id].copy()
        related["overlap"] = related.genres.map(lambda value: len(genres & set(value.split("|"))))
        related = related.loc[related.overlap > 0].sort_values(
            ["overlap", "rating_count", "movieId"], ascending=[False, False, True]
        )
        movie["related"] = movie_records(related.head(6))
        return movie

    @router.post("/predictions", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        try:
            return service.predict(request.user_id, request.movie_id, request.algorithm)
        except FileNotFoundError as error:
            raise HTTPException(503, "Model unavailable. Run: movie-rating train") from error

    app.include_router(router)
    app.include_router(account_router(settings))
    app.include_router(activity_router(settings, catalog_service))
    mount_frontend(app, PROJECT_ROOT / "frontend/dist")
    return app


app = create_app()
