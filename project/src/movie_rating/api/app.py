import json
import os
import re
import uuid
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from movie_rating.api.accounts import account_router
from movie_rating.api.activity import activity_router
from movie_rating.api.frontend import mount_frontend
from movie_rating.api.lab import lab_router
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

    @router.get("/experiments/latest")
    def comparison():
        run = get_run()
        test_path = run / "test.json"
        analysis_path = run / "analysis.json"
        stability_path = run / "stability.json"
        return {
            "validation": json.loads((run / "validation.json").read_text(encoding="utf-8")),
            "test": json.loads(test_path.read_text(encoding="utf-8"))
            if test_path.is_file()
            else None,
            "analysis": json.loads(analysis_path.read_text(encoding="utf-8"))
            if analysis_path.is_file()
            else None,
            "stability": json.loads(stability_path.read_text(encoding="utf-8"))
            if stability_path.is_file()
            else None,
        }

    @router.get("/experiments/exports/{document}")
    def experiment_export(document: Literal["stability", "seeds", "temporal", "bootstrap"]):
        relative = {
            "stability": "stability.json",
            "seeds": "stability/seed-runs.csv",
            "temporal": "stability/temporal-runs.csv",
            "bootstrap": "stability/bootstrap.csv",
        }[document]
        path = get_run() / relative
        if not path.is_file():
            raise HTTPException(404, "Kết quả này chưa được tạo.")
        media_type = "application/json" if document == "stability" else "text/csv"
        return FileResponse(path, media_type=media_type, filename=path.name)

    @router.get("/experiments/documents/{document}")
    def experiment_document(
        document: Literal["protocol", "challenges", "figures-stability", "results"]
    ):
        name = {
            "protocol": "experimental-protocol.md",
            "challenges": "experimental-challenges.md",
            "figures-stability": "experiment-figures-and-stability.md",
            "results": "experimental-results.md",
        }[document]
        path = PROJECT_ROOT / "docs" / name
        if not path.is_file():
            raise HTTPException(404, "Tài liệu thực nghiệm chưa có sẵn.")
        return FileResponse(path, media_type="text/markdown; charset=utf-8", filename=name)

    app.include_router(router)
    app.include_router(lab_router(settings))
    app.include_router(account_router(settings))
    app.include_router(activity_router(settings, catalog_service))
    mount_frontend(app, PROJECT_ROOT / "frontend/dist")
    return app


app = create_app()
