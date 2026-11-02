from typing import Literal

import pandas as pd
from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from movie_rating.api.accounts import check_origin
from movie_rating.services.accounts import AccountStore
from movie_rating.services.catalog import CatalogService, movie_records
from movie_rating.services.recommendations import RecommendationService
from movie_rating.settings import Algorithm, Settings


class RatingRequest(BaseModel):
    rating: float = Field(ge=0.5, le=5)

    @field_validator("rating")
    @classmethod
    def half_step(cls, value):
        if value * 2 != int(value * 2):
            raise ValueError("Điểm đánh giá phải theo bước 0,5.")
        return value


def activity_router(settings: Settings, catalog: CatalogService) -> APIRouter:
    router = APIRouter(prefix="/api/v1")
    store = AccountStore(settings.artifacts_dir / "application.sqlite3")
    recommendations = RecommendationService(settings, catalog)

    def load_catalog():
        try:
            return catalog.load()
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(503, str(error)) from error

    def optional_user(movie_session: str | None = Cookie(default=None)):
        return store.user_for_session(movie_session)

    def current_user(user=Depends(optional_user)):
        if not user:
            raise HTTPException(401, "Vui lòng đăng nhập để đánh giá phim và xem lịch sử của bạn.")
        return user

    @router.get("/me/ratings")
    def history(
        offset: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=100),
        user=Depends(current_user),
    ):
        records = store.ratings_for_user(user["id"])
        if not records:
            return {"items": [], "total": 0, "offset": offset, "limit": limit}
        frame = pd.DataFrame(records).rename(
            columns={"movie_id": "movieId", "rating": "user_rating"}
        )
        frame = frame.merge(load_catalog().movies, on="movieId", how="inner")
        frame = frame.sort_values(["updated_at", "movieId"], ascending=[False, False])
        page = frame.iloc[offset : offset + limit]
        items = movie_records(page)
        for item, (_, row) in zip(items, page.iterrows()):
            item.update(user_rating=float(row.user_rating), updated_at=int(row.updated_at))
        return {"items": items, "total": len(frame), "offset": offset, "limit": limit}

    @router.get("/me/ratings/{movie_id}")
    def rating(movie_id: int, user=Depends(current_user)):
        row = next(
            (row for row in store.ratings_for_user(user["id"]) if row["movie_id"] == movie_id), None
        )
        return row or {"movie_id": movie_id, "rating": None, "updated_at": None}

    @router.put("/me/ratings/{movie_id}")
    def save(movie_id: int, body: RatingRequest, request: Request, user=Depends(current_user)):
        check_origin(request)
        if movie_id not in set(load_catalog().movies.movieId):
            raise HTTPException(404, "Không tìm thấy phim trong dataset.")
        return store.save_rating(user["id"], movie_id, body.rating)

    @router.delete("/me/ratings/{movie_id}")
    def remove(movie_id: int, request: Request, user=Depends(current_user)):
        check_origin(request)
        store.delete_rating(user["id"], movie_id)
        return {"ok": True}

    @router.get("/recommendations")
    def recommend(
        limit: int = Query(20, ge=1, le=20),
        algorithm: Algorithm | Literal["auto"] = "auto",
        user=Depends(optional_user),
    ):
        load_catalog()
        records = store.ratings_for_user(user["id"]) if user else []
        try:
            profile = store.dataset_profile(user["id"]) if user else None
            return {
                **recommendations.recommend(records, limit, algorithm, dataset_profile=profile),
                "authenticated": user is not None,
            }
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(503, str(error)) from error

    @router.get("/discovery/genres")
    def genre_shelves(
        limit: int = Query(5, ge=1, le=5), movies_per_genre: int = Query(10, ge=1, le=10)
    ):
        data = load_catalog()
        counts = {item["name"]: item["movie_count"] for item in data.summary["genres"]}
        preferred = ["Action", "Comedy", "Sci-Fi", "Animation", "Drama"]
        genres = list(dict.fromkeys([name for name in preferred if name in counts] + list(counts)))[
            :limit
        ]
        ranked = data.movies.sort_values(["rating_count", "movieId"], ascending=[False, True])
        return {
            "items": [
                {
                    "name": name,
                    "movie_count": counts[name],
                    "movies": movie_records(
                        ranked.loc[ranked.genres.map(lambda value: name in value.split("|"))].head(
                            movies_per_genre
                        )
                    ),
                }
                for name in genres
            ]
        }

    return router
