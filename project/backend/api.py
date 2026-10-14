from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.recommender import MODEL_DIR, Recommender


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("movielens_api")
MODEL_KEYS = ("linear", "xgboost", "svd", "ncf")
MODEL_NAMES = {
    "linear": "Linear Regression",
    "xgboost": "XGBoost",
    "svd": "SVD / Matrix Factorization",
    "ncf": "Neural Collaborative Filtering",
}

app = FastAPI(title="MovieLens AI Recommendation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Recommender | None = None
if (MODEL_DIR / "metadata.joblib").exists():
    try:
        engine = Recommender(MODEL_DIR)
    except Exception:
        LOGGER.exception("Could not load trained MovieLens model artifacts")


class HistoryRating(BaseModel):
    movieId: int = Field(gt=0)
    rating: float = Field(ge=0.5, le=5.0)
    timestamp: int = Field(gt=0)


class PredictionRequest(BaseModel):
    userId: int | None = Field(default=None, ge=1)
    history: list[HistoryRating] = Field(default_factory=list)
    model: Literal["linear", "xgboost", "svd", "ncf"] = "ncf"


def require_engine() -> Recommender:
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Chưa có model artifact. Chạy python -m backend.train trước khi mở API.",
        )
    return engine


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ready" if engine is not None else "models_not_trained",
        "modelsLoaded": engine is not None,
    }


@app.get("/api/models")
def model_metrics() -> dict[str, object]:
    model_engine = require_engine()
    result = []
    for key in MODEL_KEYS:
        metrics = model_engine.metrics.get(key, {})
        result.append(
            {
                "key": key,
                "name": MODEL_NAMES[key],
                "trainingSeconds": metrics.get("training_seconds"),
                "sampledTrainRows": metrics.get("sampled_train_rows"),
                "validation": metrics.get("validation"),
                "test": metrics.get("test"),
            }
        )
    return {
        "models": result,
        "baseline": model_engine.metrics.get("baseline"),
        "dataset": "MovieLens 25M · split theo thời gian 70/15/15",
        "predictionMethod": "ratings + genres + timestamp",
    }


@app.post("/api/recommendations")
def recommendations(request: PredictionRequest) -> dict[str, object]:
    model_engine = require_engine()
    try:
        return model_engine.recommend(
            request.model,
            request.userId,
            [row.model_dump() for row in request.history],
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/compare")
def compare(request: PredictionRequest) -> dict[str, object]:
    model_engine = require_engine()
    history = [row.model_dump() for row in request.history]
    output = {}
    for key in MODEL_KEYS:
        rows, latency_ms = model_engine.predict_candidates(
            key, request.userId, history, list(model_engine.movie_by_id)
        )
        output[key] = {
            "name": MODEL_NAMES[key],
            "recommendations": rows,
            "latencyMs": round(latency_ms, 3),
            "metrics": model_engine.metrics.get(key, {}),
        }
    return {
        "models": output,
        "baseline": model_engine.metrics.get("baseline", {}),
        "userId": request.userId,
        "historyCount": len(history),
    }
