from pydantic import BaseModel, Field

from movie_rating.settings import Algorithm


class PredictionRequest(BaseModel):
    user_id: int = Field(gt=0)
    movie_id: int = Field(gt=0)
    algorithm: Algorithm | None = None


class PredictionResponse(BaseModel):
    user_id: int
    movie_id: int
    algorithm: Algorithm
    predicted_rating: float = Field(ge=0.5, le=5)
    cold_start: bool
    run_id: str
