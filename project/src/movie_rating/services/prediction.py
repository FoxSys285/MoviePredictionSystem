import json
from functools import lru_cache

import joblib

from movie_rating.settings import ALGORITHMS, Settings
from movie_rating.training import current_run


class PredictionService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @lru_cache(maxsize=1)
    def load_model(self, run_dir: str, algorithm: str):
        # Model artifacts are produced locally by the CLI.
        return joblib.load(f"{run_dir}/{algorithm}.joblib")

    def predict(self, user_id: int, movie_id: int, algorithm: str | None = None):
        run_dir = current_run(self.settings)
        report = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
        selected = algorithm or report["best_algorithm"]
        if selected not in ALGORITHMS:
            raise ValueError("Unsupported algorithm.")
        model = self.load_model(str(run_dir), selected)
        known_user = bool((model.user_ids == user_id).any())
        known_movie = bool((model.movie_ids == movie_id).any())
        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "algorithm": selected,
            "predicted_rating": float(model.predict([user_id], [movie_id])[0]),
            "cold_start": not (known_user and known_movie),
            "run_id": report["run_id"],
        }
