import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
Algorithm = Literal["ridge", "xgboost", "svd", "ncf", "baseline", "item_knn", "nmf"]
PRIMARY_ALGORITHMS = ("ridge", "xgboost", "svd", "ncf")
ALGORITHMS = (*PRIMARY_ALGORITHMS, "baseline", "item_knn", "nmf")


class DataSettings(BaseModel):
    directory: str = "../dataset"
    user_sample_fraction: float = Field(default=0.02, gt=0, le=1)
    chunksize: int = Field(default=500_000, gt=0)
    train_fraction: float = Field(default=0.8, gt=0, lt=1)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_split(self):
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("The split must leave a nonempty test fraction.")
        return self


class ExperimentSettings(BaseModel):
    seed: int = Field(default=42, ge=0)
    artifacts_directory: str = "artifacts"
    algorithms: list[str] = Field(default_factory=lambda: list(PRIMARY_ALGORITHMS), min_length=1)

    @model_validator(mode="after")
    def validate_algorithms(self):
        if len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("Duplicate algorithms are not allowed.")
        if set(self.algorithms) - set(ALGORITHMS):
            raise ValueError(f"Supported algorithms: {ALGORITHMS}")
        return self


class Settings(BaseModel):
    data: DataSettings = Field(default_factory=DataSettings)
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)
    models: dict[str, dict[str, int | float]] = Field(default_factory=dict)

    @property
    def dataset_dir(self) -> Path:
        return (PROJECT_ROOT / os.getenv("MOVIE_DATASET_DIR", self.data.directory)).resolve()

    @property
    def artifacts_dir(self) -> Path:
        value = os.getenv("MOVIE_ARTIFACTS_DIR", self.experiment.artifacts_directory)
        return (PROJECT_ROOT / value).resolve()


def load_settings(path: str | Path | None = None) -> Settings:
    with open(path or PROJECT_ROOT / "configs/default.toml", "rb") as stream:
        return Settings.model_validate(tomllib.load(stream))
