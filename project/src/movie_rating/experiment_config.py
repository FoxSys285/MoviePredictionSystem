"""Validated configuration for the research lab, separate from serving models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ALGORITHMS = {
    "rating": ("linear", "xgboost", "svd", "ncf"),
    "classification": ("logistic", "random_forest"),
    "clustering": ("kmeans", "dbscan"),
    "association": ("apriori",),
    "pca": ("pca", "truncated_svd"),
    "reinforcement": ("q_learning", "random", "popular"),
}


class ModelParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    factors: int = Field(32, ge=2, le=128)
    hidden_size: int = Field(32, ge=4, le=256)
    learning_rate: float = Field(0.005, gt=0, le=0.1)
    xgb_learning_rate: float = Field(0.05, gt=0, le=0.3)
    ncf_learning_rate: float = Field(0.003, gt=0, le=0.1)
    regularization: float = Field(0.02, ge=0, le=10)
    n_estimators: int = Field(150, ge=10, le=500)
    max_depth: int = Field(6, ge=1, le=20)
    forest_estimators: int = Field(100, ge=10, le=500)
    forest_depth: int = Field(8, ge=1, le=20)
    c: float = Field(1, gt=0, le=100)
    max_iter: int = Field(600, ge=50, le=2000)
    min_samples_leaf: int = Field(5, ge=1, le=50)
    dropout: float = Field(0.1, ge=0, lt=1)
    batch_size: int = Field(2048, ge=32, le=16384)


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group: Literal["rating", "classification", "clustering", "association", "pca", "reinforcement"]
    seed: Literal[42] = 42
    max_users: int = Field(150, ge=10, le=5000)
    use_all_data: bool = True
    algorithms: list[str] | None = None
    epochs: int = Field(15, ge=2, le=100)
    patience: int = Field(3, ge=1, le=20)
    tune: bool = False
    memory_budget_mb: int = Field(4096, ge=1024, le=32768)
    parameters: ModelParameters = Field(default_factory=ModelParameters)
    k_values: list[int] = Field(
        default_factory=lambda: [2, 3, 4, 6, 8, 10], min_length=1, max_length=10
    )
    eps_values: list[float] = Field(
        default_factory=lambda: [1.0, 1.5, 2.0], min_length=1, max_length=10
    )
    min_samples_values: list[int] = Field(
        default_factory=lambda: [5, 10], min_length=1, max_length=5
    )
    cluster_auto_eps: bool = True
    cluster_min_coverage: float = Field(0.8, ge=0.1, le=1)
    cluster_shrinkage: float = Field(10.0, ge=1, le=100)
    cluster_eval_users: int = Field(2000, ge=10, le=5000)
    min_user_ratings: int = Field(5, ge=2, le=100)
    max_items: int = Field(80, ge=10, le=500)
    min_support: float = Field(0.08, ge=0.001, le=1)
    min_confidence: float = Field(0.3, ge=0.01, le=1)
    max_rule_length: int = Field(2, ge=2, le=3)
    components: list[int] = Field(default_factory=lambda: [2, 4, 8], min_length=1, max_length=12)
    episodes: int = Field(500, ge=10, le=20000)
    learning_rate: float = Field(0.2, gt=0, le=1)
    discount_factor: float = Field(0.9, ge=0, le=1)
    exploration_rate: float = Field(0.2, ge=0, le=1)

    @model_validator(mode="after")
    def validate_options(self):
        if self.group == "association":
            # Association mining always reads the complete prepared Train split.
            self.use_all_data = True
        allowed = ALGORITHMS[self.group]
        if self.algorithms is None:
            self.algorithms = list(allowed)
        if not self.algorithms or len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("Chọn ít nhất một thuật toán, không trùng lặp.")
        if set(self.algorithms) - set(allowed):
            raise ValueError(f"Thuật toán hợp lệ cho {self.group}: {allowed}")
        for values, low, high in [
            (self.k_values, 2, 30),
            (self.eps_values, 0.1, 20),
            (self.min_samples_values, 2, 100),
            (self.components, 1, 128),
        ]:
            if any(not low <= v <= high for v in values):
                raise ValueError(f"Giá trị danh sách phải nằm trong [{low}, {high}].")
        return self
