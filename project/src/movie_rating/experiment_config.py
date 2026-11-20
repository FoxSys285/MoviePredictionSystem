"""Validated configuration for the research lab, separate from serving models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ALGORITHMS = {
    "rating": ("linear", "xgboost", "svd", "ncf"),
    "classification": ("logistic", "random_forest"),
    "clustering": ("kmeans", "dbscan"),
    "association": ("apriori",),
    "pca": ("pca", "truncated_svd"),
    "reinforcement": ("dqn", "random", "popular"),
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
    # Legacy field accepted from older clients; isolated workers no longer enforce it.
    memory_budget_mb: int = Field(
        4096, ge=1024, le=32768, description="Deprecated; ignored by workers"
    )
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
    # Legacy Q-table fields remain readable in saved schema 2/3 reports. New
    # runs use the explicitly named DQN parameters below.
    episodes: int = Field(500, ge=1, le=20000)
    learning_rate: float = Field(0.001, gt=0, le=0.1)
    discount_factor: float = Field(0.9, ge=0, le=1)
    exploration_rate: float = Field(0.2, ge=0, le=1)
    training_sweeps: int = Field(1, ge=1, le=20)
    max_training_steps: int = Field(100000, ge=1, le=2_000_000)
    training_user_limit: int = Field(5000, ge=10, le=50000)
    action_limit: int = Field(30, ge=5, le=30)
    horizon: int = Field(5, ge=1, le=30)
    evaluation_users: int = Field(1000, ge=10, le=5000)
    replay_capacity: int = Field(50000, ge=100, le=500000)
    dqn_batch_size: int = Field(128, ge=8, le=2048)
    checkpoint_interval_steps: int = Field(5000, ge=1, le=100000)
    replay_warmup: int = Field(500, ge=1, le=50000)
    target_update_steps: int = Field(250, ge=1, le=10000)
    epsilon_start: float = Field(0.9, ge=0, le=1)
    epsilon_end: float = Field(0.05, ge=0, le=1)
    epsilon_decay_steps: int = Field(10000, ge=1, le=2_000_000)
    hidden_sizes: list[int] = Field(default_factory=lambda: [64, 64], min_length=1, max_length=4)
    reduction_mode: Literal["off", "enabled", "compare"] = "off"
    reduction_methods: list[Literal["pca", "truncated_svd"]] = Field(
        default_factory=lambda: ["pca", "truncated_svd"], min_length=1, max_length=2
    )

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
        if len(set(self.reduction_methods)) != len(self.reduction_methods):
            raise ValueError("Không được lặp phương pháp giảm chiều.")
        if any(size < 4 or size > 256 for size in self.hidden_sizes):
            raise ValueError("Mỗi lớp DQN phải có từ 4 đến 256 units.")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end phải nhỏ hơn hoặc bằng epsilon_start.")
        if self.replay_warmup > self.replay_capacity:
            raise ValueError("Replay warmup không được lớn hơn replay_capacity.")
        if self.group == "reinforcement" and self.horizon > self.action_limit:
            # Episodes are allowed to shorten when fewer actions remain, but a
            # horizon larger than the catalog limit is almost always accidental.
            raise ValueError("Horizon không được lớn hơn action_limit.")
        return self
