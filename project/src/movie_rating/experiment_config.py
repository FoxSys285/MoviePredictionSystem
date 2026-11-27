"""Validated configuration for the research lab, separate from serving models."""

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from movie_rating.data.temporal import CV_FOLDS

ALGORITHMS = {"rating": ("linear", "xgboost", "svd", "ncf")}

PRESET_VERSIONS = {"rating": 1}
PRESET_IDS = {
    group: f"{group}.standard.v{version}" for group, version in PRESET_VERSIONS.items()
}

# Standard Studio parameters are versioned; reduction_choice selects one representation.
# Research callers that omit preset_id retain the legacy explicit configuration API.
PRESET_OVERRIDES = {
    "rating": {
        "tune": False,
        "reduction_mode": "off",
        "reduction_methods": ["pca"],
        "components": [8],
    },
}

PRESET_REDUCTIONS = {
    "rating": {
        "linear": {"method": "pca", "components": 8},
        "xgboost": None,
        "svd": None,
        "ncf": None,
    },
}


def preset_reduction_plan(group: str, algorithm: str, choice: str | None = None):
    """Resolve the selected reducer within the preset's supported algorithm branches."""
    plan = deepcopy(PRESET_REDUCTIONS.get(group, {}).get(algorithm))
    if choice == "off":
        return None
    if group == "rating":
        return plan
    return None


def preset_catalog():
    """Public, read-only descriptions generated from the same values used by requests."""
    output = {}
    for group, preset_id in PRESET_IDS.items():
        defaults = {
            name: field.get_default(call_default_factory=True)
            for name, field in ExperimentRequest.model_fields.items()
            if name not in {"group", "preset_id", "algorithms"}
        }
        values = {**defaults, **deepcopy(PRESET_OVERRIDES[group])}
        params = values["parameters"]
        if isinstance(params, ModelParameters):
            params = params.model_dump()

        common = [
            {"label": "Dữ liệu", "value": "Toàn bộ dữ liệu đã chuẩn bị"},
            {"label": "Seed", "value": str(values["seed"])},
            {"label": "Train / Validation / Test", "value": "70% / 15% / 15%"},
        ]
        common.append({"label": "Đánh giá", "value": f"GridSearchCV với cv={CV_FOLDS} trên Train"})

        algorithm_settings = {}
        for algorithm in ALGORITHMS[group]:
            if algorithm == "linear":
                settings = [
                    {"label": "scaler", "value": "StandardScaler()"},
                    {"label": "fit_intercept", "value": "True"},
                ]
            elif algorithm == "xgboost":
                settings = [
                    {"label": "n_estimators", "value": str(params["n_estimators"])},
                    {"label": "max_depth", "value": str(params["max_depth"])},
                    {"label": "learning_rate", "value": str(params["xgb_learning_rate"])},
                    {"label": "n_jobs", "value": str(params["xgb_n_jobs"])},
                ]
            elif algorithm == "svd":
                settings = [
                    {"label": "factors", "value": str(params["factors"])},
                    {"label": "epochs", "value": str(values["epochs"])},
                    {"label": "learning_rate", "value": str(params["learning_rate"])},
                    {"label": "regularization", "value": str(params["regularization"])},
                ]
            else:
                settings = [
                    {"label": "factors", "value": str(params["factors"])},
                    {"label": "hidden_size", "value": str(params["hidden_size"])},
                    {"label": "epochs", "value": str(values["epochs"])},
                    {"label": "patience", "value": str(values["patience"])},
                    {"label": "batch_size", "value": str(params["batch_size"])},
                    {"label": "learning_rate", "value": str(params["ncf_learning_rate"])},
                    {"label": "regularization", "value": str(params["regularization"])},
                    {"label": "dropout", "value": str(params["dropout"])},
                ]

            algorithm_settings[algorithm] = {
                "settings": settings,
                "reduction": preset_reduction_plan(group, algorithm),
            }

        output[group] = {
            "id": preset_id,
            "version": PRESET_VERSIONS[group],
            "label": "Preset chuẩn",
            "common": common,
            "algorithms": algorithm_settings,
            "tuning": False,
        }
    return output


class ModelParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    factors: int = Field(32, ge=2, le=128)
    hidden_size: int = Field(32, ge=4, le=256)
    learning_rate: float = Field(0.005, gt=0, le=0.1)
    xgb_learning_rate: float = Field(0.05, gt=0, le=0.3)
    ncf_learning_rate: float = Field(0.003, gt=0, le=0.1)
    regularization: float = Field(0.02, ge=0, le=10)
    n_estimators: int = Field(150, ge=10, le=500)
    xgb_n_jobs: Literal[2, 4, 8] = 4
    max_depth: int = Field(6, ge=1, le=20)
    dropout: float = Field(0.1, ge=0, lt=1)
    batch_size: int = Field(2048, ge=32, le=16384)


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group: Literal["rating"] = "rating"
    preset_id: str | None = None
    reduction_choice: Literal["off", "pca"] | None = None
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
    components: list[int] = Field(default_factory=lambda: [2, 4, 8], min_length=1, max_length=12)
    reduction_mode: Literal["off", "enabled", "compare"] = "off"
    reduction_methods: list[Literal["pca"]] = Field(default_factory=lambda: ["pca"], min_length=1, max_length=1)

    @model_validator(mode="after")
    def validate_options(self):
        if self.reduction_choice is not None and self.preset_id is None:
            raise ValueError("reduction_choice chỉ dùng với preset_id.")
        if self.preset_id is not None:
            expected_preset = PRESET_IDS.get(self.group)
            if self.preset_id != expected_preset:
                raise ValueError(
                    f"Preset {self.preset_id!r} không hợp lệ cho nhóm {self.group}."
                )
            canonical = {
                name: field.get_default(call_default_factory=True)
                for name, field in type(self).model_fields.items()
                if name not in {"group", "preset_id", "algorithms", "reduction_choice"}
            }
            canonical.update(deepcopy(PRESET_OVERRIDES[self.group]))
            if self.reduction_choice is not None:
                canonical["reduction_mode"] = "off" if self.reduction_choice == "off" else "enabled"
                canonical["reduction_methods"] = [
                    "pca" if self.reduction_choice == "off" else self.reduction_choice
                ]
                canonical["components"] = [8]
                if self.reduction_choice != "off":
                    canonical["reduction_mode"] = "compare"
                    canonical["reduction_methods"] = ["pca"]
            configurable = {"group", "preset_id", "algorithms", "reduction_choice"}
            for name in self.model_fields_set - configurable:
                supplied = getattr(self, name)
                expected = canonical[name]
                if isinstance(supplied, BaseModel):
                    supplied = supplied.model_dump()
                if isinstance(expected, BaseModel):
                    expected = expected.model_dump()
                if supplied != expected:
                    raise ValueError(
                        f"Preset {self.preset_id} là cấu hình cố định; "
                        f"trường {name} không khớp preset."
                    )
            for name, value in canonical.items():
                setattr(self, name, deepcopy(value))

        allowed = ALGORITHMS[self.group]
        if self.algorithms is None:
            self.algorithms = list(allowed)
        if not self.algorithms or len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("Chọn ít nhất một thuật toán, không trùng lặp.")
        if set(self.algorithms) - set(allowed):
            raise ValueError(f"Thuật toán hợp lệ cho {self.group}: {allowed}")
        for values, low, high in [(self.components, 1, 128)]:
            if any(not low <= v <= high for v in values):
                raise ValueError(f"Giá trị danh sách phải nằm trong [{low}, {high}].")
        if len(set(self.reduction_methods)) != len(self.reduction_methods):
            raise ValueError("Không được lặp phương pháp giảm chiều.")
        return self
