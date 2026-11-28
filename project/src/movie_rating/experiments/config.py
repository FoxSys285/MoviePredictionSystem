"""Starting parameter grids for Train-only search; callers may override them."""

from copy import deepcopy

BASELINE_GRIDS = {
    "linear": {"fit_intercept": [True], "smoothing": [5.0, 20.0]},
    "xgboost": {"max_depth": [4, 6], "learning_rate": [0.05, 0.1],
                "n_estimators": [150], "n_jobs": [4]},
    "svd": {"factors": [16, 32], "regularization": [0.02, 0.05], "epochs": [20]},
    "ncf": {"factors": [16, 32], "regularization": [0.00001, 0.0001], "epochs": [15]},
}

IMPROVED_GRIDS = {
    "linear": {"alpha": [1.0, 10.0], "smoothing": [5.0, 20.0]},
    "xgboost": {"max_depth": [4, 6], "reg_lambda": [5.0, 10.0],
                "n_estimators": [200], "n_jobs": [4]},
    "svd": {"factors": [32, 64], "regularization": [0.02, 0.05], "epochs": [20]},
    "ncf": {"factors": [16, 32], "dropout": [0.1, 0.2], "epochs": [15]},
}


def default_parameter_grids(variant="baseline"):
    """Return independent dictionaries so edits do not change future runs."""
    if variant not in {"baseline", "improved"}:
        raise ValueError(f"Unknown experiment variant: {variant}")
    return deepcopy(BASELINE_GRIDS if variant == "baseline" else IMPROVED_GRIDS)
