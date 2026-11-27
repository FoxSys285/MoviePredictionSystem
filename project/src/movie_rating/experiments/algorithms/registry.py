"""Map frontend algorithm IDs and variants to their implementation classes."""

from movie_rating.experiments.algorithms.base import MovieAlgorithm
from movie_rating.experiments.algorithms.linear_regression import (
    LinearRegressionBaseline,
    LinearRegressionImproved,
)
from movie_rating.experiments.algorithms.ncf import NCFBaseline, NCFImproved
from movie_rating.experiments.algorithms.svd import SVDBaseline, SVDImproved
from movie_rating.experiments.algorithms.xgboost import XGBoostBaseline, XGBoostImproved

ALGORITHM_REGISTRY = {
    "baseline": {
        "linear": LinearRegressionBaseline,
        "xgboost": XGBoostBaseline,
        "svd": SVDBaseline,
        "ncf": NCFBaseline,
    },
    "improved": {
        "linear": LinearRegressionImproved,
        "xgboost": XGBoostImproved,
        "svd": SVDImproved,
        "ncf": NCFImproved,
    },
}


def build_algorithm(algorithm: str, variant: str = "baseline", **parameters) -> MovieAlgorithm:
    """Create the selected scaffold; fitting is implemented in a later step."""
    if variant not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown experiment variant: {variant}")
    implementations = ALGORITHM_REGISTRY[variant]
    if algorithm not in implementations:
        raise ValueError(f"Unknown experiment algorithm: {algorithm}")
    return implementations[algorithm](**parameters)
