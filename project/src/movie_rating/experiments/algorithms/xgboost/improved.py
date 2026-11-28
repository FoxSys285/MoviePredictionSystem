"""Regularized/subsampled XGBoost with additional content/preference features."""

from movie_rating.experiments.algorithms.xgboost.baseline import XGBoostBaseline


class XGBoostImproved(XGBoostBaseline):
    """Compare richer features and stronger regularization using held-out metrics."""

    rich_features = True
    defaults = {**XGBoostBaseline.defaults, "n_estimators": 200, "subsample": 0.8,
                "colsample_bytree": 0.8, "reg_lambda": 10.0, "reg_alpha": 0.1}
