"""SVD with regularized bias initialization, more factors, and decaying SGD."""

from movie_rating.experiments.algorithms.svd.baseline import SVDBaseline


class SVDImproved(SVDBaseline):
    """Keep the observed-rating objective and validation-selected epoch count."""

    defaults = {**SVDBaseline.defaults, "factors": 64, "regularization": 0.05,
                "bias_epochs": 5, "learning_rate_decay": 0.98}
