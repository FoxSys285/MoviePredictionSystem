"""NeuMF-style interaction branch, deeper MLP, dropout, and regularized bias warm-up."""

from movie_rating.experiments.algorithms.ncf.baseline import NCFBaseline


class NCFImproved(NCFBaseline):
    """Combine embedding products with nonlinear user/movie interactions."""

    product = True
    deep = True
    defaults = {**NCFBaseline.defaults, "dropout": 0.2, "regularization": 0.0001,
                "bias_epochs": 5}
