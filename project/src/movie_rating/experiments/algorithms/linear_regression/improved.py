"""Ridge regularization plus genre/year and user-genre interaction features."""

from sklearn.linear_model import Ridge

from movie_rating.experiments.algorithms.base import finite_number
from movie_rating.experiments.algorithms.linear_regression.baseline import LinearRegressionBaseline


class LinearRegressionImproved(LinearRegressionBaseline):
    """Use the same scaling protocol with a regularized linear estimator."""

    rich_features = True
    defaults = {**LinearRegressionBaseline.defaults, "alpha": 10.0}

    def _estimator(self, options):
        alpha = finite_number("alpha", options["alpha"], strict=True)
        return Ridge(alpha=alpha, fit_intercept=options["fit_intercept"], copy_X=False,
                     solver="lsqr", tol=1e-5)
