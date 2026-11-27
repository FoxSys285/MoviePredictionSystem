"""Implement baseline rating matrix factorization here using NumPy and Numba."""

from movie_rating.experiments.algorithms.base import MovieAlgorithm


class SVDBaseline(MovieAlgorithm):
    """Next step: override fit() and predict() for observed user/movie ratings."""
