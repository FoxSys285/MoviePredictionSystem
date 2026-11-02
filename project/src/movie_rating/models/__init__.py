from movie_rating.models.baseline import BiasBaseline
from movie_rating.models.factorization import NMF, SVD
from movie_rating.models.item_knn import ItemKNN
from movie_rating.models.neural import NeuralCF
from movie_rating.models.regression import RidgeRegression, XGBoostRegression

MODEL_REGISTRY = {
    "ridge": RidgeRegression,
    "xgboost": XGBoostRegression,
    "svd": SVD,
    "ncf": NeuralCF,
    "baseline": BiasBaseline,
    "item_knn": ItemKNN,
    "nmf": NMF,
}
