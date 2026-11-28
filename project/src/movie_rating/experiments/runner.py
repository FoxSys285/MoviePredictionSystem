"""Train-only grid search, validation early stopping, and one final Test pass."""

from numbers import Integral
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.utils.validation import check_is_fitted

from movie_rating.experiments.algorithms import ALGORITHM_REGISTRY, build_algorithm
from movie_rating.experiments.baseline import GlobalMeanBaseline
from movie_rating.experiments.config import default_parameter_grids
from movie_rating.experiments.data import CV_FOLDS, make_cv_folds, validate_splits
from movie_rating.experiments.evaluation import evaluate_predictions

ALGORITHM_IDS = ("linear", "xgboost", "svd", "ncf")
SCORING = {
    "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
}


class MovieRegressor(RegressorMixin, BaseEstimator):
    """Clone-safe adapter: every CV fit builds features from that fold alone."""

    def __init__(self, algorithm, variant="baseline", movies=None, parameters=None, report=None):
        self.algorithm = algorithm
        self.variant = variant
        self.movies = movies
        self.parameters = parameters
        self.report = report

    def fit(self, X, y):
        training = X.drop(columns="rating", errors="ignore").copy()
        training["rating"] = np.asarray(y, dtype=np.float64)
        self.model_ = build_algorithm(self.algorithm, self.variant, **(self.parameters or {}))
        if self.report is not None:
            self.report(f"search:{self.algorithm}",
                        f"{self.algorithm}: fit {len(training):,} ratings")
            self.model_.report = lambda message: self.report(f"search:{self.algorithm}", message)
        self.model_.fit(training, self.movies)
        self.n_features_in_ = X.shape[1]
        if self.report is not None:
            self.report(f"search:{self.algorithm}", f"{self.algorithm}: fit completed")
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        pairs = X.drop(columns="rating", errors="ignore")
        predicted = np.asarray(self.model_.predict(pairs), dtype=np.float64)
        if predicted.shape != (len(pairs),) or not np.isfinite(predicted).all():
            raise ValueError("Models must return one finite prediction per input row.")
        return predicted


def grid_search_algorithm(algorithm, train, movies, parameter_grid, *, variant="baseline",
                          cv=None, n_jobs=1, report=None):
    """Choose parameters on five Train-only folds; refit best on all Train."""
    candidates = list(ParameterGrid(parameter_grid))
    if not candidates:
        raise ValueError(f"Provide at least one parameter configuration for {algorithm}.")
    search = GridSearchCV(
        estimator=MovieRegressor(algorithm, variant=variant, movies=movies, report=report),
        param_grid={"parameters": candidates},
        scoring=SCORING,
        refit="rmse",
        cv=make_cv_folds(train) if cv is None else cv,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(train.drop(columns="rating"), train.rating.to_numpy())
    return search


def _score(model, frame):
    # Labels must never be present in the model's prediction inputs.
    return evaluate_predictions(frame.rating, model.predict(frame.drop(columns="rating")))


def run_experiment(splits, movies, parameter_grids=None, *, variant="baseline",
                   refit_train_validation=False, n_jobs=1, report=None, algorithms=None):
    """Run the selected models plus Global Mean using the frozen outer splits.

    Parameter grids can override config.py's starting grids per algorithm.
    Ensemble fitting is not enabled in this initial protocol.
    """
    if variant not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown experiment variant: {variant}")
    requested = list(ALGORITHM_IDS if algorithms is None else algorithms)
    if (not requested or len(requested) != len(set(requested))
            or set(requested) - set(ALGORITHM_IDS)):
        raise ValueError("Choose at least one unique supported algorithm.")
    chosen = [name for name in ALGORITHM_IDS if name in requested]
    if parameter_grids is None:
        parameter_grids = {name: grid for name, grid in default_parameter_grids(variant).items()
                           if name in chosen}
    if set(parameter_grids) != set(chosen):
        raise ValueError(f"Provide parameter grids for exactly {chosen}.")
    validate_splits(splits)
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    cv = make_cv_folds(train)
    reference = GlobalMeanBaseline().fit(train)
    train_mean = reference.mean_
    validation_metrics = {"global_mean": _score(reference, validation)}
    models, best_parameters, final_parameters, cv_metrics, best_iterations = {}, {}, {}, {}, {}
    search_seconds = {}

    for algorithm in chosen:
        if report is not None:
            report(f"search:{algorithm}", f"GridSearchCV 5-fold: {algorithm}")
        started = perf_counter()
        search = grid_search_algorithm(
            algorithm, train, movies, parameter_grids[algorithm], variant=variant,
            cv=cv, n_jobs=n_jobs, report=report,
        )
        search_seconds[algorithm] = perf_counter() - started
        model = search.best_estimator_.model_
        parameters = dict(search.best_params_["parameters"])
        best_parameters[algorithm] = parameters
        final_parameters[algorithm] = dict(parameters)
        index = search.best_index_
        cv_metrics[algorithm] = {
            metric: {
                "mean": float(-search.cv_results_[f"mean_test_{metric}"][index]),
                "std": float(search.cv_results_[f"std_test_{metric}"][index]),
            }
            for metric in SCORING
        }
        if model.supports_early_stopping:
            if report is not None:
                report(f"validation:{algorithm}", f"Validation / early stopping: {algorithm}")
                model.report = lambda message, name=algorithm: report(f"validation:{name}", message)
            model.fit_with_validation(train, validation, movies)
            count = getattr(model, "best_iterations_", None)
            if not isinstance(count, Integral) or isinstance(count, bool) or count < 1:
                raise ValueError(f"{algorithm} must expose best_iterations_ as a positive count.")
            if not model.iteration_parameter:
                raise ValueError(f"{algorithm} must declare its iteration_parameter.")
            best_iterations[algorithm] = int(count)
            final_parameters[algorithm][model.iteration_parameter] = int(count)
        models[algorithm] = model
        validation_metrics[algorithm] = _score(model, validation)
        if report is not None:
            report(f"finished:{algorithm}",
                   f"{algorithm}: Validation {validation_metrics[algorithm]}")

    selected = min(validation_metrics, key=lambda name: validation_metrics[name]["rmse"])
    if refit_train_validation:
        history = pd.concat([train, validation], ignore_index=True)
        history = history.sort_values(["userId", "timestamp", "movieId"], kind="stable")
        for algorithm in chosen:
            models[algorithm] = build_algorithm(algorithm, variant, **final_parameters[algorithm])
            if report is not None:
                report(f"refit:{algorithm}", f"Refit Train + Validation: {algorithm}")
                models[algorithm].report = lambda message, name=algorithm: report(
                    f"refit:{name}", message)
            # Fixed epochs/tree count: no validation or test data supplied for stopping.
            models[algorithm].fit(history, movies)
        reference.fit(history)

    # Test is first accessed for scoring only after every tuning/refit decision.
    if report is not None:
        report("test", f"Final Test evaluation: {len(chosen)} models and Global Mean")
    test_metrics = {name: _score(model, test) for name, model in models.items()}
    test_metrics["global_mean"] = _score(reference, test)
    return {
        "protocol": {
            "split": "per_user_temporal_70_15_15",
            "cv_folds": CV_FOLDS,
            "cv": "expanding_per_user_time_folds_on_train",
            "scoring": list(SCORING),
            "refit_metric": "rmse",
            "refit_train_validation": refit_train_validation,
            "test_fit_scope": "train_validation" if refit_train_validation else "train",
            "ensemble": False,
            "algorithms": chosen,
        },
        "split_report": train.attrs.get("split_report"),
        "best_parameters": best_parameters,
        "final_parameters": final_parameters,
        "best_iterations": best_iterations,
        "cv": cv_metrics,
        "search_seconds": search_seconds,
        "validation": validation_metrics,
        "selected_on_validation": selected,
        "test": test_metrics,
        "global_mean": {"validation": train_mean, "test": reference.mean_},
    }
