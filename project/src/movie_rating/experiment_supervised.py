"""Three train-only folds, validation selection, frozen refit and final test."""

import copy
import gc
import json
import shutil
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch import nn

from movie_rating.data.temporal import temporal_folds
from movie_rating.experiment_reduction import FeatureReducer, ReductionSpec, reduction_variants
from movie_rating.lab import LinearRating, chart, measured, series
from movie_rating.lab_compact import regression_scores
from movie_rating.memory import BATCH_ROWS, feature_buffer, serialized_size, workspace
from movie_rating.models.factorization import SVD
from movie_rating.models.neural import NeuralCF
from movie_rating.models.regression import (
    StreamingLinearRegression,
    XGBoostRegression,
    _BoosterPredictor,
)

NAMES = {
    "linear": "Linear Regression",
    "xgboost": "XGBoost Regressor",
    "svd": "SVD Collaborative Filtering",
    "ncf": "Neural Collaborative Filtering",
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest Classifier",
}
CV_FOLDS = 3


def classification_scores(actual, probability, threshold=0.5):
    predicted = probability >= threshold
    return {
        "accuracy": accuracy_score(actual, predicted),
        "f1": f1_score(actual, predicted, zero_division=0),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "auc_roc": roc_auc_score(actual, probability) if len(np.unique(actual)) == 2 else None,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }


def candidates(algorithm, request):
    p = request.parameters
    base = {
        "linear": {},
        "xgboost": dict(
            n_estimators=p.n_estimators,
            max_depth=p.max_depth,
            learning_rate=p.xgb_learning_rate,
        ),
        "svd": dict(
            factors=p.factors, learning_rate=p.learning_rate, regularization=p.regularization
        ),
        "ncf": dict(
            factors=p.factors,
            hidden_size=p.hidden_size,
            learning_rate=p.ncf_learning_rate,
            regularization=p.regularization,
            dropout=p.dropout,
            batch_size=p.batch_size,
        ),
        "logistic": dict(C=p.c, max_iter=p.max_iter),
        "random_forest": dict(
            n_estimators=p.forest_estimators,
            max_depth=p.forest_depth,
            min_samples_leaf=p.min_samples_leaf,
        ),
    }[algorithm]
    output = [base]
    if request.tune and algorithm != "linear":
        smaller, larger = dict(base), dict(base)
        if algorithm in ("svd", "ncf"):
            smaller["factors"] = max(2, p.factors // 2)
            larger["factors"] = min(128, p.factors * 2)
        elif algorithm == "logistic":
            smaller["C"], larger["C"] = p.c / 10, min(100, p.c * 10)
        else:
            smaller["max_depth"] = max(1, base["max_depth"] // 2)
            larger["max_depth"] = min(12, base["max_depth"] + 2)
            larger["n_estimators"] = min(500, base["n_estimators"] + 50)
        for other in (smaller, larger):
            if other not in output:
                output.append(other)
    return output


def rating_model(algorithm, config, request, epochs=None):
    if algorithm == "linear":
        return LinearRating()
    if algorithm == "xgboost":
        return XGBoostRegression(**config, n_jobs=2, seed=42)
    if algorithm == "svd":
        return SVD(**config, epochs=epochs or request.epochs, seed=42)
    return NeuralCF(**config, epochs=epochs or request.epochs, patience=request.patience, seed=42)


class ReducedFeatureRating:
    """Linear/XGBoost adapter that compresses only preference interaction blocks."""

    def __init__(self, algorithm, config, request, variant, fit_scope):
        self.algorithm, self.config, self.request = algorithm, config, request
        self.variant, self.fit_scope = variant, fit_scope

    def fit(self, train, movies, progress=None, report=None):
        self.builder = LinearRating()
        self.builder.set_movie_metadata(movies)
        self.builder.fit_statistics(train)
        self.global_mean = self.builder.global_mean
        self.base_dimensions = 11 + len(self.builder.genre_names)
        with feature_buffer(
            len(train), self.builder.feature_count,
            self.builder.iter_features(train, training=True),
        ) as raw:
            self.reducer = FeatureReducer(
                self.variant["method"], self.variant["components"], seed=42, scale=True
            ).fit(
                raw[:, self.base_dimensions:],
                feature_schema={"family": "user_preference_movie_genre_interactions",
                                "preserved": self.base_dimensions},
                fit_scope=self.fit_scope,
            )
            self.reduction_metadata = self.reducer.metadata()
            if report:
                report(
                    f"Reducer {self.reducer.method} · {self.reducer.input_dimensions}→"
                    f"{self.reducer.output_dimensions} · fit {len(train):,} rows · {self.fit_scope}."
                )
            batches = _reduced_feature_batches(raw, self.reducer, self.base_dimensions)
            if self.algorithm == "linear":
                self.estimator = StreamingLinearRegression()

                def target_batches():
                    offset = 0
                    for block in batches:
                        target = train.rating.iloc[offset : offset + len(block)].to_numpy()
                        yield block, target - self.global_mean
                        offset += len(block)
                        if progress:
                            progress(offset, len(train))

                self.estimator.fit_batches(target_batches())
            else:
                self._fit_xgboost(raw, train, batches, progress, report)
        return self

    def _fit_xgboost(self, raw, train, batches, progress, report):
        model = XGBoostRegression(**self.config, n_jobs=2, seed=42)
        self.estimator = None
        output_dimensions = self.base_dimensions + self.reducer.output_dimensions
        if len(train) < 250_000:
            with feature_buffer(len(train), output_dimensions, batches) as features:
                self.estimator = model._estimator()
                self.estimator.fit(features, train.rating.to_numpy() - self.global_mean)
            return

        import xgboost as xgb

        failure = None
        with workspace("reduced-xgboost-", native_ascii=True) as directory:
            required = len(train) * output_dimensions * 4 + 512 * 2**20
            if shutil.disk_usage(directory).free < required:
                raise OSError("Ổ đĩa cache không đủ chỗ cho XGBoost giảm chiều external-memory.")
            builder, reducer, base_dimensions = self.builder, self.reducer, self.base_dimensions

            class Iterator(xgb.DataIter):
                def __init__(self):
                    self.pass_number = 0
                    super().__init__(cache_prefix=str(directory / "cache"))
                    self.reset()

                def reset(self):
                    self.pass_number += 1
                    self.batches = _reduced_feature_batches(raw, reducer, base_dimensions)
                    self.offset = 0

                def next(self, input_data):
                    features = next(self.batches, None)
                    if features is None:
                        return False
                    stop = self.offset + len(features)
                    input_data(data=features, label=train.rating.iloc[self.offset:stop].to_numpy() - builder.global_mean)
                    self.offset = stop
                    if report:
                        report(f"XGBoost quantile cache · pass {self.pass_number} · {stop:,}/{len(train):,} rows.")
                    return True

            iterator = data = None
            try:
                iterator = Iterator()
                matrix_type = getattr(xgb, "ExtMemQuantileDMatrix", xgb.QuantileDMatrix)
                if matrix_type is xgb.QuantileDMatrix:
                    iterator.cache_prefix = None
                data = matrix_type(iterator, nthread=2)
                self.estimator = _BoosterPredictor(
                    xgb.train(model._estimator().get_xgb_params(), data,
                              num_boost_round=self.config["n_estimators"])
                )
            except Exception as error:
                failure = error.with_traceback(None)
            finally:
                if iterator is not None:
                    iterator._exception = None
                iterator = data = None
                gc.collect()
        if failure is not None:
            raise failure

    def _transform(self, features):
        return np.column_stack(
            (features[:, : self.base_dimensions],
             self.reducer.transform(features[:, self.base_dimensions:]))
        ).astype(np.float32, copy=False)

    def predict(self, user_ids, movie_ids):
        user_ids, movie_ids = np.asarray(user_ids), np.asarray(movie_ids)
        output = np.empty(len(user_ids), dtype=np.float32)
        for start in range(0, len(user_ids), BATCH_ROWS):
            stop = min(start + BATCH_ROWS, len(user_ids))
            users, items = self.builder.encode_queries(user_ids[start:stop], movie_ids[start:stop])
            features = self.builder._feature_batch(
                users, items, movie_ids=movie_ids[start:stop]
            )
            values = self.global_mean + self.estimator.predict(self._transform(features))
            values[(users < 0) & (items < 0)] = self.global_mean
            output[start:stop] = values
        return self.builder.clip(output)


class HybridRatingAdapter:
    """Funk-SVD/NCF plus a learned correction from reduced profile interactions."""

    def __init__(self, algorithm, config, request, variant, fit_scope, epochs=None, side_epochs=None):
        self.algorithm, self.config, self.request = algorithm, config, request
        self.variant, self.fit_scope = variant, fit_scope
        self.epochs = epochs or request.epochs
        self.side_epochs = side_epochs or self.epochs

    def fit(self, train, movies, validation=None, progress=None, report=None):
        self.base_model = fit_rating(
            self.algorithm, self.config, self.request, train, movies, validation,
            epochs=self.epochs, progress=progress, log=report,
        )
        self.base_best_epoch = getattr(self.base_model, "best_epoch", self.epochs)
        self.best_epoch = self.base_best_epoch
        self.builder = LinearRating()
        self.builder.set_movie_metadata(movies)
        self.builder.fit_statistics(train)
        self.global_mean = self.builder.global_mean
        self.base_dimensions = 11 + len(self.builder.genre_names)
        with feature_buffer(
            len(train), self.builder.feature_count,
            self.builder.iter_features(train, training=True),
        ) as raw:
            self.reducer = FeatureReducer(
                self.variant["method"], self.variant["components"], seed=42, scale=True
            ).fit(
                raw[:, self.base_dimensions:],
                feature_schema={"family": "user_preference_movie_genre_interactions",
                                "semantics": "leave_one_rating_out"},
                fit_scope=self.fit_scope,
            )
            self.reduction_metadata = self.reducer.metadata()
            reduced = feature_buffer(
                len(train), self.reducer.output_dimensions,
                (self.reducer.transform(raw[start : start + BATCH_ROWS, self.base_dimensions:])
                 for start in range(0, len(train), BATCH_ROWS)),
            )
            with reduced as profile_features:
                residual = train.rating.to_numpy(dtype=np.float32) - self.base_model.predict(
                    train.userId, train.movieId
                )
                if self.algorithm == "svd":
                    self.side_model = Ridge(alpha=10.0).fit(profile_features, residual)
                    self.best_side_epoch = None
                else:
                    self._fit_neural_side(profile_features, residual, validation, progress, report)
                    if report:
                        report(
                            f"Epoch đã khóa cho refit Train + Validation · base NCF="
                            f"{self.base_best_epoch} · nhánh hiệu chỉnh={self.best_side_epoch}."
                        )
        return self

    def _validation_features(self, frame):
        with feature_buffer(
            len(frame), self.builder.feature_count, self.builder.iter_features(frame)
        ) as raw:
            return np.concatenate(
                [self.reducer.transform(raw[start : start + BATCH_ROWS, self.base_dimensions:])
                 for start in range(0, len(frame), BATCH_ROWS)],
                axis=0,
            )

    def _fit_neural_side(self, features, residual, validation, progress, report):
        torch.manual_seed(42)
        self.side_model = nn.Sequential(
            nn.Linear(self.reducer.output_dimensions, self.config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(self.config["hidden_size"], 1),
        ).cpu()
        optimizer = torch.optim.Adam(
            self.side_model.parameters(), lr=self.config["learning_rate"]
        )
        rng = np.random.default_rng(42)
        validation_features = validation_residual = None
        if validation is not None and len(validation):
            validation_features = self._validation_features(validation)
            validation_residual = validation.rating.to_numpy(dtype=np.float32) - self.base_model.predict(
                validation.userId, validation.movieId
            )
        best, stale, snapshot = float("inf"), 0, None
        self.best_side_epoch = self.side_epochs
        self.learning_curve = []
        for epoch in range(self.side_epochs):
            self.side_model.train()
            order = rng.permutation(len(features))
            total = 0.0
            for start in range(0, len(order), self.config["batch_size"]):
                indices = order[start : start + self.config["batch_size"]]
                batch = torch.from_numpy(np.asarray(features[indices], dtype=np.float32))
                target = torch.from_numpy(residual[indices])
                optimizer.zero_grad(set_to_none=True)
                loss = (self.side_model(batch).squeeze(1) - target).square().mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.side_model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            score = None
            if validation_features is not None:
                self.side_model.eval()
                with torch.no_grad():
                    predicted = self.side_model(torch.from_numpy(validation_features)).squeeze(1).numpy()
                score = float(np.sqrt(np.mean((predicted - validation_residual) ** 2)))
                if score < best - 1e-6:
                    best, stale, snapshot = score, 0, copy.deepcopy(self.side_model.state_dict())
                    self.best_side_epoch = epoch + 1
                else:
                    stale += 1
            curve_row = {
                "epoch": epoch + 1,
                "train_correction_mse": total / len(features),
            }
            if score is not None:
                curve_row["validation_correction_rmse"] = score
                curve_row["validation_correction_mse"] = score**2
            self.learning_curve.append(curve_row)
            if progress:
                progress(epoch + 1, self.side_epochs)
            if report:
                report(f"NCF hybrid correction branch · epoch {epoch + 1}/{self.side_epochs}" +
                       (f" · Validation RMSE={score:.5f}" if score is not None else ""))
            if validation_features is not None and stale >= self.request.patience:
                break
        self.best_side_epoch = self.best_side_epoch or self.side_epochs
        if snapshot is not None:
            self.side_model.load_state_dict(snapshot)
        self.side_model.eval()

    def predict(self, user_ids, movie_ids):
        users, movie_ids = np.asarray(user_ids), np.asarray(movie_ids)
        output = self.base_model.predict(users, movie_ids).astype(np.float32, copy=True)
        frame = pd.DataFrame({"userId": users, "movieId": movie_ids})
        offset = 0
        with torch.no_grad():
            for raw in self.builder.iter_features(frame):
                reduced = self.reducer.transform(raw[:, self.base_dimensions:])
                if self.algorithm == "svd":
                    correction = self.side_model.predict(reduced)
                else:
                    correction = self.side_model(torch.from_numpy(reduced)).squeeze(1).numpy()
                output[offset : offset + len(raw)] += correction
                offset += len(raw)
        return np.clip(output, 0.5, 5).astype(np.float32)


def fit_rating(
    algorithm, config, request, train, movies, validation=None, epochs=None, progress=None, log=None,
    reduction_variant=None, fit_scope="train", side_epochs=None,
):
    if reduction_variant is not None:
        if algorithm in ("linear", "xgboost"):
            model = ReducedFeatureRating(algorithm, config, request, reduction_variant, fit_scope)
            model.fit(train, movies, progress=progress, report=log)
            return model
        model = HybridRatingAdapter(
            algorithm, config, request, reduction_variant, fit_scope, epochs=epochs,
            side_epochs=side_epochs,
        )
        model.fit(train, movies, validation=validation, progress=progress, report=log)
        return model
    model = rating_model(algorithm, config, request, epochs)
    if hasattr(model, "set_movie_metadata"):
        model.set_movie_metadata(movies)
    if algorithm == "ncf":
        model.fit(train, validation=validation, progress=progress, report=log)
    elif algorithm == "svd":
        model.fit(train, progress=progress, report=log)
    elif algorithm == "xgboost":
        model.fit(train, progress=progress, report=log)
    elif algorithm == "linear":
        model.fit(train, progress=progress, report=log)
    else:
        model.fit(train)
    return model


def classifier(algorithm, config):
    if algorithm == "logistic":
        return LogisticRegression(**config, solver="saga", tol=1e-3, random_state=42)
    return RandomForestClassifier(**config, n_jobs=1, random_state=42)


def _reduction_for(request):
    if request.reduction_mode == "off":
        return None
    return {"method": request.reduction_methods[0], "components": request.components[0]}


class ClassificationAdapter:
    def __init__(self, estimator, scaler=None, reducer=None, base_dimensions=None):
        self.estimator, self.scaler, self.reducer = estimator, scaler, reducer
        self.base_dimensions = base_dimensions

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, features):
        values = np.array(features, dtype=np.float32, copy=True)
        if self.scaler is not None:
            self.scaler.transform(values, copy=False)
        if self.reducer is not None:
            prefix = values[:, : self.base_dimensions]
            compressed = self.reducer.transform(values[:, self.base_dimensions :])
            values = np.column_stack((prefix, compressed)).astype(np.float32, copy=False)
        return self.estimator.predict_proba(values)


def _reduced_feature_batches(features, reducer, base_dimensions, batch_rows=BATCH_ROWS):
    for start in range(0, len(features), batch_rows):
        block = features[start : start + batch_rows]
        compressed = reducer.transform(block[:, base_dimensions:])
        yield np.column_stack((block[:, :base_dimensions], compressed)).astype(np.float32, copy=False)


def feature_fit(train, targets, movies, algorithm, config, report=None,
                reduction_variant=None, fit_scope="train"):
    if report:
        report(f"Tạo thống kê và hồ sơ từ {len(train):,} rating Train.")
    builder = LinearRating()
    builder.set_movie_metadata(movies)
    builder.fit_statistics(train)
    if report:
        report(f"Đã tạo {builder.feature_count} đặc trưng, loại nhãn của hàng huấn luyện.")
    estimator = classifier(algorithm, config)
    scaler = None
    if algorithm == "logistic":
        if report:
            report("Đang chuẩn hóa đặc trưng theo batch (StandardScaler).")
        scaler = StandardScaler(copy=False)
        for features in builder.iter_features(train, training=True):
            scaler.partial_fit(features)
    with feature_buffer(
        len(train), builder.feature_count, builder.iter_features(train, training=True)
    ) as features:
        if report:
            storage = "file memmap trên đĩa" if isinstance(features, np.memmap) else "RAM"
            report(f"Đã tạo đặc trưng {features.shape[0]:,} × {features.shape[1]} · {storage}.")
        if scaler is not None:
            for start in range(0, len(features), BATCH_ROWS):
                scaler.transform(features[start : start + BATCH_ROWS], copy=False)
        reducer = None
        if reduction_variant is not None:
            genre_count = len(builder.genre_names)
            base_dimensions = 11 + genre_count
            reducer = FeatureReducer(
                reduction_variant["method"], reduction_variant["components"], seed=42,
                scale=True,
            ).fit(
                features[:, base_dimensions:],
                feature_schema={"family": "user_preference_movie_genre_interactions", "preserved": base_dimensions},
                fit_scope=fit_scope,
            )
            if report:
                meta = reducer.metadata()
                report(
                    f"Reducer {meta['method']} · {meta['input_dimensions']}→{meta['output_dimensions']} "
                    f"features · fit {meta['fit_rows']:,} rows ở {fit_scope} · "
                    f"EVR Train={meta['explained_variance_ratio_train']:.4f}."
                )
            fit_context = feature_buffer(
                len(train), base_dimensions + reducer.output_dimensions,
                _reduced_feature_batches(features, reducer, base_dimensions),
            )
        else:
            base_dimensions = builder.feature_count
            fit_context = nullcontext(features)
        with fit_context as model_features:
            if report:
                report(
                    f"Bắt đầu tối ưu Logistic bằng SAGA · C={config['C']} · "
                    f"tối đa {config['max_iter']} vòng."
                    if algorithm == "logistic"
                    else f"Bắt đầu Random Forest · {config['n_estimators']} cây · "
                    f"depth={config['max_depth']} · min_samples_leaf={config['min_samples_leaf']}."
                )
            estimator.fit(model_features, (train.rating >= 3.5).to_numpy(dtype=np.int8))
            if report:
                report(
                    f"Đã fit Logistic · {int(estimator.n_iter_.max())} vòng tối ưu."
                    if algorithm == "logistic"
                    else f"Đã fit xong {len(estimator.estimators_)} cây Random Forest."
                )
    adapter = ClassificationAdapter(estimator, scaler, reducer, base_dimensions)
    adapter.reduction_metadata = reducer.metadata() if reducer is not None else None
    return builder, adapter, targets


def classification_predict(builder, estimator, frame):
    output = np.empty(len(frame), dtype=np.float32)
    offset = 0
    for features in builder.iter_features(frame):
        output[offset : offset + len(features)] = estimator.predict_proba(features)[:, 1]
        offset += len(features)
    return output


def numeric_summary(folds):
    return {
        key: {
            "mean": float(np.mean([r[key] for r in folds if r.get(key) is not None])),
            "std": float(np.std([r[key] for r in folds if r.get(key) is not None], ddof=1)),
            "n": sum(r.get(key) is not None for r in folds),
        }
        for key, value in folds[0].items()
        if key != "fold"
        and isinstance(value, (float, int, np.number))
        and all(r.get(key) is not None for r in folds)
    }


def _supervised_single(splits, movies, request, update, reduction_variant=None, variant_id=None):
    train, validation, test = [splits[s] for s in ("train", "validation", "test")]
    binary = request.group == "classification"
    rows, charts, cv, best_configs, tuning = [], [], {}, {}, []
    all_candidates = sum(len(candidates(a, request)) for a in request.algorithms)
    total_work = all_candidates + len(request.algorithms) * (CV_FOLDS + 1)
    step = 0
    reduction_kwargs = {"reduction_variant": reduction_variant} if reduction_variant else {}
    for algorithm in request.algorithms:
        label = f"{NAMES[algorithm]}" + (f" · {variant_id}" if variant_id else "")
        unit = "cây" if algorithm == "xgboost" else "rating" if algorithm == "linear" else "epoch"
        update(None, f"Bắt đầu {label} · {len(train):,} rating Train.")
        options = []
        for config in candidates(algorithm, request):
            update(
                None,
                f"{label} · cấu hình {len(options) + 1}: "
                f"{json.dumps(config, ensure_ascii=False, sort_keys=True)}",
            )
            update(8 + 85 * step / total_work, f"{label} · lựa chọn trên Validation")
            side_epoch = None
            if binary:
                builder, model, xs = feature_fit(
                    train,
                    [validation],
                    movies,
                    algorithm,
                    config,
                    report=lambda message: update(None, f"{NAMES[algorithm]} · {message}"),
                    **reduction_kwargs,
                )
                probability = classification_predict(builder, model, validation)
                threshold = max(
                    np.linspace(0.1, 0.9, 33),
                    key=lambda t: (
                        f1_score(validation.rating >= 3.5, probability >= t, zero_division=0),
                        -abs(t - 0.5),
                    ),
                )
                val_scores = classification_scores(validation.rating >= 3.5, probability, threshold)
                epoch = None
            else:
                model = fit_rating(
                    algorithm,
                    config,
                    request,
                    train,
                    movies,
                    validation,
                    progress=lambda epoch, total: update(
                        8 + 85 * (step + epoch / total) / total_work,
                        f"{NAMES[algorithm]} · huấn luyện/chọn trên Validation · "
                        f"{unit} {epoch}/{total}",
                    ),
                    log=lambda message: update(None, f"{NAMES[algorithm]} · {message}"),
                    **reduction_kwargs,
                )
                val_scores = regression_scores(
                    validation.rating, model.predict(validation.userId, validation.movieId)
                )
                threshold, epoch = 0.5, getattr(model, "best_epoch", request.epochs)
                side_epoch = getattr(model, "best_side_epoch", None)
            options.append(
                {
                    "config": config,
                    "cv": {},
                    "folds": [],
                    "validation": val_scores,
                    "threshold": float(threshold),
                    "epochs": epoch,
                    "side_epochs": side_epoch,
                    "learning_curve": getattr(model, "learning_curve", []),
                    "reduction": getattr(model, "reduction_metadata", None),
                }
            )
            score_name = "F1" if binary else "RMSE"
            score_value = val_scores["f1" if binary else "rmse"]
            update(
                None,
                f"{label} · Validation {score_name}={score_value:.6f}"
                + (f" · ngưỡng={threshold:.3f}" if binary else ""),
            )
            del model
            if binary:
                del builder, probability, xs
            gc.collect()
            step += 1
        key = "f1" if binary else "rmse"
        selected = sorted(
            options, key=lambda o: -o["validation"][key] if binary else o["validation"][key]
        )[0]
        update(
            None,
            f"{label} · đã chọn cấu hình trên Validation; "
            f"bắt đầu {CV_FOLDS} lượt CV thời gian.",
        )
        # Select on Validation first; CV is diagnostic, not another candidate search.
        fold_rows = []
        for index, (fit_idx, score_idx) in enumerate(temporal_folds(train, CV_FOLDS)):
            update(
                8 + 85 * step / total_work,
                f"{label} · CV thời gian {index + 1}/{CV_FOLDS}",
            )
            fit, held = train.iloc[fit_idx], train.iloc[score_idx]
            with measured() as timing:
                if binary:
                    builder, model, _ = feature_fit(
                        fit,
                        [held],
                        movies,
                        algorithm,
                        selected["config"],
                        report=lambda message: update(None, f"{NAMES[algorithm]} · {message}"),
                        **({"reduction_variant": reduction_variant,
                            "fit_scope": f"train_fold_{index + 1}"} if reduction_variant else {}),
                    )
                else:
                    model = fit_rating(
                        algorithm,
                        selected["config"],
                        request,
                        fit,
                        movies,
                        progress=lambda epoch, total: update(
                            8 + 85 * (step + epoch / total) / total_work,
                            f"{NAMES[algorithm]} · CV {index + 1}/{CV_FOLDS} · "
                            f"{unit} {epoch}/{total}",
                        ),
                        log=lambda message: update(None, f"{NAMES[algorithm]} · {message}"),
                        epochs=selected["epochs"],
                        **({"reduction_variant": reduction_variant,
                            "fit_scope": f"train_fold_{index + 1}"} if reduction_variant else {}),
                        **({"side_epochs": selected.get("side_epochs")} if selected.get("side_epochs") else {}),
                    )
            start = time.perf_counter()
            prediction = (
                classification_predict(builder, model, held)
                if binary
                else model.predict(held.userId, held.movieId)
            )
            inference_seconds = time.perf_counter() - start
            scores = (
                classification_scores(held.rating >= 3.5, prediction)
                if binary
                else regression_scores(held.rating, prediction)
            )
            fold_rows.append(
                {
                    "fold": index + 1,
                    **scores,
                    **timing,
                    "inference_seconds": inference_seconds,
                    "fit_ratings": len(fit),
                    "held_ratings": len(held),
                    "reduction": getattr(model, "reduction_metadata", None),
                }
            )
            update(
                None,
                f"{label} · xong CV {index + 1}/{CV_FOLDS} · "
                f"{score_name}={scores['f1' if binary else 'rmse']:.6f} · "
                f"fit {timing['fit_seconds']:.2f}s.",
            )
            del model, fit, held, fit_idx, score_idx, prediction
            if binary:
                del builder, _
            gc.collect()
            step += 1
        selected["cv"], selected["folds"] = numeric_summary(fold_rows), fold_rows
        tuning.extend({"algorithm": algorithm, **o} for o in options)
        cv[algorithm] = {"metrics": selected["cv"], "folds": selected["folds"]}
        best_configs[algorithm] = {
            **selected["config"],
            **({"reduction": selected["reduction"]} if selected.get("reduction") else {}),
            **({"epochs": selected["epochs"]} if algorithm in ("svd", "ncf") else {}),
            **({"side_epochs": selected["side_epochs"]}
               if algorithm == "ncf" and selected.get("side_epochs") else {}),
            **({"threshold": selected["threshold"]} if binary else {}),
        }
        rows.append(
            {
                "algorithm": algorithm,
                "name": NAMES[algorithm],
                "split": "validation",
                "reduction": selected.get("reduction"),
                **selected["validation"],
            }
        )
        update(
            8 + 85 * step / total_work,
            f"{NAMES[algorithm]} · refit Train + Validation và đánh giá Test",
        )
        combined = pd.concat([train, validation], ignore_index=True)
        update(None, f"{NAMES[algorithm]} · refit {len(combined):,} rating Train + Validation.")
        with measured() as timing:
            if binary:
                builder, final_model, xs = feature_fit(
                    combined,
                    [test],
                    movies,
                    algorithm,
                    selected["config"],
                    report=lambda message: update(None, f"{NAMES[algorithm]} · {message}"),
                    **({"reduction_variant": reduction_variant,
                        "fit_scope": "train_plus_validation"} if reduction_variant else {}),
                )
            else:
                final_model = fit_rating(
                    algorithm,
                    selected["config"],
                    request,
                    combined,
                    movies,
                    epochs=selected["epochs"],
                    progress=lambda epoch, total: update(
                        8 + 85 * (step + epoch / total) / total_work,
                        f"{NAMES[algorithm]} · refit Train + Validation · {unit} {epoch}/{total}",
                    ),
                    log=lambda message: update(None, f"{NAMES[algorithm]} · {message}"),
                    **({"reduction_variant": reduction_variant,
                        "fit_scope": "train_plus_validation"} if reduction_variant else {}),
                    **({"side_epochs": selected.get("side_epochs")}
                       if selected.get("side_epochs") else {}),
                )
        if reduction_variant is not None:
            best_configs[algorithm]["reduction"] = final_model.reduction_metadata
        start = time.perf_counter()
        predicted = (
            classification_predict(builder, final_model, test)
            if binary
            else final_model.predict(test.userId, test.movieId)
        )
        inference = time.perf_counter() - start
        scores = (
            classification_scores(test.rating >= 3.5, predicted, selected["threshold"])
            if binary
            else regression_scores(test.rating, predicted)
        )
        artifact = (builder, final_model) if binary else final_model
        rows.append(
            {
                "algorithm": algorithm,
                "name": NAMES[algorithm],
                "split": "test",
                "reduction": getattr(final_model, "reduction_metadata", None),
                **scores,
                **timing,
                "inference_seconds": inference,
                "inference_rows": len(test),
                "inference_us_per_row": inference / len(test) * 1e6,
                "model_bytes": serialized_size(artifact),
                "refit_ratings": len(combined),
            }
        )
        step += 1
        update(
            None,
            f"{NAMES[algorithm]} · Test {score_name}="
            f"{scores['f1' if binary else 'rmse']:.6f} · "
            f"fit cuối {timing['fit_seconds']:.2f}s · "
            f"RSS đỉnh {timing['peak_rss_mb']:.1f} MiB.",
        )
        if binary and len(np.unique(test.rating >= 3.5)) == 2:
            fpr, tpr, _ = roc_curve(test.rating >= 3.5, predicted)
            stride = max(1, len(fpr) // 300)
            charts.append(
                chart(
                    -1,
                    f"ROC · {label}",
                    "line",
                    "False Positive Rate",
                    "True Positive Rate",
                    [series(label, fpr[::stride], tpr[::stride])],
                    "test",
                )
            )
        if algorithm == "ncf" and selected["learning_curve"] and reduction_variant is not None:
            curve = selected["learning_curve"]
            charts.append(
                chart(
                    -1,
                    "NCF hybrid · Correction Branch / Early Stopping",
                    "line",
                    "Epoch",
                    "Correction MSE ↓",
                    [
                        series(
                            "Train correction MSE",
                            [r["epoch"] for r in curve],
                            [r["train_correction_mse"] for r in curve],
                        ),
                        series(
                            "Validation correction MSE",
                            [r["epoch"] for r in curve],
                            [r.get("validation_correction_mse") for r in curve],
                        ),
                    ],
                    "validation",
                )
            )
        elif algorithm == "ncf" and selected["learning_curve"]:
            curve = selected["learning_curve"]
            charts.append(
                chart(
                    -1,
                    "NCF · Rating MSE / Early Stopping",
                    "line",
                    "Epoch",
                    "Rating MSE ↓",
                    [
                        series(
                            "Train rating MSE",
                            [r["epoch"] for r in curve],
                            [r["train_rating_mse"] for r in curve],
                        ),
                        series(
                            "Validation rating MSE",
                            [r["epoch"] for r in curve],
                            [r["validation_rmse"] ** 2 for r in curve],
                        ),
                    ],
                    "validation",
                )
            )
        del final_model, artifact, combined, predicted
        if binary:
            del builder, xs
        gc.collect()
    winner = min(
        [r for r in rows if r["split"] == "validation"], key=lambda r: -r[key] if binary else r[key]
    )
    final = [r for r in rows if r["split"] == "test"]
    fastest = min(final, key=lambda r: r["inference_us_per_row"])
    chosen_final = next(r for r in final if r["algorithm"] == winner["algorithm"])
    chosen_cv = cv[winner["algorithm"]]["metrics"][key]
    smallest = min(final, key=lambda r: r["model_bytes"])
    notes = [
        "Chia thời gian trong từng user: lịch sử đầu Train, tiếp theo Validation, cuối Test; "
        "tỷ lệ xấp xỉ 70/15/15, giữ nguyên nhóm timestamp trùng nhau.",
        "User thiếu lịch sử hoặc không tách được ba khoảng thời gian được báo cáo riêng. "
        "Đây là đánh giá cá nhân hóa; timeline giữa các user có thể chồng lấn.",
        f"{CV_FOLDS} lượt CV dùng lịch sử mở rộng trong từng user trên Train; chỉ học quá khứ "
        "để đánh giá đoạn sau. F1 CV dùng ngưỡng 0,5 và epoch cố định.",
        "Validation chọn cấu hình/ngưỡng/epoch trước; chỉ chạy CV cho cấu hình được chọn. "
        "Refit Train + Validation, Test chỉ đánh giá cuối.",
        "Đặc trưng có tỷ lệ thích, độ phân tán rating và sở thích 20 rating gần nhất; "
        "thống kê chỉ từ phần fit, loại nhãn của chính hàng huấn luyện.",
        "Fit cuối gồm xây đặc trưng; inference phân loại gồm xây đặc trưng theo batch. "
        "RSS là bộ nhớ toàn tiến trình, kích thước model tính bằng luồng pickle.",
    ]
    if request.group == "rating" and "ncf" in request.algorithms and reduction_variant is not None:
        notes.append(
            "NCF hybrid giữ embedding và nhánh hiệu chỉnh hồ sơ nén riêng biệt. "
            "Validation chọn epoch của từng nhánh; refit Train + Validation khóa lại "
            "đúng hai số epoch đó. Đường học hiển thị MSE của nhánh hiệu chỉnh residual, "
            "không phải lỗi tổng của mô hình rating."
        )
    result = {
        "cv_protocol": "per_user_expanding_time_v1",
        "cv_folds": CV_FOLDS,
        "rows": rows,
        "charts": charts,
        "cv": cv,
        "best_configs": best_configs,
        "tuning": tuning,
        "notes": notes,
        "selection": f"Chọn bằng Validation: {winner['name']} ({key}={winner[key]:.4f}). "
        f"Khả năng tổng quát hóa: CV {key}={chosen_cv['mean']:.4f} ± {chosen_cv['std']:.4f}; "
        f"Test sau refit={chosen_final[key]:.4f}. "
        f"Mô hình được chọn cần {chosen_final['fit_seconds']:.3f}s fit, "
        f"{chosen_final['inference_us_per_row']:.2f} µs/rating và "
        f"{chosen_final['model_bytes'] / 1024:.1f} KiB lưu trữ. "
        f"Inference nhanh nhất: {fastest['name']}; model nhỏ nhất: {smallest['name']}. "
        "Ưu tiên mô hình Validation tốt nhất khi ngân sách đáp ứng các chi phí đo được; "
        "nếu cần tối ưu tài nguyên, xác nhận lại ứng viên nhỏ/nhanh trên một thí nghiệm mới. "
        "Không chọn lại bằng Test.",
    }
    if binary:
        result["class_balance"] = {
            s: {
                "negative": int((f.rating < 3.5).sum()),
                "positive": int((f.rating >= 3.5).sum()),
                "positive_fraction": float((f.rating >= 3.5).mean()),
            }
            for s, f in splits.items()
        }
    return result


def supervised(splits, movies, request, update):
    if request.reduction_mode == "off":
        return _supervised_single(splits, movies, request, update)

    spec = ReductionSpec(
        mode=request.reduction_mode,
        methods=tuple(request.reduction_methods),
        components=tuple(request.components),
        seed=request.seed,
        scale=True,
    )
    combined = {
        "cv_protocol": "per_user_expanding_time_v1",
        "cv_folds": CV_FOLDS,
        "rows": [],
        "charts": [],
        "cv": {},
        "best_configs": {},
        "tuning": [],
        "notes": [],
    }
    variants = reduction_variants(spec)
    for descriptor in variants:
        descriptor_id = descriptor["variant_id"]
        reducer = ({"method": descriptor["method"],
                    "components": descriptor["components"]}
                   if descriptor["method"] else None)
        update(None, f"Biểu diễn {descriptor_id} · cohort và folds giữ cố định.")
        result = _supervised_single(
            splits, movies, request, update,
            reduction_variant=reducer,
            variant_id=descriptor_id,
        )
        for row in result["rows"]:
            variant_id = f"{row['algorithm']}_{descriptor_id}"
            fit_scope = "train" if row["split"] == "validation" else "train_plus_validation"
            row.update(
                variant_id=variant_id,
                base_algorithm=row["algorithm"],
                representation="raw" if descriptor["method"] is None else "reduced_profile_interactions",
                reduction_method=descriptor["method"],
                components=descriptor["components"],
                feature_family="user_preference_movie_genre_interactions",
                fit_scope=fit_scope,
            )
        for algorithm, data in result["cv"].items():
            variant_id = f"{algorithm}_{descriptor_id}"
            combined["cv"][variant_id] = {
                **data,
                "variant_id": variant_id,
                "base_algorithm": algorithm,
                "representation": "raw" if descriptor["method"] is None else "reduced_profile_interactions",
                "reduction_method": descriptor["method"],
                "components": descriptor["components"],
                "feature_family": "user_preference_movie_genre_interactions",
                "fit_scope": "train_fold_specific",
            }
        for algorithm, config in result["best_configs"].items():
            variant_id = f"{algorithm}_{descriptor_id}"
            combined["best_configs"][variant_id] = {
                **config,
                "variant_id": variant_id,
                "base_algorithm": algorithm,
                "representation": "raw" if descriptor["method"] is None else "reduced_profile_interactions",
                "reduction_method": descriptor["method"],
                "components": descriptor["components"],
                "feature_family": "user_preference_movie_genre_interactions",
                "fit_scope": "train_plus_validation",
            }
        for option in result["tuning"]:
            algorithm = option["algorithm"]
            option["variant_id"] = f"{algorithm}_{descriptor_id}"
            option["base_algorithm"] = algorithm
            option["representation"] = "raw" if descriptor["method"] is None else "reduced_profile_interactions"
            option["reduction_method"] = descriptor["method"]
            option["components"] = descriptor["components"]
            option["feature_family"] = "user_preference_movie_genre_interactions"
            combined["tuning"].append(option)
        for item in result["charts"]:
            item["variant_id"] = descriptor_id
            item["title"] = f"{item['title']} · {descriptor_id}"
            combined["charts"].append(item)
        combined["rows"].extend(result["rows"])
        combined["notes"].extend(result["notes"])
        combined.setdefault("class_balance", result.get("class_balance"))

    binary = request.group == "classification"
    metric = "f1" if binary else "rmse"
    direction = max if binary else min
    validation_rows = [row for row in combined["rows"] if row["split"] == "validation"]
    winner = direction(validation_rows, key=lambda row: row[metric])
    combined["selection"] = (
        f"Biến thể được chọn bằng Validation: {winner['variant_id']} "
        f"({metric}={winner[metric]:.4f}). Test của từng biến thể được báo riêng; "
        "Test không tham gia chọn cấu hình, method hoặc số components."
    )
    combined["reduction"] = {
        "mode": request.reduction_mode,
        "variants": [
            {"variant_id": descriptor["variant_id"], "method": descriptor["method"],
             "components": descriptor["components"]}
            for descriptor in variants
        ],
        "selection_metric": metric,
        "selected_variant_id": winner["variant_id"],
        "feature_family": "user_preference_movie_genre_interactions",
    }
    return combined
