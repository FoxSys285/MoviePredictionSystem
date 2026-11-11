"""Five train-only folds, validation selection, frozen refit and final test."""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from movie_rating.lab import LinearRating, chart, measured, pair_features, series
from movie_rating.lab_compact import regression_scores
from movie_rating.models.factorization import SVD
from movie_rating.models.neural import NeuralCF
from movie_rating.models.regression import XGBoostRegression

NAMES = {
    "linear": "Linear Regression",
    "xgboost": "XGBoost Regressor",
    "svd": "SVD Collaborative Filtering",
    "ncf": "Neural Collaborative Filtering",
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest Classifier",
}


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
            n_estimators=p.n_estimators, max_depth=p.max_depth, learning_rate=p.learning_rate
        ),
        "svd": dict(
            factors=p.factors, learning_rate=p.learning_rate, regularization=p.regularization
        ),
        "ncf": dict(
            factors=p.factors,
            hidden_size=p.hidden_size,
            learning_rate=p.learning_rate,
            regularization=p.regularization,
            dropout=p.dropout,
            batch_size=p.batch_size,
        ),
        "logistic": dict(C=p.c),
        "random_forest": dict(
            n_estimators=p.n_estimators, max_depth=p.max_depth, min_samples_leaf=p.min_samples_leaf
        ),
    }[algorithm]
    output = [base]
    if request.tune and algorithm != "linear":
        other = dict(base)
        if algorithm in ("svd", "ncf"):
            other["factors"] = max(2, p.factors // 2)
        elif algorithm == "logistic":
            other["C"] = p.c / 10
        else:
            other["max_depth"] = max(1, p.max_depth // 2)
        if other != base:
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


def fit_rating(algorithm, config, request, train, movies, validation=None, epochs=None):
    model = rating_model(algorithm, config, request, epochs)
    if hasattr(model, "set_movie_metadata"):
        model.set_movie_metadata(movies)
    if algorithm == "ncf":
        model.fit(train, validation=validation)
    else:
        model.fit(train)
    return model


def classifier(algorithm, config):
    if algorithm == "logistic":
        return make_pipeline(
            StandardScaler(), LogisticRegression(**config, max_iter=600, random_state=42)
        )
    return RandomForestClassifier(**config, n_jobs=2, random_state=42)


def feature_fit(train, targets, movies, algorithm, config):
    builder, x, matrices = pair_features(train, targets, movies)
    estimator = classifier(algorithm, config).fit(x, (train.rating >= 3.5).astype(int))
    return builder, estimator, matrices


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


def supervised(splits, movies, request, update):
    train, validation, test = [splits[s] for s in ("train", "validation", "test")]
    binary = request.group == "classification"
    y = (train.rating >= 3.5).astype(int).to_numpy()
    if binary and np.bincount(y, minlength=2).min() < 5:
        raise ValueError("Mỗi lớp trong Train cần ít nhất 5 ratings cho Stratified 5-Fold.")
    splitter = (
        StratifiedKFold(5, shuffle=True, random_state=42)
        if binary
        else KFold(5, shuffle=True, random_state=42)
    )
    folds = list(splitter.split(train, y if binary else None))
    rows, charts, cv, best_configs, tuning = [], [], {}, {}, []
    all_candidates = sum(len(candidates(a, request)) for a in request.algorithms)
    total_work = all_candidates * 6 + len(request.algorithms)
    step = 0
    for algorithm in request.algorithms:
        options = []
        for config in candidates(algorithm, request):
            fold_rows = []
            for index, (fit_idx, score_idx) in enumerate(folds):
                update(
                    8 + 85 * step / total_work,
                    f"{NAMES[algorithm]} · 5-Fold {index + 1}/5 · cấu hình {len(options) + 1}",
                )
                fit, held = train.iloc[fit_idx], train.iloc[score_idx]
                with measured() as timing:
                    if binary:
                        _, model, xs = feature_fit(fit, [held], movies, algorithm, config)
                    else:
                        model = fit_rating(algorithm, config, request, fit, movies)
                start = time.perf_counter()
                prediction = (
                    model.predict_proba(xs[0])[:, 1]
                    if binary
                    else model.predict(held.userId, held.movieId)
                )
                seconds = time.perf_counter() - start
                scores = (
                    classification_scores((held.rating >= 3.5).astype(int), prediction)
                    if binary
                    else regression_scores(held.rating, prediction)
                )
                fold_rows.append(
                    {"fold": index + 1, **scores, **timing, "inference_seconds": seconds}
                )
                step += 1
            update(8 + 85 * step / total_work, f"{NAMES[algorithm]} · lựa chọn trên Validation")
            if binary:
                builder, model, xs = feature_fit(train, [validation], movies, algorithm, config)
                probability = model.predict_proba(xs[0])[:, 1]
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
                model = fit_rating(algorithm, config, request, train, movies, validation)
                val_scores = regression_scores(
                    validation.rating, model.predict(validation.userId, validation.movieId)
                )
                threshold, epoch = 0.5, getattr(model, "best_epoch", request.epochs)
            options.append(
                {
                    "config": config,
                    "cv": numeric_summary(fold_rows),
                    "folds": fold_rows,
                    "validation": val_scores,
                    "threshold": float(threshold),
                    "epochs": epoch,
                    "learning_curve": getattr(model, "learning_curve", []),
                }
            )
            step += 1
        key = "f1" if binary else "rmse"
        selected = sorted(
            options, key=lambda o: -o["validation"][key] if binary else o["validation"][key]
        )[0]
        tuning.extend({"algorithm": algorithm, **o} for o in options)
        cv[algorithm] = {"metrics": selected["cv"], "folds": selected["folds"]}
        best_configs[algorithm] = {
            **selected["config"],
            **({"epochs": selected["epochs"]} if algorithm in ("svd", "ncf") else {}),
            **({"threshold": selected["threshold"]} if binary else {}),
        }
        rows.append(
            {
                "algorithm": algorithm,
                "name": NAMES[algorithm],
                "split": "validation",
                **selected["validation"],
            }
        )
        update(
            8 + 85 * step / total_work,
            f"{NAMES[algorithm]} · refit Train + Validation và đánh giá Test",
        )
        combined = pd.concat([train, validation], ignore_index=True)
        with measured() as timing:
            if binary:
                builder, final_model, xs = feature_fit(
                    combined, [test], movies, algorithm, selected["config"]
                )
            else:
                final_model = fit_rating(
                    algorithm,
                    selected["config"],
                    request,
                    combined,
                    movies,
                    epochs=selected["epochs"],
                )
        start = time.perf_counter()
        predicted = (
            final_model.predict_proba(xs[0])[:, 1]
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
                **scores,
                **timing,
                "inference_seconds": inference,
                "inference_rows": len(test),
                "inference_us_per_row": inference / len(test) * 1e6,
                "model_bytes": len(pickle.dumps(artifact)),
                "refit_ratings": len(combined),
            }
        )
        step += 1
        if binary and len(np.unique(test.rating >= 3.5)) == 2:
            fpr, tpr, _ = roc_curve(test.rating >= 3.5, predicted)
            stride = max(1, len(fpr) // 300)
            charts.append(
                chart(
                    -1,
                    f"ROC · {NAMES[algorithm]}",
                    "line",
                    "False Positive Rate",
                    "True Positive Rate",
                    [series(NAMES[algorithm], fpr[::stride], tpr[::stride])],
                    "test",
                )
            )
        if selected["learning_curve"]:
            curve = selected["learning_curve"]
            charts.append(
                chart(
                    -1,
                    "NCF · Learning Curve / Early Stopping",
                    "line",
                    "Epoch",
                    "MSE ↓",
                    [
                        series(
                            "Train Loss",
                            [r["epoch"] for r in curve],
                            [r["train_rating_mse"] for r in curve],
                        ),
                        series(
                            "Validation Loss",
                            [r["epoch"] for r in curve],
                            [r["validation_rmse"] ** 2 for r in curve],
                        ),
                    ],
                    "validation",
                )
            )
    winner = min(
        [r for r in rows if r["split"] == "validation"], key=lambda r: -r[key] if binary else r[key]
    )
    final = [r for r in rows if r["split"] == "test"]
    fastest = min(final, key=lambda r: r["inference_us_per_row"])
    chosen_final = next(r for r in final if r["algorithm"] == winner["algorithm"])
    chosen_cv = cv[winner["algorithm"]]["metrics"][key]
    smallest = min(final, key=lambda r: r["model_bytes"])
    notes = [
        "5-Fold chỉ trên Train 70%; mỗi fold tạo lại thống kê rating, "
        "đặc trưng và scaler từ phần fit của fold.",
        "CV dùng ngưỡng xác suất 0,5 và ngân sách epoch cố định. "
        "Validation chọn cấu hình/ngưỡng/epoch; refit 85% không early-stop trên Test.",
        "Thời gian fit cuối gồm xây đặc trưng; inference đo cả batch Test "
        "(không gồm xây đặc trưng phân loại). Model size là số byte pickle trong RAM.",
        "IID ratings đo nội suy sở thích, không chứng minh dự đoán tương lai hoặc cold-start. "
        "Các fold có thể chứa cùng user/phim.",
    ]
    result = {
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
