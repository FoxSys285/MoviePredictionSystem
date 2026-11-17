"""Three train-only folds, validation selection, frozen refit and final test."""

import gc
import json
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from movie_rating.data.temporal import temporal_folds
from movie_rating.lab import LinearRating, chart, measured, series
from movie_rating.lab_compact import regression_scores
from movie_rating.memory import BATCH_ROWS, feature_buffer, serialized_size
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


def fit_rating(
    algorithm, config, request, train, movies, validation=None, epochs=None, progress=None, log=None
):
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


def feature_fit(train, targets, movies, algorithm, config, report=None):
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
        if report:
            report(
                f"Bắt đầu tối ưu Logistic bằng SAGA · C={config['C']} · "
                f"tối đa {config['max_iter']} vòng."
                if algorithm == "logistic"
                else f"Bắt đầu Random Forest · {config['n_estimators']} cây · "
                f"depth={config['max_depth']} · min_samples_leaf={config['min_samples_leaf']}."
            )
        estimator.fit(features, (train.rating >= 3.5).to_numpy(dtype=np.int8))
        if report:
            report(
                f"Đã fit Logistic · {int(estimator.n_iter_.max())} vòng tối ưu."
                if algorithm == "logistic"
                else f"Đã fit xong {len(estimator.estimators_)} cây Random Forest."
            )
    if scaler is not None:
        # Prediction receives freshly built batches, so scaler can transform in place.
        estimator = make_pipeline(scaler, estimator)
    return builder, estimator, targets


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


def supervised(splits, movies, request, update):
    train, validation, test = [splits[s] for s in ("train", "validation", "test")]
    binary = request.group == "classification"
    rows, charts, cv, best_configs, tuning = [], [], {}, {}, []
    all_candidates = sum(len(candidates(a, request)) for a in request.algorithms)
    total_work = all_candidates + len(request.algorithms) * (CV_FOLDS + 1)
    step = 0
    for algorithm in request.algorithms:
        unit = "cây" if algorithm == "xgboost" else "rating" if algorithm == "linear" else "epoch"
        update(None, f"Bắt đầu {NAMES[algorithm]} · {len(train):,} rating Train.")
        options = []
        for config in candidates(algorithm, request):
            update(
                None,
                f"{NAMES[algorithm]} · cấu hình {len(options) + 1}: "
                f"{json.dumps(config, ensure_ascii=False, sort_keys=True)}",
            )
            update(8 + 85 * step / total_work, f"{NAMES[algorithm]} · lựa chọn trên Validation")
            if binary:
                builder, model, xs = feature_fit(
                    train,
                    [validation],
                    movies,
                    algorithm,
                    config,
                    report=lambda message: update(None, f"{NAMES[algorithm]} · {message}"),
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
                )
                val_scores = regression_scores(
                    validation.rating, model.predict(validation.userId, validation.movieId)
                )
                threshold, epoch = 0.5, getattr(model, "best_epoch", request.epochs)
            options.append(
                {
                    "config": config,
                    "cv": {},
                    "folds": [],
                    "validation": val_scores,
                    "threshold": float(threshold),
                    "epochs": epoch,
                    "learning_curve": getattr(model, "learning_curve", []),
                }
            )
            score_name = "F1" if binary else "RMSE"
            score_value = val_scores["f1" if binary else "rmse"]
            update(
                None,
                f"{NAMES[algorithm]} · Validation {score_name}={score_value:.6f}"
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
            f"{NAMES[algorithm]} · đã chọn cấu hình trên Validation; "
            f"bắt đầu {CV_FOLDS} lượt CV thời gian.",
        )
        # Select on Validation first; CV is diagnostic, not another candidate search.
        fold_rows = []
        for index, (fit_idx, score_idx) in enumerate(temporal_folds(train, CV_FOLDS)):
            update(
                8 + 85 * step / total_work,
                f"{NAMES[algorithm]} · CV thời gian {index + 1}/{CV_FOLDS}",
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
                }
            )
            update(
                None,
                f"{NAMES[algorithm]} · xong CV {index + 1}/{CV_FOLDS} · "
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
                )
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
