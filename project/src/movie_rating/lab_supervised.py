"""Explicit-rating and binary-classification benchmarks on shared temporal splits."""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from movie_rating.lab import (LinearRating, chart, cost_charts, latency, measured,
                              pair_features, series, summarize)
from movie_rating.models.baseline import BiasBaseline
from movie_rating.models.factorization import SVD
from movie_rating.models.neural import NeuralCF
from movie_rating.models.regression import XGBoostRegression


def rating_scores(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(abs(error))),
            'bias': float(np.mean(error))}


def by_value(actual, predicted, groups):
    frame = pd.DataFrame({'group': groups, 'error': np.asarray(predicted) - np.asarray(actual)})
    return [{'x': str(key), 'y': float(abs(part.error).mean()), 'count': len(part)}
            for key, part in frame.groupby('group', observed=True, sort=True)]


def ranking(model, train, heldout, seed, max_users=15):
    """Full ranking over trained items; identical sampled users across every model."""
    candidates = np.sort(train.movieId.unique())
    positive = heldout.loc[(heldout.rating >= 4) & heldout.movieId.isin(candidates)]
    eligible = np.sort(positive.userId.unique())
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(eligible, min(max_users, len(eligible)), replace=False))
    history = train.groupby('userId').movieId.agg(set).to_dict()
    ks = [1, 5, 10, 20]
    recall, ndcg, coverage, timings = {k: [] for k in ks}, {k: [] for k in ks}, set(), []
    for user in chosen:
        unseen = candidates[~np.isin(candidates, list(history.get(user, set())))]
        relevant = set(positive.loc[positive.userId == user, 'movieId']) & set(unseen)
        if not relevant or not len(unseen):
            continue
        start = time.perf_counter()
        scores = model.predict(np.full(len(unseen), user), unseen)
        ranked = unseen[np.argsort(-scores, kind='stable')[:20]]
        timings.append((time.perf_counter() - start) * 1000)
        coverage.update(ranked)
        for k in ks:
            hits = np.asarray([int(m in relevant) for m in ranked[:k]])
            recall[k].append(float(hits.sum() / len(relevant)))
            ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
            ndcg[k].append(float(np.sum(hits / np.log2(np.arange(len(hits)) + 2)) / ideal))
    return {'ks': ks, 'recall': [float(np.mean(recall[k])) if recall[k] else None for k in ks],
            'ndcg': [float(np.mean(ndcg[k])) if ndcg[k] else None for k in ks],
            'ranking_users': len(timings), 'candidate_movies': len(candidates),
            'coverage_at_20': len(coverage) / max(1, len(candidates)),
            'topn_p95_ms': float(np.percentile(timings, 95)) if timings else None}


def paired_intervals(frame, predictions, reference, seed, repetitions=200):
    users, inverse = np.unique(frame.userId, return_inverse=True)
    if len(users) < 2:
        return []
    count = np.bincount(inverse)
    truth = frame.rating.to_numpy()
    sums = {name: (np.bincount(inverse, weights=(pred - truth)**2),
                   np.bincount(inverse, weights=abs(pred - truth))) for name, pred in predictions.items()}
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(users), np.full(len(users), 1 / len(users)), size=repetitions)
    denominators = weights @ count
    result = []
    for name in predictions:
        if name == reference:
            continue
        for j, metric in enumerate(('rmse', 'mae')):
            a = weights @ sums[name][j] / denominators
            b = weights @ sums[reference][j] / denominators
            if metric == 'rmse':
                a, b = np.sqrt(a), np.sqrt(b)
            low, high = np.quantile(a - b, [0.025, 0.975])
            result.append({'algorithm': name, 'reference': reference, 'metric': metric,
                           'ci_low': low, 'ci_high': high, 'users': len(users),
                           'repetitions': repetitions, 'unit': 'paired user bootstrap; first model seed'})
    return result


def rating(splits, movies, request, update):
    train, validation = splits['train'], splits['validation']
    factories = [
        ('global_mean', 'Global Mean · baseline', lambda s: None),
        ('bias_baseline', 'Bias Baseline', lambda s: BiasBaseline(epochs=request.epochs, seed=s)),
        ('linear', 'Linear Regression', lambda s: LinearRating()),
        ('xgboost', 'XGBoost Regressor', lambda s: XGBoostRegression(n_estimators=70, n_jobs=2, seed=s)),
        ('svd', 'SVD/MF có bias', lambda s: SVD(epochs=request.epochs, factors=16, seed=s)),
        ('ncf', 'NCF hồi quy', lambda s: NeuralCF(epochs=request.epochs, factors=12, hidden_size=24, seed=s)),
    ]
    rows, charts, predictions = [], [], {'validation': {}, 'test': {}}
    ranking_data = {'validation': {}, 'test': {}}
    details = {'validation': {}, 'test': {}}
    curve = None
    for repeat in range(request.repeats):
        for index, (algorithm, name, factory) in enumerate(factories):
            update(8 + 82 * (repeat * len(factories) + index) / (request.repeats * len(factories)),
                   f'Huấn luyện {name} · seed {request.seed + repeat}')
            model = factory(request.seed + repeat)
            with measured() as timing:
                if model is not None:
                    if hasattr(model, 'set_movie_metadata'):
                        model.set_movie_metadata(movies)
                    if algorithm == 'ncf':
                        model.fit(train, validation=validation)
                    else:
                        model.fit(train)
            mean = float(train.rating.mean())

            def predict(frame):
                return model.predict(frame.userId, frame.movieId) if model is not None else np.full(len(frame), mean)

            sample = validation.iloc[:1]
            costs = {**timing, **latency(lambda: predict(sample)),
                     'model_bytes': len(pickle.dumps(model if model is not None else mean))}
            for split in ('validation', 'test'):
                frame = splits[split]
                pred = predict(frame)
                row = {'algorithm': algorithm, 'name': name, 'seed': request.seed + repeat,
                       'split': split, **rating_scores(frame.rating, pred), **costs}
                known = frame.userId.isin(train.userId.unique()) & frame.movieId.isin(train.movieId.unique())
                row['known_pair_fraction'] = float(known.mean())
                if repeat == 0:
                    predictions[split][algorithm] = pred
                    if model is not None:
                        ranks = ranking(model, train, frame, request.seed)
                        row.update(topn_p95_ms=ranks['topn_p95_ms'], ndcg_at_10=ranks['ndcg'][2],
                                   recall_at_10=ranks['recall'][2], ranking_users=ranks['ranking_users'],
                                   coverage_at_20=ranks['coverage_at_20'])
                        ranking_data[split][name] = ranks
                    details[split][algorithm] = by_value(frame.rating, pred, frame.rating)
                rows.append(row)
            if algorithm == 'ncf' and repeat == 0:
                curve = model.learning_curve
    summary = []
    for split in ('validation', 'test'):
        split_rows = summarize([r for r in rows if r['split'] == split])
        summary.extend(split_rows)
        for field in ('rmse', 'mae'):
            pts = [{'x': r['name'], 'y': r[field], **({'error': r['std'][field]} if field in r['std'] else {})} for r in split_rows]
            charts.append(chart(0, field.upper() + ' theo mô hình', 'bar', 'Mô hình', field.upper() + ' ↓',
                                [{'label': field.upper(), 'points': pts}], split))
        charts.append(chart(1, 'MAE theo rating thật', 'line', 'Rating thật', 'MAE ↓',
                            [{'label': name, 'points': [{'x': float(p['x']), 'y': p['y'], 'count': p['count']} for p in details[split][aid]]} for aid, name, _ in factories], split))
        bins = np.linspace(-4.5, 4.5, 19)
        charts.append(chart(3, 'Phân phối phần dư', 'line', 'Dự đoán − rating thật', 'Tỷ lệ',
                            [series(name, (bins[:-1] + bins[1:]) / 2,
                                    np.histogram(predictions[split][aid] - splits[split].rating.to_numpy(), bins=bins)[0] / len(splits[split])) for aid, name, _ in factories], split))
        counts = splits[split].userId.map(train.groupby('userId').size()).fillna(0)
        groups = pd.cut(counts, [-1, 0, 10, 50, 200, np.inf], labels=['0', '1–10', '11–50', '51–200', '>200'])
        charts.append(chart(4, 'MAE theo lịch sử user trong train', 'bar', 'Số rating train / user', 'MAE ↓',
                            [{'label': name, 'points': by_value(splits[split].rating, predictions[split][aid], groups)} for aid, name, _ in factories], split))
        for metric in ('ndcg', 'recall'):
            charts.append(chart(-2, metric.upper() + '@K · xếp hạng toàn bộ phim train chưa xem', 'line', 'K', metric.upper() + '@K ↑',
                                [series(name, ranks['ks'], ranks[metric]) for name, ranks in ranking_data[split].items()], split))
    if curve:
        charts.append(chart(2, 'Đường học NCF', 'line', 'Epoch', 'MSE ↓',
                            [series('Train MSE', [r['epoch'] for r in curve], [r['train_mse'] for r in curve]),
                             series('Validation MSE', [r['epoch'] for r in curve], [r['validation_rmse']**2 for r in curve])], 'train',
                            note='Train loss đo trên mini-batch có dropout, chưa clip; validation đo sau epoch, tắt dropout và clip. Chênh lệch hai đường không tự chứng minh overfitting.'))
    val_rows = [r for r in summary if r['split'] == 'validation']
    selected = min(val_rows, key=lambda r: r['rmse'])['algorithm']
    bootstrap = {s: paired_intervals(splits[s], predictions[s], selected, request.seed) for s in ('validation', 'test')}
    charts.extend(cost_charts(val_rows))
    return {'rows': summary, 'seed_rows': rows, 'charts': charts, 'bootstrap': bootstrap,
            'selection': f'Mô hình có RMSE validation trung bình thấp nhất: {selected}. Test không được dùng để chọn.',
            'notes': ['Linear/XGBoost dùng cùng nguồn rating và metadata thể loại; SVD/NCF ở cấu hình này chỉ dùng rating. Đây là so sánh các pipeline có biểu diễn và thông tin đầu vào khác nhau.',
                      'Chẩn đoán chi tiết và xếp hạng dùng seed đầu; bảng RMSE/MAE tổng hợp các seed. Chỉ có std khi chạy từ 2 seed.',
                      'p50/p95 đo 12 lần lặp của cùng một cặp user–movie sau warm-up; chưa mô tả tải đa người dùng. Top-N p95 đo riêng qua các user ranking.',
                      'Train features loại rating của chính mẫu khỏi thống kê; rating thiếu không được điền 0.',
                      'Top-N đánh giá tối đa 15 user có rating ≥ 4 giữ lại, trên toàn bộ phim train chưa có trong lịch sử train; các mục chưa có positive giữ lại được xử lý là không phù hợp theo giao thức offline.']}


def classification_metrics(y, probability, threshold):
    predicted = np.asarray(probability) >= threshold
    two_classes = len(np.unique(y)) == 2
    return {'macro_f1': f1_score(y, predicted, average='macro', labels=[0, 1], zero_division=0),
            'precision': precision_score(y, predicted, zero_division=0),
            'recall': recall_score(y, predicted, zero_division=0),
            'accuracy': accuracy_score(y, predicted), 'balanced_accuracy': balanced_accuracy_score(y, predicted),
            'roc_auc': roc_auc_score(y, probability) if two_classes else None,
            'average_precision': average_precision_score(y, probability) if np.any(y) else None}


def classification(splits, movies, request, update):
    train = splits['train']
    update(8, 'Xây dựng đặc trưng từ train, loại target của từng mẫu')
    builder, x_train, matrices = pair_features(train, [splits['validation'], splits['test']], movies)
    labels = {s: (f.rating.to_numpy() >= request.positive_threshold).astype(int) for s, f in splits.items()}
    if len(np.unique(labels['train'])) < 2:
        raise ValueError('Train chỉ có một lớp với ngưỡng đã chọn. Hãy tăng mẫu hoặc đổi ngưỡng rating.')
    rows, charts = [], []
    predictions = {'validation': {}, 'test': {}}
    factories = [('majority', 'Lớp đa số · baseline', lambda seed: None),
                 ('logistic', 'Logistic Regression', lambda seed: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))),
                 ('random_forest', 'Random Forest Classifier', lambda seed: RandomForestClassifier(n_estimators=90, min_samples_leaf=3, n_jobs=2, random_state=seed))]
    for repeat in range(request.repeats):
        for index, (algorithm, name, factory) in enumerate(factories):
            update(15 + 73 * (repeat * 3 + index) / (request.repeats * 3), f'Huấn luyện {name} · seed {request.seed + repeat}')
            estimator = factory(request.seed + repeat)
            with measured() as timing:
                if estimator is not None:
                    estimator.fit(x_train, labels['train'])
            p_train = float(labels['train'].mean())
            probs = [estimator.predict_proba(x)[:, 1] if estimator is not None else np.full(len(x), p_train) for x in matrices]
            if estimator is None:
                threshold = 0.5
            else:
                thresholds = np.linspace(0.1, 0.9, 33)
                threshold = float(max(thresholds, key=lambda t: f1_score(labels['validation'], probs[0] >= t, labels=[0, 1], average='macro', zero_division=0)))
            costs = {**timing, 'model_bytes': len(pickle.dumps((builder, estimator) if estimator is not None else p_train)),
                     **latency(lambda: estimator.predict_proba(matrices[0][:1]) if estimator is not None else np.array([p_train]))}
            for split, probability in zip(('validation', 'test'), probs):
                rows.append({'algorithm': algorithm, 'name': name, 'split': split,
                             'seed': request.seed + repeat, 'threshold': threshold,
                             **classification_metrics(labels[split], probability, threshold), **costs})
                if repeat == 0:
                    predictions[split][algorithm] = probability
                    matrix = confusion_matrix(labels[split], probability >= threshold, labels=[0, 1])
                    charts.append(chart(0, f'Ma trận nhầm lẫn · {name}', 'heatmap', 'Nhãn dự đoán', 'Nhãn thật', split=split,
                                        matrix=matrix.tolist(), row_labels=['Không thích', 'Thích'], column_labels=['Không thích', 'Thích'], normalize_rows=True))
    summary = []
    for split in ('validation', 'test'):
        values = summarize([r for r in rows if r['split'] == split])
        summary.extend(values)
        charts.append(chart(1, 'Precision / Recall / Macro-F1', 'bar', 'Mô hình', 'Điểm ↑',
                            [series(field, [r['name'] for r in values], [r[field] for r in values]) for field in ('precision', 'recall', 'macro_f1')], split))
        pr_series, roc_series = [], []
        for algorithm, name, _ in factories:
            probability = predictions[split][algorithm]
            if len(np.unique(labels[split])) != 2:
                continue
            p, r, _ = precision_recall_curve(labels[split], probability)
            chosen = np.unique(np.linspace(0, len(p) - 1, min(120, len(p))).astype(int))
            pr_series.append(series(name, r[chosen][::-1], p[chosen][::-1]))
            fpr, tpr, _ = roc_curve(labels[split], probability)
            chosen = np.unique(np.linspace(0, len(fpr) - 1, min(120, len(fpr))).astype(int))
            roc_series.append(series(name, fpr[chosen], tpr[chosen]))
        pr_series.append(series('Tỷ lệ lớp thích', [0, 1], [float(labels[split].mean())] * 2))
        charts.append(chart(2, 'Precision–Recall curve', 'line', 'Recall', 'Precision', pr_series, split))
        roc_series.append(series('Ngẫu nhiên', [0, 1], [0, 1]))
        charts.append(chart(3, 'ROC curve', 'line', 'False Positive Rate', 'True Positive Rate', roc_series, split))
    val_rows = [r for r in summary if r['split'] == 'validation']
    selected = max(val_rows, key=lambda r: r['macro_f1'])['algorithm']
    charts.extend(cost_charts(val_rows))
    return {'rows': summary, 'seed_rows': rows, 'charts': charts,
            'class_balance': {s: {'positive': int(y.sum()), 'negative': int(len(y)-y.sum())} for s, y in labels.items()},
            'selection': f'Chọn theo Macro-F1 validation: {selected}. Ngưỡng xác suất của mỗi seed được khóa trước test.',
            'notes': ['Nhãn thích được tạo từ ngưỡng rating đã công bố; rating chưa quan sát không được coi là không thích.',
                      'AP tính bằng average_precision_score, không thay bằng diện tích PR nội suy. AUC không xác định khi tập đánh giá chỉ có một lớp.',
                      'Độ trễ phân loại chỉ đo estimator trên vector đặc trưng đã tạo, chưa gồm dựng đặc trưng từ user/movie. Kích thước hiện vật gồm cả bộ dựng đặc trưng cần cho pipeline.',
                      'Confusion matrix ghi số lượng; tỷ lệ chuẩn hóa theo hàng được hiển thị kèm. Đường cong và ma trận dùng seed đầu; điểm tổng hợp dùng tất cả seed.']}
