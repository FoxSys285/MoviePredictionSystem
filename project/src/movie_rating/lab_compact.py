"""One requested metric per chart, with honest held-out validation/test scores."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, calinski_harabasz_score,
                             davies_bouldin_score, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from movie_rating.lab import CATALOG, LinearRating, chart, pair_features, series, summarize
from movie_rating.models.factorization import SVD
from movie_rating.models.neural import NeuralCF
from movie_rating.models.regression import XGBoostRegression


def regression_scores(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not len(actual):
        return dict.fromkeys(('mae', 'mse', 'rmse', 'r2'))
    error = predicted-actual
    mse = float(np.mean(error**2))
    total = float(np.sum((actual-actual.mean())**2))
    return {'mae': float(np.mean(abs(error))), 'mse': mse, 'rmse': float(np.sqrt(mse)),
            'r2': float(1-np.sum(error**2)/total) if len(actual) > 1 and total > 0 else None}


def comparison_charts(group_id, rows):
    group = next(g for g in CATALOG['groups'] if g['id'] == group_id)
    algorithms = list(dict.fromkeys((r['algorithm'], r['name']) for r in rows))
    lookup = {(r['algorithm'], r['split']): r for r in rows}
    return [chart(i, metric['name'], 'bar', 'Thuật toán / cấu hình', metric['name'],
                  [series(split.title(), [name for _, name in algorithms],
                          [lookup.get((aid, split), {}).get(metric['key']) for aid, _ in algorithms])
                   for split in ('validation', 'test')], 'comparison', metric_key=metric['key'])
            for i, metric in enumerate(group['metrics'])]


def report(group, rows, notes=()):
    metrics = next(g['metrics'] for g in CATALOG['groups'] if g['id'] == group)
    allowed = ('algorithm', 'name', 'split', *(m['key'] for m in metrics))
    rows = [{key: row.get(key) for key in allowed} for row in rows]
    return {'schema_version': 2, 'rows': rows, 'charts': comparison_charts(group, rows), 'notes': list(notes)}


def rating(splits, movies, request, update):
    train = splits['train']
    models = [
        ('linear', 'Linear Regression', lambda seed: LinearRating()),
        ('xgboost', 'XGBoost Regressor', lambda seed: XGBoostRegression(n_estimators=70, n_jobs=2, seed=seed)),
        ('svd', 'SVD/MF có bias', lambda seed: SVD(epochs=request.epochs, factors=16, seed=seed)),
        ('ncf', 'NCF hồi quy', lambda seed: NeuralCF(epochs=request.epochs, factors=12, hidden_size=24, seed=seed)),
    ]
    seed_rows = []
    for repeat in range(request.repeats):
        for index, (aid, name, factory) in enumerate(models):
            update(10+80*(repeat*4+index)/(request.repeats*4), f'Huấn luyện {name}')
            model = factory(request.seed+repeat)
            if hasattr(model, 'set_movie_metadata'):
                model.set_movie_metadata(movies)
            if aid == 'ncf':
                model.fit(train, validation=splits['validation'])
            else:
                model.fit(train)
            for split in ('validation', 'test'):
                frame = splits[split]
                predicted = model.predict(frame.userId, frame.movieId)
                seed_rows.append({'algorithm': aid, 'name': name, 'split': split,
                                  **regression_scores(frame.rating, predicted)})
    rows = [r for split in ('validation', 'test') for r in summarize([r for r in seed_rows if r['split'] == split])]
    return report('rating', rows, ['Linear/XGBoost dùng thêm thể loại; SVD/NCF chỉ dùng rating. R² có thể âm; N/A khi rating thật không có phương sai.'])


def classification(splits, movies, request, update):
    update(8, 'Tạo đặc trưng từ train')
    _, x_train, heldouts = pair_features(splits['train'], [splits['validation'], splits['test']], movies)
    labels = {s: (f.rating.to_numpy() >= request.positive_threshold).astype(int) for s, f in splits.items()}
    if len(np.unique(labels['train'])) < 2:
        raise ValueError('Train chỉ có một lớp. Hãy tăng mẫu hoặc đổi ngưỡng rating thích.')
    seed_rows = []
    for repeat in range(request.repeats):
        estimators = [
            ('logistic', 'Logistic Regression', make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=request.seed+repeat))),
            ('random_forest', 'Random Forest Classifier', RandomForestClassifier(n_estimators=90, min_samples_leaf=3, n_jobs=2, random_state=request.seed+repeat)),
        ]
        for index, (aid, name, estimator) in enumerate(estimators):
            update(15+70*(repeat*2+index)/(request.repeats*2), f'Huấn luyện {name}')
            estimator.fit(x_train, labels['train'])
            probabilities = [estimator.predict_proba(x)[:, 1] for x in heldouts]
            thresholds = np.linspace(.1, .9, 33)
            threshold = max(thresholds, key=lambda t: (f1_score(labels['validation'], probabilities[0] >= t, zero_division=0), -abs(t-.5)))
            for split, probability in zip(('validation', 'test'), probabilities):
                predicted, actual = probability >= threshold, labels[split]
                seed_rows.append({'algorithm': aid, 'name': name, 'split': split,
                                  'accuracy': accuracy_score(actual, predicted),
                                  'precision': precision_score(actual, predicted, zero_division=0),
                                  'recall': recall_score(actual, predicted, zero_division=0),
                                  'f1': f1_score(actual, predicted, zero_division=0)})
    rows = [r for split in ('validation', 'test') for r in summarize([r for r in seed_rows if r['split'] == split])]
    return report('classification', rows, ['Precision, Recall, F1 tính cho lớp thích. Ngưỡng xác suất được chọn theo F1 validation và giữ nguyên khi đánh giá test.'])


def object_matrix(frame, movies, request, genres, eligible_movies=None):
    metadata = movies.set_index('movieId')
    key = 'movieId' if request.cluster_object == 'movies' else 'userId'
    stats = frame.groupby(key).rating.agg(['mean', 'count', 'std']).fillna(0)
    if key == 'movieId':
        if eligible_movies is None:
            stats = stats.sort_values('count', ascending=False, kind='stable').head(request.max_items)
        else:
            stats = stats.reindex([m for m in eligible_movies if m in stats.index])
        values = [[float(g in str(metadata.loc[m, 'genres']).split('|')) for g in genres] for m in stats.index]
    else:
        genre_map = {int(m): set(str(s).split('|')) for m, s in zip(movies.movieId, movies.genres)}
        histories = {u: f for u, f in frame.groupby('userId')}
        values = [[float(np.mean([g in genre_map[int(m)] for m in histories[u].movieId])) for g in genres] for u in stats.index]
    x = np.column_stack([np.asarray(values).reshape(len(stats), len(genres)), stats['mean'], np.log1p(stats['count']), stats['std']])
    return x, stats.index.to_numpy()


def clustering_scores(x, labels):
    mask = labels >= 0
    count = len(set(labels[mask]))
    if count < 2 or mask.sum() <= count:
        return dict.fromkeys(('silhouette', 'chi', 'dbi'))
    return {'silhouette': float(silhouette_score(x[mask], labels[mask])),
            'chi': float(calinski_harabasz_score(x[mask], labels[mask])),
            'dbi': float(davies_bouldin_score(x[mask], labels[mask]))}


def assign_dbscan(model, x):
    if not len(x) or not len(model.core_sample_indices_):
        return np.full(len(x), -1, dtype=int)
    distances, indices = NearestNeighbors(n_neighbors=1).fit(model.components_).kneighbors(x)
    labels = model.labels_[model.core_sample_indices_[indices[:, 0]]].copy()
    labels[distances[:, 0] > model.eps] = -1
    return labels


def clustering(splits, movies, request, update):
    genres = sorted({g for s in movies.genres.fillna('') for g in s.split('|') if g and g != '(no genres listed)'})
    raw, ids = object_matrix(splits['train'], movies, request, genres)
    if len(raw) < 4:
        raise ValueError('Cần ít nhất 4 đối tượng trong train để phân cụm.')
    scaler = StandardScaler().fit(raw)
    train_x = scaler.transform(raw)
    heldouts = {}
    for split in ('validation', 'test'):
        x, _ = object_matrix(splits[split], movies, request, genres, ids if request.cluster_object == 'movies' else None)
        heldouts[split] = scaler.transform(x) if len(x) else x
    seed_rows = []
    for repeat in range(request.repeats):
        models = [
            ('kmeans', 'K-Means', KMeans(n_clusters=min(request.n_clusters, len(raw)-1), n_init=10, random_state=request.seed+repeat)),
            ('dbscan', 'DBSCAN', DBSCAN(eps=request.eps, min_samples=request.min_samples)),
        ]
        for index, (aid, name, model) in enumerate(models):
            update(15+70*(repeat*2+index)/(request.repeats*2), f'Phân cụm {name}')
            model.fit(train_x)
            for split, x in heldouts.items():
                assigned = (model.predict(x) if len(x) else np.array([], dtype=int)) if aid == 'kmeans' else assign_dbscan(model, x)
                seed_rows.append({'algorithm': aid, 'name': name, 'split': split, **clustering_scores(x, assigned)})
    rows = [r for split in ('validation', 'test') for r in summarize([r for r in seed_rows if r['split'] == split])]
    return report('clustering', rows, ['Mô hình và scaler chỉ fit trên train. Vector valid/test được tạo riêng theo cùng đặc trưng. DBSCAN gán theo core train gần nhất trong eps; điểm nhiễu không tính chỉ số. N/A khi không đủ hai cụm hợp lệ.'])


def rule_scores(transactions, rules):
    if not rules or not len(transactions):
        return dict.fromkeys(('support', 'confidence', 'lift'))
    supports, confidences, lifts = [], [], []
    for antecedents, consequents in rules:
        a = transactions.loc[:, list(antecedents)].all(axis=1).to_numpy()
        b = transactions.loc[:, list(consequents)].all(axis=1).to_numpy()
        support = float(np.mean(a & b))
        supports.append(support)
        if a.any():
            confidence = support/float(a.mean())
            confidences.append(confidence)
            if b.any():
                lifts.append(confidence/float(b.mean()))
    return {'support': float(np.mean(supports)),
            'confidence': float(np.mean(confidences)) if confidences else None,
            'lift': float(np.mean(lifts)) if lifts else None}


def association(splits, movies, request, update):
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

    positive = splits['train'].loc[splits['train'].rating >= request.positive_threshold]
    items = list(positive.groupby('movieId').size().sort_values(ascending=False, kind='stable').head(request.max_items).index)
    if not items:
        raise ValueError('Không có phim tích cực trong train với ngưỡng rating đã chọn.')
    transactions = {}
    for split, frame in splits.items():
        positive = frame.loc[frame.rating >= request.positive_threshold]
        transactions[split] = pd.crosstab(positive.userId, positive.movieId).reindex(index=np.sort(frame.userId.unique()), columns=items, fill_value=0).fillna(0).astype(bool)
    rows = []
    for index, (aid, name, miner) in enumerate([('apriori', 'Apriori', apriori), ('fp_growth', 'FP-Growth', fpgrowth)]):
        update(15+index*40, f'Khai phá {name} trên train')
        patterns = miner(transactions['train'], min_support=request.min_support, use_colnames=True, max_len=2)
        generated = association_rules(patterns, num_itemsets=len(transactions['train']), metric='confidence', min_threshold=.3) if len(patterns) else pd.DataFrame()
        rules = list(zip(generated.antecedents, generated.consequents)) if len(generated) else []
        for split in ('validation', 'test'):
            rows.append({'algorithm': aid, 'name': name, 'split': split, **rule_scores(transactions[split], rules)})
    return report('association', rows, ['Chỉ số là trung bình trên tập luật cố định khai phá từ train (confidence ≥ 0,3, tối đa 2 phim/luật). Confidence/Lift không xác định được bỏ qua khi lấy trung bình. Cùng ngưỡng, Apriori và FP-Growth cho cùng tập luật nên các chỉ số có thể bằng nhau.'])


def pca_experiment(splits, movies, request, update):
    update(8, 'Tạo đặc trưng và chuẩn hóa bằng train')
    _, raw, heldouts = pair_features(splits['train'], [splits['validation'], splits['test']], movies)
    scaler = StandardScaler().fit(raw)
    x = scaler.transform(raw)
    maximum = min(x.shape[1], len(x)-1)
    choices = sorted({k for k in [1, 2, 4, 8, 16, maximum] if 0 < k <= maximum})
    rows = []
    for index, k in enumerate([0, *choices]):
        update(15+75*index/(len(choices)+1), f'PCA · {k} chiều' if k else 'Baseline không PCA')
        transform = PCA(n_components=k, svd_solver='full').fit(x) if k else None
        z = transform.transform(x) if transform is not None else x
        estimator = Ridge(alpha=1).fit(z, splits['train'].rating)
        for split, original in zip(('validation', 'test'), heldouts):
            input_x = scaler.transform(original)
            z_hold = transform.transform(input_x) if transform is not None else input_x
            prediction = np.clip(estimator.predict(z_hold), .5, 5)
            rows.append({'algorithm': f'pca_{k}' if k else 'no_pca',
                         'name': f'PCA · {k} chiều' if k else 'Không PCA', 'split': split,
                         'explained_variance_ratio': float(transform.explained_variance_ratio_.sum()) if transform is not None else 1.,
                         'reconstruction_error': float(np.mean((input_x-transform.inverse_transform(z_hold))**2)) if transform is not None else 0.,
                         'downstream_performance': regression_scores(splits[split].rating, prediction)['rmse']})
    return report('pca', rows, ['EVR được học trên train và dùng chung cho valid/test. Reconstruction Error tính trên đặc trưng chuẩn hóa. Downstream Model Performance là RMSE của cùng Ridge alpha=1, thấp hơn tốt hơn.'])


HANDLERS = {'rating': rating, 'classification': classification, 'clustering': clustering,
            'association': association, 'pca': pca_experiment}
