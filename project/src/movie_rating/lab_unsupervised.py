"""Shared representations and explicit scopes for clustering, pattern mining and PCA."""

import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from movie_rating.lab import (chart, cost_charts, latency, measured, object_features,
                              pair_features, series, summarize)
from movie_rating.lab_supervised import rating_scores


def cluster_metrics(x, labels, seed=42):
    mask = labels >= 0
    count = len(set(labels[mask]))
    valid = count >= 2 and mask.sum() > count
    return {'silhouette': silhouette_score(x[mask], labels[mask], sample_size=min(1000, int(mask.sum())), random_state=seed) if valid else None,
            'davies_bouldin': davies_bouldin_score(x[mask], labels[mask]) if valid else None,
            'clusters': count, 'coverage': float(mask.mean()), 'noise_fraction': float((~mask).mean())}


def clustering(splits, movies, request, update):
    x, raw, labels, features = object_features(splits['train'], movies, request)
    if len(x) < 4:
        raise ValueError('Cần ít nhất 4 đối tượng trong train để phân cụm. Hãy tăng mẫu.')
    n_clusters = min(request.n_clusters, len(x)-1)
    projected = PCA(n_components=min(2, x.shape[1]), random_state=request.seed).fit_transform(x)
    rows, charts, seed_labels = [], [], {}
    first_labels = {}
    for repeat in range(request.repeats):
        for index, (algorithm, name, model) in enumerate([
            ('kmeans', 'K-Means', KMeans(n_clusters=n_clusters, n_init=10, random_state=request.seed+repeat)),
            ('dbscan', 'DBSCAN', DBSCAN(eps=request.eps, min_samples=request.min_samples)),
        ]):
            update(10 + 45 * (repeat * 2 + index) / (request.repeats * 2), f'Phân cụm {name} · seed {request.seed+repeat}')
            with measured() as timing:
                assigned = model.fit_predict(x)
            row = {'algorithm': algorithm, 'name': name, 'split': 'train', 'seed': request.seed+repeat,
                   **cluster_metrics(x, assigned, request.seed), **timing,
                   'model_bytes': len(pickle.dumps(model))}
            rows.append(row)
            seed_labels.setdefault(algorithm, []).append(assigned)
            if repeat:
                continue
            first_labels[algorithm] = assigned
            unique = sorted(set(assigned))
            charts.append(chart(1, f'Kích thước cụm · {name}', 'bar', 'Cụm', 'Số đối tượng',
                                [series(name, ['Nhiễu' if c == -1 else f'Cụm {c}' for c in unique], [int((assigned == c).sum()) for c in unique])]))
            valid = [c for c in unique if c != -1]
            charts.append(chart(2, f'Đặc điểm cụm · {name}', 'heatmap', 'Đặc trưng', 'Cụm',
                                matrix=[np.mean(x[assigned == c], axis=0).tolist() for c in valid],
                                row_labels=[f'Cụm {c} · n={int((assigned == c).sum())}' for c in valid], column_labels=features,
                                note='Trung bình đặc trưng đã chuẩn hóa (z-score), cùng thang đo cho hai thuật toán.'))
            charts.append(chart(3, f'Phép chiếu chung PCA 2D · {name}', 'scatter', 'PC1', 'PC2',
                                [{'label': 'Nhiễu' if c == -1 else f'Cụm {c}',
                                  'points': [{'x': float(projected[i, 0]), 'y': float(projected[i, 1]), 'label': labels[i]} for i in np.flatnonzero(assigned == c)]} for c in unique]))
    summary = summarize(rows)
    for row in summary:
        runs = seed_labels[row['algorithm']]
        row['stability_ari'] = float(np.mean([adjusted_rand_score(runs[0], r) for r in runs[1:]])) if len(runs) > 1 else None
    mask = first_labels['dbscan'] >= 0
    common = []
    if mask.sum() >= 3:
        for aid in ('kmeans', 'dbscan'):
            common.append({'algorithm': aid, 'points': int(mask.sum()), **cluster_metrics(x[mask], first_labels[aid][mask], request.seed)})
    for metric, title in [('silhouette', 'Silhouette'), ('davies_bouldin', 'Davies–Bouldin')]:
        charts.append(chart(0, title, 'bar', 'Thuật toán', title,
                            [series(title, [r['name'] for r in summary], [r[metric] for r in summary])]))
    update(65, 'Khảo sát K và độ nhạy eps/min_samples')
    ks = list(range(2, min(12, len(x)-1) + 1))
    inertia = []
    for k in ks:
        update(65 + k, f'Khảo sát K-Means · K={k}')
        inertia.append(float(KMeans(n_clusters=k, n_init=5, random_state=request.seed).fit(x).inertia_))
    charts.append(chart(4, 'Elbow của K-Means', 'line', 'K', 'Inertia', [series('K-Means', ks, inertia)]))
    eps_values = sorted(set([round(request.eps * 0.6, 3), request.eps, round(request.eps * 1.4, 3)]))
    ms_values = sorted(set([max(2, request.min_samples-2), request.min_samples, request.min_samples+2]))
    coverage = []
    cluster_counts = []
    for ms in ms_values:
        line, counts = [], []
        for eps in eps_values:
            update(85, f'Khảo sát DBSCAN · eps={eps}, min_samples={ms}')
            assigned = DBSCAN(eps=eps, min_samples=ms).fit_predict(x)
            line.append(float((assigned >= 0).mean()))
            counts.append(len(set(assigned)-{-1}))
        coverage.append(series(f'min_samples={ms}', eps_values, line))
        cluster_counts.append(series(f'min_samples={ms}', eps_values, counts))
    charts.append(chart(4, 'Độ nhạy DBSCAN · tỷ lệ gán cụm', 'line', 'eps', 'Tỷ lệ gán cụm', coverage))
    charts.append(chart(4, 'Độ nhạy DBSCAN · số cụm', 'line', 'eps', 'Số cụm', cluster_counts))
    charts.extend(cost_charts(summary))
    return {'rows': summary, 'seed_rows': rows, 'charts': charts, 'common_subset': common,
            'objects': len(x), 'feature_names': features,
            'selection': 'Đối chiếu chất lượng hình học, tỷ lệ bao phủ, ý nghĩa và độ ổn định. Không xếp hạng tuyệt đối bằng một chỉ số.',
            'notes': ['Hai thuật toán dùng cùng vector đặc trưng train và chuẩn hóa chung. Phép chiếu PCA chỉ minh họa, không dùng để tính chỉ số.',
                      'DBSCAN: Silhouette/DBI chỉ tính trên điểm không nhiễu; xem bảng tập điểm chung với K-Means để tránh lợi thế do bỏ điểm khó.',
                      'N/A nghĩa là không đủ ít nhất hai cụm hợp lệ; không thay chỉ số không xác định bằng 0.',
                      f'Đối tượng phân cụm: {request.cluster_object}; đặc trưng gồm thể loại, rating trung bình, log số rating và độ phân tán từ train.']}


def association(splits, movies, request, update):
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

    train = splits['train']
    positive = train.loc[train.rating >= request.positive_threshold]
    popular = positive.groupby('movieId').size().sort_values(ascending=False, kind='stable')
    items = list(map(int, popular.head(request.max_items).index))
    users = np.sort(train.userId.unique())
    if not items or len(users) < 3:
        raise ValueError('Không đủ giao dịch/phim tích cực để khai phá. Hãy tăng mẫu hoặc hạ ngưỡng rating.')
    transactions = pd.crosstab(positive.userId, positive.movieId).reindex(index=users, columns=items, fill_value=0).fillna(0).astype(bool)
    names = movies.set_index('movieId').title.to_dict()
    supports = sorted(set([round(max(.02, request.min_support * .5), 4), request.min_support, round(min(.9, request.min_support * 1.5), 4)]))
    factories = [('apriori', 'Apriori', apriori), ('fp_growth', 'FP-Growth', fpgrowth)]
    rows, charts, patterns, rules_by_algo, sweeps = [], [], {}, {}, {}
    equality = []
    max_len = 2
    for ai, (algorithm, name, miner) in enumerate(factories):
        sweep = []
        for si, support in enumerate(supports):
            update(10 + 55 * (ai * len(supports) + si) / (2 * len(supports)), f'{name} · min_support={support}')
            with measured() as timing:
                frequent = miner(transactions, min_support=support, use_colnames=True, max_len=max_len)
            rule_start = __import__('time').perf_counter()
            rules = association_rules(frequent, num_itemsets=len(transactions), metric='confidence', min_threshold=.3) if len(frequent) else pd.DataFrame()
            rule_seconds = __import__('time').perf_counter() - rule_start
            sweep.append({'support': support, **timing, 'rules': len(rules), 'itemsets': len(frequent), 'rule_seconds': rule_seconds})
            if support == request.min_support:
                patterns[algorithm] = {tuple(sorted(v)): float(s) for s, v in zip(frequent.support, frequent.itemsets)}
                rules_by_algo[algorithm] = rules
                row = {'algorithm': algorithm, 'name': name, 'split': 'train', 'itemsets': len(frequent), 'rules': len(rules),
                       'rule_seconds': rule_seconds, 'min_support': support, **timing,
                       'model_bytes': len(pickle.dumps((frequent, rules)))}
                rows.append(row)
        sweeps[algorithm] = sweep
    same_keys = patterns['apriori'].keys() == patterns['fp_growth'].keys()
    same_support = same_keys and all(abs(v - patterns['fp_growth'][key]) < 1e-10 for key, v in patterns['apriori'].items())
    for row in rows:
        row['same_itemsets'] = bool(same_support)
    for field, title, unit, index in [('fit_seconds', 'Thời gian khai phá theo min_support', 'Giây ↓', 0),
                                      ('peak_rss_mb', 'Bộ nhớ theo min_support', 'MiB RSS tiến trình', 1),
                                      ('rules', 'Số luật theo min_support', 'Số luật', 4)]:
        charts.append(chart(index, title, 'line', 'min_support', unit,
                            [series(name, [r['support'] for r in sweeps[aid]], [r[field] for r in sweeps[aid]]) for aid, name, _ in factories]))
    rules = rules_by_algo['fp_growth']
    public_rules = []
    if len(rules):
        ordered = rules.sort_values(['lift', 'confidence', 'support'], ascending=False, kind='stable')
        for r in ordered.head(100).itertuples():
            a, b = [int(x) for x in sorted(r.antecedents)], [int(x) for x in sorted(r.consequents)]
            public_rules.append({'antecedents': [str(names.get(x, x)) for x in a], 'consequents': [str(names.get(x, x)) for x in b],
                                 'support': r.support, 'confidence': r.confidence, 'lift': r.lift})
        sampled = ordered if len(ordered) <= 400 else ordered.sample(400, random_state=request.seed)
        charts.append(chart(2, 'Chất lượng luật · tập kết quả chung', 'scatter', 'Support', 'Confidence',
                            [{'label': 'Luật', 'points': [{'x': float(r.support), 'y': float(r.confidence), 'value': float(r.lift),
                                                         'label': ' + '.join(str(names.get(x, x)) for x in r.antecedents) + ' → ' + ' + '.join(str(names.get(x, x)) for x in r.consequents)} for r in sampled.itertuples()]}],
                            color_value='Lift', note=f'Hiển thị {len(sampled)}/{len(rules)} luật; số liệu tính trên toàn bộ tập luật.'))
    else:
        charts.append(chart(2, 'Chất lượng luật', 'scatter', 'Support', 'Confidence', note='Không có luật vượt ngưỡng confidence 0,3; hãy điều chỉnh mẫu hoặc min_support.'))
    sizes = sorted(set([max(2, len(users)//3), max(2, 2*len(users)//3), len(users)]))
    order = np.random.default_rng(request.seed).permutation(len(users))
    for aid, name, miner in factories:
        times = []
        for size in sizes:
            update(85, f'Khả năng mở rộng {name} · {size} giao dịch')
            with measured() as timing:
                miner(transactions.iloc[order[:size]], min_support=request.min_support, use_colnames=True, max_len=max_len)
            times.append(timing['fit_seconds'])
        equality.append(series(name, sizes, times))
    charts.append(chart(3, 'Thời gian theo số giao dịch', 'line', 'Số giao dịch', 'Giây ↓', equality))
    charts.extend(cost_charts(rows))
    return {'rows': rows, 'charts': charts, 'rules': public_rules, 'same_itemsets': bool(same_support),
            'transactions': len(users), 'transaction_items': len(items), 'sweeps': sweeps,
            'selection': 'Hai thuật toán cho cùng tập phổ biến và support.' if same_support else 'Kết quả chưa tương đương; cần kiểm tra trước khi so hiệu suất.',
            'notes': [f'Giới hạn công bố: {len(items)} phim tích cực phổ biến nhất trong train, độ dài tập mục tối đa {max_len}, confidence ≥ 0,3. Không khai phá toàn danh mục.',
                      'Tất cả user train tạo thành mẫu số support, kể cả giao dịch rỗng sau khi lọc phim/ngưỡng rating.',
                      'Thời gian khai phá không gồm preprocessing và sinh luật; thời gian sinh luật được báo riêng. Số luật không phải càng nhiều càng tốt.',
                      'Bảng chỉ hiển thị tối đa 100 luật; Lift cao trên support thấp chưa chứng minh luật đáng tin cậy.']}


def pca_experiment(splits, movies, request, update):
    update(8, 'Tạo cùng đặc trưng hồi quy và chuẩn hóa bằng train')
    builder, train_x, matrices = pair_features(splits['train'], [splits['validation'], splits['test']], movies)
    scaler = StandardScaler().fit(train_x)
    x = scaler.transform(train_x)
    heldouts = {s: scaler.transform(a) for s, a in zip(('validation', 'test'), matrices)}
    maximum = min(x.shape[1], len(x)-1)
    if maximum < 2:
        raise ValueError('Không đủ hạng dữ liệu để đánh giá PCA.')
    full = PCA(n_components=maximum, svd_solver='full').fit(x)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    choices = sorted(set([k for k in [1, 2, 4, 8, 16, maximum] if k <= maximum]))
    rows, charts = [], []
    for index, k in enumerate([0, *choices]):
        update(15 + 70 * index / (1 + len(choices)), f'PCA {k} thành phần' if k else 'Baseline không PCA')
        with measured() as timing:
            transform = PCA(n_components=k, svd_solver='full').fit(x) if k else None
            z = transform.transform(x) if transform is not None else x
            estimator = Ridge(alpha=1).fit(z, splits['train'].rating)
        for split, input_x in heldouts.items():
            z_hold = transform.transform(input_x) if transform is not None else input_x
            predicted = np.clip(estimator.predict(z_hold), .5, 5)
            def inference():
                sample = input_x[:1]
                return estimator.predict(transform.transform(sample) if transform is not None else sample)
            rows.append({'algorithm': 'pca' if k else 'no_pca', 'name': f'PCA · {k} chiều' if k else 'Không PCA · baseline', 'split': split,
                         'components': k or x.shape[1], 'dimension_reduction': 1-k/x.shape[1] if k else 0,
                         'variance_retained': float(cumulative[k-1]) if k else 1,
                         'reconstruction_mse': float(np.mean((input_x-transform.inverse_transform(z_hold))**2)) if transform is not None else 0,
                         **rating_scores(splits[split].rating, predicted), **timing, **latency(inference),
                         'model_bytes': len(pickle.dumps((builder, scaler, transform, estimator)))})
    charts.append(chart(0, 'Phương sai giải thích tích lũy', 'line', 'Số thành phần k', 'Tỷ lệ phương sai',
                        [series('PCA', range(1, maximum+1), cumulative), series('Mốc 90%', [1, maximum], [.9, .9]), series('Mốc 95%', [1, maximum], [.95, .95])]))
    for split in ('validation', 'test'):
        values = [r for r in rows if r['split'] == split]
        base = values[0]
        pc = values[1:]
        charts.append(chart(1, 'RMSE sau PCA so với không PCA', 'line', 'Số thành phần k', 'RMSE ↓',
                            [series('PCA + Ridge', [r['components'] for r in pc], [r['rmse'] for r in pc]),
                             series('Không PCA + Ridge', [choices[0], choices[-1]], [base['rmse']]*2)], split))
        charts.append(chart(2, 'Lỗi tái tạo trên đặc trưng đã chuẩn hóa', 'line', 'Số thành phần k', 'MSE tái tạo ↓',
                            [series('PCA', [r['components'] for r in pc], [r['reconstruction_mse'] for r in pc])], split))
    val_rows = [r for r in rows if r['split'] == 'validation']
    pc = val_rows[1:]
    charts.append(chart(3, 'Thời gian toàn pipeline PCA + Ridge', 'line', 'Số thành phần k', 'Giây ↓',
                        [series('PCA + Ridge', [r['components'] for r in pc], [r['fit_seconds'] for r in pc]),
                         series('Không PCA', [choices[0], choices[-1]], [val_rows[0]['fit_seconds']]*2)]))
    projected = full.transform(x)
    chosen = np.random.default_rng(request.seed).choice(len(x), min(350, len(x)), replace=False)
    truth = splits['train'].rating.to_numpy()
    charts.append(chart(4, 'PC1–PC2 · biểu diễn train', 'scatter', f'PC1 · {full.explained_variance_ratio_[0]:.1%}', f'PC2 · {full.explained_variance_ratio_[1]:.1%}',
                        [{'label': title, 'points': [{'x': float(projected[i, 0]), 'y': float(projected[i, 1])} for i in chosen if predicate(truth[i])]} for title, predicate in [('Rating < 3', lambda y: y < 3), ('3 ≤ rating < 4', lambda y: 3 <= y < 4), ('Rating ≥ 4', lambda y: y >= 4)]]))
    eligible = [r for r in pc if r['variance_retained'] >= .9 and r['rmse'] <= val_rows[0]['rmse']*1.01]
    selected = min(eligible, key=lambda r: r['components']) if eligible else min(val_rows, key=lambda r: r['rmse'])
    charts.extend(cost_charts(val_rows))
    return {'rows': rows, 'charts': charts,
            'selection': f"Chọn trên validation: {selected['name']}. Ưu tiên ít chiều nhất giữ ≥90% phương sai và RMSE không tăng quá 1% so baseline; nếu không đạt, dùng RMSE validation thấp nhất.",
            'notes': ['PCA chỉ fit bằng train. Reconstruction error đo trên không gian đặc trưng đã chuẩn hóa, không phải sai số rating.',
                      'Mô hình sau PCA và baseline đều là Ridge với alpha=1; không đưa ID dạng số liên tục vào PCA.',
                      'Chi phí pipeline bao gồm fit PCA và Ridge, không gồm bước tạo đặc trưng chung. Mốc 90% là quy tắc thiết kế của thực nghiệm này, không phải bảo đảm cho mọi dữ liệu.']}
