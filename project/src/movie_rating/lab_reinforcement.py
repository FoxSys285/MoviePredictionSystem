"""A declared train-derived simulator; rewards are estimates, never counterfactual truth."""

import pickle

import numpy as np
from sklearn.cluster import KMeans

from movie_rating.lab import chart, cost_charts, latency, measured, series, summarize


def simulator(train, movies, seed):
    actions = train.groupby('movieId').size().sort_values(ascending=False, kind='stable').head(12)
    if len(actions) < 4:
        raise ValueError('Cần ít nhất 4 phim trong train cho môi trường tuần tự.')
    metadata = movies.set_index('movieId')
    genres = sorted({g for m in actions.index for g in str(metadata.loc[m, 'genres']).split('|')})
    action_genres = np.asarray([[g in str(metadata.loc[m, 'genres']).split('|') for g in genres] for m in actions.index], dtype=float)
    all_genres = {int(m): set(str(s).split('|')) for m, s in zip(movies.movieId, movies.genres)}
    global_mean = float(train.rating.mean())
    preferences = []
    for _, history in train.groupby('userId'):
        values = []
        for g in genres:
            ratings = history.loc[history.movieId.map(lambda m: g in all_genres[int(m)]), 'rating']
            values.append(float((ratings.sum() + 5*global_mean) / (len(ratings)+5)))
        preferences.append(values)
    clusterer = KMeans(n_clusters=min(3, len(preferences)), n_init=5, random_state=seed).fit(preferences)
    weights = np.bincount(clusterer.labels_, minlength=len(clusterer.cluster_centers_)).astype(float)
    weights /= weights.sum()
    item_mean = train.groupby('movieId').rating.mean().reindex(actions.index).to_numpy()
    preference_score = clusterer.cluster_centers_ @ action_genres.T / np.maximum(action_genres.sum(axis=1), 1)
    estimated_rating = np.clip(.65*preference_score + .35*item_mean, .5, 5)
    expected = (estimated_rating-.5)/4.5
    primary = np.argmax(action_genres, axis=1)
    return expected, weights, primary, [str(metadata.loc[m, 'title']) for m in actions.index]


def evaluate_policy(policy, q, expected, weights, genres, seed, episodes=60, horizon=5):
    """Same profile/noise per episode for every policy; fresh from training episode seeds."""
    rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(seed+901)
    returns, coverage, duplicate, invalid, actions_total = [], set(), 0, 0, 0
    action_count = expected.shape[1]
    for _ in range(episodes):
        profile = int(rng.choice(len(weights), p=weights))
        noise = rng.normal(0, .03, horizon)
        mask, last, total = 0, -1, 0.0
        for step in range(horizon):
            valid = [a for a in range(action_count) if not mask & (1 << a)]
            if policy == 'random':
                action = int(action_rng.choice(valid))
            elif policy == 'popular':
                action = valid[0]
            elif policy == 'greedy':
                action = max(valid, key=lambda a: expected[profile, a] - (.15 if last >= 0 and genres[a] == genres[last] else 0))
            else:
                values = q.get((profile, mask, last), np.zeros(action_count))
                action = max(valid, key=lambda a: values[a])
            duplicate += int(bool(mask & (1 << action)))
            invalid += int(action not in valid)
            reward = expected[profile, action] - (.15 if last >= 0 and genres[action] == genres[last] else 0) + noise[step]
            total += (.9**step) * reward
            mask |= 1 << action
            last = action
            coverage.add(action)
            actions_total += 1
        returns.append(total)
    return {'return_mean': float(np.mean(returns)), 'episode_return_std': float(np.std(returns, ddof=1)),
            'coverage': len(coverage)/action_count, 'repeat_fraction': duplicate/max(1, actions_total),
            'invalid_fraction': invalid/max(1, actions_total), 'evaluation_episodes': episodes}


def reinforcement(splits, movies, request, update):
    update(8, 'Ước lượng môi trường từ train: 12 phim phổ biến và hồ sơ thể loại')
    expected, weights, genres, action_names = simulator(splits['train'], movies, request.seed)
    horizon = min(5, len(action_names))
    names = {'q_learning': 'Q-Learning', 'random': 'Random · baseline', 'popular': 'Most Popular · baseline', 'greedy': 'Tham lam 1 bước · baseline'}
    rows, trajectories, seed_results = [], [], []
    evaluation_seed = request.seed + 500_000
    for repeat in range(request.repeats):
        rng = np.random.default_rng(request.seed+repeat)
        q = {}
        checkpoints = sorted(set(np.linspace(0, request.episodes, 11).astype(int)))
        curve = []
        with measured() as timing:
            for episode in range(request.episodes+1):
                if episode in checkpoints:
                    update(12 + 73 * (repeat + episode/request.episodes) / request.repeats,
                           f'Q-Learning · seed {request.seed+repeat} · episode {episode}/{request.episodes}')
                    report = evaluate_policy('q_learning', q, expected, weights, genres, evaluation_seed, horizon=horizon)
                    curve.append({'steps': episode*horizon, **report})
                if episode == request.episodes:
                    break
                profile = int(rng.choice(len(weights), p=weights))
                mask, last = 0, -1
                epsilon = max(.08, 1 - episode/(request.episodes*.8))
                for step in range(horizon):
                    state = (profile, mask, last)
                    values = q.setdefault(state, np.zeros(expected.shape[1]))
                    valid = [a for a in range(expected.shape[1]) if not mask & (1 << a)]
                    if rng.random() < epsilon:
                        action = int(rng.choice(valid))
                    else:
                        best = max(values[a] for a in valid)
                        action = int(rng.choice([a for a in valid if values[a] == best]))
                    reward = expected[profile, action] - (.15 if last >= 0 and genres[action] == genres[last] else 0) + rng.normal(0, .03)
                    next_mask = mask | (1 << action)
                    next_valid = [a for a in range(expected.shape[1]) if not next_mask & (1 << a)]
                    future = q.get((profile, next_mask, action), np.zeros(expected.shape[1]))
                    target = reward + (.9*max(future[a] for a in next_valid) if step < horizon-1 else 0)
                    values[action] += .2 * (target-values[action])
                    mask, last = next_mask, action
        trajectories.append(curve)
        for policy, name in names.items():
            update(88, f'Đánh giá chính sách {name} trên cùng episode mô phỏng')
            report = evaluate_policy(policy, q, expected, weights, genres, evaluation_seed, horizon=horizon)
            if policy == 'q_learning':
                costs = timing
            else:
                with measured() as costs:
                    baseline_parameters = expected.copy()
            delay = latency(lambda: evaluate_policy(policy, q, expected, weights, genres, evaluation_seed, episodes=1, horizon=horizon), repetitions=8)
            rows.append({'algorithm': policy, 'name': name, 'split': 'simulation', 'seed': request.seed+repeat,
                         **report, **costs, **delay, 'model_bytes': len(pickle.dumps(q if policy == 'q_learning' else expected))})
            seed_results.append({'policy': name, 'seed': request.seed+repeat, 'return': report['return_mean']})
    summary = summarize(rows)
    curves = np.asarray([[point['return_mean'] for point in curve] for curve in trajectories])
    points = [{'x': point['steps'], 'y': float(curves[:, i].mean()),
               **({'error': float(curves[:, i].std(ddof=1))} if request.repeats > 1 else {})} for i, point in enumerate(trajectories[0])]
    lines = [{'label': 'Q-Learning', 'points': points}]
    max_steps = request.episodes*horizon
    for row in summary:
        if row['algorithm'] != 'q_learning':
            lines.append(series(row['name'], [0, max_steps], [row['return_mean']]*2))
    final_points = [{'x': r['name'], 'y': r['return_mean'], **({'error': r['std']['return_mean']} if 'return_mean' in r['std'] else {})} for r in summary]
    popular = next(r['return_mean'] for r in summary if r['algorithm'] == 'popular')
    reached = next((p['x'] for p in points if p['y'] >= popular), None)
    for row in summary:
        row['steps_to_popular'] = reached if row['algorithm'] == 'q_learning' else None
    charts = [
        chart(0, 'Evaluation return theo số bước tương tác', 'line', 'Số bước huấn luyện', 'Return chiết khấu ↑', lines, 'simulation', note='Vùng/thanh sai số là std qua seed nếu chạy ≥2 seed; không phải khoảng tin cậy 95%.'),
        chart(1, 'Return cuối giữa các chính sách', 'bar', 'Chính sách', 'Return chiết khấu ↑', [{'label': 'Return', 'points': final_points}], 'simulation'),
        chart(2, 'Return trung bình của từng seed huấn luyện', 'scatter', 'Seed', 'Return ↑',
              [series(name, [r['seed'] for r in seed_results if r['policy'] == name], [r['return'] for r in seed_results if r['policy'] == name]) for name in names.values()], 'simulation'),
        chart(3, 'Độ phủ danh mục mô phỏng', 'bar', 'Chính sách', 'Tỷ lệ phim ứng viên', [series('Độ phủ', [r['name'] for r in summary], [r['coverage'] for r in summary])], 'simulation'),
        chart(3, 'Tỷ lệ lặp / hành động không hợp lệ', 'bar', 'Chính sách', 'Tỷ lệ ↓', [series('Lặp', [r['name'] for r in summary], [r['repeat_fraction'] for r in summary]), series('Không hợp lệ', [r['name'] for r in summary], [r['invalid_fraction'] for r in summary])], 'simulation'),
        *cost_charts(summary, 'simulation'),
    ]
    return {'rows': summary, 'seed_rows': rows, 'charts': charts, 'actions': action_names,
            'selection': 'Đối chiếu return và độ ổn định trong cùng simulator; không suy ra hiệu quả với người dùng thực.',
            'notes': ['MÔ PHỎNG: các reward là ước lượng từ rating/thể loại trong train, không phải phản hồi thực cho hành động chưa quan sát.',
                      'Trạng thái = nhóm hồ sơ thể loại + tập hành động đã chọn + phim trước; hành động = một trong tối đa 12 phim phổ biến train. Episode dài 5 bước, gamma=0,9, alpha=0,2.',
                      'Reward = rating ước lượng chuẩn hóa về [0;1] − 0,15 nếu lặp thể loại liên tiếp + nhiễu Gaussian sigma=0,03. Phim đã chọn bị loại khỏi tập hành động, nên tỷ lệ lặp/không hợp lệ kỳ vọng bằng 0 theo thiết kế.',
                      'Các chính sách được đánh giá bằng cùng 60 profile/nhiễu episode trên seed đánh giá tách huấn luyện; Q-Learning dùng epsilon=0 khi đánh giá. Greedy biết reward kỳ vọng một bước của simulator.',
                      'Thời gian Q-Learning gồm đánh giá định kỳ; độ trễ đo một episode 5 bước, không phải một lần dự đoán rating. Test MovieLens không được dùng xây simulator.']}
