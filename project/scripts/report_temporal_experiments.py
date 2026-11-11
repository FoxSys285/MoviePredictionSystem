"""Independent, logged temporal experiments for the graduation report.

Does not replace app model pointers or the IID experiment history.
Run from repository root: project/.venv/Scripts/python.exe -X utf8 this_file.py
"""

import hashlib
import json
import pickle
import platform
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import joblib
import numpy as np
import optuna
import pandas as pd
import psutil
from gymnasium import spaces
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from movie_rating.data.prepare import chronological_split
from movie_rating.experiment_data import ExperimentData, dataset_statistics
from movie_rating.experiment_supervised import (
    classification_scores, feature_fit, fit_rating, numeric_summary,
)
from movie_rating.experiment_unsupervised import (
    association, clustering, genre_profiles, reduction,
)
from movie_rating.lab import LabRequest, chart, measured, native, series
from movie_rating.lab_compact import regression_scores
from movie_rating.models.item_knn import ItemKNN
from movie_rating.settings import load_settings

SEED = 42
MIN_USER, MIN_ITEM = 5, 3


def filter_training(frame):
    """Fit-only iterative filtering; never remove difficult held-out ratings."""
    current = frame.copy()
    while True:
        users = current.groupby('userId').size()
        items = current.groupby('movieId').size()
        kept = current.loc[current.userId.isin(users[users >= MIN_USER].index)
                           & current.movieId.isin(items[items >= MIN_ITEM].index)]
        if len(kept) == len(current):
            return current.reset_index(drop=True)
        current = kept
        if len(current) < 20:
            raise ValueError('Too few training rows after fit-only cold-start filtering.')


class FrameRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, algorithm='linear', config=None, movies=None, epochs=12):
        self.algorithm, self.config, self.movies, self.epochs = algorithm, config, movies, epochs

    def fit(self, X, y):
        frame = X.copy()
        frame['rating'] = np.asarray(y)
        frame = filter_training(frame)
        self.model_ = fit_rating(self.algorithm, self.config or {},
                                SimpleNamespace(epochs=self.epochs, patience=3), frame, self.movies)
        return self

    def predict(self, X):
        return self.model_.predict(X.userId, X.movieId)


class FrameClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, algorithm='logistic', config=None, movies=None):
        self.algorithm, self.config, self.movies = algorithm, config, movies

    def fit(self, X, y):
        # Actual observed rating is fit-only input for historical aggregates.
        frame = filter_training(X)
        self.builder_, self.model_, _ = feature_fit(
            frame, [], self.movies, self.algorithm, self.config or {})
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        u, i = self.builder_.encode_queries(X.userId, X.movieId)
        return self.model_.predict_proba(self.builder_._features(u, i))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


def temporal_folds(train):
    """Split timestamp groups, so equal timestamps never cross fold boundaries."""
    timestamps = np.sort(train.timestamp.unique())
    output = []
    for a, b in TimeSeriesSplit(n_splits=5).split(timestamps):
        left = np.flatnonzero(train.timestamp.isin(timestamps[a]))
        right = np.flatnonzero(train.timestamp.isin(timestamps[b]))
        assert train.iloc[left].timestamp.max() < train.iloc[right].timestamp.min()
        output.append((left, right))
    return output


def fit_classifier(frame, targets, movies, algorithm, config):
    filtered = filter_training(frame)
    return feature_fit(filtered, targets, movies, algorithm, config)


def scores_for(algorithm, model, frame, builder=None, threshold=.5):
    start = time.perf_counter()
    if builder is None:
        prediction = model.predict(frame.userId, frame.movieId)
        seconds = time.perf_counter() - start
        scores = regression_scores(frame.rating, prediction)
    else:
        u, i = builder.encode_queries(frame.userId, frame.movieId)
        prediction = model.predict_proba(builder._features(u, i))[:, 1]
        seconds = time.perf_counter() - start
        return classification_scores(frame.rating >= 3.5, prediction, threshold), seconds, prediction
    return scores, seconds, prediction


def cv_report(algorithm, config, train, movies, folds, epochs=12):
    rows = []
    binary = algorithm in ('logistic', 'random_forest')
    for index, (a, b) in enumerate(folds):
        fit, held = train.iloc[a], train.iloc[b]
        with measured() as timing:
            if binary:
                builder, model, _ = fit_classifier(fit, [], movies, algorithm, config)
            else:
                builder = None
                model = fit_rating(algorithm, config, SimpleNamespace(epochs=epochs, patience=3),
                                   filter_training(fit), movies)
        scores, seconds, _ = scores_for(algorithm, model, held, builder)
        rows.append({'fold': index+1, 'fit_rows': len(filter_training(fit)),
                     'score_rows': len(held), **scores, **timing, 'inference_seconds': seconds})
    return {'folds': rows, 'metrics': numeric_summary(rows)}


def supervised_report(splits, movies, destination, log):
    train, val, test = [splits[s] for s in ('train', 'validation', 'test')]
    folds = temporal_folds(train)
    rows, cv, configs, tuning, curves, predictions = [], {}, {}, {}, {}, {}
    request = SimpleNamespace(epochs=12, patience=3)
    searches = {
        'linear': [{}],
        'xgboost': [dict(n_estimators=100, max_depth=d, learning_rate=.05) for d in (3, 6)],
        'svd': [dict(factors=k, learning_rate=.005, regularization=.02) for k in (8, 16)],
        'logistic': [dict(C=c) for c in (.1, 1.)],
        'random_forest': [dict(n_estimators=100, max_depth=d, min_samples_leaf=3) for d in (6, 12)],
    }
    for algorithm, candidates in searches.items():
        binary = algorithm in ('logistic', 'random_forest')
        log(f'GridSearchCV / 5 temporal folds: {algorithm}')
        estimator = FrameClassifier(algorithm, movies=movies) if binary else FrameRegressor(algorithm, movies=movies)
        scoring = 'f1' if binary else 'neg_root_mean_squared_error'
        search = GridSearchCV(estimator, {'config': candidates}, scoring=scoring, cv=folds,
                              n_jobs=1, refit=False, error_score='raise')
        y = (train.rating >= 3.5).astype(int) if binary else train.rating
        search.fit(train, y)
        tuning[algorithm] = native(search.cv_results_)
        choices = []
        for config in candidates:
            if binary:
                builder, model, _ = fit_classifier(train, [], movies, algorithm, config)
                _, _, probability = scores_for(algorithm, model, val, builder)
                threshold = max(np.linspace(.1, .9, 33), key=lambda t: (
                    f1_score(val.rating >= 3.5, probability >= t, zero_division=0), -abs(t-.5)))
                score = classification_scores(val.rating >= 3.5, probability, threshold)
            else:
                model = fit_rating(algorithm, config, request, filter_training(train), movies)
                score, _, _ = scores_for(algorithm, model, val)
                threshold = .5
            choices.append({'config': config, 'validation': score, 'threshold': float(threshold)})
        selected = min(choices, key=lambda o: -o['validation']['f1'] if binary else o['validation']['rmse'])
        config, threshold = selected['config'], selected['threshold']
        tuning[algorithm]['validation_choices'] = choices
        configs[algorithm] = {**config, **({'threshold': threshold} if binary else {})}
        cv[algorithm] = cv_report(algorithm, config, train, movies, folds)
        rows.append({'algorithm': algorithm, 'split': 'validation', **selected['validation']})
        combined = filter_training(pd.concat([train, val], ignore_index=True))
        with measured() as timing:
            if binary:
                builder, model, _ = fit_classifier(combined, [], movies, algorithm, config)
            else:
                builder = None
                model = fit_rating(algorithm, config, request, combined, movies)
        scores, seconds, predicted = scores_for(algorithm, model, test, builder, threshold)
        artifact = (builder, model) if binary else model
        joblib.dump(artifact, destination / f'{algorithm}.joblib', compress=3)
        rows.append({'algorithm': algorithm, 'split': 'test', **scores, **timing,
                     'inference_seconds': seconds, 'inference_us_per_row': seconds/len(test)*1e6,
                     'model_bytes': len(pickle.dumps(artifact)), 'refit_rows': len(combined)})
        predictions[algorithm] = predicted

    log('Optuna TPE: NCF / 8 trials / Train only fitting + Validation early stopping')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3))
    filtered = filter_training(train)
    def objective(trial):
        config = dict(factors=trial.suggest_categorical('factors', [8, 16, 32]),
                      hidden_size=trial.suggest_categorical('hidden_size', [16, 32]),
                      learning_rate=trial.suggest_float('learning_rate', .001, .01, log=True),
                      regularization=trial.suggest_float('regularization', .0001, .02, log=True),
                      dropout=trial.suggest_categorical('dropout', [.1, .2]), batch_size=1024)
        model = fit_rating('ncf', config, request, filtered, movies, validation=val)
        score, _, _ = scores_for('ncf', model, val)
        trial.set_user_attr('best_epoch', model.best_epoch)
        trial.set_user_attr('curve', model.learning_curve)
        trial.set_user_attr('validation', score)
        return score['rmse']
    study.optimize(objective, n_trials=8, n_jobs=1)
    config = {**study.best_params, 'batch_size': 1024}
    epoch = study.best_trial.user_attrs['best_epoch']
    configs['ncf'] = {**config, 'epochs': epoch}
    curves['ncf'] = study.best_trial.user_attrs['curve']
    cv['ncf'] = cv_report('ncf', config, train, movies, folds, epochs=12)
    rows.append({'algorithm': 'ncf', 'split': 'validation', **study.best_trial.user_attrs['validation']})
    tuning['ncf'] = [{'trial': t.number, 'params': t.params, 'value': t.value,
                      'best_epoch': t.user_attrs.get('best_epoch')} for t in study.trials]
    combined = filter_training(pd.concat([train, val], ignore_index=True))
    with measured() as timing:
        model = fit_rating('ncf', config, request, combined, movies, epochs=epoch)
    scores, seconds, prediction = scores_for('ncf', model, test)
    joblib.dump(model, destination/'ncf.joblib', compress=3)
    rows.append({'algorithm': 'ncf', 'split': 'test', **scores, **timing,
                 'inference_seconds': seconds, 'inference_us_per_row': seconds/len(test)*1e6,
                 'model_bytes': len(pickle.dumps(model)), 'refit_rows': len(combined)})
    predictions['ncf'] = prediction
    # Baselines use exactly the same fit-only filtering policy.
    for aid in ('mean', 'item_knn'):
        log(f'Rating baseline: {aid}')
        for split, source in [('validation', train), ('test', pd.concat([train, val], ignore_index=True))]:
            fit = filter_training(source)
            with measured() as timing:
                model = ItemKNN(neighbors=20, seed=42).fit(fit) if aid == 'item_knn' else float(fit.rating.mean())
            start = time.perf_counter()
            predicted = model.predict(splits[split].userId, splits[split].movieId) if aid == 'item_knn' else np.full(len(splits[split]), model)
            seconds = time.perf_counter()-start
            rows.append({'algorithm': aid, 'split': split,
                         **regression_scores(splits[split].rating, predicted), **timing,
                         'inference_seconds': seconds, 'inference_us_per_row': seconds/len(predicted)*1e6,
                         'model_bytes': len(pickle.dumps(model))})
    for aid, prediction in predictions.items():
        np.save(destination/f'{aid}-test-predictions.npy', prediction)
    fold_indices = [{'fold': i+1, 'fit_rows': len(a), 'score_rows': len(b),
                     'fit_max_timestamp': int(train.iloc[a].timestamp.max()),
                     'score_min_timestamp': int(train.iloc[b].timestamp.min())} for i, (a,b) in enumerate(folds)]
    return {'rows': rows, 'cv': cv, 'best_configs': configs, 'tuning': tuning,
            'learning_curves': curves, 'fold_protocol': fold_indices,
            'class_balance': {s: {'positive': int((f.rating >= 3.5).sum()),
                                  'negative': int((f.rating < 3.5).sum())} for s,f in splits.items()}}


class MovieHistoryEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, histories, genres, horizon=5):
        self.histories, self.genres, self.horizon = histories, genres, horizon
        self.users = sorted(histories)
        self.action_space = spaces.Discrete(len(genres))
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.int64),
            high=np.array([max(self.users), (1 << len(genres))-1, max(genres)+1, 1], dtype=np.int64),
            dtype=np.int64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.user = (options or {}).get('user', self.users[int(self.np_random.integers(len(self.users)))])
        self.mask, self.last, self.liked, self.step_count = 0, 0, 0, 0
        return self.observation(), {'observed': False}

    def observation(self):
        return np.array([self.user, self.mask, self.last, self.liked], dtype=np.int64)

    def valid_actions(self):
        return [a for a in range(len(self.genres)) if not self.mask & (1 << a)]

    def step(self, action):
        if action not in self.valid_actions():
            raise ValueError('Repeated/invalid movie recommendation')
        known = action in self.histories[self.user]
        reward = self.histories[self.user].get(action, 0.)
        self.mask |= 1 << int(action)
        self.last, self.liked = self.genres[action]+1, int(reward > 0)
        self.step_count += 1
        return self.observation(), float(reward), self.step_count >= self.horizon, False, {'observed': known}


def gym_reinforcement(splits, movies, destination, log):
    from movie_rating.experiment_reinforcement import environment
    log('Gymnasium custom environment / Q-Learning 5000 episodes')
    actions = splits['train'].groupby('movieId').size().sort_values(ascending=False, kind='stable').head(30).index.tolist()
    metadata = movies.set_index('movieId')
    names = [str(metadata.loc[m, 'genres']).split('|')[0] for m in actions]
    genres = [sorted(set(names)).index(g) for g in names]
    histories = {s: environment(frame, actions) for s,frame in splits.items()}
    envs = {s: MovieHistoryEnv(h, genres) for s,h in histories.items()}
    rng, q = np.random.default_rng(42), {}
    def choose(policy, state, valid, random):
        if policy == 'random': return int(random.choice(valid))
        if policy == 'popular': return valid[0]
        values = q.get(tuple(state), {})
        best = max(values.get(a, 0.) for a in valid)
        return int(random.choice([a for a in valid if values.get(a, 0.) == best]))
    def evaluate(policy, split):
        env = envs[split]
        scenario, action_rng = np.random.default_rng(100042), np.random.default_rng(100043)
        rewards, observed, elapsed = [], 0, 0.
        for _ in range(100):
            user = env.users[int(scenario.integers(len(env.users)))]
            state, _ = env.reset(seed=42, options={'user': user})
            total, done = 0., False
            while not done:
                start = time.perf_counter()
                action = choose(policy, state, env.valid_actions(), action_rng)
                elapsed += time.perf_counter()-start
                state, reward, done, _, info = env.step(action)
                total += reward
                observed += int(info['observed'])
            rewards.append(total)
        return {'cumulative_reward': float(sum(rewards)), 'average_reward': float(sum(rewards)/500),
                'episode_reward_mean': float(np.mean(rewards)), 'episode_reward_std': float(np.std(rewards, ddof=1)),
                'observed_feedback_fraction': observed/500, 'inference_seconds': elapsed,
                'evaluation_episodes': 100, 'horizon': 5}
    curve, train_curve = [], []
    with measured() as timing:
        for episode in range(1, 5001):
            state, _ = envs['train'].reset(seed=42+episode)
            total, done = 0., False
            while not done:
                env = envs['train']
                valid = env.valid_actions()
                action = int(rng.choice(valid)) if rng.random() < .2 else choose('q_learning', state, valid, rng)
                next_state, reward, done, _, _ = env.step(action)
                future = max((q.get(tuple(next_state), {}).get(a, 0.) for a in env.valid_actions()), default=0.) if not done else 0.
                values = q.setdefault(tuple(state), {})
                values[action] = values.get(action, 0.) + .2*(reward+.9*future-values.get(action, 0.))
                state, total = next_state, total+reward
            train_curve.append(total)
            if episode % 250 == 0: curve.append({'episode': episode, **evaluate('q_learning', 'validation')})
    rows = []
    for policy in ('q_learning', 'random', 'popular'):
        for split in ('validation', 'test'):
            rows.append({'algorithm': policy, 'split': split, **evaluate(policy, split),
                         'fit_seconds': timing['fit_seconds'] if policy == 'q_learning' else None,
                         'model_bytes': len(pickle.dumps(q if policy == 'q_learning' else actions))})
    joblib.dump(q, destination/'q_learning.joblib', compress=3)
    return {'rows': rows, 'curve': curve, 'train_rewards': train_curve,
            'parameters': {'alpha': .2, 'gamma': .9, 'epsilon': .2, 'episodes': 5000,
                           'actions': 30, 'evaluation_episodes': 100, 'horizon': 5},
            'action_movies': actions, 'environment': 'Gymnasium MovieHistoryEnv'}


def reduction_downstream(splits, movies, result, log):
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    log('Downstream rating from reconstructed user-genre profiles')
    raw, users, genres = genre_profiles(splits['train'], movies, 5)
    scaler = StandardScaler().fit(raw)
    x = scaler.transform(raw)
    user_map, genre_map = {int(u): i for i,u in enumerate(users)}, {g:i for i,g in enumerate(genres)}
    movie_map = {int(row.movieId): [genre_map[g] for g in row.genres.split('|') if g in genre_map] for row in movies.itertuples()}
    mean = float(splits['train'].rating.mean())
    rows = []
    for aid in ('pca', 'truncated_svd'):
        k = result['best_configs'][aid]['selected_components']
        model = PCA(k, svd_solver='full') if aid == 'pca' else TruncatedSVD(k, random_state=42)
        restored = scaler.inverse_transform(model.inverse_transform(model.fit_transform(x)))
        for split in ('validation','test'):
            frame = splits[split]
            predicted = []
            for r in frame.itertuples():
                u, gs = user_map.get(int(r.userId)), movie_map.get(int(r.movieId), [])
                predicted.append(float(np.mean(restored[u, gs])) if u is not None and gs else mean)
            rows.append({'algorithm': aid, 'split': split, 'components': k,
                         **regression_scores(frame.rating, np.clip(predicted,.5,5))})
    knn = ItemKNN(neighbors=20, seed=42).fit(splits['train'])
    for split in ('validation','test'):
        f = splits[split]
        rows.append({'algorithm': 'mean', 'split': split,
                     **regression_scores(f.rating, np.full(len(f), mean))})
        rows.append({'algorithm': 'item_knn', 'split': split,
                     **regression_scores(f.rating, knn.predict(f.userId, f.movieId))})
    return rows


def run():
    settings = load_settings()
    run_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%S')
    destination = settings.artifacts_dir / 'report-temporal' / run_id
    destination.mkdir(parents=True)
    def log(message):
        print(message, flush=True)
        with (destination/'progress.log').open('a',encoding='utf-8') as out: out.write(message+'\n')
    log('Loading 300 reproducibly sampled prepared MovieLens users')
    cached = ExperimentData(settings.artifacts_dir/'data')
    iid_parts, movies, provenance = cached.load(LabRequest(group='rating', max_users=300))
    ratings = pd.concat(list(iid_parts.values()),ignore_index=True).sort_values(['timestamp','userId','movieId'],kind='stable')
    splits = chronological_split(ratings,.7,.15)
    for name,frame in splits.items(): frame.to_parquet(destination/f'{name}.parquet',index=False)
    filtered = filter_training(splits['train'])
    refit_filtered = filter_training(pd.concat([splits['train'], splits['validation']], ignore_index=True))
    provenance.update(protocol='global_temporal_70_15_15_report_v1', sample=dataset_statistics(ratings,movies),
                      splits={s:{'ratings':len(f),'users':int(f.userId.nunique()),'movies':int(f.movieId.nunique()),
                                 'min_timestamp':int(f.timestamp.min()),'max_timestamp':int(f.timestamp.max()),
                                 'known_user_fraction':float(f.userId.isin(filtered.userId).mean()),
                                 'known_item_fraction':float(f.movieId.isin(filtered.movieId).mean())} for s,f in splits.items()},
                      fit_filter={'min_user':MIN_USER,'min_item':MIN_ITEM,'train_retained':len(filtered),
                                  'train_removed':len(splits['train'])-len(filtered),'refit_retained':len(refit_filtered),
                                  'test_known_user_refit':float(splits['test'].userId.isin(refit_filtered.userId).mean()),
                                  'test_known_item_refit':float(splits['test'].movieId.isin(refit_filtered.movieId).mean()),
                                  'policy':'fit-only iterative; heldouts retained'})
    output = {'run_id':run_id,'provenance':provenance,'supervised':supervised_report(splits,movies,destination,log)}
    for group,handler in [('clustering',clustering),('association',association),('pca',reduction)]:
        log('Unsupervised '+group)
        request=LabRequest(group=group,max_users=300,components=[2,4,8,12,16])
        output[group]=handler(splits,movies,request,lambda *_:None)
    from movie_rating.lab_compact import rule_scores
    rules=output['association']['rules']
    items=sorted({m for rule in rules for m in [*rule['antecedent_ids'],*rule['consequent_ids']]})
    heldout_rules=[]
    for split in ('validation','test'):
        frame=splits[split]
        liked=frame.loc[frame.rating>=4]
        transactions=pd.crosstab(liked.userId,liked.movieId).reindex(
            index=np.sort(frame.userId.unique()),columns=items,fill_value=0).fillna(0).astype(bool)
        pairs=[(frozenset(rule['antecedent_ids']),frozenset(rule['consequent_ids'])) for rule in rules]
        coverage=float(np.mean([transactions.loc[:,list(a)].all(axis=1).any() for a,_ in pairs])) if pairs else 0.
        heldout_rules.append({'split':split,**rule_scores(transactions,pairs),
                              'transactions':len(transactions),'rule_count':len(rules),
                              'antecedent_coverage':coverage})
    output['association']['heldout']=heldout_rules
    output['pca']['downstream']=reduction_downstream(splits,movies,output['pca'],log)
    output['reinforcement']=gym_reinforcement(splits,movies,destination,log)
    output['runtime']={'python':platform.python_version(),'platform':platform.platform(),
                       'processor':platform.processor(),'cpu_count':psutil.cpu_count(),
                       'ram_bytes':psutil.virtual_memory().total,'device':'CPU',
                       'libraries':{name:version(name) for name in ['numpy','pandas','scikit-learn','torch','xgboost','mlxtend','optuna','gymnasium']},
                       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=destination/'results.json'
    path.write_text(json.dumps(native(output),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    rows=[]
    for group in ('supervised','clustering','association','pca','reinforcement'):
        rows.extend({'group':group,**{k:json.dumps(native(v),ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()}} for r in output[group]['rows'])
    pd.DataFrame(rows).to_csv(destination/'metrics.csv',index=False,encoding='utf-8-sig')
    (settings.artifacts_dir/'report-latest.json').write_text(json.dumps({'run_id':run_id,'directory':str(destination)},ensure_ascii=False),encoding='utf-8')
    log('COMPLETED '+str(path))


if __name__=='__main__':
    run()
