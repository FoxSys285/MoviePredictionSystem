"""Bounded, reproducible experiment jobs. Production prediction artifacts are untouched."""

import csv
import hashlib
import io
import json
import os
import pickle
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import pandas as pd
import psutil
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from movie_rating.data.prepare import chronological_split
from movie_rating.models.regression import FeatureRegression
from movie_rating.windows_dlls import enable_unicode_dll_paths

enable_unicode_dll_paths()

GROUPS = ('rating', 'classification', 'clustering', 'association', 'pca', 'reinforcement')
CATALOG = json.loads(Path(__file__).with_name('lab_catalog.json').read_text(encoding='utf-8'))


class LabRequest(BaseModel):
    group: Literal['rating', 'classification', 'clustering', 'association', 'pca', 'reinforcement']
    seed: int = Field(default=42, ge=0, le=1_000_000)
    max_users: int = Field(default=150, ge=10, le=1000)
    repeats: int = Field(default=1, ge=1, le=3)
    epochs: int = Field(default=8, ge=2, le=30)
    positive_threshold: float = Field(default=4, ge=1, le=5, multiple_of=0.5)
    n_clusters: int = Field(default=4, ge=2, le=12)
    eps: float = Field(default=1.5, ge=0.1, le=10)
    min_samples: int = Field(default=5, ge=2, le=30)
    max_items: int = Field(default=80, ge=10, le=120)
    min_support: float = Field(default=0.08, ge=0.02, le=0.5)
    episodes: int = Field(default=1200, ge=100, le=5000)
    cluster_object: Literal['movies', 'users'] = 'movies'


class Cancelled(Exception):
    pass


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(path, value):
    temporary = path.with_suffix('.' + uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(native(value), ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


@contextmanager
def measured():
    """Peak RSS is process-wide, sampled during this operation; not model-only RAM."""
    process = psutil.Process()
    done = threading.Event()
    stats = {'peak_rss_mb': process.memory_info().rss / 2**20}

    def sample():
        while not done.wait(0.025):
            stats['peak_rss_mb'] = max(stats['peak_rss_mb'], process.memory_info().rss / 2**20)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats['fit_seconds'] = time.perf_counter() - start
        stats['peak_rss_mb'] = max(stats['peak_rss_mb'], process.memory_info().rss / 2**20)
        done.set()
        sampler.join(timeout=1)


def latency(call, repetitions=12):
    call()
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        call()
        timings.append((time.perf_counter() - start) * 1000)
    return {'latency_p50_ms': np.median(timings), 'latency_p95_ms': np.percentile(timings, 95)}


class LinearRating(FeatureRegression):
    def _estimator(self):
        return LinearRegression(n_jobs=1)


def pair_features(train, frames, movies):
    builder = LinearRating()
    builder.set_movie_metadata(movies)
    builder.fit(train)
    u, i = builder.encode_queries(train.userId, train.movieId)
    x_train = builder._features(u, i, exclude=train.rating.to_numpy())
    matrices = []
    for frame in frames:
        u, i = builder.encode_queries(frame.userId, frame.movieId)
        matrices.append(builder._features(u, i))
    return builder, x_train, matrices


def object_features(train, movies, request):
    """A shared, train-only representation used by both clustering algorithms."""
    genre_names = sorted({g for s in movies.genres.fillna('') for g in s.split('|') if g and g != '(no genres listed)'})
    metadata = movies.set_index('movieId')
    stats = train.groupby('movieId').rating.agg(['mean', 'count', 'std']).fillna(0)
    if request.cluster_object == 'movies':
        stats = stats.sort_values(['count'], ascending=False, kind='stable').head(request.max_items)
        ids = stats.index.to_numpy()
        genre_values = np.asarray([[float(g in str(metadata.loc[m, 'genres']).split('|')) for g in genre_names] for m in ids])
        x = np.column_stack([genre_values, stats['mean'], np.log1p(stats['count']), stats['std']])
        labels = [str(metadata.loc[m, 'title']) for m in ids]
    else:
        stats = train.groupby('userId').rating.agg(['mean', 'count', 'std']).fillna(0)
        ids = stats.index.to_numpy()
        genre_map = {int(m): set(str(g).split('|')) for m, g in zip(movies.movieId, movies.genres)}
        genre_values = []
        by_user = {u: frame for u, frame in train.groupby('userId')}
        for user in ids:
            frame = by_user[user]
            genre_values.append([float(np.mean([float(g in genre_map[int(m)]) for m in frame.movieId])) for g in genre_names])
        x = np.column_stack([genre_values, stats['mean'], np.log1p(stats['count']), stats['std']])
        labels = [f'User {u}' for u in ids]
    names = [*genre_names, 'Rating trung bình', 'log(1 + số rating)', 'Độ lệch chuẩn rating']
    return StandardScaler().fit_transform(x), np.asarray(x), labels, names


def points(xs, ys):
    return [{'x': native(x), 'y': native(y)} for x, y in zip(xs, ys)]


def chart(index, title, kind, x_label, y_label, series=None, split='train', **kwargs):
    return {'index': index, 'title': title, 'kind': kind, 'x_label': x_label,
            'y_label': y_label, 'series': series or [], 'split': split, **kwargs}


def series(label, xs, ys, **kwargs):
    return {'label': label, 'points': points(xs, ys), **kwargs}


def cost_charts(rows, split='train'):
    output = []
    for field, title, unit in [('fit_seconds', 'Thời gian huấn luyện / khai phá', 'Giây ↓'),
                               ('peak_rss_mb', 'Bộ nhớ đỉnh RSS của tiến trình', 'MiB'),
                               ('latency_p95_ms', 'Độ trễ p95', 'ms ↓')]:
        values = [r for r in rows if r.get(field) is not None]
        if values:
            output.append(chart(-1, title, 'bar', 'Phương pháp / cấu hình', unit,
                                [series(title, [r['name'] for r in values], [r[field] for r in values])], split))
    quality = next((key for key in ('rmse', 'macro_f1', 'silhouette', 'return_mean') if any(r.get(key) is not None for r in rows)), None)
    tradeoff = [r for r in rows if quality and r.get(quality) is not None and r.get('latency_p95_ms') is not None]
    if tradeoff:
        output.append(chart(-1, 'Chất lượng và độ trễ', 'scatter', 'p95 (ms) ↓', quality,
                            [series(r['name'], [r['latency_p95_ms']], [r[quality]]) for r in tradeoff], split,
                            note='Đối chiếu trong cùng giao thức đo. Với RMSE, thấp hơn tốt hơn; với F1/return, cao hơn tốt hơn.'))
    return output


def summarize(rows):
    """Average independent model seeds, preserving details from the first seed."""
    result = []
    for algorithm in dict.fromkeys(r['algorithm'] for r in rows):
        runs = [r for r in rows if r['algorithm'] == algorithm]
        row = dict(runs[0])
        row['n_runs'] = len(runs)
        row['std'] = {}
        for key in runs[0]:
            values = [r.get(key) for r in runs]
            if key not in ('seed', 'threshold') and all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in values):
                row[key] = float(np.mean(values))
                if len(values) > 1:
                    row['std'][key] = float(np.std(values, ddof=1))
        result.append(row)
    return result


class LabService:
    def __init__(self, settings):
        self.settings = settings
        self.directory = settings.artifacts_dir / 'experiment-lab'
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='movie-lab')
        self.active = None
        self.cancel_event = threading.Event()

    def catalog(self):
        data = self.settings.artifacts_dir / 'data'
        manifest = data / 'manifest.json'
        return {**CATALOG, 'data_ready': all((data / f'{s}.parquet').is_file() for s in ['train', 'validation', 'test', 'movies']),
                'dataset': json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else None}

    def list_runs(self, group=None):
        if not self.directory.exists():
            return []
        runs = []
        for path in sorted(self.directory.glob('*/job.json'), reverse=True)[:100]:
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if group is None or job['group'] == group:
                if job['status'] in ('queued', 'running') and job.get('process_id') != os.getpid():
                    job.update(status='interrupted', message='Máy chủ đã khởi động lại. Hãy chạy lại thực nghiệm.')
                runs.append(job)
        return runs

    def get(self, run_id):
        if not run_id or any(c not in '0123456789abcdefT-' for c in run_id):
            raise FileNotFoundError('Không tìm thấy lần chạy.')
        path = self.directory / run_id / 'job.json'
        if not path.is_file():
            raise FileNotFoundError('Không tìm thấy lần chạy.')
        job = json.loads(path.read_text(encoding='utf-8'))
        result = path.with_name('result.json')
        if result.is_file():
            job['result'] = json.loads(result.read_text(encoding='utf-8'))
        if job['status'] in ('queued', 'running') and job.get('process_id') != os.getpid():
            job.update(status='interrupted', message='Máy chủ đã khởi động lại. Hãy chạy lại thực nghiệm.')
        return job

    def start(self, request):
        if not self.catalog()['data_ready']:
            raise FileNotFoundError('Chưa có dữ liệu đã chuẩn bị. Hãy chuẩn bị dữ liệu trước khi thực nghiệm.')
        with self.lock:
            if self.active is not None:
                raise RuntimeError('Đang có một thực nghiệm chạy. Hãy đợi hoàn tất hoặc dừng lần chạy đó.')
            run_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
            folder = self.directory / run_id
            folder.mkdir(parents=True)
            job = {'id': run_id, 'group': request.group, 'status': 'queued', 'progress': 0,
                   'message': 'Đang chuẩn bị dữ liệu', 'created_at': datetime.now(UTC).isoformat(),
                   'request': request.model_dump(), 'process_id': os.getpid(), 'cancel_requested': False}
            save_json(folder / 'job.json', job)
            self.active = run_id
            self.cancel_event.clear()
            self.executor.submit(self._run, job, request)
            return job

    def cancel(self, run_id):
        with self.lock:
            job = self.get(run_id)
            if self.active == run_id and job['status'] in ('queued', 'running'):
                self.cancel_event.set()
                job.update(cancel_requested=True, message='Đang dừng sau bước tính toán hiện tại…')
                save_json(self.directory / run_id / 'job.json', job)
            return job

    def _load(self, request):
        data = self.settings.artifacts_dir / 'data'
        frames = [pd.read_parquet(data / f'{s}.parquet') for s in ('train', 'validation', 'test')]
        all_ratings = pd.concat(frames, ignore_index=True)
        users = np.sort(all_ratings.userId.unique())
        rng = np.random.default_rng(request.seed)
        selected = np.sort(rng.choice(users, min(request.max_users, len(users)), replace=False))
        sampled = all_ratings.loc[all_ratings.userId.isin(selected)].copy()
        # Preserve whole sampled histories and recompute strictly global temporal cuts.
        splits = chronological_split(sampled, 0.7, 0.15)
        movies = pd.read_parquet(data / 'movies.parquet')
        manifest = data / 'manifest.json'
        fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
        provenance = {'source': 'Prepared MovieLens sample; complete histories of sampled users',
                      'prepared_ratings': len(all_ratings), 'sampled_ratings': len(sampled),
                      'sampled_users': len(selected), 'data_seed': request.seed,
                      'data_manifest_sha256': fingerprint,
                      'sample_sha256': hashlib.sha256(pd.util.hash_pandas_object(sampled, index=False).values.tobytes()).hexdigest(),
                      'splits': {s: {'ratings': len(f), 'users': f.userId.nunique(), 'movies': f.movieId.nunique(),
                                     'min_timestamp': int(f.timestamp.min()), 'max_timestamp': int(f.timestamp.max())} for s, f in splits.items()}}
        return splits, movies, native(provenance)

    def _run(self, job, request):
        path = self.directory / job['id']

        def update(progress, message):
            if self.cancel_event.is_set():
                raise Cancelled()
            job.update(status='running', progress=int(progress), message=message)
            save_json(path / 'job.json', job)

        try:
            update(2, 'Lấy mẫu toàn bộ lịch sử user và chia theo thời gian')
            splits, movies, provenance = self._load(request)
            from movie_rating.lab_supervised import classification, rating
            from movie_rating.lab_unsupervised import association, clustering, pca_experiment
            from movie_rating.lab_reinforcement import reinforcement

            handlers = {'rating': rating, 'classification': classification, 'clustering': clustering,
                        'association': association, 'pca': pca_experiment, 'reinforcement': reinforcement}
            result = handlers[request.group](splits, movies, request, update)
            update(98, 'Lưu bảng tiêu chí, số liệu biểu đồ và dấu vết lần chạy')
            result.update(provenance=provenance, group=request.group, config=request.model_dump(),
                          notes=[*result.get('notes', []),
                                 'Kết quả thuộc mẫu user công bố ở trên; không đại diện tự động cho toàn bộ MovieLens 25M.',
                                 'RAM là RSS tiến trình lấy mẫu trong phép đo, có thể gồm bộ nhớ của máy chủ; không phải RAM riêng của mô hình.',
                                 'Chọn cấu hình trên validation; test chỉ báo cáo đánh giá cuối, không dùng để tuning tiếp.'])
            result['runtime'] = {'python': platform.python_version(),
                                 'libraries': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'torch', 'xgboost', 'mlxtend']},
                                 'code_sha256': hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes() for name in ['lab.py', 'lab_supervised.py', 'lab_unsupervised.py', 'lab_reinforcement.py', 'lab_catalog.json'])).hexdigest()}
            save_json(path / 'result.json', result)
            job.update(status='completed', progress=100, message='Đã hoàn tất', finished_at=datetime.now(UTC).isoformat())
        except Cancelled:
            job.update(status='cancelled', message='Đã dừng; không công bố kết quả chạy dở.', finished_at=datetime.now(UTC).isoformat())
        except Exception as error:
            job.update(status='failed', message=f'{type(error).__name__}: {error}', finished_at=datetime.now(UTC).isoformat())
        finally:
            with self.lock:
                save_json(path / 'job.json', job)
                self.active = None

    def csv_export(self, run_id):
        job = self.get(run_id)
        rows = job.get('result', {}).get('rows', [])
        flattened = [{k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in rows]
        fields = list(dict.fromkeys(k for row in flattened for k in row))
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
        return '\ufeff' + buffer.getvalue()
