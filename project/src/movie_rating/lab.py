"""Bounded, reproducible experiment jobs. Production prediction artifacts are untouched."""

import csv
import hashlib
import io
import json
import os
import platform
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.preprocessing import StandardScaler

from movie_rating.experiment_config import ALGORITHMS, ExperimentRequest
from movie_rating.experiment_data import ExperimentData
from movie_rating.experiment_logs import RunLogger, log_page
from movie_rating.models.regression import FeatureRegression, StreamingLinearRegression
from movie_rating.windows_dlls import enable_unicode_dll_paths

enable_unicode_dll_paths()

GROUPS = tuple(ALGORITHMS)
CATALOG = json.loads(Path(__file__).with_name('lab_catalog.json').read_text(encoding='utf-8'))


LabRequest = ExperimentRequest


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
    try:
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                # Windows readers/antivirus can briefly deny replacement of job.json.
                if attempt == 7:
                    raise
                time.sleep(.01*(attempt+1))
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    """Windows replacement/antivirus may briefly deny a simultaneous reader."""
    for attempt in range(8):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.01 * (attempt + 1))


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
        return StreamingLinearRegression()


def pair_features(train, frames, movies):
    builder = LinearRating()
    builder.set_movie_metadata(movies)
    builder.fit_statistics(train)
    u, i = builder.encode_queries(train.userId, train.movieId)
    x_train = builder._feature_batch(
        u, i, exclude=train.rating.to_numpy(),
        timestamps=train.timestamp.to_numpy() if 'timestamp' in train else None,
        movie_ids=train.movieId.to_numpy())
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
        self.data = ExperimentData(settings.artifacts_dir / 'data')

    def catalog(self):
        data = self.settings.artifacts_dir / 'data'
        manifest = data / 'manifest.json'
        ready = all((data / f'{s}.parquet').is_file() for s in ['train', 'validation', 'test', 'movies'])
        return {**CATALOG, 'data_ready': ready,
                'statistics': self.data.catalog_statistics() if ready else None,
                'dataset': json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else None}

    def list_runs(self, group=None):
        if not self.directory.exists():
            return []
        runs = []
        for path in sorted(self.directory.glob('*/job.json'), reverse=True)[:100]:
            try:
                job = read_json(path)
            except (OSError, ValueError):
                continue
            if job.get('schema_version') not in (2, 3) or job['group'] not in GROUPS:
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
        job = read_json(path)
        result = path.with_name('result.json')
        if result.is_file():
            job['result'] = read_json(result)
        if job['status'] in ('queued', 'running') and job.get('process_id') != os.getpid():
            job.update(status='interrupted', message='Máy chủ đã khởi động lại. Hãy chạy lại thực nghiệm.')
        return job

    def start(self, request):
        requested_at = datetime.now(UTC).isoformat()
        if not self.catalog()['data_ready']:
            raise FileNotFoundError('Chưa có dữ liệu đã chuẩn bị. Hãy chuẩn bị dữ liệu trước khi thực nghiệm.')
        with self.lock:
            if self.active is not None:
                raise RuntimeError('Đang có một thực nghiệm chạy. Hãy đợi hoàn tất hoặc dừng lần chạy đó.')
            run_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
            folder = self.directory / run_id
            folder.mkdir(parents=True)
            job = {'id': run_id, 'group': request.group, 'schema_version': 3, 'status': 'queued', 'progress': 0,
                   'message': 'Đang chuẩn bị dữ liệu', 'created_at': requested_at,
                   'request': request.model_dump(), 'process_id': os.getpid(), 'cancel_requested': False}
            save_json(folder / 'job.json', job)
            if request.group in ('rating', 'classification'):
                RunLogger(folder).write(
                    f"Đã tạo thực nghiệm {request.group} · thuật toán: {', '.join(request.algorithms)}",
                    progress=0)
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
                if job['group'] in ('rating', 'classification'):
                    RunLogger(self.directory / run_id).write('Đã nhận yêu cầu dừng.', 'warning')
            return job

    def logs_directory(self, run_id):
        if not run_id or any(c not in '0123456789abcdefT-' for c in run_id):
            raise FileNotFoundError('Không tìm thấy lần chạy.')
        directory = self.directory / run_id
        if not (directory / 'job.json').is_file():
            raise FileNotFoundError('Không tìm thấy lần chạy.')
        return directory

    def logs(self, run_id, cursor=0, limit=200):
        return log_page(self.logs_directory(run_id), cursor, limit)

    def _load(self, request):
        return self.data.load(request)

    def _run(self, job, request):
        try:
            if self.data.catalog_statistics()['ratings'] >= 1_000_000:
                from movie_rating.experiment_worker import run_isolated

                run_isolated(self, job, request)
            else:
                self._run_inline(job, request)
        except Exception as error:
            # Includes failures creating a scratch directory or starting a Windows worker.
            with self.lock:
                if self.active == job['id']:
                    job.update(status='failed', message=f'{type(error).__name__}: {error}',
                               finished_at=datetime.now(UTC).isoformat())
                    if request.group in ('rating', 'classification'):
                        RunLogger(self.directory / job['id']).write(job['message'], 'error')
                    save_json(self.directory / job['id'] / 'job.json', job)
                    self.active = None

    def _run_inline(self, job, request):
        path = self.directory / job['id']
        logger = RunLogger(path) if request.group in ('rating', 'classification') else None
        last_write = 0.0
        previous_message = None

        def update(progress, message):
            nonlocal last_write, previous_message
            if self.cancel_event.is_set():
                raise Cancelled()
            previous_progress = job['progress']
            next_progress = previous_progress if progress is None else max(
                previous_progress, int(progress))
            job.update(status='running', progress=next_progress, message=message)
            if logger and message != previous_message:
                logger.write(message, progress=next_progress if progress is not None else None)
                previous_message = message
            now = time.monotonic()
            if now - last_write >= 1 or next_progress != previous_progress:
                save_json(path / 'job.json', job)
                last_write = now

        try:
            message = (
                'Đọc dữ liệu; chuẩn bị lịch sử theo user cho hồi quy/phân loại'
                if request.use_all_data
                else 'Lấy mẫu lịch sử MovieLens, chia 70/15/15 với seed 42'
            )
            update(2, message)
            splits, movies, provenance = self._load(request)
            update(None, 'Đã nạp dữ liệu · ' + ' · '.join(
                f'{name}: {len(frame):,} rating' for name, frame in splits.items()))
            from movie_rating.experiment_pipeline import execute

            result = execute(splits, movies, request, update)
            update(98, 'Lưu bảng tiêu chí, số liệu biểu đồ và dấu vết lần chạy')
            data_note = (
                'Kết quả dùng toàn bộ dữ liệu đã chuẩn bị.'
                if request.use_all_data
                else 'Kết quả thuộc mẫu user công bố ở trên; không đại diện tự động cho toàn bộ MovieLens 25M.'
            )
            result.update(provenance=provenance, group=request.group, config=request.model_dump(),
                          notes=[*result.get('notes', []), data_note,
                                 'Chọn cấu hình trên validation; test chỉ báo cáo đánh giá cuối, không dùng để tuning tiếp.'])
            device = 'CPU'
            if (
                request.group == 'rating'
                and 'ncf' in request.algorithms
                and torch.cuda.is_available()
            ):
                device = f'CUDA · {torch.cuda.get_device_name(0)}'
            result['runtime'] = {'python': platform.python_version(),
                                 'config_sha256': hashlib.sha256(json.dumps(request.model_dump(), sort_keys=True).encode()).hexdigest(),
                                 'platform': platform.platform(), 'processor': platform.processor(),
                                 'cpu_count': os.cpu_count(), 'ram_bytes': psutil.virtual_memory().total,
                                 'device': device, 'time_unit': 'seconds',
                                 'measurement': 'perf_counter; fit includes preprocessing; RSS process-wide',
                                 'libraries': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'torch', 'xgboost', 'mlxtend']},
                                 'code_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.rglob('*.py'))) + Path(__file__).with_name('lab_catalog.json').read_bytes()).hexdigest()}
            save_json(path / 'result.json', result)
            job.update(status='completed', progress=100, message='Đã hoàn tất', finished_at=datetime.now(UTC).isoformat())
            if logger:
                logger.write('Đã hoàn tất và lưu kết quả thực nghiệm.', 'success', progress=100)
        except Cancelled:
            job.update(status='cancelled', message='Đã dừng; không công bố kết quả chạy dở.', finished_at=datetime.now(UTC).isoformat())
            if logger:
                logger.write(job['message'], 'warning')
        except Exception as error:
            job.update(status='failed', message=f'{type(error).__name__}: {error}', finished_at=datetime.now(UTC).isoformat())
            if logger:
                logger.write(job['message'], 'error')
                logger.write(traceback.format_exc(limit=8), 'error')
        finally:
            if self.data.full_statistics and self.data.full_statistics['ratings'] >= 1_000_000:
                self.data.sample = self.data.sample_key = None
            with self.lock:
                save_json(path / 'job.json', job)
                self.active = None

    def csv_export(self, run_id):
        job = self.get(run_id)
        flattened = self.export_rows(job)
        for rule in job.get('result', {}).get('rules', []):
            flattened.append({'record_type': 'association_rule', **{
                key: json.dumps(native(value), ensure_ascii=False) if isinstance(value, list) else value
                for key, value in rule.items()}})
        fields = list(dict.fromkeys(k for row in flattened for k in row))
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
        return '\ufeff' + buffer.getvalue()

    @staticmethod
    def export_rows(job):
        result = job.get('result', {})
        rows = []
        for row in result.get('rows', []):
            flat = {k: json.dumps(native(v), ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()}
            for key, stats in result.get('cv', {}).get(row['algorithm'], {}).get('metrics', {}).items():
                flat[f'cv_{key}_mean'] = stats['mean']
                flat[f'cv_{key}_std'] = stats['std']
            configs = result.get('best_configs', {})
            config = configs.get(row['algorithm'])
            if config is None:
                config = next((v for k, v in configs.items()
                               if row['algorithm'].startswith(k + '_')), {})
            flat['best_config'] = json.dumps(config)
            rows.append(flat)
        return rows

    def excel_export(self, run_id):
        job = self.get(run_id)
        result = job['result']
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            pd.DataFrame(self.export_rows(job)).to_excel(writer, sheet_name='Metrics', index=False)
            if result.get('rules'):
                rules = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                          for k, v in r.items()} for r in result['rules']]
                pd.DataFrame(rules).to_excel(writer, sheet_name='Rules', index=False)
            if result.get('cv'):
                folds = [{'algorithm': a, **{k: v for k, v in row.items() if not isinstance(v, list)}}
                         for a, data in result['cv'].items() for row in data['folds']]
                pd.DataFrame(folds).to_excel(writer, sheet_name='CV folds', index=False)
            pd.DataFrame({'key': ['request', 'provenance', 'runtime', 'selection'],
                          'value': [json.dumps(result.get(k, job.get(k)), ensure_ascii=False)
                                    for k in ['request', 'provenance', 'runtime', 'selection']]}).to_excel(
                                        writer, sheet_name='Protocol', index=False)
            # Preserve movie titles literally even if a title starts with '='.
            for sheet in writer.book:
                for cells in sheet:
                    for cell in cells:
                        if isinstance(cell.value, str):
                            cell.data_type = 's'
        return buffer.getvalue()
