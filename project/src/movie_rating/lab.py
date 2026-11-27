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
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import psutil
import torch

from movie_rating.experiment_config import ALGORITHMS, ExperimentRequest, preset_catalog
from movie_rating.experiment_data import ExperimentData
from movie_rating.experiment_logs import LOGGED_GROUPS, RunLogger, log_page
from movie_rating.windows_dlls import enable_unicode_dll_paths

enable_unicode_dll_paths()

GROUPS = tuple(ALGORITHMS)
CATALOG = json.loads(Path(__file__).with_name('lab_catalog.json').read_text(encoding='utf-8'))
CATALOG['groups'] = [group for group in CATALOG['groups'] if group['id'] == 'rating']
PRESETS = preset_catalog()


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


def points(xs, ys):
    return [{'x': native(x), 'y': native(y)} for x, y in zip(xs, ys)]


def chart(index, title, kind, x_label, y_label, series=None, split='train', **kwargs):
    return {'index': index, 'title': title, 'kind': kind, 'x_label': x_label,
            'y_label': y_label, 'series': series or [], 'split': split, **kwargs}


def series(label, xs, ys, **kwargs):
    return {'label': label, 'points': points(xs, ys), **kwargs}


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
        return {**CATALOG, 'data_ready': ready, 'presets': PRESETS,
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
            if job.get('schema_version') not in (2, 3, 4) or job['group'] not in GROUPS:
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
        if job['status'] in ('queued', 'running') and job.get('process_id') != os.getpid():
            job.update(status='interrupted', message='Máy chủ đã khởi động lại. Hãy chạy lại thực nghiệm.')
        result = path.with_name('result.json')
        if result.is_file():
            job['result'] = read_json(result)
            if job.get('group') == 'rating' and job.get('status') == 'completed' and not job['result'].get('baseline'):
                from movie_rating.experiment_baseline import baseline_from_provenance

                with self.lock:
                    saved = read_json(result)
                    if not saved.get('baseline'):
                        try:
                            baseline = baseline_from_provenance(
                                saved.get('provenance', {}), self.settings.artifacts_dir,
                            )
                            if baseline:
                                saved['baseline'] = baseline
                                save_json(result, saved)
                        except (OSError, ValueError, KeyError):
                            pass
                    job['result'] = saved
        from movie_rating.experiment_resume import ProgressStore

        # The progress snapshot is atomically replaced by the worker. Reading it
        # from every UI poll can keep the destination open on Windows and block
        # os.replace; only terminal runs need resume metadata.
        progress = (
            ProgressStore.summary(path.parent)
            if job["status"] in ("cancelled", "failed", "interrupted", "completed")
            else None
        )
        if progress:
            job['completed_work_units'] = progress['completed_work_units']
            job['epoch_checkpoint_available'] = progress['active_fit_checkpoint']
        job['resume_available'] = bool(
            progress
            and job.get('group') == 'rating'
            and job.get('status') in ('cancelled', 'failed', 'interrupted')
            and not job.get('resume_blocked')
            and not result.exists()
        )
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
            job = {'id': run_id, 'group': request.group, 'schema_version': 4, 'status': 'queued', 'progress': 0,
                   'message': 'Đang chuẩn bị dữ liệu', 'created_at': requested_at,
                   'request': request.model_dump(), 'process_id': os.getpid(), 'cancel_requested': False,
                   'resume_count': 0}
            save_json(folder / 'job.json', job)
            if request.group in LOGGED_GROUPS:
                RunLogger(folder).write(
                    f"Đã tạo thực nghiệm {request.group} · thuật toán: {', '.join(request.algorithms)}",
                    progress=0)
            self.active = run_id
            self.cancel_event.clear()
            self.executor.submit(self._run, job, request)
            return job

    def resume(self, run_id):
        with self.lock:
            job = self.get(run_id)
            if self.active is not None:
                raise RuntimeError('Đang có một thực nghiệm chạy. Hãy đợi hoàn tất trước khi tiếp tục.')
            if job.get('group') != 'rating':
                raise RuntimeError('Only regression runs can resume from checkpoints.')
            if job.get('status') not in ('cancelled', 'failed', 'interrupted'):
                raise RuntimeError('Chỉ có thể tiếp tục lần chạy đã dừng, lỗi hoặc bị gián đoạn.')
            if not job.get('resume_available'):
                raise RuntimeError(
                    'Lần chạy này không có checkpoint tương thích. Hãy tạo một thực nghiệm mới; '
                    'các lần chạy cũ không có checkpoint không thể khôi phục.'
                )
            folder = self.directory / run_id
            if (folder / 'result.json').exists():
                raise RuntimeError('Lần chạy đã có báo cáo hoàn tất và không cần tiếp tục.')
            request = ExperimentRequest.model_validate(job['request'])
            job.update(
                status='queued',
                message='Đang xác minh dữ liệu và tiếp tục từ các checkpoint đã lưu',
                cancel_requested=False,
                process_id=os.getpid(),
                resume_count=job.get('resume_count', 0) + 1,
            )
            job.pop('finished_at', None)
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
                if job['group'] in LOGGED_GROUPS:
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
                    if request.group in LOGGED_GROUPS:
                        RunLogger(self.directory / job['id']).write(job['message'], 'error')
                    save_json(self.directory / job['id'] / 'job.json', job)
                    self.active = None

    def _run_inline(self, job, request):
        path = self.directory / job['id']
        logger = RunLogger(path) if request.group in LOGGED_GROUPS else None
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
                'Read prepared per-user chronological rating history'
                if request.use_all_data
                else 'Sample MovieLens user histories and split them 70/15/15 with seed 42'
            )
            update(2, message)
            splits, movies, provenance = self._load(request)
            update(None, 'Đã nạp dữ liệu · ' + ' · '.join(
                f'{name}: {len(frame):,} rating' for name, frame in splits.items()))
            from movie_rating.experiment_pipeline import execute
            from movie_rating.experiment_resume import ProgressStore, resume_fingerprint

            fingerprint = resume_fingerprint(request, splits, movies, provenance)
            progress_store = ProgressStore(path, fingerprint)
            if job.get('resume_count', 0):
                update(None, 'Data, splits, parameters, seed, libraries, and code verified for resume.')
            from movie_rating.native_threads import limit_native_threadpools

            thread_limit = request.parameters.xgb_n_jobs if 'xgboost' in request.algorithms else 2
            with limit_native_threadpools(
                thread_limit,
                include_xgboost='xgboost' in request.algorithms,
            ):
                result = execute(
                    splits, movies, request, update, progress_store=progress_store
                )
            update(98, 'Lưu bảng tiêu chí, số liệu biểu đồ và dấu vết lần chạy')
            data_note = (
                'Kết quả dùng toàn bộ dữ liệu đã chuẩn bị.'
                if request.use_all_data
                else 'Kết quả thuộc mẫu user công bố ở trên; không đại diện tự động cho toàn bộ MovieLens 25M.'
            )
            protocol_note = (
                'GridSearchCV with cv=5 selects parameters using Train only; Validation compares models; '
                'Test reports final held-out scores. Linear Regression scales features with StandardScaler.'
            )
            result.update(provenance=provenance, group=request.group, config=request.model_dump(),
                          notes=[*result.get('notes', []), data_note, protocol_note])
            result['sample_sha256'] = provenance.get('sample_sha256')
            result['protocol'] = provenance.get('protocol')
            for row in result.get('rows', []):
                row.setdefault('variant_id', row.get('algorithm'))
                row.setdefault('base_algorithm', row.get('algorithm'))
                row.setdefault('representation', 'raw')
                row.setdefault('reduction_method', None)
                row.setdefault('components', None)
                row.setdefault('fit_scope', 'train')
                row.setdefault('sample_sha256', result['sample_sha256'])
                row.setdefault('protocol', result['protocol'])
            for key, config in result.get('best_configs', {}).items():
                if isinstance(config, dict):
                    config.setdefault('variant_id', key)
                    config.setdefault('protocol', result['protocol'])
                    config.setdefault('sample_sha256', result['sample_sha256'])
            device = 'CPU'
            if 'ncf' in request.algorithms and torch.cuda.is_available():
                device = f'CUDA · {torch.cuda.get_device_name(0)}'
            result['runtime'] = {'python': platform.python_version(),
                                 'config_sha256': hashlib.sha256(json.dumps(request.model_dump(), sort_keys=True).encode()).hexdigest(),
                                 'platform': platform.platform(), 'processor': platform.processor(),
                                 'cpu_count': os.cpu_count(), 'ram_bytes': psutil.virtual_memory().total,
                                 'device': device, 'time_unit': 'seconds',
                                 'xgb_n_jobs': request.parameters.xgb_n_jobs
                                 if 'xgboost' in request.algorithms
                                 else None,
                                 'threadpool_limit': thread_limit,
                                 'torch_cuda_runtime': torch.version.cuda,
                                 'measurement': 'perf_counter; fit includes preprocessing; RSS process-wide',
                                 'libraries': {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'torch', 'xgboost']},
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
            from movie_rating.experiment_resume import ResumeStateMismatch

            job.update(
                status='failed', message=f'{type(error).__name__}: {error}',
                finished_at=datetime.now(UTC).isoformat(),
                resume_blocked=isinstance(error, ResumeStateMismatch),
            )
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
            variant_id = row.get('variant_id') or row['algorithm']
            cv_data = result.get('cv', {}).get(variant_id) or result.get('cv', {}).get(row['algorithm'], {})
            for key, stats in cv_data.get('metrics', {}).items():
                flat[f'cv_{key}_mean'] = stats['mean']
                flat[f'cv_{key}_std'] = stats['std']
            configs = result.get('best_configs', {})
            config = configs.get(variant_id) or configs.get(row['algorithm'])
            if config is None:
                config = next((v for k, v in configs.items()
                               if row['algorithm'].startswith(k + '_')), {})
            flat['best_config'] = json.dumps(config, ensure_ascii=False)
            rows.append(flat)
        return rows

    def excel_export(self, run_id):
        job = self.get(run_id)
        result = job['result']
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            pd.DataFrame(self.export_rows(job)).to_excel(writer, sheet_name='Metrics', index=False)
            if result.get('cv'):
                folds = [{'algorithm': a, **{k: v for k, v in row.items() if not isinstance(v, list)}}
                         for a, data in result['cv'].items() for row in data['folds']]
                pd.DataFrame(folds).to_excel(writer, sheet_name='CV folds', index=False)
            pd.DataFrame({'key': ['request', 'provenance', 'runtime', 'selection', 'reduction'],
                          'value': [json.dumps(result.get(k, job.get(k)), ensure_ascii=False)
                                    for k in ['request', 'provenance', 'runtime', 'selection', 'reduction']]}).to_excel(
                                        writer, sheet_name='Protocol', index=False)
            # Preserve movie titles literally even if a title starts with '='.
            for sheet in writer.book:
                for cells in sheet:
                    for cell in cells:
                        if isinstance(cell.value, str):
                            cell.data_type = 's'
        return buffer.getvalue()
