"""Bounded workspaces and disk-backed feature buffers for model training."""

import os
import pickle
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from movie_rating.settings import PROJECT_ROOT

BATCH_ROWS = 65_536
MEMMAP_THRESHOLD_BYTES = 64 * 2**20


def scratch_root(native_ascii=False):
    root = Path(os.environ.get("MOVIE_SCRATCH_DIR", PROJECT_ROOT / "artifacts" / "scratch"))
    root.mkdir(parents=True, exist_ok=True)
    if native_ascii and os.name == "nt" and not str(root).isascii():
        # dmlc's Windows file API cannot open UTF-8 cache paths. Prefer an 8.3
        # alias on the same drive, then a writable ASCII temporary directory.
        import ctypes

        buffer = ctypes.create_unicode_buffer(32768)
        length = ctypes.windll.kernel32.GetShortPathNameW(str(root), buffer, len(buffer))
        if length and length < len(buffer) and buffer.value.isascii():
            return Path(buffer.value)
        for location in (tempfile.gettempdir(), os.environ.get("PUBLIC", "")):
            if location and str(location).isascii():
                candidate = Path(location) / "movie-rating-scratch"
                try:
                    candidate.mkdir(parents=True, exist_ok=True)
                    return candidate
                except OSError:
                    continue
        raise OSError("XGBoost cần thư mục cache không dấu trên Windows; đặt MOVIE_SCRATCH_DIR.")
    return root


@contextmanager
def workspace(prefix="features-", native_ascii=False):
    root = scratch_root(native_ascii=native_ascii)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=root) as directory:
        yield Path(directory)


@contextmanager
def feature_buffer(rows, columns, batches):
    """Small matrices stay in RAM; large matrices have a single float32 file backing."""
    size = rows * columns * 4
    with workspace() as directory:
        if size > MEMMAP_THRESHOLD_BYTES:
            if shutil.disk_usage(directory).free < size + 512 * 2**20:
                raise OSError("Ổ đĩa không đủ chỗ lưu ma trận đặc trưng tạm.")
            matrix = np.memmap(
                directory / "features.bin", dtype="float32", mode="w+", shape=(rows, columns)
            )
        else:
            matrix = np.empty((rows, columns), dtype=np.float32)
        try:
            offset = 0
            for features in batches:
                stop = offset + len(features)
                matrix[offset:stop] = features
                offset = stop
            if offset != rows:
                raise ValueError("Feature batches do not match the training rows.")
            if isinstance(matrix, np.memmap):
                matrix.flush()
            yield matrix
        finally:
            if isinstance(matrix, np.memmap):
                matrix._mmap.close()


class _Counter:
    def __init__(self):
        self.size = 0

    def write(self, value):
        self.size += len(value)
        return len(value)


def serialized_size(value):
    """Count pickle bytes without keeping another model-sized byte string in RAM."""
    counter = _Counter()
    pickle.Pickler(counter, protocol=pickle.DEFAULT_PROTOCOL).dump(value)
    return counter.size
