"""Recover NumPy/SciPy DLL paths mangled by Windows module enumeration.

Some Python 3.12 / threadpoolctl combinations return '?' for characters in
Vietnamese installation paths. Resolve only missing DLLs against the actual
installed package directories; do not suppress failures for existing paths.
"""

import os
from pathlib import Path


def repair_dll_path(filepath):
    if os.name != 'nt' or '?' not in filepath or Path(filepath).is_file():
        return filepath
    import numpy
    import scipy

    filename = Path(filepath).name
    marker = '\\site-packages\\'
    index = filepath.lower().find(marker)
    if index >= 0:
        candidate = Path(numpy.__file__).resolve().parent.parent / filepath[index + len(marker):]
        if candidate.is_file():
            return str(candidate)
    for module, directory in ((numpy, 'numpy.libs'), (scipy, 'scipy.libs')):
        candidate = Path(module.__file__).resolve().parent.parent / directory / filename
        if candidate.is_file():
            return str(candidate)
    return filepath


def enable_unicode_dll_paths():
    if os.name != 'nt':
        return
    from threadpoolctl import ThreadpoolController

    if getattr(ThreadpoolController, '_movie_unicode_paths', False):
        return
    original = ThreadpoolController._make_controller_from_path

    def load(self, filepath):
        return original(self, repair_dll_path(filepath))

    ThreadpoolController._make_controller_from_path = load
    ThreadpoolController._movie_unicode_paths = True
