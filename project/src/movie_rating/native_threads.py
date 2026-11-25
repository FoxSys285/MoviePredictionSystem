"""Thread-pool limits that also work when Windows paths contain non-ASCII characters."""

import os
from contextlib import contextmanager
from pathlib import Path

import psutil
from threadpoolctl import ThreadpoolController


class _WindowsPathThreadpoolController(ThreadpoolController):
    """Repair module paths returned through the Windows ANSI process API."""

    def _make_controller_from_path(self, filepath):
        if os.name == "nt" and not os.path.exists(filepath):
            basename = Path(filepath).name.casefold()
            try:
                mapped_paths = psutil.Process().memory_maps()
            except psutil.Error:
                mapped_paths = ()
            filepath = next(
                (
                    module.path
                    for module in mapped_paths
                    if Path(module.path).name.casefold() == basename
                    and os.path.isfile(module.path)
                ),
                filepath,
            )
        return super()._make_controller_from_path(filepath)


@contextmanager
def limit_native_threadpools(limits, *, include_xgboost=False):
    """Apply a native thread limit and yield the effective threadpool inventory.

    XGBoost's own ``n_jobs`` remains its primary setting. Import it before taking
    the inventory so threadpoolctl also sees any OpenMP runtime it loads.
    """
    if include_xgboost:
        import xgboost  # noqa: F401

    controller_class = (
        _WindowsPathThreadpoolController if os.name == "nt" else ThreadpoolController
    )
    controller = controller_class()
    with controller.limit(limits=limits):
        yield controller.info()
