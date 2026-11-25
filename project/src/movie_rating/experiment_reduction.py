"""Train-only, memory-bounded reducers shared by Experiment Studio adapters."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ReductionSpec:
    mode: str = "off"
    methods: tuple[str, ...] = ("pca",)
    components: tuple[int, ...] = (4, 8)
    seed: int = 42
    scale: bool = True

    def __post_init__(self):
        if self.mode not in {"off", "enabled", "compare"}:
            raise ValueError("Reduction mode phải là off, enabled hoặc compare.")
        if not self.methods or set(self.methods) - {"pca", "truncated_svd"}:
            raise ValueError("Reduction method phải là pca hoặc truncated_svd.")
        if not self.components or any(int(k) < 1 for k in self.components):
            raise ValueError("Components phải là số nguyên dương.")


class FeatureReducer:
    """Exact dense/sparse PCA or uncentered SVD using feature-sized moments.

    It reads every fit row in bounded batches and only stores feature covariance,
    never an N-by-N or user-by-item matrix.
    """

    def __init__(self, method="pca", components=2, seed=42, scale=True, batch_rows=65_536):
        if method not in {"pca", "truncated_svd"}:
            raise ValueError("method phải là pca hoặc truncated_svd.")
        if not isinstance(components, int) or components < 1:
            raise ValueError("components phải là số nguyên dương.")
        self.method = method
        self.components = components
        self.seed = int(seed)
        self.scale = bool(scale)
        self.batch_rows = int(batch_rows)
        self.fitted = False
        self.transform_seconds = 0.0
        self.transform_calls = 0

    @staticmethod
    def _batch(x, start, stop):
        chunk = x[start:stop]
        if sparse.issparse(chunk):
            chunk = chunk.toarray()
        array = np.asarray(chunk, dtype=np.float64)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("Reducer cần ma trận số hữu hạn, hai chiều.")
        return array

    def fit(self, X_fit, feature_schema=None, fit_scope="train"):
        started = time.perf_counter()
        rows, columns = X_fit.shape
        if rows < 2 or columns < 1:
            raise ValueError("Cần ít nhất hai hàng và một đặc trưng để fit reducer.")
        if self.components > min(rows - 1, columns):
            raise ValueError(
                f"components={self.components} vượt giới hạn dữ liệu "
                f"min(rows-1, features)={min(rows - 1, columns)}."
            )

        total = np.zeros(columns, dtype=np.float64)
        squares = np.zeros((columns, columns), dtype=np.float64)
        count = 0
        for start in range(0, rows, self.batch_rows):
            block = self._batch(X_fit, start, min(start + self.batch_rows, rows))
            total += block.sum(axis=0)
            squares += block.T @ block
            count += len(block)
        mean = total / count
        variance = np.maximum(np.diag(squares / count - np.outer(mean, mean)), 0)
        if not np.any(variance > 1e-12):
            raise ValueError("Feature family không có phương sai; không thể giảm chiều.")
        scale = np.sqrt(variance) if self.scale else np.ones(columns, dtype=np.float64)
        scale[scale < 1e-8] = 1.0
        self.input_mean_ = np.zeros(columns, dtype=np.float32)
        self.input_scale_ = scale.astype(np.float32)

        if self.method == "pca":
            covariance = (squares / count - np.outer(mean, mean)) / scale[:, None] / scale[None, :]
            self.center_ = (mean / scale).astype(np.float32)
        else:
            covariance = squares / count / scale[:, None] / scale[None, :]
            self.center_ = np.zeros(columns, dtype=np.float32)
        values, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        tolerance = max(float(values[0]) * 1e-10, 1e-12)
        rank = int(np.sum(values > tolerance))
        if rank < self.components:
            raise ValueError(
                f"Feature rank={rank}; components={self.components} không khả dụng."
            )
        self.components_ = vectors[:, : self.components].T.astype(np.float32)
        total_variance = float(np.sum(variance / np.square(scale)))
        if self.method == "pca":
            projected_variance = values[: self.components]
        else:
            # TruncatedSVD keeps the uncentered projection, but its explained
            # variance is var(X @ components.T), not the captured second moment.
            projected_mean = (mean / scale) @ vectors[:, : self.components]
            projected_variance = np.maximum(values[: self.components] - projected_mean**2, 0.0)
        self.explained_variance_ = projected_variance.astype(np.float32)
        self.total_variance_ = total_variance
        self.explained_variance_ratio_ = (
            projected_variance / max(total_variance, 1e-12)
        ).astype(np.float32)
        self.input_dimensions = int(columns)
        self.output_dimensions = int(self.components)
        self.fit_rows = int(count)
        self.feature_schema = feature_schema
        self.fit_scope = str(fit_scope)
        self.fit_seconds = time.perf_counter() - started
        fingerprint_data = {
            "method": self.method,
            "components": self.components,
            "rows": count,
            "scope": self.fit_scope,
            "schema": feature_schema,
        }
        digest = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True, default=str).encode())
        digest.update(self.components_.tobytes())
        digest.update(self.input_scale_.tobytes())
        self.fingerprint = digest.hexdigest()
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Reducer chưa được fit.")
        started = time.perf_counter()
        if len(X.shape) != 2 or X.shape[1] != self.input_dimensions:
            raise ValueError(
                f"Input có shape {X.shape}; cần (n, {self.input_dimensions})."
            )
        output = np.empty((X.shape[0], self.output_dimensions), dtype=np.float32)
        for start in range(0, X.shape[0], self.batch_rows):
            stop = min(start + self.batch_rows, X.shape[0])
            block = self._batch(X, start, stop)
            block = (block - self.input_mean_) / self.input_scale_
            output[start:stop] = (block - self.center_) @ self.components_.T
        self.transform_seconds += time.perf_counter() - started
        self.transform_calls += 1
        return output

    def inverse_transform(self, Z):
        if not self.fitted:
            raise RuntimeError("Reducer chưa được fit.")
        z = np.asarray(Z, dtype=np.float32)
        if z.ndim != 2 or z.shape[1] != self.output_dimensions:
            raise ValueError("Latent vector có số chiều không khớp với reducer.")
        values = z @ self.components_ + self.center_
        return values * self.input_scale_ + self.input_mean_

    def metadata(self):
        if not self.fitted:
            raise RuntimeError("Reducer chưa được fit.")
        return {
            "method": self.method,
            "components": self.components,
            "input_dimensions": self.input_dimensions,
            "output_dimensions": self.output_dimensions,
            "explained_variance_ratio_train": float(self.explained_variance_ratio_.sum()),
            "explained_variance_train": float(self.explained_variance_.sum()),
            "total_variance_train": self.total_variance_,
            "explained_variance_definition": (
                "projected_component_variance / total_variance_in_scaled_input"
            ),
            "centers_input": self.method == "pca",
            "fit_rows": self.fit_rows,
            "fit_scope": self.fit_scope,
            "scale": self.scale,
            "fit_seconds": self.fit_seconds,
            "transform_seconds": self.transform_seconds,
            "transform_calls": self.transform_calls,
            "fingerprint": self.fingerprint,
        }


def reduction_variants(spec: ReductionSpec):
    """Return stable raw/reduced representation descriptors for one run."""
    if spec.mode == "off":
        return [{"variant_id": "raw", "method": None, "components": None}]
    reduced = [
        {"variant_id": f"{method}_{k}", "method": method, "components": int(k)}
        for method in spec.methods
        for k in spec.components
    ]
    return ([{"variant_id": "raw", "method": None, "components": None}, *reduced]
            if spec.mode == "compare" else reduced)

