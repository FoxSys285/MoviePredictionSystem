"""Exact Euclidean DBSCAN without retaining the quadratic neighborhood graph."""

import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn import config_context
from sklearn.neighbors import NearestNeighbors

QUERY_BATCH = 512


class Neighborhoods:
    """Reuse a full-data index and exact k-th distances across eps candidates."""

    def __init__(self, x):
        self.x = x
        self.index = NearestNeighbors(algorithm="brute", n_jobs=1).fit(x)
        self.distances = {}
        self.neighbor_distances = None
        self.neighbor_indices = None

    def prepare(self, min_samples, update=lambda *_: None):
        missing = sorted({k for k in min_samples if k not in self.distances})
        if not missing:
            return
        for k in missing:
            self.distances[k] = np.full(len(self.x), np.inf, dtype=np.float64)
        valid = [k for k in missing if k <= len(self.x)]
        if not valid:
            return
        width = max(valid)
        if self.neighbor_distances is not None and self.neighbor_distances.shape[1] >= width:
            for k in valid:
                self.distances[k][:] = self.neighbor_distances[:, k - 1]
            return
        self.neighbor_distances = np.empty((len(self.x), width), dtype=np.float64)
        self.neighbor_indices = np.empty((len(self.x), width), dtype=np.int32)
        for start in range(0, len(self.x), QUERY_BATCH):
            stop = min(start + QUERY_BATCH, len(self.x))
            # Brute-force BLAS is much faster than a kd-tree for the 38-dimensional
            # taste vectors. The reducer retains only k results per query row.
            with config_context(working_memory=64):
                distances, indices = self.index.kneighbors(self.x[start:stop], n_neighbors=width)
            self.neighbor_distances[start:stop] = distances
            self.neighbor_indices[start:stop] = indices
            for k in valid:
                self.distances[k][start:stop] = distances[:, k - 1]
            update(None, f"DBSCAN · khoảng cách láng giềng đầy đủ {stop:,}/{len(self.x):,} user")


class CoreNeighbors:
    def __init__(self, components):
        self.index = NearestNeighbors(algorithm="brute", n_jobs=1).fit(components)

    def kneighbors(self, x):
        with config_context(working_memory=64):
            return self.index.kneighbors(x, n_neighbors=1)


class FullDBSCAN:
    """Same density/core connectivity as DBSCAN, with O(n*d) retained storage.

    Connected components are expanded against a tree of undiscovered core points.
    Rebuilding after half the points are discovered avoids enumerating the same
    dense edges repeatedly. Border points choose the earliest adjacent component,
    matching the usual input-order DBSCAN convention.
    """

    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, x, neighborhoods=None, update=lambda *_: None):
        if neighborhoods is None:
            neighborhoods = Neighborhoods(x)
        if neighborhoods.x is not x:
            raise ValueError("DBSCAN index must belong to the complete fit matrix.")
        neighborhoods.prepare([self.min_samples], update)
        core = neighborhoods.distances[self.min_samples] <= self.eps
        self.core_sample_indices_ = np.flatnonzero(core)
        labels = np.full(len(x), -1, dtype=np.int32)
        core_ids = self.core_sample_indices_
        remaining = len(core_ids)
        update(
            None,
            f"DBSCAN · xác định core: {remaining:,}/{len(x):,} user · "
            f"eps={self.eps:.5f} · min_samples={self.min_samples}",
        )
        update(
            None,
            "DBSCAN · nối cụm core trên toàn bộ user Train; không giữ đồ thị khoảng cách đầy đủ",
        )
        # Known k-neighbor edges are a subset of the exact eps graph. Connecting
        # these first is safe and often proves that a dense core graph is already
        # connected, without enumerating its billions of redundant edges.
        fragments = []
        membership = np.full(len(x), -1, dtype=np.int32)
        if remaining:
            core_position = np.full(len(x), -1, dtype=np.int32)
            core_position[core_ids] = np.arange(len(core_ids))
            indices = neighborhoods.neighbor_indices[core_ids]
            valid = (neighborhoods.neighbor_distances[core_ids] <= self.eps) & core[indices]
            row = np.broadcast_to(np.arange(len(core_ids))[:, None], valid.shape)[valid]
            column = core_position[indices[valid]]
            graph = csr_matrix(
                (np.ones(len(row), dtype=np.uint8), (row, column)),
                shape=(len(core_ids), len(core_ids)),
            )
            count, groups = connected_components(graph, directed=False)
            order = np.argsort(groups, kind="stable")
            boundaries = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1, len(order)]
            fragments = [core_ids[order[boundaries[i] : boundaries[i + 1]]] for i in range(count)]
            membership[core_ids] = groups
            del graph, indices, valid, row, column, core_position
        tree_ids = core_ids
        tree = cKDTree(x[tree_ids]) if remaining else None
        cluster = 0
        last_update = 0.0
        for seed in core_ids:
            if labels[seed] >= 0:
                continue
            initial = fragments[membership[seed]]
            labels[initial] = cluster
            remaining -= len(initial)
            stack = list(initial)
            while stack and remaining:
                point = stack.pop()
                now = time.monotonic()
                if now - last_update >= 0.5:
                    update(
                        None,
                        f"DBSCAN · nối cụm core {len(core_ids) - remaining:,}/{len(core_ids):,}",
                    )
                    last_update = now
                if remaining * 2 < len(tree_ids):
                    tree_ids = core_ids[labels[core_ids] < 0]
                    tree = cKDTree(x[tree_ids])
                neighbors = tree.query_ball_point(x[point], self.eps, return_sorted=False)
                discovered = tree_ids[np.asarray(neighbors, dtype=np.intp)]
                discovered = discovered[labels[discovered] < 0]
                if len(discovered):
                    for group in np.unique(membership[discovered]):
                        members = fragments[group]
                        labels[members] = cluster
                        remaining -= len(members)
                        stack.extend(members)
            cluster += 1
            if not remaining:
                break
        del tree
        self.components_ = np.ascontiguousarray(x[core_ids])
        if len(core_ids):
            borders = np.flatnonzero(~core)
            for start in range(0, len(borders), QUERY_BATCH):
                ids = borders[start : start + QUERY_BATCH]
                # Every eps neighbor of a non-core point is in its cached first k
                # neighbors, since its k-th distance exceeds eps. No second full
                # distance search is needed. Border points never propagate labels.
                indices = neighborhoods.neighbor_indices[ids]
                valid = (neighborhoods.neighbor_distances[ids] <= self.eps) & core[indices]
                candidates = np.where(valid, labels[indices], np.iinfo(np.int32).max)
                assigned = candidates.min(axis=1)
                assigned[assigned == np.iinfo(np.int32).max] = -1
                labels[ids] = assigned
                update(
                    None,
                    f"DBSCAN · gán border {min(start + len(ids), len(borders)):,}/{len(borders):,}",
                )
        self.labels_ = labels
        self.n_features_in_ = x.shape[1]
        update(
            None,
            f"DBSCAN · hoàn tất gán nhãn {len(labels):,} user · "
            f"{cluster} cụm · noise {np.mean(labels < 0):.2%}",
        )
        return self
