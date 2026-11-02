import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, diags

from movie_rating.models.baseline import BiasBaseline


@njit(cache=True)
def _predict_neighbors(users, movies, ptr, cols, residual, sim_ptr, sim_cols, sim_data):
    corrections = np.zeros(len(users), dtype=np.float32)
    for row in range(len(users)):
        u, i = users[row], movies[row]
        if u < 0 or i < 0:
            continue
        left, right = ptr[u], sim_ptr[i]
        numerator, denominator = 0.0, 0.0
        while left < ptr[u + 1] and right < sim_ptr[i + 1]:
            if cols[left] == sim_cols[right]:
                weight = sim_data[right]
                numerator += weight * residual[left]
                denominator += abs(weight)
                left += 1
                right += 1
            elif cols[left] < sim_cols[right]:
                left += 1
            else:
                right += 1
        if denominator > 1e-8:
            corrections[row] = numerator / denominator
    return corrections


class ItemKNN(BiasBaseline):
    """Top-k positive cosine neighbors on train-only bias residuals."""

    def __init__(self, neighbors=40, block_size=64, seed=42):
        super().__init__(seed=seed)
        if neighbors < 1 or block_size < 1:
            raise ValueError("neighbors and block_size must be positive.")
        self.neighbors, self.block_size = neighbors, block_size

    def fit(self, ratings):
        super().fit(ratings)
        users, movies, values = self.encode_training(ratings)
        residual = values - self.global_mean - self.user_bias[users] - self.movie_bias[movies]
        self.history = csr_matrix(
            (residual.astype(np.float32), (users, movies)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )
        self.history.sort_indices()
        items = self.history.T.tocsr()
        norms = np.sqrt(np.asarray(items.multiply(items).sum(axis=1)).ravel())
        normalized = (diags(1 / np.maximum(norms, 1e-8)) @ items).tocsr()
        # Only materialize block_size x n_items, never n_items x n_items.
        indices, weights, indptr = [], [], [0]
        for start in range(0, len(self.movie_ids), self.block_size):
            end = min(start + self.block_size, len(self.movie_ids))
            block = (normalized[start:end] @ normalized.T).toarray()
            for offset, scores in enumerate(block):
                scores[start + offset] = 0
                k = min(self.neighbors, len(scores))
                top = np.argpartition(scores, len(scores) - k)[-k:]
                top = np.sort(top[scores[top] > 0])
                indices.extend(top.tolist())
                weights.extend(scores[top].tolist())
                indptr.append(len(indices))
        self.similarity = csr_matrix(
            (
                np.asarray(weights, dtype=np.float32),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(len(self.movie_ids), len(self.movie_ids)),
        )
        return self

    def predict(self, user_ids, movie_ids):
        users, movies = self.encode_queries(user_ids, movie_ids)
        # Use unclipped biases so clipping happens once after the correction.
        scores = np.full(len(users), self.global_mean)
        known_u, known_i = users >= 0, movies >= 0
        scores[known_u] += self.user_bias[users[known_u]]
        scores[known_i] += self.movie_bias[movies[known_i]]
        scores += _predict_neighbors(
            users,
            movies,
            self.history.indptr,
            self.history.indices,
            self.history.data,
            self.similarity.indptr,
            self.similarity.indices,
            self.similarity.data,
        )
        return self.clip(scores)
