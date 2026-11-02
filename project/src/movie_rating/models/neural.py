"""Neural collaborative filtering adapted to explicit 0.5–5 ratings on CPU."""

import copy

import numpy as np
import torch
from torch import nn

from movie_rating.models.baseline import BiasBaseline


class RatingNetwork(nn.Module):
    def __init__(self, users, items, factors, hidden_size, dropout):
        super().__init__()
        self.users = nn.Embedding(users, factors)
        self.items = nn.Embedding(items, factors)
        self.mlp = nn.Sequential(
            nn.Linear(3 * factors, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        nn.init.normal_(self.users.weight, std=0.05)
        nn.init.normal_(self.items.weight, std=0.05)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def score(self, user_vectors, items):
        item_vectors = self.items(items)
        return self.mlp(
            torch.cat(
                [
                    user_vectors,
                    item_vectors,
                    user_vectors * item_vectors,
                ],
                dim=1,
            )
        ).squeeze(-1)

    def forward(self, users, items):
        return self.score(self.users(users), items)


class NeuralCF(BiasBaseline):
    def __init__(
        self,
        factors=16,
        hidden_size=32,
        epochs=12,
        batch_size=2048,
        learning_rate=0.003,
        regularization=0.01,
        dropout=0.1,
        patience=3,
        profile_steps=60,
        seed=42,
    ):
        super().__init__(epochs=epochs, regularization=regularization, seed=seed)
        if min(factors, hidden_size, batch_size, learning_rate, patience, profile_steps) <= 0:
            raise ValueError("Neural network parameters must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.factors, self.hidden_size, self.batch_size = factors, hidden_size, batch_size
        self.learning_rate, self.dropout, self.patience = learning_rate, dropout, patience
        self.profile_steps, self.seed = profile_steps, seed

    def fit(self, ratings, validation=None):
        super().fit(ratings)
        users, items, values = self.encode_training(ratings)
        torch.set_num_threads(4)
        # Keep initialization and dropout reproducible without advancing callers' RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network = RatingNetwork(
                len(self.user_ids),
                len(self.movie_ids),
                self.factors,
                self.hidden_size,
                self.dropout,
            )
            optimizer = torch.optim.AdamW(
                self.network.parameters(), lr=self.learning_rate, weight_decay=self.regularization
            )
            user_tensor, item_tensor = torch.from_numpy(users), torch.from_numpy(items)
            residual = torch.tensor(
                values - self.global_mean - self.user_bias[users] - self.movie_bias[items],
                dtype=torch.float32,
            )
            rng = np.random.default_rng(self.seed)
            best, stale, checkpoint = float("inf"), 0, None
            self.learning_curve = []
            self.best_epoch = self.epochs
            for epoch in range(self.epochs):
                self.network.train()
                total = 0.0
                order = rng.permutation(len(values))
                for start in range(0, len(values), self.batch_size):
                    rows = order[start : start + self.batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    error = self.network(user_tensor[rows], item_tensor[rows]) - residual[rows]
                    loss = error.square().mean()
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach()) * len(rows)
                self.network.eval()
                row = {"epoch": epoch + 1, "train_mse": total / len(values)}
                if validation is not None and len(validation):
                    predicted = self.predict(validation.userId, validation.movieId)
                    score = float(np.sqrt(np.mean((validation.rating.to_numpy() - predicted) ** 2)))
                    row["validation_rmse"] = score
                    if score < best - 1e-5:
                        best, stale = score, 0
                        self.best_epoch = epoch + 1
                        checkpoint = copy.deepcopy(self.network.state_dict())
                    else:
                        stale += 1
                self.learning_curve.append(row)
                if stale >= self.patience:
                    break
            if checkpoint is not None:
                self.network.load_state_dict(checkpoint)
            self.network.eval()
        return self

    def predict(self, user_ids, movie_ids):
        users, items = self.encode_queries(user_ids, movie_ids)
        output = np.full(len(users), self.global_mean, dtype=np.float32)
        known_u, known_i = users >= 0, items >= 0
        output[known_u] += self.user_bias[users[known_u]]
        output[known_i] += self.movie_bias[items[known_i]]
        indices = np.flatnonzero(known_u & known_i)
        with torch.no_grad():
            for start in range(0, len(indices), self.batch_size):
                rows = indices[start : start + self.batch_size]
                output[rows] += self.network(
                    torch.from_numpy(users[rows]),
                    torch.from_numpy(items[rows]),
                ).numpy()
        return self.clip(output)

    def predict_profile(self, history):
        """Fit only a fresh user's embedding and bias; trained weights stay unchanged."""
        index = {int(movie): position for position, movie in enumerate(self.movie_ids)}
        available = [row for row in history if row["movie_id"] in index]
        if not available:
            return None
        indices = np.array([index[row["movie_id"]] for row in available])
        items = torch.from_numpy(indices)
        target = torch.tensor([row["rating"] for row in available], dtype=torch.float32)
        base = torch.tensor(self.global_mean + self.movie_bias[indices], dtype=torch.float32)
        with torch.enable_grad():
            embedding = nn.Parameter(torch.zeros(1, self.factors))
            bias = nn.Parameter((target - base).mean().detach())
            optimizer = torch.optim.Adam([embedding, bias], lr=0.03)
            for _ in range(self.profile_steps):
                optimizer.zero_grad(set_to_none=True)
                scores = base + bias + self.network.score(embedding.expand(len(items), -1), items)
                loss = (scores - target).square().mean() + self.regularization * (
                    embedding.square().sum() + bias.square()
                )
                embedding.grad, bias.grad = torch.autograd.grad(loss, (embedding, bias))
                optimizer.step()
        output = np.empty(len(self.movie_ids), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(output), self.batch_size):
                end = min(start + self.batch_size, len(output))
                correction = self.network.score(
                    embedding.expand(end - start, -1), torch.arange(start, end)
                ).numpy()
                output[start:end] = (
                    self.global_mean + self.movie_bias[start:end] + float(bias) + correction
                )
        return self.clip(output)
