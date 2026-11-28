"""Neural Collaborative Filtering for explicit ratings, trained with PyTorch."""

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from movie_rating.experiments.algorithms.base import (
    MovieAlgorithm,
    finite_number,
    positive_integer,
)
from movie_rating.experiments.data import validate_ratings
from movie_rating.experiments.evaluation import rmse
from movie_rating.experiments.features import RatingIndex, fit_biases


@contextmanager
def _torch_threads(count):
    previous = torch.get_num_threads()
    if count != previous:
        torch.set_num_threads(count)
    try:
        yield
    finally:
        if count != previous:
            torch.set_num_threads(previous)


class _RatingNetwork(nn.Module):
    def __init__(self, users, items, factors, hidden_size, dropout, *, product=False, deep=False):
        super().__init__()
        self.product = product
        self.users = nn.Embedding(users, factors)
        self.items = nn.Embedding(items, factors)
        self.user_bias = nn.Embedding(users, 1)
        self.item_bias = nn.Embedding(items, 1)
        layers = [nn.Linear((3 if product else 2) * factors, hidden_size), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        if deep:
            reduced = max(4, hidden_size // 2)
            layers += [nn.Linear(hidden_size, reduced), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            hidden_size = reduced
        layers.append(nn.Linear(hidden_size, 1))
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.users.weight, std=0.05)
        nn.init.normal_(self.items.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def interaction(self, users, items):
        u, i = self.users(users), self.items(items)
        pieces = [u, i, u * i] if self.product else [u, i]
        return self.mlp(torch.cat(pieces, dim=1)).squeeze(-1)

    def forward(self, users, items):
        return (self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1)
                + self.interaction(users, items))


class NCFBaseline(MovieAlgorithm):
    """User/movie embeddings + MLP, rating MSE, AdamW, and train-only ID maps."""

    supports_early_stopping = True
    iteration_parameter = "epochs"
    product = False
    deep = False
    defaults = {
        "factors": 32, "hidden_size": 64, "epochs": 15, "batch_size": 2048,
        "learning_rate": 0.003, "regularization": 0.00001, "dropout": 0.0,
        "patience": 3, "min_delta": 0.0001, "seed": 42, "device": "auto", "n_jobs": 4,
        "bias_epochs": 0, "bias_regularization": 10.0, "gradient_clip": 5.0,
    }

    def fit(self, ratings, movies=None):
        return self._fit(ratings)

    def fit_with_validation(self, ratings, validation, movies=None):
        validate_ratings(validation)
        return self._fit(ratings, validation)

    def _fit(self, ratings, validation=None):
        options = self.configuration(self.defaults)
        for name in ("factors", "hidden_size", "epochs", "batch_size", "patience", "n_jobs"):
            positive_integer(name, options[name])
        for name in ("learning_rate", "bias_regularization", "gradient_clip"):
            finite_number(name, options[name], strict=True)
        for name in ("regularization", "min_delta", "dropout"):
            finite_number(name, options[name])
        if options["dropout"] >= 1:
            raise ValueError("dropout must be less than 1.")
        if not isinstance(options["bias_epochs"], int) or options["bias_epochs"] < 0:
            raise ValueError("bias_epochs must be a nonnegative integer.")
        if options["device"] not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda.")
        device = options["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        self.device_ = torch.device(device)
        self.options_ = options
        self.index_ = RatingIndex()
        users, items, values = self.index_.fit(ratings)
        devices = [torch.cuda.current_device()] if device == "cuda" else []
        with _torch_threads(options["n_jobs"]), torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(options["seed"])
            if device == "cuda":
                torch.cuda.manual_seed(options["seed"])
            self.network_ = _RatingNetwork(
                len(self.index_.user_ids_), len(self.index_.movie_ids_), options["factors"],
                options["hidden_size"], options["dropout"], product=self.product, deep=self.deep,
            ).to(self.device_)
            bu, bi = fit_biases(
                users, items, values, self.index_.mean_, len(self.index_.user_ids_),
                len(self.index_.movie_ids_), epochs=options["bias_epochs"],
                regularization=options["bias_regularization"],
            )
            with torch.no_grad():
                self.network_.user_bias.weight[:, 0].copy_(torch.from_numpy(bu))
                self.network_.item_bias.weight[:, 0].copy_(torch.from_numpy(bi))
            optimizer = torch.optim.AdamW(
                self.network_.parameters(), lr=options["learning_rate"],
                weight_decay=options["regularization"],
            )
            self._train(users, items, values, optimizer, validation)
        self.network_.eval()
        return self

    def _train(self, users, items, values, optimizer, validation):
        options = self.options_
        rng = np.random.default_rng(options["seed"])
        self.learning_curve_, self.best_iterations_ = [], options["epochs"]
        best, stale, checkpoint = float("inf"), 0, None
        for epoch in range(options["epochs"]):
            self.network_.train()
            order = rng.permutation(len(values))
            total = 0.0
            for start in range(0, len(values), options["batch_size"]):
                rows = order[start:start + options["batch_size"]]
                u = torch.from_numpy(users[rows].astype(np.int64)).to(self.device_)
                i = torch.from_numpy(items[rows].astype(np.int64)).to(self.device_)
                target = torch.from_numpy(values[rows] - np.float32(self.index_.mean_)).to(
                    self.device_)
                optimizer.zero_grad(set_to_none=True)
                loss = (self.network_(u, i) - target).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("NCF loss is nonfinite; reduce learning_rate.")
                loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), options["gradient_clip"])
                optimizer.step()
                total += float(loss.detach().cpu()) * len(rows)
            self.network_.eval()
            row = {"epoch": epoch + 1, "online_train_mse": total / len(values)}
            if validation is not None:
                score = rmse(validation.rating, self.predict(validation))
                row["validation_rmse"] = score
                if score < best - options["min_delta"]:
                    best, stale = score, 0
                    self.best_iterations_ = epoch + 1
                    checkpoint = {name: value.detach().cpu().clone()
                                  for name, value in self.network_.state_dict().items()}
                else:
                    stale += 1
            self.learning_curve_.append(row)
            self.emit_progress(f"NCF epoch {epoch + 1}/{options['epochs']} ({self.device_})")
            if validation is not None and stale >= options["patience"]:
                break
        if checkpoint is not None:
            self.network_.load_state_dict(checkpoint)

    def predict(self, pairs):
        self.require_fitted("network_")
        users, items = self.index_.transform(pairs)
        output = np.empty(len(pairs), dtype=np.float64)
        self.network_.eval()
        with _torch_threads(self.options_["n_jobs"]), torch.no_grad():
            for start in range(0, len(pairs), self.options_["batch_size"]):
                stop = start + self.options_["batch_size"]
                u, i = users[start:stop], items[start:stop]
                known_u, known_i = u >= 0, i >= 0
                ut = torch.from_numpy(np.maximum(u, 0)).to(self.device_)
                it = torch.from_numpy(np.maximum(i, 0)).to(self.device_)
                ub = self.network_.user_bias(ut).squeeze(-1).cpu().numpy()
                ib = self.network_.item_bias(it).squeeze(-1).cpu().numpy()
                interaction = self.network_.interaction(ut, it).cpu().numpy()
                output[start:stop] = (self.index_.mean_ + ub * known_u + ib * known_i
                                      + interaction * (known_u & known_i))
        return np.clip(output, 0.5, 5.0)
