"""Temporal offline recommendation with a small, shared-feature DQN policy."""

from __future__ import annotations

import copy
import hashlib
import pickle
import time
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch import nn

from movie_rating.experiment_reduction import FeatureReducer, ReductionSpec, reduction_variants
from movie_rating.lab import chart, measured, series


def environment(frame, actions):
    """Observed rewards by catalog position; missing feedback stays unknown/zero."""
    positions = {int(movie): i for i, movie in enumerate(actions)}
    histories = {}
    for user, history in frame.groupby("userId", sort=True):
        histories[int(user)] = {
            positions[int(row.movieId)]: (float(row.rating) - 2.75) / 2.25
            for row in history.itertuples()
            if int(row.movieId) in positions
        }
    return histories


def split_support_query(frame, support_fraction=0.8):
    """Split each user's Train history in time, keeping tied timestamps together."""
    if "timestamp" not in frame:
        raise ValueError("RL cần timestamp để tạo support/query theo thời gian.")
    support, query, excluded = {}, {}, 0
    for user, group in frame.groupby("userId", sort=True):
        ordered = group.sort_values(["timestamp", "movieId"], kind="stable")
        if len(ordered) < 3:
            excluded += 1
            continue
        boundary = max(1, min(len(ordered) - 1, int(np.floor(len(ordered) * support_fraction))))
        cut = ordered.timestamp.iloc[boundary - 1]
        left = ordered.loc[ordered.timestamp <= cut]
        right = ordered.loc[ordered.timestamp > cut]
        if left.empty or right.empty:
            excluded += 1
            continue
        support[int(user)] = left
        query[int(user)] = right
    return support, query, {"excluded_users": excluded, "reason": "short or inseparable timestamp blocks"}


def support_global_mean(support, fallback):
    rows = pd.concat(support.values(), ignore_index=True) if support else None
    return float(rows.rating.mean()) if rows is not None and len(rows) else float(fallback)


def _genre_matrix(actions, movies):
    metadata = movies.set_index("movieId")
    genre_names = sorted(
        {
            genre
            for value in metadata.genres.fillna("")
            for genre in str(value).split("|")
            if genre and genre != "(no genres listed)"
        }
    )
    if not genre_names:
        genre_names = ["unknown"]
    genre_index = {genre: index for index, genre in enumerate(genre_names)}
    matrix = np.zeros((len(actions), len(genre_names)), dtype=np.float32)
    for position, movie in enumerate(actions):
        raw = str(metadata.loc[movie, "genres"] if movie in metadata.index else "")
        for genre in raw.split("|"):
            if genre in genre_index:
                matrix[position, genre_index[genre]] = 1.0
    return genre_names, matrix


def _profile(frame, movie_genres, movie_ids, global_mean):
    genre_count = movie_genres.shape[1]
    values = np.zeros(genre_count * 2, dtype=np.float32)
    if frame is None or not len(frame):
        return values, 0.0
    sums = np.zeros(genre_count, dtype=np.float64)
    counts = np.zeros(genre_count, dtype=np.float64)
    indices = movie_ids.get_indexer(frame.movieId)
    known = indices >= 0
    if known.any():
        genre_values = movie_genres[indices[known]]
        centered = frame.rating.to_numpy(dtype=np.float64)[known] - global_mean
        sums = centered @ genre_values
        counts = genre_values.sum(axis=0, dtype=np.float64)
    values[:genre_count] = (sums / np.maximum(counts, 1)).astype(np.float32)
    values[genre_count:] = (counts / max(1, len(frame))).astype(np.float32)
    return values, min(1.0, len(frame) / 20.0)


def _movie_genre_features(movies, genre_names):
    movie_ids = pd.Index(movies.movieId.to_numpy())
    columns = {genre: index for index, genre in enumerate(genre_names)}
    values = np.zeros((len(movies), len(genre_names)), dtype=np.float32)
    for row, raw in enumerate(movies.genres.fillna("")):
        for genre in str(raw).split("|"):
            if genre in columns:
                values[row, columns[genre]] = 1.0
    return movie_ids, values


def build_episodes(train, heldout, actions, action_genres, genres, movie_ids, movie_genres,
                   horizon, seed, limit, return_meta=False):
    positions = {int(movie): i for i, movie in enumerate(actions)}
    global_mean = float(train.rating.mean())
    wanted_users = heldout.userId.unique()
    background = {
        int(user): frame
        for user, frame in train.loc[train.userId.isin(wanted_users)].groupby("userId", sort=False)
    }
    episodes = []
    excluded_no_actions = 0
    for user, query in heldout.groupby("userId", sort=True):
        user = int(user)
        history = background.get(user)
        profile, fullness = _profile(history, movie_genres, movie_ids, global_mean)
        seen = set()
        if history is not None:
            seen = {positions[int(movie)] for movie in history.movieId if int(movie) in positions}
        valid = tuple(index for index in range(len(actions)) if index not in seen)
        if not valid:
            excluded_no_actions += 1
            continue
        rewards = {
            positions[int(row.movieId)]: (float(row.rating) - 2.75) / 2.25
            for row in query.itertuples()
            if int(row.movieId) in positions
        }
        steps = min(horizon, len(valid))
        if steps < 1:
            continue
        episodes.append(
            {
                "episode_id": f"{user}:{int(query.timestamp.min()) if 'timestamp' in query else 0}",
                "user_id": user,
                "profile": profile,
                "fullness": fullness,
                "seen": seen,
                "valid": valid,
                "rewards": rewards,
                "steps": steps,
            }
        )
    rng = np.random.default_rng(seed)
    eligible_count = len(episodes)
    if len(episodes) > limit:
        selected = np.sort(rng.choice(len(episodes), limit, replace=False))
        episodes = [episodes[index] for index in selected]
    metadata = {
        "total_users": int(heldout.userId.nunique()),
        "eligible_users": eligible_count,
        "sampled_users": len(episodes),
        "excluded_no_valid_action": excluded_no_actions,
    }
    return (episodes, metadata) if return_meta else episodes


class QNetwork(nn.Module):
    def __init__(self, state_size, actions, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        previous = state_size
        for size in hidden_sizes:
            layers.extend((nn.Linear(previous, size), nn.ReLU()))
            previous = size
        layers.append(nn.Linear(previous, actions))
        self.layers = nn.Sequential(*layers)

    def forward(self, states):
        return self.layers(states)


def checkpoint_snapshot(model):
    """Copy the complete CPU state so subsequent optimizer updates cannot mutate it."""
    return copy.deepcopy({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})


def restore_checkpoint(model, state):
    model.load_state_dict(state)
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def bellman_targets(rewards, dones, next_q_values, next_valid, gamma):
    """Masked DQN targets; terminal/all-invalid transitions never bootstrap."""
    values = np.asarray(next_q_values, dtype=np.float32).copy()
    valid = np.asarray(next_valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("next_q_values and next_valid phải cùng shape.")
    values[~valid] = -np.inf
    has_action = valid.any(axis=1)
    maximum = np.zeros(len(values), dtype=np.float32)
    maximum[has_action] = np.max(values[has_action], axis=1)
    reward_array = np.asarray(rewards, dtype=np.float32)
    done_array = np.asarray(dones, dtype=bool)
    return reward_array + float(gamma) * maximum * (~done_array & has_action)


class ReplayBuffer:
    def __init__(self, capacity):
        self.rows = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, next_valid, done):
        self.rows.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                np.asarray(next_valid, dtype=bool).copy(),
                bool(done),
            )
        )

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self.rows), min(batch_size, len(self.rows)), replace=False)
        return [self.rows[int(index)] for index in indices]

    def __len__(self):
        return len(self.rows)


def _state(episode, chosen, left, previous_reward, observed, previous_genre, action_genres, genre_count):
    mask = np.zeros(len(action_genres), dtype=np.float32)
    blocked = set(episode["seen"]) | set(chosen)
    if blocked:
        mask[list(blocked)] = 1.0
    last = np.zeros(genre_count, dtype=np.float32)
    if previous_genre is not None:
        last[previous_genre] = 1.0
    return np.concatenate(
        [
            episode["profile"],
            np.asarray([episode["fullness"]], dtype=np.float32),
            mask,
            np.asarray([left, previous_reward, float(observed)], dtype=np.float32),
            last,
        ]
    )


def _valid_mask(size, used, initial_seen=()):
    blocked = set(initial_seen) | set(used)
    return np.asarray([index not in blocked for index in range(size)], dtype=bool)


def _choose(model, state, valid, rng, epsilon=0.0):
    choices = np.flatnonzero(valid)
    if not len(choices):
        return None
    if model == "random" or rng.random() < epsilon:
        return int(rng.choice(choices))
    if model == "popular":
        return int(choices[0])
    with torch.no_grad():
        values = model(torch.from_numpy(state).reshape(1, -1))[0].cpu().numpy()
    values[~valid] = -np.inf
    best = np.flatnonzero(values == np.max(values))
    return int(rng.choice(best))


def _train_episode(episode, online, target, optimizer, replay, rng, config, global_step, total_budget,
                   action_genres, genre_count, update_network=True):
    chosen, total, observed_count, rows = set(), 0.0, 0, []
    state = _state(episode, chosen, episode["steps"] / config["horizon"], 0.0, False,
                   None, action_genres, genre_count)
    for step in range(episode["steps"]):
        valid = _valid_mask(len(action_genres), chosen, episode["seen"])
        progress = min(1.0, global_step / max(1, config["epsilon_decay_steps"]))
        epsilon = config["epsilon_end"] + (config["epsilon_start"] - config["epsilon_end"]) * np.exp(-5 * progress)
        action = _choose(online, state, valid, rng, epsilon)
        if action is None:
            break
        observed = action in episode["rewards"]
        reward = episode["rewards"].get(action, 0.0)
        chosen.add(action)
        next_valid = _valid_mask(len(action_genres), chosen, episode["seen"])
        terminal = step + 1 >= episode["steps"] or not next_valid.any()
        next_state = _state(
            episode,
            chosen,
            max(0, (episode["steps"] - step - 1) / config["horizon"]),
            reward if observed else 0.0,
            observed,
            action_genres[action],
            action_genres,
            genre_count,
        )
        replay.add(state, action, reward, next_state, next_valid, terminal)
        if update_network and len(replay) >= config["replay_warmup"]:
            batch = replay.sample(config["batch_size"], rng)
            states = torch.from_numpy(np.stack([row[0] for row in batch]))
            actions = torch.tensor([row[1] for row in batch], dtype=torch.long)
            rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32)
            next_states = torch.from_numpy(np.stack([row[3] for row in batch]))
            next_valid = torch.from_numpy(np.stack([row[4] for row in batch]))
            dones = np.asarray([row[5] for row in batch], dtype=bool)
            q_values = online(states).gather(1, actions[:, None]).squeeze(1)
            with torch.no_grad():
                target_values = target(next_states).cpu().numpy()
                targets = torch.from_numpy(
                    bellman_targets(rewards.numpy(), dones, target_values, next_valid.numpy(), config["gamma"])
                )
            loss = nn.functional.smooth_l1_loss(q_values, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            optimizer.step()
            rows.append(float(loss.detach()))
        global_step += 1
        if update_network and global_step % config["target_update_steps"] == 0:
            target.load_state_dict(online.state_dict())
        total += reward
        observed_count += int(observed)
        state = next_state
    return total, observed_count, len(chosen), global_step, rows


def evaluate(policy, model, episode_data, action_genres=None, genre_count=None, seed=100042, horizon=5,
             legacy_episode_count=None, **kwargs):
    """Frozen-policy evaluation on fixed user episodes; rewards never train the model."""
    if isinstance(episode_data, dict):
        # Keep the small helper contract used by schema 2/3 regression tests.
        count = legacy_episode_count or kwargs.get("episodes") or 100
        users = list(episode_data)
        scenario_rng, policy_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
        returns, observed, actions = [], 0, 0
        for _ in range(count):
            user = users[int(scenario_rng.integers(len(users)))]
            history, chosen, total = episode_data[user], set(), 0.0
            for _step in range(horizon):
                valid = [action for action in range(len(action_genres)) if action not in chosen]
                if policy == "random":
                    action = int(policy_rng.choice(valid))
                elif policy == "popular":
                    action = valid[0]
                else:
                    action = int(policy_rng.choice(valid))
                total += history.get(action, 0.0)
                observed += int(action in history)
                actions += 1
                chosen.add(action)
            returns.append(total)
        return {
            "cumulative_reward": float(sum(returns)),
            "average_reward": float(sum(returns) / max(1, actions)),
            "observed_feedback_fraction": observed / max(1, actions),
        }
    rng = np.random.default_rng(seed)
    returns, user_rewards, observed, total_steps, infer_seconds = [], {}, 0, 0, 0.0
    hits, dcg, ideal, eligible_positives, candidate_sizes = [], [], [], 0, []
    for episode in episode_data:
        chosen, total, observed_here, selected, gains = set(), 0.0, 0, [], []
        state = _state(episode, chosen, episode["steps"] / horizon, 0.0, False,
                       None, action_genres, genre_count)
        for step in range(episode["steps"]):
            valid = _valid_mask(len(action_genres), chosen, episode["seen"])
            start = time.perf_counter()
            action_model = policy if policy in {"random", "popular"} else model
            action = _choose(action_model, state, valid, rng, 1.0 if policy == "random" else 0.0)
            infer_seconds += time.perf_counter() - start
            if action is None:
                break
            observed_action = action in episode["rewards"]
            reward = episode["rewards"].get(action, 0.0)
            positive = observed_action and reward > 0
            selected.append(action)
            gains.append(int(positive))
            chosen.add(action)
            observed += int(observed_action)
            observed_here += int(observed_action)
            total_steps += 1
            total += reward
            next_state = _state(
                episode,
                chosen,
                max(0, (episode["steps"] - step - 1) / horizon),
                reward if observed_action else 0.0,
                observed_action,
                action_genres[action],
                action_genres,
                genre_count,
            )
            state = next_state
        returns.append(total)
        user_rewards[episode["user_id"]] = total / max(1, len(selected))
        candidate_sizes.append(len(episode["valid"]))
        relevant = {
            action
            for action, reward in episode["rewards"].items()
            if reward > 0 and action in episode["valid"]
        }
        eligible_positives += len(relevant)
        hits.append(len(set(selected) & relevant) / len(relevant) if relevant else None)
        ordered_gain = [int(action in relevant) for action in selected]
        dcg.append(sum(gain / np.log2(index + 2) for index, gain in enumerate(ordered_gain)))
        ideal.append(sum(1 / np.log2(index + 2) for index in range(min(len(relevant), len(selected)))))
    nonnull_recall = [value for value in hits if value is not None]
    ndcg = [value / denom for value, denom in zip(dcg, ideal) if denom > 0]
    denominator = max(1, total_steps)
    return {
        "cumulative_reward": float(sum(returns)),
        "average_reward": float(sum(returns) / denominator) if total_steps else None,
        "episode_reward_mean": float(np.mean(returns)) if returns else None,
        "episode_reward_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "observed_feedback_fraction": observed / denominator,
        "observed_actions": int(observed),
        "actual_steps": int(total_steps),
        "evaluation_episodes": len(episode_data),
        "evaluation_users": len(user_rewards),
        "user_coverage": len(user_rewards) / max(1, len(episode_data)),
        "candidate_pool_size": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
        "eligible_positives": int(eligible_positives),
        "recall_at_horizon": float(np.mean(nonnull_recall)) if nonnull_recall else None,
        "ndcg_at_horizon": float(np.mean(ndcg)) if ndcg else None,
        "users_without_eligible_positive": len(hits) - len(nonnull_recall),
        "inference_seconds": infer_seconds,
        "inference_rows": total_steps,
        "user_rewards": user_rewards,
    }


def _bootstrap_paired(reference, candidate, seed=42):
    users = sorted(set(reference) & set(candidate))
    if len(users) < 10:
        return {"users": len(users), "mean_delta": None, "ci95": None, "reason": "fewer than 10 paired users"}
    delta = np.asarray([candidate[user] - reference[user] for user in users], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(1000, dtype=np.float64)
    for index in range(len(samples)):
        samples[index] = rng.choice(delta, len(delta), replace=True).mean()
    return {
        "users": len(users),
        "mean_delta": float(delta.mean()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "reason": None,
    }


def _episode_fingerprint(episodes):
    return hashlib.sha256("\n".join(episode["episode_id"] for episode in episodes).encode()).hexdigest()


def _reinforcement_single(splits, movies, request, update, reduction_variant=None):
    train = splits["train"]
    actions = (
        train.groupby("movieId").size().sort_values(ascending=False, kind="stable")
        .head(min(request.action_limit, request.max_items, 30)).index.tolist()
    )
    if len(actions) < request.horizon:
        raise ValueError(f"Chỉ có {len(actions)} phim trong catalog; cần ít nhất {request.horizon}.")
    genres, action_genre_matrix = _genre_matrix(actions, movies)
    movie_ids, movie_genres = _movie_genre_features(movies, genres)
    action_genres = np.argmax(action_genre_matrix, axis=1).tolist()
    all_training_users = np.sort(train.userId.unique())
    eligible_mask = train.groupby("userId").timestamp.nunique().reindex(all_training_users, fill_value=0) >= 2
    eligible_user_ids = all_training_users[eligible_mask.to_numpy()]
    rng = np.random.default_rng(42)
    if len(eligible_user_ids) > request.training_user_limit:
        selected = np.sort(rng.choice(eligible_user_ids, request.training_user_limit, replace=False))
    else:
        selected = eligible_user_ids
    training_frame = train.loc[train.userId.isin(selected)]
    support, query, support_exclusions = split_support_query(training_frame)
    train_episodes = []
    train_global_mean = float(train.rating.mean())
    profile_global_mean = support_global_mean(support, train_global_mean)
    positions = {int(movie): index for index, movie in enumerate(actions)}
    for user in sorted(support):
        history, future = support[user], query[user]
        profile, fullness = _profile(history, movie_genres, movie_ids, profile_global_mean)
        seen = {positions[int(movie)] for movie in history.movieId if int(movie) in positions}
        valid = tuple(index for index in range(len(actions)) if index not in seen)
        if not valid:
            continue
        rewards = {
            positions[int(row.movieId)]: (float(row.rating) - 2.75) / 2.25
            for row in future.itertuples()
            if int(row.movieId) in positions
        }
        steps = min(request.horizon, len(valid))
        train_episodes.append(
            {"episode_id": f"train:{user}", "user_id": user, "profile": profile,
             "fullness": fullness, "seen": seen, "valid": valid,
             "rewards": rewards, "steps": steps}
        )
    total_training_users = len(eligible_user_ids)
    eligible_training_users = len(train_episodes)
    if not train_episodes:
        raise ValueError("Không có user Train đủ điều kiện tạo support/query và action hợp lệ.")
    rng.shuffle(train_episodes)
    training_episodes = []
    for _ in range(request.training_sweeps):
        training_episodes.extend(train_episodes)
    if request.max_training_steps is not None:
        capped, steps = [], 0
        for episode in training_episodes:
            if steps >= request.max_training_steps:
                break
            remaining = request.max_training_steps - steps
            episode = dict(episode, steps=min(episode["steps"], remaining))
            if episode["steps"]:
                capped.append(episode)
                steps += episode["steps"]
        training_episodes = capped
    max_steps = sum(episode["steps"] for episode in training_episodes)
    val_episodes, validation_cohort = build_episodes(
        train, splits["validation"], actions, action_genres, genres, movie_ids, movie_genres,
        request.horizon, 100042, request.evaluation_users, return_meta=True,
    )
    test_episodes, test_cohort = build_episodes(
        train, splits["test"], actions, action_genres, genres, movie_ids, movie_genres,
        request.horizon, 200042, request.evaluation_users, return_meta=True,
    )

    reducer = None
    reduction_metadata = None
    if reduction_variant is not None:
        profile_matrix = np.stack([episode["profile"] for episode in train_episodes])
        reducer = FeatureReducer(
            reduction_variant["method"], reduction_variant["components"], seed=request.seed,
            scale=True,
        ).fit(
            profile_matrix,
            feature_schema={"family": "dqn_support_preference_and_frequency",
                            "profile_dimensions": 2 * len(genres)},
            fit_scope="train_support_profiles",
        )
        reduction_metadata = reducer.metadata()
        for episode in training_episodes + val_episodes + test_episodes:
            episode["profile"] = reducer.transform(episode["profile"].reshape(1, -1))[0]
        update(None, f"DQN reducer {reducer.method} · {reducer.input_dimensions}→"
               f"{reducer.output_dimensions} · fit {reducer.fit_rows:,} Train support profiles.")

    names = {"dqn": "DQN (Q-Learning xấp xỉ bằng mạng)",
             "random": "Random Recommendation", "popular": "Popularity Recommendation"}
    selected_policy = "dqn" in request.algorithms
    best_state, best_reducer, best_validation, checkpoint_step = None, None, -np.inf, 0
    train_curve, validation_curve, loss_curve = [], [], []
    global_step = 0
    with measured() as timing:
        torch.manual_seed(42)
        profile_dimensions = len(training_episodes[0]["profile"])
        state_size = profile_dimensions + 1 + len(actions) + 3 + len(genres)
        online = QNetwork(state_size, len(actions), request.hidden_sizes).cpu()
        target = copy.deepcopy(online).eval()
        optimizer = torch.optim.Adam(online.parameters(), lr=request.learning_rate)
        replay = ReplayBuffer(request.replay_capacity)
        replay_rng = np.random.default_rng(2026)
        checkpoints = set(range(request.checkpoint_interval_steps, max_steps + 1, request.checkpoint_interval_steps))
        checkpoints.add(max_steps)
        checkpoint_stride = sorted(checkpoints)
        next_checkpoint = checkpoint_stride.pop(0) if checkpoint_stride else max_steps
        if selected_policy:
            for episode_index, episode in enumerate(training_episodes, 1):
                episode_return, _, actual_steps, global_step, losses = _train_episode(
                    episode, online, target, optimizer, replay, replay_rng,
                    {"horizon": request.horizon, "epsilon_start": request.epsilon_start,
                     "epsilon_end": request.epsilon_end, "epsilon_decay_steps": request.epsilon_decay_steps,
                     "gamma": request.discount_factor, "replay_warmup": request.replay_warmup,
                     "batch_size": request.dqn_batch_size,
                     "target_update_steps": request.target_update_steps},
                    global_step, max_steps, action_genres, len(genres), True,
                )
                train_curve.append((global_step, episode_return / max(1, actual_steps)))
                loss_curve.extend(losses)
                should_checkpoint = global_step >= next_checkpoint or episode_index == len(training_episodes)
                if episode_index % 100 == 0 or episode_index == len(training_episodes):
                    update(
                        15 + 65 * episode_index / len(training_episodes),
                        f"DQN · episode {episode_index:,}/{len(training_episodes):,} · "
                        f"{global_step:,} training steps · replay {len(replay):,}/{request.replay_capacity:,}",
                    )
                if should_checkpoint:
                    validation_report = evaluate("dqn", online, val_episodes, action_genres, len(genres),
                                                 seed=300042, horizon=request.horizon)
                    score = validation_report["average_reward"]
                    validation_curve.append((global_step, score))
                    update(
                        15 + 65 * episode_index / len(training_episodes),
                        f"DQN · checkpoint Validation {global_step:,}/{max_steps:,} steps · "
                        f"reward/action={'N/A' if score is None else f'{score:.5f}'} · "
                        f"coverage={validation_report['observed_feedback_fraction']:.2%}",
                    )
                    if score is not None and score > best_validation:
                        best_validation = score
                        best_state = checkpoint_snapshot(online)
                        best_reducer = copy.deepcopy(reducer)
                        checkpoint_step = global_step
                    while checkpoint_stride and checkpoint_stride[0] <= global_step:
                        checkpoint_stride.pop(0)
                    next_checkpoint = checkpoint_stride[0] if checkpoint_stride else max_steps + 1
            if best_state is None:
                best_state = checkpoint_snapshot(online)
                best_reducer = copy.deepcopy(reducer)
                checkpoint_step = global_step
            restored_checkpoint_fingerprint = restore_checkpoint(online, best_state)
            reducer = best_reducer
        else:
            restored_checkpoint_fingerprint = None
        best_model_bytes = len(pickle.dumps((online.state_dict(), reducer)))

    policies = list(request.algorithms)
    policy_models = {"dqn": online, "random": None, "popular": None}
    reports = {}
    rows = []
    completed_evaluations = 0
    total_evaluations = max(1, len(policies) * 2)
    for policy in policies:
        for split_name, episodes in (("validation", val_episodes), ("test", test_episodes)):
            report = evaluate(
                policy, policy_models[policy], episodes, action_genres, len(genres),
                seed=100042 if split_name == "validation" else 200042,
                horizon=request.horizon,
            )
            reports[(policy, split_name)] = report
            completed_evaluations += 1
            update(
                80 + 15 * completed_evaluations / total_evaluations,
                f"{names[policy]} · {split_name.title()} đánh giá đóng băng · "
                f"{report['evaluation_episodes']:,} episodes · "
                f"phản hồi quan sát={report['observed_feedback_fraction']:.2%}",
            )
            row = {k: v for k, v in report.items() if k != "user_rewards"}
            row.update(
                algorithm=policy, variant_id=policy, base_algorithm=policy,
                representation="raw_profile" if policy == "dqn" else "not_applicable",
                reduction_method=None, components=None, split=split_name, name=names[policy],
                fit_seconds=float(timing["fit_seconds"]) if policy == "dqn" else 0.0,
                peak_rss_mb=timing["peak_rss_mb"],
                model_bytes=best_model_bytes if policy == "dqn" else 0,
                training_steps=global_step if policy == "dqn" else 0,
                learned_users=min(eligible_training_users, len(training_episodes)) if policy == "dqn" else 0,
                total_training_users=int(total_training_users),
                eligible_training_users=int(eligible_training_users),
                training_user_coverage=len(training_episodes) / max(1, eligible_training_users)
                    if policy == "dqn" else 1.0,
                checkpoint_step=checkpoint_step if policy == "dqn" else None,
                eligible_evaluation_users=(validation_cohort if split_name == "validation" else test_cohort)["eligible_users"],
                total_evaluation_users=(validation_cohort if split_name == "validation" else test_cohort)["total_users"],
                excluded_no_valid_action=(validation_cohort if split_name == "validation" else test_cohort)["excluded_no_valid_action"],
                user_coverage=(validation_cohort if split_name == "validation" else test_cohort)["eligible_users"]
                    / max(1, (validation_cohort if split_name == "validation" else test_cohort)["total_users"]),
                reduction=reduction_metadata if policy == "dqn" else None,
                feature_family="dqn_support_preference_and_frequency" if policy == "dqn" else None,
                fit_scope="train_support_profiles" if policy == "dqn" else "not_applicable",
                preprocessing_seconds=(reduction_metadata["fit_seconds"] if reduction_metadata else 0.0)
                    if policy == "dqn" else 0.0,
            )
            rows.append(row)

    validation_rows = [
        row for row in rows if row["split"] == "validation" and row["average_reward"] is not None
    ]
    winner = max(validation_rows, key=lambda row: row["average_reward"]) if validation_rows else None
    paired = {}
    reference = reports[("popular", "test")] if "popular" in policies else reports[(policies[0], "test")]
    reference_id = "popular" if "popular" in policies else policies[0]
    for policy in policies:
        if policy == reference_id:
            continue
        paired[policy] = _bootstrap_paired(reference["user_rewards"], reports[(policy, "test")]["user_rewards"])
    charts = []
    if selected_policy:
        charts.extend(
            [
                chart(-1, "Reward mỗi hành động theo checkpoint DQN", "line", "Training step", "Average Reward / action",
                      [series("Validation", *zip(*validation_curve))] if validation_curve else [], "validation"),
                chart(-1, "Loss SmoothL1 trong replay update", "line", "Optimizer update", "Huber loss ↓",
                      [series("DQN", range(1, len(loss_curve) + 1), loss_curve)], "train"),
            ]
        )
    return {
        "rows": rows,
        "charts": charts,
        "actions": [str(movies.set_index("movieId").loc[movie, "title"]) for movie in actions],
        "best_configs": {
            policy: (
                {"policy": "DQN", "learning_rate": request.learning_rate,
                 "gamma": request.discount_factor, "epsilon_start": request.epsilon_start,
                 "epsilon_end": request.epsilon_end, "epsilon_decay_steps": request.epsilon_decay_steps,
                 "hidden_sizes": request.hidden_sizes, "replay_capacity": request.replay_capacity,
                 "batch_size": request.dqn_batch_size, "target_update_steps": request.target_update_steps,
                 "checkpoint_step_selected_by_validation": checkpoint_step,
                 "restored_checkpoint_sha256": restored_checkpoint_fingerprint,
                 "training_sweeps": request.training_sweeps, "training_steps": global_step,
                 "replay_rows": len(replay), "reduction": reduction_metadata,
                 "fit_scope": "train_support_profiles" if reducer else "not_applicable"}
                if policy == "dqn"
                else {"seed": 42, "definition": "random action from the same valid catalog"}
                if policy == "random"
                else {"ranking": "rating count descending on Train"}
            )
            for policy in policies
        },
        "selection": (
            f"Chọn bằng Validation: {winner['name']} với reward/action="
            f"{winner['average_reward']:.5f}. Test báo cáo sau khi khóa checkpoint; "
            "đọc cùng observed-feedback coverage, user coverage và ranking trong candidate catalog."
            if winner else
            "Không có user Validation với action hợp lệ trong catalog; reward/ranking là N/A "
            "và checkpoint DQN cuối chỉ được giữ để báo cáo, không thể chọn bằng Validation."
        ),
        "rl_protocol": {
            "name": "temporal_support_query_dqn_v1",
            "training_split": "Train: support trước timestamp, query sau support",
            "evaluation": "Train history frozen; Validation/Test feedback only within each episode",
            "checkpoint_selection": "Validation average reward per actual action",
            "checkpoint_step": checkpoint_step,
            "test_used_for_selection": False,
            "action_limit_requested": request.action_limit,
            "action_limit_actual": len(actions),
            "horizon": request.horizon,
            "train_support_query_users": eligible_training_users,
            "train_excluded_support_query_users": support_exclusions["excluded_users"],
            "train_sweeps_requested": request.training_sweeps,
            "train_steps_budget": request.max_training_steps,
            "train_steps_actual": global_step,
            "validation_sampled_users": len(val_episodes),
            "test_sampled_users": len(test_episodes),
            "validation_episode_ids_sha256": _episode_fingerprint(val_episodes),
            "test_episode_ids_sha256": _episode_fingerprint(test_episodes),
            "validation_cohort": validation_cohort,
            "test_cohort": test_cohort,
            "training_user_limit": request.training_user_limit,
            "reduction": reduction_metadata,
            "reduction_fit_scope": "train_support_profiles" if reducer else "not_applicable",
        },
        "paired_test_uncertainty": {"reference_policy": reference_id, "policies": paired},
        "notes": [
            "Giao thức RL mới dùng split per_user_temporal_70_15_15_v2. Trong Train, mỗi episode chia support/query theo timestamp; state ban đầu chỉ lấy support và reward chỉ lấy query.",
            "Observation chứa hồ sơ sở thích/tần suất genre, độ đầy đủ, action mask, bước còn lại, phản hồi trước và cờ observed. Không đưa userId vào mạng; action mask không bị giảm chiều.",
            "DQN dùng MLP nhỏ trên CPU, replay buffer hữu hạn, target network, Huber loss, gradient clipping và epsilon schedule. Checkpoint độc lập được chọn bằng Validation rồi phục hồi trước Test.",
            f"Catalog có {len(actions)} action từ popularity Train; giới hạn yêu cầu {request.action_limit}, tối đa 30. Mọi policy dùng cùng cohort, episode IDs, catalog và mask ban đầu.",
            "Phim đã xem trong Train support bị loại khỏi candidate. Candidate không bị lọc theo việc Validation/Test có rating; phản hồi thiếu nhận reward 0 và cờ observed=false.",
            f"Training sweep={request.training_sweeps}; {global_step:,} bước học trên {len(training_episodes):,}/{eligible_training_users:,} episode đủ điều kiện. Reward/episode chia theo số bước thực tế.",
            "Reward (rating−2,75)/2,25 là mô phỏng theo rating quan sát, thiếu phản hồi=0; không có propensity/IPS và không phải đánh giá counterfactual không thiên lệch.",
            "Môi trường có reward tĩnh theo rating, không mô hình hóa thay đổi sở thích hoặc giữ chân. Recall/NDCG chỉ tính trong candidate pool được báo cáo; user không có positive/action hợp lệ là N/A.",
            "Train, Validation và Test giữ cùng hồ sơ Train nền; trọng số DQN đóng băng ở Validation/Test. Phản hồi tuần tự chỉ còn trong episode hiện tại.",
            "Q-table q_learning trong lịch sử schema 2/3 giữ nguyên tên và giao thức cũ; dqn là định danh thuật toán mới.",
            "Profile episode Train ban đầu chỉ dùng support; trung bình chuẩn hóa cũng fit trên support, không đọc query rating.",
        ],
    }


def reinforcement(splits, movies, request, update):
    if request.reduction_mode == "off" or "dqn" not in request.algorithms:
        result = _reinforcement_single(splits, movies, request, update)
        if request.reduction_mode != "off":
            result["reduction"] = {
                "mode": request.reduction_mode,
                "variants": [],
                "selected_variant_id": None,
                "reason": "DQN chưa được chọn; Random/Popularity không cần profile.",
            }
        return result

    spec = ReductionSpec(
        mode=request.reduction_mode,
        methods=tuple(request.reduction_methods),
        components=tuple(request.components),
        seed=request.seed,
        scale=True,
    )
    variants = reduction_variants(spec)
    output = {
        "rows": [], "charts": [], "best_configs": {}, "notes": [],
        "paired_test_uncertainty": {"reference_policy": "popular", "policies": {}},
        "reduction": {
            "mode": request.reduction_mode, "variants": [],
            "selected_variant_id": None, "selection_metric": "average_reward",
            "feature_family": "dqn_support_preference_and_frequency",
        },
    }
    baseline_ids = set()
    runs = []
    for descriptor in variants:
        descriptor_id = descriptor["variant_id"]
        reducer_spec = (
            {"method": descriptor["method"], "components": descriptor["components"]}
            if descriptor["method"] else None
        )
        update(None, f"DQN · biểu diễn {descriptor_id}; episode và ngân sách giữ cố định.")
        result = _reinforcement_single(splits, movies, request, update, reducer_spec)
        dqn_variant_id = f"dqn_{descriptor_id}"
        output["reduction"]["variants"].append({
            "variant_id": descriptor_id, "method": descriptor["method"],
            "components": descriptor["components"],
            "validation_episode_ids_sha256": result["rl_protocol"]["validation_episode_ids_sha256"],
            "test_episode_ids_sha256": result["rl_protocol"]["test_episode_ids_sha256"],
            "training_steps": result["rl_protocol"]["train_steps_actual"],
            "training_user_coverage": next(
                row["training_user_coverage"] for row in result["rows"]
                if row["algorithm"] == "dqn"
            ),
        })
        for row in result["rows"]:
            if row["algorithm"] == "dqn":
                row.update(
                    variant_id=dqn_variant_id, base_algorithm="dqn",
                    representation="raw_profile" if not descriptor["method"] else "reduced_profile",
                    reduction_method=descriptor["method"], components=descriptor["components"],
                    fit_scope="train_support_profiles",
                )
                row["name"] = f"DQN · {descriptor_id}"
            else:
                row.update(
                    variant_id=row["algorithm"], base_algorithm=row["algorithm"],
                    representation="not_applicable", reduction_method=None, components=None,
                    fit_scope="not_applicable",
                )
            if row["algorithm"] in {"random", "popular"}:
                identity = (row["algorithm"], row["split"])
                if identity in baseline_ids:
                    continue
                baseline_ids.add(identity)
            output["rows"].append(row)

        for algorithm, config in result["best_configs"].items():
            variant_id = dqn_variant_id if algorithm == "dqn" else algorithm
            if algorithm in {"random", "popular"} and algorithm in output["best_configs"]:
                continue
            output["best_configs"][variant_id] = {
                **config, "variant_id": variant_id, "base_algorithm": algorithm,
                "representation": (
                    "raw_profile" if algorithm == "dqn" and not descriptor["method"]
                    else "reduced_profile" if algorithm == "dqn" else "not_applicable"
                ),
                "reduction_method": descriptor["method"] if algorithm == "dqn" else None,
                "components": descriptor["components"] if algorithm == "dqn" else None,
            }
        for policy, value in result["paired_test_uncertainty"]["policies"].items():
            key = dqn_variant_id if policy == "dqn" else policy
            output["paired_test_uncertainty"]["policies"][key] = value
        for figure in result["charts"]:
            figure["variant_id"] = descriptor_id
            figure["title"] = f"{figure['title']} · {descriptor_id}"
            output["charts"].append(figure)
        output["notes"].extend(result["notes"])
        runs.append(result)

    output["rows"].sort(key=lambda row: (row["variant_id"], row["split"]))
    validation = [
        row for row in output["rows"]
        if row["split"] == "validation" and row["average_reward"] is not None
    ]
    winner = max(validation, key=lambda row: row["average_reward"]) if validation else None
    output["selection"] = (
        f"Chọn bằng Validation: {winner['variant_id']} với reward/action="
        f"{winner['average_reward']:.5f}; Test chỉ đánh giá cuối."
        if winner else
        "Không có user Validation với action hợp lệ; reward/ranking là N/A và không thể chọn biến thể."
    )
    output["rl_protocol"] = dict(runs[0]["rl_protocol"])
    output["rl_protocol"]["reduction_variants"] = list(output["reduction"]["variants"])
    dqn_validation = [row for row in validation if row["base_algorithm"] == "dqn"]
    selected_dqn = max(dqn_validation, key=lambda row: row["average_reward"]) if dqn_validation else None
    output["reduction"]["selected_variant_id"] = selected_dqn["variant_id"] if selected_dqn else None
    output["notes"] = list(dict.fromkeys(output["notes"]))
    output["schema_version"] = 4
    return output
