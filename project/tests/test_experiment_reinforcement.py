import copy

import numpy as np
import pandas as pd
import pytest
import torch

from movie_rating.experiment_config import ExperimentRequest
from movie_rating.experiment_pipeline import execute
from movie_rating.experiment_reinforcement import (
    QNetwork,
    ReplayBuffer,
    _choose,
    _movie_genre_features,
    _profile,
    bellman_targets,
    build_episodes,
    checkpoint_snapshot,
    evaluate,
    restore_checkpoint,
    split_support_query,
    support_global_mean,
)


def _dataset():
    rows = []
    for user in range(1, 31):
        movies = range(1, 7) if user <= 20 else range(1, 9)
        for movie in movies:
            rows.append((user, movie, 1.0 + ((user + movie) % 9) * 0.5, user * 100 + movie))
    train = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    validation = pd.DataFrame(
        [(user, movie, 4.5, user * 100 + 100 + movie) for user in range(1, 21) for movie in (7, 8)],
        columns=train.columns,
    )
    test = pd.DataFrame(
        [(user, movie, 4.0, user * 100 + 200 + movie) for user in range(1, 21) for movie in (7, 8)],
        columns=train.columns,
    )
    movies = pd.DataFrame(
        {"movieId": range(1, 9), "title": [f"Movie {movie}" for movie in range(1, 9)],
         "genres": ["Action" if movie % 2 else "Comedy" for movie in range(1, 9)]}
    )
    return {"train": train, "validation": validation, "test": test}, movies


def _request():
    return ExperimentRequest(
        group="reinforcement", algorithms=["dqn", "random", "popular"],
        action_limit=8, horizon=3, evaluation_users=20, training_user_limit=30,
        max_training_steps=50, replay_capacity=100, dqn_batch_size=8, replay_warmup=1,
        checkpoint_interval_steps=2, target_update_steps=2, epsilon_decay_steps=10,
    )


def test_train_support_query_uses_time_and_excludes_query_from_initial_profile():
    splits, movies = _dataset()
    support, query, excluded = split_support_query(splits["train"])
    user = min(support)
    assert support[user].timestamp.max() < query[user].timestamp.min()
    genres = sorted({part for value in movies.genres for part in value.split("|")})
    movie_ids, movie_features = _movie_genre_features(movies, genres)
    first, _ = _profile(support[user], movie_features, movie_ids, splits["train"].rating.mean())
    changed_query = query[user].copy()
    changed_query.loc[:, "rating"] = 5.0 - changed_query.rating
    second, _ = _profile(support[user], movie_features, movie_ids, splits["train"].rating.mean())
    np.testing.assert_array_equal(first, second)
    assert excluded["excluded_users"] == 0


def test_query_ratings_do_not_change_train_profile_centering():
    splits, _ = _dataset()
    support, query, _ = split_support_query(splits["train"])
    center = support_global_mean(support, splits["train"].rating.mean())
    original_full_mean = float(splits["train"].rating.mean())
    changed = {user: frame.copy() for user, frame in query.items()}
    for frame in changed.values():
        frame.loc[:, "rating"] = 5.0
    assert support_global_mean(support, splits["train"].rating.mean()) == center
    assert float(pd.concat(changed.values()).rating.mean()) != pytest.approx(original_full_mean)


def test_heldout_rating_does_not_change_candidate_mask_or_initial_state():
    splits, movies = _dataset()
    train, validation = splits["train"], splits["validation"]
    actions = list(range(1, 9))
    genres = sorted(set(movies.genres))
    movie_ids, movie_features = _movie_genre_features(movies, genres)
    matrix = np.asarray([[float(genre in str(movies.loc[movies.movieId == movie, "genres"].iloc[0]))
                          for genre in genres] for movie in actions], dtype=np.float32)
    action_genres = matrix.argmax(axis=1).tolist()
    original = build_episodes(train, validation, actions, action_genres, genres,
                              movie_ids, movie_features, 3, 7, 20)
    altered = validation.copy()
    altered.loc[altered.index[0], "rating"] = 0.5
    changed = build_episodes(train, altered, actions, action_genres, genres,
                             movie_ids, movie_features, 3, 7, 20)
    assert [episode["valid"] for episode in original] == [episode["valid"] for episode in changed]
    assert [episode["profile"].tolist() for episode in original] == [episode["profile"].tolist() for episode in changed]
    assert [episode["episode_id"] for episode in original] == [episode["episode_id"] for episode in changed]
    assert original[0]["rewards"] != changed[0]["rewards"] or any(
        original[index]["rewards"] != changed[index]["rewards"] for index in range(len(original))
    )


def test_dqn_masks_actions_and_bellman_terminal_targets():
    target = bellman_targets(
        [1, 2, 3], [False, True, False],
        [[100, 2, 4], [5, 8, 9], [1, 2, 3]],
        [[False, True, False], [True, True, True], [False, False, False]], 0.5,
    )
    np.testing.assert_allclose(target, [2, 2, 3])

    class Fixed:
        def __call__(self, state):
            return torch.tensor([[10.0, 3.0, 20.0]])

    valid = np.asarray([False, True, False])
    assert _choose(Fixed(), np.zeros(1, dtype=np.float32), valid, np.random.default_rng(1)) == 1
    assert _choose(Fixed(), np.zeros(1, dtype=np.float32), np.zeros(3, dtype=bool),
                   np.random.default_rng(1)) is None


def test_checkpoint_snapshot_is_independent_and_restored():
    model = QNetwork(4, 3, (8,))
    snapshot = checkpoint_snapshot(model)
    expected = restore_checkpoint(model, snapshot)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    restored = restore_checkpoint(model, snapshot)
    assert restored == expected
    assert all(torch.equal(model.state_dict()[key], snapshot[key]) for key in snapshot)


def test_replay_capacity_is_bounded_and_states_are_copied():
    replay = ReplayBuffer(2)
    state = np.asarray([1, 2], dtype=np.float32)
    replay.add(state, 0, 1, state, [True, False], False)
    state[0] = 9
    replay.add(state, 1, 0, state, [False, False], True)
    replay.add(state, 0, -1, state, [True, False], False)
    assert len(replay) == 2
    assert replay.rows[0][1] == 1
    assert replay.rows[0][0][0] == 9


def test_frozen_evaluation_and_real_dqn_run_restore_validation_checkpoint():
    splits, movies = _dataset()
    request = _request()
    result = execute(splits, movies, request, lambda *_: None)
    assert result["schema_version"] == 4
    assert result["rl_protocol"]["name"] == "temporal_support_query_dqn_v1"
    assert result["rl_protocol"]["train_steps_actual"] == 50
    assert result["rl_protocol"]["validation_sampled_users"] == 20
    assert result["rl_protocol"]["test_sampled_users"] == 20
    assert result["rl_protocol"]["validation_episode_ids_sha256"]
    assert result["best_configs"]["dqn"]["restored_checkpoint_sha256"]
    assert result["best_configs"]["dqn"]["checkpoint_step_selected_by_validation"] <= 50
    assert all(row["actual_steps"] == 40 for row in result["rows"] if row["split"] == "test")
    assert all(row["user_coverage"] == 1 for row in result["rows"] if row["split"] == "test")

    # Evaluation consumes observed feedback but must not mutate the frozen model.
    from movie_rating.experiment_reinforcement import _genre_matrix, _movie_genre_features

    actions = splits["train"].groupby("movieId").size().sort_values(ascending=False, kind="stable").head(8).index.tolist()
    genre_names, matrix = _genre_matrix(actions, movies)
    movie_ids, movie_features = _movie_genre_features(movies, genre_names)
    action_genres = matrix.argmax(axis=1).tolist()
    from movie_rating.experiment_reinforcement import build_episodes

    episodes = build_episodes(splits["train"], splits["test"], actions, action_genres, genre_names,
                              movie_ids, movie_features, 3, 42, 20)
    model = QNetwork(len(genre_names) * 2 + 1 + len(actions) + 3 + len(genre_names), len(actions))
    before = copy.deepcopy(model.state_dict())
    evaluate("dqn", model, episodes, action_genres, len(genre_names), horizon=3)
    assert all(torch.equal(model.state_dict()[key], before[key]) for key in before)


@pytest.mark.parametrize("method", ["pca", "truncated_svd"])
def test_dqn_reduction_compares_same_episodes_and_keeps_mask_raw(method):
    splits, movies = _dataset()
    request = _request().model_copy(update={
        "reduction_mode": "compare", "reduction_methods": [method], "components": [1]
    })
    result = execute(splits, movies, request, lambda *_: None)
    variants = result["reduction"]["variants"]
    assert [row["variant_id"] for row in variants] == ["raw", f"{method}_1"]
    assert len({row["validation_episode_ids_sha256"] for row in variants}) == 1
    assert len({row["test_episode_ids_sha256"] for row in variants}) == 1
    assert len({row["training_steps"] for row in variants}) == 1
    assert {row["variant_id"] for row in result["rows"] if row["base_algorithm"] == "dqn"} == {
        "dqn_raw", f"dqn_{method}_1"
    }
    assert len([row for row in result["rows"] if row["base_algorithm"] == "popular"]) == 2
    reduced = next(row for row in result["rows"] if row["variant_id"] == f"dqn_{method}_1")
    assert reduced["reduction"]["fit_scope"] == "train_support_profiles"
    assert reduced["reduction"]["fit_rows"] == result["rl_protocol"]["train_support_query_users"]
    assert result["best_configs"][f"dqn_{method}_1"]["reduction"]["fingerprint"]
    assert result["rl_protocol"]["action_limit_actual"] == 8
    assert result["schema_version"] == 4


def test_no_action_cohort_is_reported_as_not_applicable():
    splits, movies = _dataset()
    # Every held-out user has already seen every Train catalog action.
    splits["train"] = pd.concat(
        [splits["train"], splits["validation"].loc[splits["validation"].userId <= 20]],
        ignore_index=True,
    )
    request = _request().model_copy(update={"action_limit": 8})
    # The full train history for users 1–20 still contains the complete candidate catalog.
    result = execute(splits, movies, request, lambda *_: None)
    assert result["rl_protocol"]["test_sampled_users"] == 0
    test_rows = [row for row in result["rows"] if row["split"] == "test"]
    assert all(row["average_reward"] is None for row in test_rows)
    assert all(row["excluded_no_valid_action"] == 20 for row in test_rows)
