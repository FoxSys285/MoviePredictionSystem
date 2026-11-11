"""Tabular sequential recommendation with observed MovieLens rewards only."""

import pickle
import time

import numpy as np

from movie_rating.lab import chart, measured, series


def environment(frame, actions):
    positions = {int(m): i for i, m in enumerate(actions)}
    histories = {}
    for user, history in frame.groupby("userId", sort=True):
        histories[int(user)] = {
            positions[int(row.movieId)]: (float(row.rating) - 2.75) / 2.25
            for row in history.itertuples()
            if int(row.movieId) in positions
        }
    return histories


def action_for(policy, q, state, valid, rng):
    if policy == "random":
        return int(rng.choice(valid))
    if policy == "popular":
        return valid[0]
    values = q.get(state, {})
    best = max(values.get(a, 0.0) for a in valid)
    return int(rng.choice([a for a in valid if values.get(a, 0.0) == best]))


def evaluate(policy, q, histories, genres, seed=100042, episodes=100, horizon=5):
    users = list(histories)
    scenario_rng, policy_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    returns, observed, timings = [], 0, 0.0
    for _ in range(episodes):
        user = users[int(scenario_rng.integers(len(users)))]
        feedback = histories[user]
        state, chosen, total, mask = (user, 0, -1, 0), set(), 0.0, 0
        for step in range(horizon):
            valid = [a for a in range(len(genres)) if a not in chosen]
            start = time.perf_counter()
            action = action_for(policy, q, state, valid, policy_rng)
            timings += time.perf_counter() - start
            reward = feedback.get(action, 0.0)
            observed += int(action in feedback)
            total += reward
            chosen.add(action)
            mask |= 1 << action
            state = (user, mask, genres[action], int(reward > 0))
        returns.append(total)
    return {
        "cumulative_reward": float(np.sum(returns)),
        "average_reward": float(np.sum(returns) / (episodes * horizon)),
        "episode_reward_mean": float(np.mean(returns)),
        "episode_reward_std": float(np.std(returns, ddof=1)),
        "observed_feedback_fraction": observed / (episodes * horizon),
        "evaluation_episodes": episodes,
        "inference_seconds": timings,
        "inference_rows": episodes * horizon,
    }


def reinforcement(splits, movies, request, update):
    actions = (
        splits["train"]
        .groupby("movieId")
        .size()
        .sort_values(ascending=False, kind="stable")
        .head(min(30, request.max_items))
        .index.tolist()
    )
    if len(actions) < 5:
        raise ValueError("Cần ít nhất 5 phim Train cho episode 5 bước.")
    metadata = movies.set_index("movieId")
    names = {
        "q_learning": "Q-Learning",
        "random": "Random Recommendation",
        "popular": "Popularity Recommendation",
    }
    genre_names = sorted({str(metadata.loc[m, "genres"]).split("|")[0] for m in actions})
    genres = [genre_names.index(str(metadata.loc[m, "genres"]).split("|")[0]) for m in actions]
    histories = {s: environment(f, actions) for s, f in splits.items()}
    rng = np.random.default_rng(42)
    q, curve, train_curve = {}, [], []
    with measured() as timing:
        if "q_learning" in request.algorithms:
            users = list(histories["train"])
            checkpoints = set(
                np.linspace(1, request.episodes, min(20, request.episodes)).astype(int)
            )
            for episode in range(1, request.episodes + 1):
                user = users[int(rng.integers(len(users)))]
                state, chosen, total, mask = (user, 0, -1, 0), set(), 0.0, 0
                feedback = histories["train"][user]
                for step in range(5):
                    valid = [a for a in range(len(actions)) if a not in chosen]
                    action = (
                        int(rng.choice(valid))
                        if rng.random() < request.exploration_rate
                        else action_for("q_learning", q, state, valid, rng)
                    )
                    reward = feedback.get(action, 0.0)
                    mask |= 1 << action
                    next_state = (user, mask, genres[action], int(reward > 0))
                    next_valid = [a for a in valid if a != action]
                    future = max(
                        (q.get(next_state, {}).get(a, 0.0) for a in next_valid), default=0.0
                    )
                    target = reward + (request.discount_factor * future if step < 4 else 0.0)
                    values = q.setdefault(state, {})
                    values[action] = values.get(action, 0.0) + request.learning_rate * (
                        target - values.get(action, 0.0)
                    )
                    total += reward
                    chosen.add(action)
                    state = next_state
                train_curve.append(total)
                if episode in checkpoints:
                    update(
                        15 + 65 * episode / request.episodes,
                        f"Q-Learning · episode {episode}/{request.episodes}",
                    )
                    report = evaluate("q_learning", q, histories["validation"], genres, seed=100042)
                    curve.append((episode, report["average_reward"]))
    rows = []
    for policy in request.algorithms:
        if policy == "q_learning":
            costs = timing
        else:
            with measured() as costs:
                baseline = np.random.default_rng(42) if policy == "random" else list(actions)
        for split in ("validation", "test"):
            report = evaluate(
                policy,
                q,
                histories[split],
                genres,
                seed=100042 if split == "validation" else 200042,
            )
            rows.append(
                {
                    "algorithm": policy,
                    "name": names[policy],
                    "split": split,
                    **report,
                    **costs,
                    "model_bytes": len(pickle.dumps(q if policy == "q_learning" else baseline)),
                }
            )
    winner = max([r for r in rows if r["split"] == "validation"], key=lambda r: r["average_reward"])
    charts = []
    if curve:
        charts.append(
            chart(
                -1,
                "Reward đánh giá theo Episode",
                "line",
                "Episode huấn luyện",
                "Average Reward ↑",
                [
                    series("Q-Learning / Validation", *zip(*curve)),
                    *[
                        series(
                            names[p],
                            [1, request.episodes],
                            [
                                next(
                                    r["average_reward"]
                                    for r in rows
                                    if r["algorithm"] == p and r["split"] == "validation"
                                )
                            ]
                            * 2,
                        )
                        for p in request.algorithms
                        if p != "q_learning"
                    ],
                ],
                "validation",
            )
        )
        charts.append(
            chart(
                -1,
                "Reward thực tế trong episode huấn luyện",
                "line",
                "Episode",
                "Cumulative Reward",
                [series("Train", range(1, len(train_curve) + 1), train_curve)],
                "train",
            )
        )
    return {
        "rows": rows,
        "charts": charts,
        "actions": [str(metadata.loc[m, "title"]) for m in actions],
        "best_configs": {
            policy: (
                {
                    "alpha": request.learning_rate,
                    "gamma": request.discount_factor,
                    "epsilon": request.exploration_rate,
                    "episodes": request.episodes,
                }
                if policy == "q_learning"
                else {"seed": 42}
                if policy == "random"
                else {"ranking": "rating count descending on Train"}
            )
            for policy in request.algorithms
        },
        "selection": f"Reward Validation cao nhất: {winner['name']}. "
        "Kết luận chỉ áp dụng cho môi trường offline được mô tả; "
        "kiểm tra observed-feedback coverage trước khi sử dụng.",
        "notes": [
            "State = (userId, bitmask phim đã chọn, genre phim trước, phản hồi thích/không thích); "
            "state đầu=(userId,0,-1,0). Action = tối đa 30 phim phổ biến Train, "
            "loại phim đã chọn trong episode. Q-table chỉ lưu state đã gặp, không tạo bảng dense. "
            "Policy epsilon-greedy khi học, greedy khi đánh giá.",
            "Reward quan sát = (rating − 2,75)/2,25; action thiếu phản hồi lịch sử nhận 0 "
            "theo quy ước môi trường, không phải rating dự đoán hay bằng chứng user không thích. "
            "Không thêm nhiễu hoặc sinh rating giả.",
            "Mọi policy dùng cùng 100 user-episode và horizon 5 trên từng split, "
            "seed đánh giá tách Train. Báo cáo observed_feedback_fraction "
            "để thấy độ thưa và thiên lệch phản hồi.",
            "Không có propensity logging nên đây là mô phỏng offline, "
            "không phải đánh giá nhân quả/counterfactual. Test không dùng chọn policy, "
            "reward không chứng minh hiệu quả online.",
            "Training time Q-Learning gồm checkpoint Validation; baseline đo khởi tạo policy, "
            "catalog/popularity chung lấy từ Train. Inference chỉ đo chọn action, "
            "loại truy cập reward.",
        ],
    }
