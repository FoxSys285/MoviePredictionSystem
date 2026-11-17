"""User taste clusters from past history, with coverage and future ranking checks."""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from movie_rating.experiment_unsupervised import cluster_scores, projected_chart
from movie_rating.lab import chart, measured, series
from movie_rating.memory import check_memory, serialized_size
from movie_rating.models.full_dbscan import CoreNeighbors, FullDBSCAN, Neighborhoods

PROFILE_BATCH = 262_144
# These thresholds switch implementations; neither limits the number of fit users.
BOUNDED_DBSCAN_USERS = 5000
MINIBATCH_USERS = 20_000
CLUSTER_ITEM_DENSE_BYTES = 64 * 2**20
SCORE_USERS = 2000


def metric_text(value):
    return "N/A" if value is None else f"{value:.5f}"


def report_scores(algorithm, stage, quality, update):
    update(
        None,
        f"{algorithm} · {stage} · "
        f"Silhouette={metric_text(quality['silhouette'])} · "
        f"DBI={metric_text(quality['dbi'])} · "
        f"{quality['clusters']} cụm · bao phủ {quality['coverage']:.2%} · "
        f"noise={quality['noise_fraction']:.2%} · "
        f"{quality['assigned_users']:,}/{quality['evaluated_users']:,} user được gán",
    )


@dataclass
class TasteProfiles:
    users: np.ndarray
    genres: list[str]
    preferences: np.ndarray
    frequencies: np.ndarray
    prior_frequency: np.ndarray
    rating_counts: np.ndarray
    user_means: np.ndarray
    excluded_users: int

    @property
    def features(self):
        return np.column_stack((self.preferences, self.frequencies - self.prior_frequency)).astype(
            np.float32
        )


def taste_profiles(frame, movies, minimum=5, shrinkage=10.0, update=lambda *_: None):
    """Accumulate user/genre statistics in batches; never explode the rating table."""
    all_users, counts = np.unique(frame.userId.to_numpy(), return_counts=True)
    eligible = counts >= minimum
    users = all_users[eligible]
    if len(users) < 4:
        raise ValueError("Cần ít nhất 4 user có đủ ratings Train để phân cụm.")
    counts = counts[eligible]
    genres = sorted(
        {
            g
            for value in movies.genres.fillna("")
            for g in value.split("|")
            if g and g != "(no genres listed)"
        }
    )
    if len(genres) < 2:
        raise ValueError("Cần ít nhất hai thể loại phim để tạo hồ sơ sở thích.")
    movie_index = pd.Index(movies.movieId)
    user_index = pd.Index(users)
    # Row zero also handles missing catalog metadata without accidental negative indexing.
    indicator = np.zeros((len(movies) + 1, len(genres)), dtype=bool)
    for column, genre in enumerate(genres):
        indicator[1:, column] = (
            movies.genres.fillna("").str.split("|").map(lambda parts: genre in parts)
        )
    sums = np.zeros((len(users), len(genres)), dtype=np.float64)
    genre_counts = np.zeros_like(sums, dtype=np.int32)
    user_sums = np.zeros(len(users), dtype=np.float64)
    for start in range(0, len(frame), PROFILE_BATCH):
        check_memory()
        batch = frame.iloc[start : start + PROFILE_BATCH]
        uc = user_index.get_indexer(batch.userId)
        keep = uc >= 0
        uc = uc[keep]
        ratings = batch.rating.to_numpy()[keep]
        mc = movie_index.get_indexer(batch.movieId)[keep] + 1
        user_sums += np.bincount(uc, weights=ratings, minlength=len(users))
        for column in range(len(genres)):
            present = indicator[mc, column]
            genre_counts[:, column] += np.bincount(uc[present], minlength=len(users))
            sums[:, column] += np.bincount(
                uc[present], weights=ratings[present], minlength=len(users)
            )
        update(
            5 + 12 * min(1, (start + len(batch)) / len(frame)),
            f"Hồ sơ sở thích · {min(start + len(batch), len(frame)):,}/{len(frame):,} ratings",
        )
    means = user_sums / counts
    preferences = (sums - genre_counts * means[:, None]) / (genre_counts + shrinkage)
    prior = genre_counts.sum(axis=0, dtype=np.float64) / counts.sum()
    frequencies = (genre_counts + shrinkage * prior) / (counts[:, None] + shrinkage)
    return TasteProfiles(
        users,
        genres,
        preferences.astype(np.float32),
        frequencies.astype(np.float32),
        prior.astype(np.float32),
        counts,
        means,
        int((~eligible).sum()),
    )


@dataclass
class TasteScale:
    center: np.ndarray
    scale: np.ndarray
    genre_count: int

    @classmethod
    def fit(cls, x):
        half = x.shape[1] // 2
        floor = np.r_[np.full(half, 0.25), np.full(half, 0.1)]
        return cls(x.mean(axis=0), np.maximum(x.std(axis=0), floor), half)

    def transform(self, x):
        result = ((x - self.center) / self.scale).astype(np.float32)
        result[:, self.genre_count :] *= 0.5
        return result


@dataclass
class ClusterModel:
    algorithm: str
    model: object
    nearest: object
    fit_users: int

    def predict(self, x):
        labels = np.full(len(x), -1, dtype=np.int32)
        size = 512 if self.algorithm == "dbscan" else 65_536
        for start in range(0, len(x), size):
            check_memory()
            batch = x[start : start + size]
            if self.algorithm == "kmeans":
                labels[start : start + len(batch)] = self.model.predict(batch)
            elif self.nearest is not None:
                distance, index = self.nearest.kneighbors(batch)
                assigned = self.model.labels_[self.model.core_sample_indices_[index[:, 0]]].copy()
                assigned[distance[:, 0] > self.model.eps] = -1
                labels[start : start + len(batch)] = assigned
        return labels


def fit_cluster(
    algorithm, config, x, seed=42, minibatch=False, neighborhoods=None, update=lambda *_: None
):
    fit = x
    if algorithm == "dbscan":
        model = (
            FullDBSCAN(**config).fit(fit, neighborhoods, update)
            if len(fit) > BOUNDED_DBSCAN_USERS
            else DBSCAN(**config, n_jobs=1).fit(fit)
        )
    elif minibatch or len(fit) > MINIBATCH_USERS:
        model = MiniBatchKMeans(
            **config, n_init=3, batch_size=2048, max_iter=100, random_state=seed
        )
        # Every row is visited on every pass; partial_fit is not user sampling.
        rng = np.random.default_rng(seed)
        for epoch in range(3):
            order = rng.permutation(len(fit))
            for start in range(0, len(fit), 2048):
                check_memory()
                model.partial_fit(fit[order[start : start + 2048]])
            update(None, f"K-Means · lượt {epoch + 1}/3 đã đọc {len(fit):,}/{len(fit):,} user")
    else:
        model = KMeans(**config, n_init=10, random_state=seed).fit(fit)
    nearest = None
    if algorithm == "dbscan" and len(model.components_):
        nearest = CoreNeighbors(model.components_)
    return ClusterModel(algorithm, model, nearest, len(fit))


def scores(x, labels):
    result = cluster_scores(x, labels, 42)
    names, counts = np.unique(labels[labels >= 0], return_counts=True)
    assigned = int((labels >= 0).sum())
    result.update(
        coverage=assigned / len(labels) if len(labels) else 0,
        assigned_users=assigned,
        evaluated_users=len(labels),
        cluster_sizes={int(n): int(c) for n, c in zip(names, counts)},
        smallest_cluster=int(counts.min()) if len(counts) else 0,
        largest_cluster_fraction=float(counts.max() / assigned) if assigned else None,
    )
    return result


def candidate_key(score, minimum_coverage):
    """No amount of silhouette compensates for failing the coverage requirement."""
    valid = score["silhouette"] is not None and score["clusters"] >= 2
    qualified = valid and score["coverage"] >= minimum_coverage
    return (
        qualified,
        valid,
        score["silhouette"] if qualified else score["coverage"],
        score["silhouette"] if valid else -2,
        -(score["dbi"] if score["dbi"] is not None else float("inf")),
    )


def eps_candidates(x, min_samples, request, neighborhoods=None):
    if not request.cluster_auto_eps:
        return sorted(set(request.eps_values))
    if neighborhoods is None:
        neighborhoods = Neighborhoods(x)
    neighborhoods.prepare([min_samples])
    distances = neighborhoods.distances[min_samples]
    return sorted({max(0.0001, float(e)) for e in np.quantile(distances, [0.5, 0.75, 0.9, 0.98])})


def stability(algorithm, config, x, reference, update):
    probe = np.sort(np.random.default_rng(42).choice(len(x), min(SCORE_USERS, len(x)), False))
    base = reference.predict(x[probe])
    aris, overlaps = [], []
    for seed in (43, 44):
        update(None, f"{algorithm} · kiểm tra độ ổn định seed {seed}")
        take = max(config.get("n_clusters", 2), int(0.8 * len(x)))
        subset = np.random.default_rng(seed).choice(len(x), take, False)
        other = fit_cluster(
            algorithm,
            config,
            x[subset],
            seed,
            minibatch=isinstance(reference.model, MiniBatchKMeans),
            update=update,
        ).predict(x[probe])
        joint = (base >= 0) & (other >= 0)
        overlaps.append(float(joint.mean()))
        if (
            joint.sum() >= 3
            and len(np.unique(base[joint])) > 1
            and len(np.unique(other[joint])) > 1
        ):
            ari = float(adjusted_rand_score(base[joint], other[joint]))
            aris.append(ari)
        else:
            ari = None
        update(
            None,
            f"{algorithm} · ổn định seed {seed}: ARI={metric_text(ari)} · "
            f"bao phủ chung {joint.mean():.2%} ({joint.sum():,}/{len(joint):,} user)",
        )
    return {
        "adjusted_rand_mean": float(np.mean(aris)) if aris else None,
        "joint_coverage_mean": float(np.mean(overlaps)),
        "comparisons": len(aris),
        "seeds": [42, 43, 44],
        "probe_users": len(probe),
    }


def ranking_scores(recommendations, relevance, k=10):
    if not relevance:
        return None, None
    hits = len(set(recommendations[:k]) & relevance)
    ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevance))) + 2))
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommendations[:k]) if item in relevance)
    return hits / len(relevance), float(dcg / ideal)


def evaluate_ranking(splits, profiles, assignments, request, update):
    """Full Train counts; a fixed, disclosed user cohort bounds offline evaluation memory."""
    users = profiles.users
    cohort = np.sort(
        np.random.default_rng(42).choice(
            users, min(request.cluster_eval_users, len(users)), replace=False
        )
    )
    cohort_set = set(map(int, cohort))
    seen = {int(u): set() for u in cohort}
    # Only Train catalog items can become recommendations; future-only items remain misses.
    items = np.sort(splits["train"].movieId.unique())
    item_index, user_index = pd.Index(items), pd.Index(users)
    popularity = np.zeros(len(items), dtype=np.float64)
    counts = {}
    for aid, labels in assignments.items():
        shape = (max(0, int(labels.max()) + 1), len(items))
        counts[aid] = (
            sparse.csr_matrix(shape, dtype=np.float32)
            if shape[0] * shape[1] * 4 > CLUSTER_ITEM_DENSE_BYTES
            else np.zeros(shape, dtype=np.float32)
        )
    for start in range(0, len(splits["train"]), PROFILE_BATCH):
        check_memory()
        batch = splits["train"].iloc[start : start + PROFILE_BATCH]
        uc = user_index.get_indexer(batch.userId)
        mc = item_index.get_indexer(batch.movieId)
        positive = (batch.rating.to_numpy() >= 3.5) & (uc >= 0)
        popularity += np.bincount(mc[positive], minlength=len(items))
        for aid, labels in assignments.items():
            valid = positive.copy()
            valid[uc >= 0] &= labels[uc[uc >= 0]] >= 0
            if valid.any():
                indices = (labels[uc[valid]], mc[valid])
                if sparse.issparse(counts[aid]):
                    counts[aid] += sparse.coo_matrix(
                        (np.ones(valid.sum(), dtype=np.float32), indices),
                        shape=counts[aid].shape,
                    ).tocsr()
                else:
                    np.add.at(counts[aid], indices, 1)
        selected = batch.loc[batch.userId.isin(cohort_set), ["userId", "movieId"]]
        for user, group in selected.groupby("userId", sort=False):
            seen[int(user)].update(map(int, group.movieId))
        update(
            82 + 8 * min(1, (start + len(batch)) / len(splits["train"])),
            "Gợi ý theo cụm · thống kê phim đã thích trên Train",
        )
    # Smoothed probability of a like per cluster member, toward global popularity.
    prior = popularity / len(users)
    popular_order = items[np.argsort(-popularity, kind="stable")]
    orders = {}
    cohort_codes = user_index.get_indexer(cohort)
    maximum_seen = max(map(len, seen.values()), default=0)
    for aid, matrix in counts.items():
        sizes = np.bincount(assignments[aid][assignments[aid] >= 0], minlength=matrix.shape[0])
        orders[aid] = {}
        for c in np.unique(assignments[aid][cohort_codes]):
            if c < 0:
                continue
            row = matrix.getrow(c).toarray().ravel() if sparse.issparse(matrix) else matrix[c]
            ordered = np.argsort(-(row + 20 * prior) / (sizes[c] + 20), kind="stable")
            orders[aid][int(c)] = items[ordered[: maximum_seen + 10]]
    del counts
    rows = []
    for split in ("validation", "test"):
        frame = splits[split]
        positive = frame.loc[(frame.rating >= 3.5) & frame.userId.isin(cohort_set)]
        relevant = {
            int(u): set(map(int, group.movieId)) - seen[int(u)]
            for u, group in positive.groupby("userId", sort=False)
        }
        relevant = {u: movies for u, movies in relevant.items() if movies}
        for aid in ["popularity", *assignments]:
            recalls, ndcgs, grouped_recalls, grouped_ndcgs = [], [], [], []
            for user, code in zip(cohort, cohort_codes):
                user = int(user)
                if user not in relevant:
                    continue
                cluster = -1 if aid == "popularity" else int(assignments[aid][code])
                order = popular_order if cluster < 0 else orders[aid][cluster]
                chosen = []
                for item in order:
                    if int(item) not in seen[user]:
                        chosen.append(int(item))
                        if len(chosen) == 10:
                            break
                recall, ndcg = ranking_scores(chosen, relevant[user])
                recalls.append(recall)
                ndcgs.append(ndcg)
                if cluster >= 0:
                    grouped_recalls.append(recall)
                    grouped_ndcgs.append(ndcg)
            rows.append(
                {
                    "algorithm": aid,
                    "split": split,
                    "recall_at_10": float(np.mean(recalls)) if recalls else None,
                    "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else None,
                    "evaluated_users": len(recalls),
                    "sampled_users": len(cohort),
                    "clustered_users": len(grouped_recalls),
                    "clustered_recall_at_10": float(np.mean(grouped_recalls))
                    if grouped_recalls
                    else None,
                    "clustered_ndcg_at_10": float(np.mean(grouped_ndcgs))
                    if grouped_ndcgs
                    else None,
                }
            )
            update(
                None,
                f"{aid} · {split.title()} · "
                f"Recall@10={metric_text(rows[-1]['recall_at_10'])} · "
                f"NDCG@10={metric_text(rows[-1]['ndcg_at_10'])} · "
                f"{len(recalls):,}/{len(cohort):,} user có phản hồi hợp lệ · "
                f"{len(grouped_recalls):,} user có cụm",
            )
    return {
        "k": 10,
        "rows": rows,
        "cohort_seed": 42,
        "cohort_users": len(cohort),
        "candidate_movies": len(items),
        "positive_threshold": 3.5,
        "protocol": "frozen_train_clusters_full_train_catalog_future_positives_v1",
    }


def clustering(splits, movies, request, update):
    profiles = taste_profiles(
        splits["train"], movies, request.min_user_ratings, request.cluster_shrinkage, update
    )
    raw = profiles.features
    update(
        None,
        f"Đã tạo hồ sơ sở thích Train: {len(raw):,} user · "
        f"{int(profiles.rating_counts.sum()):,} ratings · {len(profiles.genres)} genres · "
        f"{raw.shape[1]} đặc trưng · loại {profiles.excluded_users:,} user thiếu lịch sử",
    )
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(raw))
    # Hold out USERS with the same past-history representation, not fragments of future ratings.
    held_count = max(2, min(SCORE_USERS, len(raw) // 5))
    fit_indices, held_indices = shuffled[held_count:], shuffled[:held_count]
    scale = TasteScale.fit(raw[fit_indices])
    x = scale.transform(raw)
    fit_x = x[fit_indices]
    neighborhoods = None
    if "dbscan" in request.algorithms:
        update(
            None,
            f"DBSCAN · chuẩn bị khoảng cách láng giềng cho toàn bộ "
            f"{len(fit_x):,} user fit ứng viên · "
            f"min_samples={sorted(set(request.min_samples_values))}",
        )
        neighborhoods = Neighborhoods(fit_x)
        neighborhoods.prepare([m for m in request.min_samples_values if m <= len(fit_x)], update)
    rows, search, winners, charts, assignments = [], [], [], [], {}
    for aid in request.algorithms:
        name = "K-Means" if aid == "kmeans" else "DBSCAN"
        update(
            None,
            f"Bắt đầu {name} · {len(fit_indices):,} user fit ứng viên · "
            f"{held_count:,} user giữ lại · bao phủ yêu cầu ≥{request.cluster_min_coverage:.0%}",
        )
        configs = []
        if aid == "kmeans":
            configs = [
                {"n_clusters": k} for k in sorted(set(request.k_values)) if k <= len(fit_indices)
            ]
        else:
            support = fit_x
            for m in sorted(set(request.min_samples_values)):
                if m <= len(support):
                    configs.extend(
                        {"eps": e, "min_samples": m}
                        for e in eps_candidates(support, m, request, neighborhoods)
                    )
        best, best_key = None, None
        for index, config in enumerate(configs):
            progress = 20 + 60 * (request.algorithms.index(aid) + 0.7 * index / len(configs)) / len(
                request.algorithms
            )
            update(progress, f"{aid} · khảo sát {config}")
            with measured() as timing:
                candidate = fit_cluster(
                    aid,
                    config,
                    fit_x,
                    minibatch=len(x) > MINIBATCH_USERS,
                    neighborhoods=neighborhoods,
                    update=update,
                )
            train_labels = candidate.model.labels_ if aid == "dbscan" else candidate.predict(fit_x)
            train_score = scores(x[fit_indices], train_labels)
            held_score = scores(x[held_indices], candidate.predict(x[held_indices]))
            qualified = candidate_key(held_score, request.cluster_min_coverage)[0]
            report_scores(name, f"user giữ lại · {config}", held_score, update)
            search.append(
                {
                    "algorithm": aid,
                    "config": config,
                    **train_score,
                    **timing,
                    "validation_silhouette": held_score["silhouette"],
                    "validation_dbi": held_score["dbi"],
                    "validation_coverage": held_score["coverage"],
                    "validation_users": len(held_indices),
                    "qualified": qualified,
                }
            )
            key = candidate_key(held_score, request.cluster_min_coverage)
            if best_key is None or key > best_key:
                best, best_key = config, key
            del candidate
        if best is None:
            raise ValueError(f"Không có cấu hình {aid} hợp lệ với số user Train hiện tại.")
        update(
            None,
            f"{name} · cấu hình chọn: {best} · "
            + (
                "đạt điều kiện trên user giữ lại"
                if best_key[0]
                else "chưa đạt điều kiện; chỉ báo kết quả chẩn đoán"
            ),
        )
        progress = 20 + 60 * (request.algorithms.index(aid) + 0.8) / len(request.algorithms)
        update(progress, f"{aid} · fit cấu hình đã chọn trên hồ sơ lịch sử Train")
        with measured() as timing:
            model = fit_cluster(aid, best, x, update=update)
        labels = model.model.labels_.copy() if aid == "dbscan" else model.predict(x)
        inference_x = x[held_indices]
        start = time.perf_counter()
        model.predict(inference_x)
        inference = time.perf_counter() - start
        quality = scores(x, labels)
        qualified = bool(best_key[0]) and candidate_key(quality, request.cluster_min_coverage)[0]
        report_scores(name, f"fit cuối {model.fit_users:,}/{len(x):,} user Train", quality, update)
        update(
            None,
            f"{name} · fit cuối mất {timing['fit_seconds']:.3f}s · "
            + ("đạt điều kiện chọn" if qualified else "chưa đạt điều kiện chọn"),
        )
        stable = stability(aid, best, x, model, update)
        update(
            None,
            f"{name} · độ ổn định: ARI trung bình="
            f"{metric_text(stable['adjusted_rand_mean'])} · "
            f"{stable['comparisons']}/2 lượt hợp lệ",
        )
        rows.append(
            {
                "algorithm": aid,
                "name": "K-Means" if aid == "kmeans" else "DBSCAN",
                "split": "train_history",
                **quality,
                **timing,
                "inference_seconds": inference,
                "inference_rows": len(inference_x),
                "model_bytes": serialized_size((scale, model)),
                "fit_users": model.fit_users,
                "train_ratings": int(profiles.rating_counts.sum()),
                "fit_fraction": model.fit_users / len(x),
                "qualified": qualified,
                "stability_ari": stable["adjusted_rand_mean"],
            }
        )
        descriptions = []
        for cluster in np.unique(labels):
            mask = labels == cluster
            relative = profiles.preferences[mask].mean(axis=0)
            top = np.argsort(-relative, kind="stable")[:5]
            descriptions.append(
                {
                    "cluster": int(cluster),
                    "users": int(mask.sum()),
                    "genre_means": {profiles.genres[i]: float(relative[i]) for i in top},
                }
            )
        winners.append(
            {
                "algorithm": aid,
                "config": best,
                "profiles": descriptions,
                "qualified": qualified,
                "stability": stable,
            }
        )
        assignments[aid] = labels
        chosen = np.sort(rng.choice(len(x), min(500, len(x)), False))
        pca_fit = x[np.sort(rng.choice(len(x), min(5000, len(x)), False))]
        projection = PCA(2, random_state=42).fit(pca_fit).transform(x[chosen])
        charts.append(
            projected_chart(f"{aid} · hồ sơ lịch sử / PCA 2D", projection, labels[chosen])
        )
        del model
        update(None, f"Hoàn tất {name} · đã thu thập kết quả và giải phóng mô hình fit")
    update(
        82,
        f"Đánh giá Recall@10 / NDCG@10 trên sở thích tương lai · "
        f"cohort tối đa {request.cluster_eval_users:,} user · baseline Popularity",
    )
    recommendation = evaluate_ranking(splits, profiles, assignments, request, update)
    eligible = {w["algorithm"] for w in winners if w["qualified"]}
    validation = [
        r
        for r in recommendation["rows"]
        if r["split"] == "validation" and r["algorithm"] in eligible and r["ndcg_at_10"] is not None
    ]
    baseline = next(
        r
        for r in recommendation["rows"]
        if r["algorithm"] == "popularity" and r["split"] == "validation"
    )
    best = max(validation, key=lambda r: r["ndcg_at_10"]) if validation else None
    selection = "Chưa có mô hình đạt đồng thời điều kiện phân cụm và tỷ lệ bao phủ yêu cầu."
    if eligible and not validation:
        selection = (
            "Có mô hình đạt điều kiện phân cụm nhưng chưa đủ phản hồi "
            "Validation để chọn theo NDCG@10."
        )
    if best:
        exceeds = baseline["ndcg_at_10"] is not None and best["ndcg_at_10"] > baseline["ndcg_at_10"]
        selection = (
            f"Trong các mô hình đạt bao phủ ≥{request.cluster_min_coverage:.0%} và có ≥2 cụm, "
            f"{best['algorithm']} có NDCG@10 Validation cao nhất ({best['ndcg_at_10']:.4f}). "
            + (
                "Vượt baseline Popularity trên Validation."
                if exceeds
                else "Chưa vượt baseline Popularity trên Validation; "
                "chưa có bằng chứng cải thiện gợi ý."
            )
        )
    update(None, "Lựa chọn mô hình · " + selection)
    charts.append(
        chart(
            -1,
            "Kích thước cụm / Train",
            "bar",
            "Cụm",
            "Users",
            [
                series(
                    w["algorithm"],
                    [str(p["cluster"]) if p["cluster"] >= 0 else "Noise" for p in w["profiles"]],
                    [p["users"] for p in w["profiles"]],
                )
                for w in winners
            ],
            "train_history",
        )
    )
    return {
        "rows": rows,
        "search": search,
        "clusters": winners,
        "charts": charts,
        "best_configs": {w["algorithm"]: w["config"] for w in winners},
        "selection": selection,
        "recommendation": recommendation,
        "clustering_protocol": "past_taste_full_train_future_ranking_v2",
        "cluster_min_coverage": request.cluster_min_coverage,
        "profile_kind": "relative_genre_preference_and_frequency",
        "profile_excluded_users": profiles.excluded_users,
        "training": {
            "split": "train",
            "ratings": len(splits["train"]),
            "eligible_ratings": int(profiles.rating_counts.sum()),
            "eligible_users": len(x),
            "scope": "all_eligible_train_users",
        },
        "notes": [
            f"Hồ sơ Train: rating thể loại trừ rating trung bình user, shrinkage "
            f"{request.cluster_shrinkage:g}; thể loại chưa xem có sở thích tương đối bằng 0. "
            "Ghép tần suất xem thể loại có làm trơn, trọng số 0,5 "
            "sau chuẩn hóa có sàn độ lệch chuẩn.",
            f"Loại {profiles.excluded_users} user có dưới "
            f"{request.min_user_ratings} ratings Train. "
            "Không dựng hồ sơ từ các mẩu rating Validation/Test. "
            "Hình học trong bảng đo trên Train; "
            "chọn cấu hình bằng user giữ lại từ chính hồ sơ quá khứ (không dùng rating tương lai).",
            f"Tuning dùng toàn bộ {len(fit_indices):,} hồ sơ Train ngoài "
            f"{held_count:,} user giữ lại; fit cuối dùng toàn bộ {len(x):,} user đủ lịch sử. "
            "Silhouette tối đa 2.000 user không noise. "
            f"Yêu cầu bao phủ ≥{request.cluster_min_coverage:.0%} "
            "và ≥2 cụm trước khi xếp hạng. Không có cấu hình đạt thì báo chưa đạt.",
            "DBSCAN xác định core và nối cụm trên toàn bộ hồ sơ Train, "
            "truy vấn láng giềng theo batch, không giữ đồ thị khoảng cách toàn bộ trong RAM. "
            "eps tự động lấy phân vị 50/75/90/98% khoảng cách láng giềng thứ min_samples "
            "(gồm chính điểm đó), hoặc dùng danh sách thủ công khi tắt tự động.",
            "K-Means dùng MiniBatchKMeans khi hơn 20.000 hồ sơ. Độ ổn định: ARI trên user "
            "cùng được gán cụm, so với hai lần fit lại trên 80% user với seed 43/44; "
            "đọc cùng tỷ lệ bao phủ chung. PCA chỉ minh họa, không dùng tính điểm.",
            "Training Time chỉ gồm fit mô hình cuối và tạo index core DBSCAN; "
            "không gồm tạo hồ sơ, tuning hay kiểm tra ổn định. Đồng hồ lượt chạy gồm toàn bộ bước.",
            f"Gợi ý từ toàn bộ catalog Train, bỏ phim user đã đánh giá trong Train. "
            f"Cohort cố định tối đa {request.cluster_eval_users:,} user seed 42; chỉ chấm user có "
            "phim tương lai chưa xem được rating ≥3,5. "
            "Phim chỉ xuất hiện tương lai vẫn tính là miss. "
            "Recall/NDCG là trung bình theo user; DBSCAN noise dùng Popularity dự phòng. "
            "Báo thêm điểm riêng user có cụm; đây là đánh giá offline trên phản hồi quan sát.",
            "Giữ nguyên mô hình và lịch sử Train cho cả Validation/Test. Chọn thuật toán bằng "
            "NDCG@10 Validation sau điều kiện bao phủ; Test chỉ báo cáo cuối, không chọn tham số. "
            "Số liệu theo giao thức mới không so trực tiếp với lịch sử phân cụm cũ.",
        ],
    }
