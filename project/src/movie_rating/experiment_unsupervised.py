"""Genre user profiles, Apriori transactions and distinct PCA/TruncatedSVD."""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from movie_rating.lab import chart, measured, series


def genre_profiles(frame, movies, minimum):
    """Mean observed rating per genre; missing genres use that user's mean."""
    genres = sorted(
        {
            g
            for s in movies.genres.fillna("")
            for g in s.split("|")
            if g and g != "(no genres listed)"
        }
    )
    eligible = frame.groupby("userId").size().loc[lambda n: n >= minimum].index
    frame = frame.loc[frame.userId.isin(eligible)]
    if not len(frame) or len(genres) < 2:
        raise ValueError("Không đủ user/thể loại. Hãy tăng mẫu hoặc giảm minimum ratings.")
    genre_map = movies[["movieId", "genres"]].copy()
    genre_map["genre"] = genre_map.genres.str.split("|")
    expanded = frame.merge(genre_map[["movieId", "genre"]].explode("genre"), on="movieId")
    profile = expanded.groupby(["userId", "genre"]).rating.mean().unstack().reindex(columns=genres)
    means = frame.groupby("userId").rating.mean().reindex(profile.index)
    profile = profile.T.fillna(means).T
    return profile.to_numpy(), profile.index.to_numpy(), genres


def cluster_scores(x, labels, seed):
    valid = labels >= 0
    count = len(np.unique(labels[valid]))
    scores = {
        "clusters": count,
        "noise_fraction": float(np.mean(~valid)) if len(labels) else None,
        "silhouette": None,
        "dbi": None,
    }
    if 1 < count < valid.sum():
        # Bound the O(n²) silhouette distance matrix.
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(np.flatnonzero(valid), min(2000, valid.sum()), replace=False))
        if 1 < len(np.unique(labels[chosen])) < len(chosen):
            scores["silhouette"] = float(silhouette_score(x[chosen], labels[chosen]))
        scores["dbi"] = float(davies_bouldin_score(x[valid], labels[valid]))
    return scores


def projected_chart(title, x, labels, split="train"):
    names = np.unique(labels)
    chosen = np.random.default_rng(42).choice(len(x), min(500, len(x)), replace=False)
    return chart(
        -1,
        title,
        "scatter",
        "PC1",
        "PC2",
        [
            series(
                "Noise" if k < 0 else f"Cụm {k}",
                x[chosen[labels[chosen] == k], 0],
                x[chosen[labels[chosen] == k], 1],
            )
            for k in names
        ],
        split,
    )


def clustering(splits, movies, request, update):
    raw, users, genres = genre_profiles(splits["train"], movies, request.min_user_ratings)
    if len(raw) < 4:
        raise ValueError("Cần ít nhất 4 user hợp lệ cho phân cụm.")
    scaler = StandardScaler().fit(raw)
    x = scaler.transform(raw)
    projection = PCA(2, random_state=42).fit_transform(x)
    held = {}
    for s in ("validation", "test"):
        try:
            a, _, _ = genre_profiles(splits[s], movies, 1)
            held[s] = scaler.transform(a)
        except ValueError:
            held[s] = np.empty((0, x.shape[1]))
    configs = []
    if "kmeans" in request.algorithms:
        configs.extend(
            ("kmeans", {"n_clusters": k}) for k in sorted(set(request.k_values)) if k < len(x)
        )
    if "dbscan" in request.algorithms:
        configs.extend(
            ("dbscan", {"eps": eps, "min_samples": m})
            for eps in sorted(set(request.eps_values))
            for m in sorted(set(request.min_samples_values))
        )
    rows, details, charts, winners = [], [], [], []
    for algorithm in request.algorithms:
        options = []
        for aid, config in configs:
            if aid != algorithm:
                continue
            update(12 + 65 * len(details) / max(1, len(configs)), f"{aid} · {config}")
            model = (
                KMeans(**config, n_init=10, random_state=42)
                if aid == "kmeans"
                else DBSCAN(**config)
            )
            with measured() as timing:
                model.fit(x)
            # Build the explicit DBSCAN extension once; index construction belongs to fit.
            start = time.perf_counter()
            nearest = (
                NearestNeighbors(n_neighbors=1).fit(model.components_)
                if aid == "dbscan" and len(model.components_)
                else None
            )
            timing["fit_seconds"] += time.perf_counter() - start

            def predict(new_x):
                if not len(new_x):
                    return np.array([], dtype=int)
                if aid == "kmeans":
                    return model.predict(new_x)
                if nearest is None:
                    return np.full(len(new_x), -1)
                distance, index = nearest.kneighbors(new_x)
                assigned = model.labels_[model.core_sample_indices_[index[:, 0]]].copy()
                assigned[distance[:, 0] > model.eps] = -1
                return assigned

            score = cluster_scores(x, model.labels_, 42)
            detail = {"algorithm": aid, "config": config, **score, **timing}
            details.append(detail)
            val_labels = predict(held["validation"])
            val = cluster_scores(held["validation"], val_labels, 42)
            detail["validation_silhouette"] = val["silhouette"]
            options.append((val, score, config, model, nearest, timing))
        if not options:
            raise ValueError("Không có cấu hình phân cụm hợp lệ với số user hiện tại.")
        # Prefer validation quality; deterministic train fallback if validation is degenerate.
        selected = max(
            options,
            key=lambda o: (
                o[0]["silhouette"] is not None,
                o[0]["silhouette"]
                if o[0]["silhouette"] is not None
                else o[1]["silhouette"]
                if o[1]["silhouette"] is not None
                else -2,
                -(o[1]["noise_fraction"] or 0),
            ),
        )
        val, score, config, model, nearest, timing = selected
        winners.append(
            {"algorithm": algorithm, "config": config, "validation": val, "train": score}
        )
        for split, new_x in held.items():
            start = time.perf_counter()
            # The closure refers to the last candidate; assign explicitly to the selected model.
            if algorithm == "kmeans":
                labels = model.predict(new_x) if len(new_x) else np.array([], dtype=int)
            elif nearest is not None and len(new_x):
                distance, index = nearest.kneighbors(new_x)
                labels = model.labels_[model.core_sample_indices_[index[:, 0]]].copy()
                labels[distance[:, 0] > model.eps] = -1
            else:
                labels = np.full(len(new_x), -1)
            seconds = time.perf_counter() - start
            rows.append(
                {
                    "algorithm": algorithm,
                    "name": "K-Means" if algorithm == "kmeans" else "DBSCAN",
                    "split": split,
                    **cluster_scores(new_x, labels, 42),
                    **timing,
                    "inference_seconds": seconds,
                    "inference_rows": len(new_x),
                    "model_bytes": len(pickle.dumps((scaler, model, nearest))),
                }
            )
        charts.append(projected_chart(f"{algorithm} · PCA 2D / Train", projection, model.labels_))
        profiles = []
        for cluster in np.unique(model.labels_):
            subset = raw[model.labels_ == cluster]
            top = np.argsort(subset.mean(axis=0))[::-1][:5]
            profiles.append(
                {
                    "cluster": int(cluster),
                    "users": len(subset),
                    "genre_means": {genres[i]: float(subset[:, i].mean()) for i in top},
                }
            )
        winners[-1]["profiles"] = profiles
    eligible = [w for w in winners if w["validation"]["silhouette"] is not None]
    selected = max(eligible, key=lambda w: w["validation"]["silhouette"]) if eligible else None
    return {
        "rows": rows,
        "charts": charts,
        "search": details,
        "clusters": winners,
        "best_configs": {w["algorithm"]: w["config"] for w in winners},
        "selection": f"Silhouette Validation cao nhất: {selected['algorithm']}."
        if selected
        else "Validation không đủ cụm hợp lệ; chưa có cơ sở chọn mô hình triển khai.",
        "notes": [
            f"Vector user = rating trung bình từng genre ({len(genres)} chiều). "
            f"Train loại user dưới {request.min_user_ratings} ratings; "
            "genre thiếu điền bằng rating trung bình user. Scaler chỉ fit trên Train.",
            "Silhouette loại noise và lấy mẫu tối đa 2.000 user với seed 42. "
            "Noise vẫn được báo cáo riêng; chỉ số đẹp với noise cao cần thận trọng.",
            "DBSCAN inference là phép mở rộng: gán core Train gần nhất trong eps, "
            "ngoài eps là noise. Thời gian gồm truy vấn nearest-neighbor, "
            "index được tính trong fit.",
            "Validation/Test là profile từ ratings giữ lại của user; "
            "không đại diện toàn bộ user mới. PCA 2D chỉ dùng để hiển thị.",
        ],
    }


def association(splits, movies, request, update):
    from mlxtend.frequent_patterns import apriori, association_rules

    frame = splits["train"]
    positive = frame.loc[frame.rating >= 4.0]
    items = (
        positive.groupby("movieId")
        .size()
        .sort_values(ascending=False, kind="stable")
        .head(request.max_items)
        .index
    )
    transactions = (
        pd.crosstab(positive.userId, positive.movieId)
        .reindex(index=np.sort(frame.userId.unique()), columns=items, fill_value=0)
        .fillna(0)
        .astype(bool)
    )
    update(20, "Apriori · rating ≥ 4,0 · transactions gồm cả user không có phim thích")
    with measured() as timing:
        patterns = (
            apriori(
                transactions,
                min_support=request.min_support,
                use_colnames=True,
                max_len=request.max_rule_length,
                low_memory=True,
            )
            if len(items)
            else pd.DataFrame()
        )
        rules = (
            association_rules(
                patterns,
                num_itemsets=len(transactions),
                metric="confidence",
                min_threshold=request.min_confidence,
            )
            if len(patterns)
            else pd.DataFrame()
        )
    metadata = movies.set_index("movieId").title.to_dict()
    records = []
    if len(rules):
        for _, r in rules.sort_values(
            ["lift", "confidence", "support"], ascending=False
        ).iterrows():
            records.append(
                {
                    "antecedents": [str(metadata[m]) for m in sorted(r.antecedents)],
                    "consequents": [str(metadata[m]) for m in sorted(r.consequents)],
                    "antecedent_ids": sorted(r.antecedents),
                    "consequent_ids": sorted(r.consequents),
                    "support": float(r.support),
                    "confidence": float(r.confidence),
                    "lift": float(r.lift),
                }
            )
    row = {
        "algorithm": "apriori",
        "name": "Apriori",
        "split": "train",
        **timing,
        "rule_count": len(records),
        "itemsets": len(patterns),
        "transactions": len(transactions),
        **{
            key: float(rules[key].mean()) if len(rules) else None
            for key in ("support", "confidence", "lift")
        },
    }
    charts = [
        chart(
            -1,
            "Top luật theo Lift",
            "bar",
            "Luật",
            "Lift ↑",
            [
                series(
                    "Lift",
                    [f"Luật {i + 1}" for i in range(min(12, len(records)))],
                    [r["lift"] for r in records[:12]],
                )
            ],
            "train",
        )
    ]
    best = records[0] if records else None
    return {
        "rows": [row],
        "charts": charts,
        "rules": records,
        "best_configs": {
            "apriori": {
                "min_support": request.min_support,
                "min_confidence": request.min_confidence,
                "max_items": request.max_items,
                "max_length": request.max_rule_length,
            }
        },
        "selection": (
            "Luật đầu: "
            + " + ".join(best["antecedents"])
            + " → "
            + " + ".join(best["consequents"])
            + f" (lift={best['lift']:.3f}). "
            + (
                "Có liên kết đồng xuất hiện dương; có thể tạo ứng viên gợi ý "
                "và xác nhận trên dữ liệu mới."
                if best["lift"] > 1
                else "Lift không vượt 1; chưa có bằng chứng liên kết dương để ưu tiên gợi ý."
            )
        )
        if best
        else "Không có luật đạt ngưỡng; có thể giảm support/confidence hoặc tăng số user.",
        "notes": [
            "Chỉ phim rating ≥4,0 tạo mục trong transaction. "
            "Giữ cả transaction rỗng của user Train trong mẫu số support.",
            "Chỉ khai phá trên Train; Support/Confidence/Lift mô tả đồng xuất hiện, "
            "không chứng minh quan hệ nhân quả.",
            "Không có inference chuẩn cho Apriori: "
            "báo cáo execution time của khai phá và sinh luật. "
            "Catalog giới hạn top phim theo số rating thích Train; "
            "ngưỡng support áp dụng trên catalog này.",
        ],
    }


def reduction(splits, movies, request, update):
    raw, _, genres = genre_profiles(splits["train"], movies, request.min_user_ratings)
    scaler = StandardScaler().fit(raw)
    x = scaler.transform(raw)
    maximum = min(x.shape[1], len(x) - 1)
    if maximum < 1 or float(np.var(x, axis=0).sum()) <= 1e-12:
        raise ValueError("Vector user–genre không có đủ phương sai để đánh giá giảm chiều.")
    choices = sorted({k for k in request.components if k <= maximum})
    if not choices:
        raise ValueError(f"Components tối đa với dữ liệu hiện tại: {maximum}.")
    held = {
        s: scaler.transform(genre_profiles(splits[s], movies, 1)[0]) for s in ("validation", "test")
    }
    rows, charts, configs = [], [], {}
    for aid in request.algorithms:
        curve = []
        for k in choices:
            update(15 + 65 * len(curve) / len(choices), f"{aid} · {k} components")
            model = PCA(k, svd_solver="full") if aid == "pca" else TruncatedSVD(k, random_state=42)
            with measured() as timing:
                z = model.fit_transform(x)
            variance = float(model.explained_variance_ratio_.sum())
            curve.append((k, variance))
            for split, original in held.items():
                start = time.perf_counter()
                transformed = model.transform(original)
                seconds = time.perf_counter() - start
                rows.append(
                    {
                        "algorithm": f"{aid}_{k}",
                        "name": f"{'PCA' if aid == 'pca' else 'TruncatedSVD'} · {k}",
                        "split": split,
                        "components": k,
                        "explained_variance_ratio": variance,
                        "reconstruction_error": float(
                            np.mean((original - model.inverse_transform(transformed)) ** 2)
                        ),
                        **timing,
                        "inference_seconds": seconds,
                        "model_bytes": len(pickle.dumps((scaler, model))),
                    }
                )
            if k == choices[-1] and z.shape[1] >= 2:
                charts.append(
                    projected_chart(
                        f"{aid} · sau giảm chiều", z[:, :2], np.zeros(len(z), dtype=int)
                    )
                )
        charts.append(
            chart(
                -1,
                f"{aid} · Explained variance theo components",
                "line",
                "Components",
                "Explained variance ↑",
                [series(aid, *zip(*curve))],
                "train",
            )
        )
        configs[aid] = {"components_tested": choices, "input_genres": len(genres)}
    charts.append(
        projected_chart(
            "Trước giảm chiều · hai genre chuẩn hóa đầu", x[:, :2], np.zeros(len(x), dtype=int)
        )
    )
    charts[-1].update(x_label=genres[0], y_label=genres[1])
    validation = [r for r in rows if r["split"] == "validation"]
    eligible = [r for r in validation if r["explained_variance_ratio"] >= 0.9]
    winner = (
        min(eligible, key=lambda r: (r["components"], r["reconstruction_error"]))
        if eligible
        else min(validation, key=lambda r: r["reconstruction_error"])
    )
    for aid in configs:
        options = [r for r in validation if r["algorithm"].startswith(aid + "_")]
        eligible_options = [r for r in options if r["explained_variance_ratio"] >= 0.9]
        selected = (
            min(eligible_options, key=lambda r: (r["components"], r["reconstruction_error"]))
            if eligible_options
            else min(options, key=lambda r: r["reconstruction_error"])
        )
        configs[aid]["selected_components"] = selected["components"]
    return {
        "rows": rows,
        "charts": charts,
        "best_configs": configs,
        "selection": f"Chọn trên Validation: {winner['name']}. "
        "Ưu tiên ít chiều giữ ≥90% phương sai; nếu không đạt, chọn reconstruction error thấp nhất.",
        "notes": [
            "PCA và TruncatedSVD dùng cùng vector rating trung bình user–genre chuẩn hóa; "
            "không tạo ma trận user–item dense.",
            "PCA có centering nội bộ; TruncatedSVD không center nội bộ "
            "(đầu vào chung đã chuẩn hóa) và dùng randomized SVD. Đây là giảm chiều đặc trưng, "
            "khác hoàn toàn Funk SVD Collaborative Filtering ở Regression.",
            "Explained variance học trên Train; reconstruction MSE đo riêng trên Validation/Test. "
            "Không coi explained variance là độ chính xác dự đoán rating.",
        ],
    }
