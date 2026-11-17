"""Genre user profiles, Apriori transactions and distinct PCA/TruncatedSVD."""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import davies_bouldin_score, silhouette_score
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
    # Keep the historical import path; the active implementation has its own module.
    from movie_rating.experiment_clustering import clustering as run_clustering

    return run_clustering(splits, movies, request, update)


def association(splits, movies, request, update):
    from mlxtend.frequent_patterns import apriori, association_rules

    frame = splits["train"]
    update(12, f"Apriori · quét toàn bộ {len(frame):,} ratings Train; lọc phim rating ≥ 4,0.")
    with measured() as timing:
        positive = frame.loc[frame.rating >= 4.0, ["userId", "movieId"]]
        counts = positive.groupby("movieId").size().sort_values(ascending=False, kind="stable")
        items = counts.head(request.max_items).index
        users = np.sort(frame.userId.unique())
        update(20, f"Có {len(positive):,} lượt thích / {len(counts):,} phim; "
               f"chọn {len(items):,} phim ứng viên từ toàn bộ Train "
               f"(max_items={request.max_items}).")
        selected = positive.loc[positive.movieId.isin(items)]
        # Allocate only the bounded candidate catalog, never users × every observed movie.
        values = np.zeros((len(users), len(items)), dtype=bool)
        if len(selected):
            values[np.searchsorted(users, selected.userId.to_numpy()),
                   items.get_indexer(selected.movieId)] = True
        transactions = pd.DataFrame(values, index=users, columns=items)
        empty_transactions = int((~values.any(axis=1)).sum())
        update(32, f"Transactions: {len(users):,} users × {len(items):,} phim; "
               f"giữ {empty_transactions:,} transaction rỗng trong mẫu số support; "
               f"ma trận boolean {values.nbytes / 2**20:.2f} MiB.")
        update(40, f"Bắt đầu Apriori · min_support={request.min_support:g} · "
               f"độ dài itemset tối đa={request.max_rule_length}.")
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
        update(65, f"Apriori tìm được {len(patterns):,} tập phim phổ biến.")
        if len(patterns):
            for length, count in patterns.itemsets.map(len).value_counts().sort_index().items():
                update(None, f"Itemset {length} phim: {count:,} tập đạt support.")
        update(72, f"Sinh luật · min_confidence={request.min_confidence:g}; "
               "Support/Confidence/Lift chỉ đo trên Train.")
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
    update(85, f"Đã sinh {len(rules):,} luật; sắp xếp theo Lift, Confidence và Support.")
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
    data_summary = {
        "mode": "full_prepared_train",
        "train_ratings": len(frame),
        "train_users": len(users),
        "train_movies": int(frame.movieId.nunique()),
        "positive_ratings": len(positive),
        "positive_movies": len(counts),
        "candidate_movies": len(items),
        "retained_positive_ratings": len(selected),
        "empty_transactions": empty_transactions,
        "transaction_bytes": values.nbytes,
    }
    if len(records):
        update(92, f"Trung bình {len(records):,} luật Train · Support={row['support']:.6f} · "
               f"Confidence={row['confidence']:.6f} · Lift={row['lift']:.6f}.")
    else:
        update(92, "Không có luật đạt ngưỡng; kết quả rỗng được lưu cùng số liệu Train.")
    update(95, f"Hoàn tất khai phá trong {timing['fit_seconds']:.3f}s; "
           "Validation/Test được giữ riêng, không gộp vào Train.")
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
        "association_data": data_summary,
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
            f"Đọc toàn bộ {len(frame):,} ratings / {len(users):,} users của Train đã chuẩn bị; "
            "không lấy mẫu user. movies.parquet chỉ cung cấp tên phim; "
            "không gộp Validation/Test vào khai phá luật.",
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
