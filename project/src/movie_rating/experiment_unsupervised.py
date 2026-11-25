"""Genre user profiles, Apriori transactions and distinct PCA/TruncatedSVD."""

import math
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from movie_rating.experiment_reduction import FeatureReducer, ReductionSpec, reduction_variants
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


def _association_global(splits, movies, request, update):
    from mlxtend.frequent_patterns import apriori, association_rules

    pipeline_started = time.perf_counter()
    frame = splits["train"]
    update(12, f"Apriori · quét toàn bộ {len(frame):,} ratings Train; lọc phim rating ≥ 4,0.")
    with measured() as timing:
        feature_started = time.perf_counter()
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
        feature_creation_seconds = time.perf_counter() - feature_started
        update(32, f"Transactions: {len(users):,} users × {len(items):,} phim; "
               f"giữ {empty_transactions:,} transaction rỗng trong mẫu số support; "
               f"ma trận boolean {values.nbytes / 2**20:.2f} MiB.")
        update(40, f"Bắt đầu Apriori · min_support={request.min_support:g} · "
               f"độ dài itemset tối đa={request.max_rule_length}.")
        model_started = time.perf_counter()
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
        model_training_seconds = time.perf_counter() - model_started
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
        "feature_creation_seconds": feature_creation_seconds,
        "model_training_seconds": model_training_seconds,
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
        "pipeline_timings": {
            "feature_creation_seconds": feature_creation_seconds,
            "reducer_fit_seconds": 0.0,
            "transform_seconds": 0.0,
            "model_training_seconds": model_training_seconds,
            "evaluation_seconds": 0.0,
            "total_pipeline_seconds": time.perf_counter() - pipeline_started,
        },
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


def _mine_apriori_rules(transactions, request):
    from mlxtend.frequent_patterns import apriori, association_rules

    patterns = (
        apriori(
            transactions,
            min_support=request.min_support,
            use_colnames=True,
            max_len=request.max_rule_length,
            low_memory=True,
        )
        if transactions.shape[1]
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
    return patterns, rules


def _rule_list(rules):
    if rules is None or not len(rules):
        return []
    return [
        {
            "antecedent_ids": set(row.antecedents),
            "consequent_ids": set(row.consequents),
            "support": float(row.support),
            "confidence": float(row.confidence),
            "lift": float(row.lift),
        }
        for row in rules.itertuples()
    ]


def _recommend_from_rules(known, rules, popular_order, limit=10):
    scores = {}
    for rule in rules:
        if rule["antecedent_ids"] <= known:
            priority = (rule["lift"], rule["confidence"], rule["support"])
            for item in rule["consequent_ids"] - known:
                if item not in scores or priority > scores[item]:
                    scores[item] = priority
    if scores:
        return sorted(scores, key=lambda item: (*scores[item], -int(item)), reverse=True)[:limit]
    return [item for item in popular_order if item not in known][:limit]


def _rule_ranking(recommendations, relevant, k=10):
    if not relevant:
        return None, None
    hits = len(set(recommendations[:k]) & relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    dcg = sum(
        1 / np.log2(index + 2)
        for index, item in enumerate(recommendations[:k])
        if item in relevant
    )
    return hits / len(relevant), dcg / ideal if ideal else None


def _association_grouped(splits, movies, request, update, global_result, descriptor):
    from movie_rating.experiment_clustering import TasteScale, taste_profiles

    pipeline_started = time.perf_counter()
    feature_started = time.perf_counter()
    train = splits["train"]
    frame = train.loc[train.rating >= 4.0, ["userId", "movieId"]]
    counts = frame.groupby("movieId").size().sort_values(ascending=False, kind="stable")
    items = counts.head(request.max_items).index.to_numpy()
    users = np.sort(train.userId.unique())
    selected = frame.loc[frame.movieId.isin(items)]
    values = np.zeros((len(users), len(items)), dtype=bool)
    if len(selected) and len(items):
        values[
            np.searchsorted(users, selected.userId.to_numpy()),
            pd.Index(items).get_indexer(selected.movieId),
        ] = True
    transactions = pd.DataFrame(values, index=users, columns=items)
    transaction_feature_seconds = time.perf_counter() - feature_started
    global_rules = [
        {
            "antecedent_ids": set(rule["antecedent_ids"]),
            "consequent_ids": set(rule["consequent_ids"]),
            "support": float(rule["support"]),
            "confidence": float(rule["confidence"]),
            "lift": float(rule["lift"]),
        }
        for rule in global_result.get("rules", [])
    ]
    item_names = movies.set_index("movieId").title.to_dict()
    popular_order = [int(x) for x in items[np.argsort(-values.sum(axis=0), kind="stable")]]

    profile_started = time.perf_counter()
    profiles = taste_profiles(
        train, movies, request.min_user_ratings, request.cluster_shrinkage,
        lambda _, message: update(None, f"Apriori phân nhóm · {message}"),
    )
    scaler = TasteScale.fit(profiles.features)
    x = scaler.transform(profiles.features)
    profile_feature_seconds = time.perf_counter() - profile_started
    reducer = None
    reduction_metadata = None
    if descriptor["method"]:
        reducer = FeatureReducer(
            descriptor["method"], descriptor["components"], seed=request.seed, scale=True
        ).fit(
            x,
            feature_schema={"family": "relative_genre_preference_and_frequency",
                            "genres": profiles.genres},
            fit_scope="train_all_eligible_users",
        )
        x = reducer.transform(x)
        reduction_metadata = reducer.metadata()
        update(None, f"Apriori nhóm · reducer {reducer.method} · "
               f"{reducer.input_dimensions}→{reducer.output_dimensions} · "
               f"fit {reducer.fit_rows:,} Train users.")
    global_code = pd.Series(-1, index=users, dtype=np.int32)
    global_code.loc[profiles.users] = np.arange(len(profiles.users), dtype=np.int32)
    group_minimum = max(5, math.ceil(1 / request.min_support))
    grouped = []
    k_values = sorted({k for k in request.k_values if 2 <= k <= len(profiles.users)})
    if not k_values:
        raise ValueError("Apriori phân nhóm cần ít nhất hai user Train đủ điều kiện và k hợp lệ.")

    for k in k_values:
        fit_started = time.perf_counter()
        update(None, f"Apriori nhóm {descriptor['variant_id']} · K-Means k={k}.")
        model = KMeans(n_clusters=k, n_init=10, random_state=request.seed).fit(x)
        labels = model.labels_.astype(np.int32)
        cluster_of_user = global_code.copy()
        cluster_of_user.loc[profiles.users] = labels
        rule_by_group, group_summary = {}, []
        combined_group_rules = []
        for cluster in range(k):
            member_users = profiles.users[labels == cluster]
            group_tx = transactions.loc[transactions.index.isin(member_users)]
            if len(group_tx) < group_minimum:
                group_summary.append({"group": cluster, "users": len(group_tx),
                                      "status": "fallback_global_small_group", "rule_count": 0})
                continue
            patterns, rules_frame = _mine_apriori_rules(group_tx, request)
            group_rules = _rule_list(rules_frame)
            for rule in group_rules:
                union = list(rule["antecedent_ids"] | rule["consequent_ids"])
                antecedent = list(rule["antecedent_ids"])
                support_global = float(transactions[union].all(axis=1).mean())
                antecedent_global = float(transactions[antecedent].all(axis=1).mean())
                consequent_global = float(
                    transactions[list(rule["consequent_ids"])].all(axis=1).mean()
                )
                rule["group"] = cluster
                rule["support_in_group"] = rule["support"]
                rule["support_global"] = support_global
                rule["confidence_global"] = (
                    support_global / antecedent_global if antecedent_global else 0.0
                )
                rule["lift_global"] = (
                    rule["confidence_global"] / consequent_global if consequent_global else 0.0
                )
                rule["antecedents"] = [item_names.get(item, str(item)) for item in sorted(rule["antecedent_ids"])]
                rule["consequents"] = [item_names.get(item, str(item)) for item in sorted(rule["consequent_ids"])]
                combined_group_rules.append(rule)
            rule_by_group[cluster] = group_rules
            group_summary.append({"group": cluster, "users": len(group_tx),
                                  "status": "mined" if len(group_rules) else "fallback_global_no_rules",
                                  "itemsets": len(patterns), "rule_count": len(group_rules)})

        variant_fit_seconds = time.perf_counter() - fit_started

        variant_id = f"apriori_grouped_{descriptor['variant_id']}_k{k}"
        model_rules = {
            "global": global_rules,
            "groups": rule_by_group,
        }
        evaluation_started = time.perf_counter()
        global_recommendations = []
        recommendation_rows = []
        for split in ("validation", "test"):
            held = splits[split]
            held_users = np.sort(held.userId.unique())
            cohort = np.sort(
                np.random.default_rng(42).choice(
                    held_users, min(request.cluster_eval_users, len(held_users)), replace=False
                )
            )
            cohort_set = set(map(int, cohort))
            history_frame = train.loc[train.userId.isin(cohort_set), ["userId", "movieId"]]
            history = {
                int(user): set(map(int, group.movieId))
                for user, group in history_frame.groupby("userId", sort=False)
            }
            future = held.loc[
                (held.rating >= 4.0) & held.userId.isin(cohort_set), ["userId", "movieId"]
            ]
            relevant = {
                int(user): set(map(int, group.movieId)) - history.get(int(user), set())
                for user, group in future.groupby("userId", sort=False)
            }
            relevant = {user: positive for user, positive in relevant.items() if positive}
            all_positive = sum(len(positive) for positive in relevant.values())
            pool_positive = sum(len(positive & set(items)) for positive in relevant.values())
            recall_scores, ndcg_scores = [], []
            global_recall_scores, global_ndcg_scores = [], []
            group_assigned, fallback_count, rec_coverage = 0, 0, 0
            for user in cohort:
                user = int(user)
                if user not in relevant:
                    continue
                known = history.get(user, set()) & set(items)
                baseline_chosen = _recommend_from_rules(known, global_rules, popular_order)
                base_recall, base_ndcg = _rule_ranking(baseline_chosen, relevant[user])
                global_recall_scores.append(base_recall)
                global_ndcg_scores.append(base_ndcg)
                code = int(cluster_of_user.get(user, -1))
                cluster_rules = rule_by_group.get(code, []) if code >= 0 else []
                if cluster_rules:
                    chosen = _recommend_from_rules(known, cluster_rules, popular_order)
                    group_assigned += 1
                else:
                    fallback_count += 1
                    chosen = _recommend_from_rules(known, global_rules, popular_order)
                rec_coverage += bool(chosen)
                recall, ndcg = _rule_ranking(chosen, relevant[user])
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
            recommendation_rows.append({
                "algorithm": "apriori_grouped", "split": split,
                "recall_at_10": float(np.mean(recall_scores)) if recall_scores else None,
                "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else None,
                "evaluated_users": len(recall_scores), "sampled_users": len(cohort),
                "group_assigned_users": group_assigned,
                "group_coverage": group_assigned / len(recall_scores) if recall_scores else 0.0,
                "recommendation_coverage": rec_coverage / len(recall_scores) if recall_scores else 0.0,
                "fallback_users": fallback_count,
                "candidate_pool_size": len(items),
                "eligible_positives": all_positive,
                "candidate_eligible_positives": pool_positive,
                "candidate_positive_coverage": pool_positive / all_positive if all_positive else None,
                "variant_id": variant_id,
                "base_algorithm": "apriori",
                "representation": "grouped_reduced_profiles" if descriptor["method"] else "grouped_raw_profiles",
                "reduction_method": descriptor["method"], "components": descriptor["components"],
                "feature_family": "relative_genre_preference_and_frequency",
                "fit_scope": "train",
            })
            global_recommendations.append({
                "algorithm": "apriori_global", "split": split,
                "recall_at_10": float(np.mean(global_recall_scores)) if global_recall_scores else None,
                "ndcg_at_10": float(np.mean(global_ndcg_scores)) if global_ndcg_scores else None,
                "evaluated_users": len(global_recall_scores), "sampled_users": len(cohort),
                "candidate_pool_size": len(items), "eligible_positives": all_positive,
                "candidate_eligible_positives": pool_positive,
                "candidate_positive_coverage": pool_positive / all_positive if all_positive else None,
                "variant_id": "apriori_global", "base_algorithm": "apriori",
                "representation": "global_transactions", "reduction_method": None,
                "components": None, "fit_scope": "train",
            })
            update(None, f"{variant_id} · {split.title()} · NDCG@10="
                   f"{'N/A' if not ndcg_scores else f'{np.mean(ndcg_scores):.4f}'} · "
                   f"{len(recall_scores):,}/{len(cohort):,} user có positive · "
                   f"fallback {fallback_count:,}.")
        validation = recommendation_rows[0]
        support_values = [r["support_in_group"] for r in combined_group_rules]
        support_global_values = [r["support_global"] for r in combined_group_rules]
        evaluation_seconds = time.perf_counter() - evaluation_started
        fit_metadata = {
            "method": descriptor["method"], "components": descriptor["components"],
            "input_dimensions": reduction_metadata["input_dimensions"] if reduction_metadata else x.shape[1],
            "output_dimensions": reduction_metadata["output_dimensions"] if reduction_metadata else x.shape[1],
            "fit_scope": "train_all_eligible_users", "fit_rows": len(profiles.users),
            "feature_family": "relative_genre_preference_and_frequency",
            "fingerprint": reduction_metadata["fingerprint"] if reduction_metadata else None,
            "feature_creation_seconds": transaction_feature_seconds + profile_feature_seconds,
            "reducer_fit_seconds": reduction_metadata["fit_seconds"] if reduction_metadata else 0.0,
            "transform_seconds": reduction_metadata["transform_seconds"] if reduction_metadata else 0.0,
            "model_training_seconds": variant_fit_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_pipeline_seconds": time.perf_counter() - pipeline_started,
        }
        grouped.append({
            "variant_id": variant_id, "base_algorithm": "apriori", "k": k,
            "descriptor": descriptor, "recommendation": recommendation_rows,
            "global_recommendation": global_recommendations,
            "groups": group_summary,
            "rules": [
                {**rule,
                 "antecedent_ids": sorted(rule["antecedent_ids"]),
                 "consequent_ids": sorted(rule["consequent_ids"])}
                for rule in combined_group_rules
            ],
            "fit_metadata": fit_metadata,
            "validation_ndcg": validation["ndcg_at_10"],
            "validation_coverage": validation["group_coverage"],
            "config": {"n_clusters": k, "min_support": request.min_support,
                       "min_confidence": request.min_confidence,
                       "minimum_group_users": group_minimum,
                       "reducer": fit_metadata},
            "rule_summary": {
                "rules": len(combined_group_rules),
                "support_in_group_mean": float(np.mean(support_values)) if support_values else None,
                "support_global_mean": float(np.mean(support_global_values)) if support_global_values else None,
                "fallback_groups": sum(g["status"].startswith("fallback") for g in group_summary),
                "empty_transactions": int((~values.any(axis=1)).sum()),
            },
            "model_bytes": len(pickle.dumps((scaler, reducer, model, model_rules))),
            "fit_seconds": variant_fit_seconds,
            "reduction_fit_seconds": reduction_metadata["fit_seconds"] if reduction_metadata else 0.0,
        })
        del model, model_rules

    return grouped, global_rules, transactions, len(items), profiles


def association(splits, movies, request, update):
    pipeline_started = time.perf_counter()
    if request.reduction_mode == "off":
        result = _association_global(splits, movies, request, update)
        result["pipeline_timings"]["total_pipeline_seconds"] = time.perf_counter() - pipeline_started
        return result

    def global_progress(progress, message):
        mapped = None if progress is None else 3 + 33 * progress / 100
        update(mapped, message)

    result = _association_global(splits, movies, request, global_progress)
    result["rows"][0].update(
        variant_id="apriori_global", base_algorithm="apriori",
        representation="global_transactions", reduction_method=None, components=None,
        fit_scope="train",
    )
    result["best_configs"] = {"apriori_global": result["best_configs"]["apriori"]}
    spec = ReductionSpec(
        mode=request.reduction_mode,
        methods=tuple(request.reduction_methods),
        components=tuple(request.components),
        seed=request.seed,
        scale=True,
    )
    variants = reduction_variants(spec)
    grouped_variants = []
    all_group_rules = []
    ranking_rows = []
    model_rows = []
    recommendation = {"k": 10, "rows": [], "candidate_movies": result["association_data"]["candidate_movies"],
                      "protocol": "grouped_train_rules_frozen_validation_test_v1"}
    seen_baseline = set()
    for index, descriptor in enumerate(variants):
        update(38 + 50 * index / len(variants),
               f"Apriori theo nhóm · biểu diễn {descriptor['variant_id']}.")
        grouped, global_rules, transactions, candidate_count, profiles = _association_grouped(
            splits, movies, request,
            lambda p, message: update(38 + 57 * (index + (0 if p is None else p / 100)) / len(variants), message),
            result, descriptor,
        )
        for variant in grouped:
            variant_id = variant["variant_id"]
            grouped_variants.append(variant)
            all_group_rules.extend({**rule, "variant_id": variant_id} for rule in variant["rules"])
            output_row = {
                "algorithm": "apriori_grouped", "name": f"Apriori theo nhóm · {variant['descriptor']['variant_id']} · k={variant['k']}",
                "split": "train", "variant_id": variant_id, "base_algorithm": "apriori",
                "representation": "grouped_reduced_profiles" if variant["descriptor"]["method"] else "grouped_raw_profiles",
                "reduction_method": variant["descriptor"]["method"],
                "components": variant["descriptor"]["components"],
                "feature_family": "relative_genre_preference_and_frequency",
                "fit_scope": "train_all_eligible_users",
                "support": None, "confidence": None, "lift": None,
                "rule_count": variant["rule_summary"]["rules"],
                "support_in_group_mean": variant["rule_summary"]["support_in_group_mean"],
                "support_global_mean": variant["rule_summary"]["support_global_mean"],
                "cluster_count": variant["k"],
                "group_fallbacks": variant["rule_summary"]["fallback_groups"],
                "transactions": result["association_data"]["train_users"],
                "model_bytes": variant["model_bytes"],
                "fit_seconds": variant["fit_seconds"],
                "model_training_seconds": variant["fit_metadata"]["model_training_seconds"],
                "reduction_fit_seconds": variant["reduction_fit_seconds"],
                "reduction": variant["fit_metadata"],
            }
            model_rows.append(output_row)
            result["best_configs"][variant_id] = {
                **variant["config"], "variant_id": variant_id,
                "base_algorithm": "apriori", "representation": output_row["representation"],
                "reduction_method": output_row["reduction_method"],
                "components": output_row["components"],
                "fit_scope": "train_all_eligible_users",
            }
            ranking_rows.extend(variant["recommendation"])
            for rec in variant["recommendation"]:
                result["charts"].append(chart(
                    -1, f"{variant_id} · NDCG@10", "bar", "Split", "NDCG@10 ↑",
                    [series("Apriori theo nhóm", [rec["split"]], [rec["ndcg_at_10"]])],
                    rec["split"], metric_key="ndcg_at_10",
                ))
        # Recommendation from the global rule set is identical in every representation;
        # preserve one shared baseline record per holdout split.
        if not seen_baseline:
            for rec in grouped[0]["global_recommendation"]:
                baseline_rec = dict(rec)
                recommendation["rows"].append(baseline_rec)
                seen_baseline.add(rec["split"])
        for variant in grouped:
            recommendation["rows"].extend(variant["recommendation"])

    result["rows"].extend(model_rows)
    result["charts"].append(chart(
        -1, "Apriori nhóm · NDCG@10", "bar", "Biến thể / split", "NDCG@10 ↑",
        [series("Validation", [r["variant_id"] for r in ranking_rows if r["split"] == "validation"],
                [r["ndcg_at_10"] for r in ranking_rows if r["split"] == "validation"]),
         series("Test", [r["variant_id"] for r in ranking_rows if r["split"] == "test"],
                [r["ndcg_at_10"] for r in ranking_rows if r["split"] == "test"])],
        "comparison", metric_key="ndcg_at_10",
    ))
    recommendation["candidate_movies"] = result["association_data"]["candidate_movies"]
    recommendation["rows"] = sorted(
        recommendation["rows"], key=lambda row: (row["variant_id"], row["split"])
    )
    result["recommendation"] = recommendation
    result["group_rules"] = all_group_rules
    result["grouped_variants"] = grouped_variants
    result["association_data"]["grouped"] = {
        "eligible_profile_users": int(len(profiles.users)),
        "excluded_profile_users": int(profiles.excluded_users),
        "candidate_movies": int(recommendation["candidate_movies"]),
        "minimum_group_users": max(5, math.ceil(1 / request.min_support)),
        "support_denominator": "users in each group, including empty transactions",
        "fallback": "global rules then Train popularity when no group rule matches",
    }
    validation_rows = [row for row in recommendation["rows"] if row["split"] == "validation"]
    eligible = [row for row in validation_rows if row["ndcg_at_10"] is not None]
    winner = max(eligible, key=lambda row: (row["ndcg_at_10"], row.get("group_coverage", 0))) if eligible else None
    result["selection"] = (
        f"Chọn bằng NDCG@10 Validation: {winner['variant_id']} "
        f"({winner['ndcg_at_10']:.4f}); Test chỉ báo cáo cuối. "
        "Apriori toàn cục được giữ làm baseline trên cùng candidate catalog và cohort."
        if winner else
        "Không có positive Validation hợp lệ để xếp hạng Apriori; giữ luật toàn cục và N/A cho ranking."
    )
    result["reduction"] = {
        "mode": request.reduction_mode,
        "variants": [{"variant_id": d["variant_id"], "method": d["method"], "components": d["components"]}
                     for d in variants],
        "selected_variant_id": winner["variant_id"] if winner else None,
        "selection_metric": "ndcg_at_10",
        "feature_family": "relative_genre_preference_and_frequency",
        "transaction_reduction": False,
    }
    result["selected_variant_id"] = winner["variant_id"] if winner else None
    grouped_timings = [variant["fit_metadata"] for variant in grouped_variants]
    global_timings = result.get("pipeline_timings", {})
    result["pipeline_timings"] = {
        "feature_creation_seconds": global_timings.get("feature_creation_seconds", 0.0)
        + sum(item["feature_creation_seconds"] for item in grouped_timings),
        "reducer_fit_seconds": sum(item["reducer_fit_seconds"] for item in grouped_timings),
        "transform_seconds": sum(item["transform_seconds"] for item in grouped_timings),
        "model_training_seconds": global_timings.get("model_training_seconds", 0.0)
        + sum(item["model_training_seconds"] for item in grouped_timings),
        "evaluation_seconds": sum(item["evaluation_seconds"] for item in grouped_timings),
        "total_pipeline_seconds": time.perf_counter() - pipeline_started,
    }
    result["notes"].extend([
        "Apriori toàn cục vẫn fit trên toàn bộ prepared Train; biến thể phân nhóm fit reducer, K-Means và luật chỉ từ Train.",
        "PCA/TruncatedSVD chỉ nén hồ sơ sở thích/tần suất user; transaction phim vẫn boolean nguyên bản.",
        "Support phân nhóm có mẫu số là user trong nhóm, kể cả transaction rỗng. Support toàn cục được lưu riêng; không so trực tiếp hai mẫu số.",
        "User thiếu hồ sơ hoặc nhóm thiếu user/rules dùng luật toàn cục; khi không khớp luật thì fallback Popularity.",
        "Validation/Test ranking giới hạn trong candidate catalog top phim thích Train; báo candidate_pool_size, positive coverage, user coverage và fallback.",
    ])
    update(95, "Apriori toàn cục và các biến thể phân nhóm đã hoàn tất; Test không dùng để chọn.")
    return result


def reduction(splits, movies, request, update):
    raw, _, genres = genre_profiles(splits["train"], movies, request.min_user_ratings)
    scaler = StandardScaler().fit(raw)
    x = scaler.transform(raw)
    update(10, f"Fit scaler/PCA trên {len(x):,} profile Train; Validation/Test chỉ được transform.")
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
            reconstruction_errors = {}
            for split, original in held.items():
                start = time.perf_counter()
                transformed = model.transform(original)
                seconds = time.perf_counter() - start
                reconstruction_error = float(
                    np.mean((original - model.inverse_transform(transformed)) ** 2)
                )
                reconstruction_errors[split] = reconstruction_error
                rows.append(
                    {
                        "algorithm": f"{aid}_{k}",
                        "name": f"{'PCA' if aid == 'pca' else 'TruncatedSVD'} · {k}",
                        "split": split,
                        "components": k,
                        "explained_variance_ratio": variance,
                        "reconstruction_error": reconstruction_error,
                        **timing,
                        "inference_seconds": seconds,
                        "model_bytes": len(pickle.dumps((scaler, model))),
                    }
                )
            update(
                15 + 65 * (len(curve)) / len(choices),
                f"{aid} · Explained variance={variance:.4f} · "
                f"reconstruction Validation={reconstruction_errors['validation']:.6f} · "
                f"Test={reconstruction_errors['test']:.6f} (diagnostic only)",
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
