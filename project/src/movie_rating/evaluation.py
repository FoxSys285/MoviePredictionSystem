import time

import numpy as np


def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metrics require nonempty arrays of the same shape.")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Metrics require finite ratings and predictions.")
    error = actual - predicted
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))}


def evaluate_model(model, ratings) -> dict:
    started = time.perf_counter()
    predictions = model.predict(ratings.userId, ratings.movieId)
    seconds = time.perf_counter() - started
    known = np.isin(ratings.userId, model.user_ids) & np.isin(ratings.movieId, model.movie_ids)
    return {
        **regression_metrics(ratings.rating, predictions),
        "prediction_seconds": seconds,
        "ratings": len(ratings),
        "known_pair_fraction": float(known.mean()),
        "warm": regression_metrics(ratings.rating[known], predictions[known])
        if known.any()
        else None,
        "cold": regression_metrics(ratings.rating[~known], predictions[~known])
        if (~known).any()
        else None,
    }


def evaluate_ranking(model, heldout, history, k=10, max_users=100, seed=42, details=False):
    """Full train-catalogue ranking on a fixed sample of eligible warm users.

    Unobserved items are candidates, not asserted dislikes. These are offline metrics
    against observed held-out positives (rating >= 4), averaged equally per user.
    """
    relevant = (
        heldout.loc[
            (heldout.rating >= 4)
            & heldout.userId.isin(model.user_ids)
            & heldout.movieId.isin(model.movie_ids)
        ]
        .groupby("userId")
        .movieId.agg(set)
        .to_dict()
    )
    seen = history.groupby("userId").movieId.agg(set).to_dict()
    relevant = {user: items - seen.get(user, set()) for user, items in relevant.items()}
    eligible = sorted(user for user, items in relevant.items() if items)
    rng = np.random.default_rng(seed)
    users = np.sort(rng.choice(eligible, min(max_users, len(eligible)), replace=False))
    precision, recall, ndcg, latencies = [], [], [], []
    recommended = set()
    candidate_counts = []
    for user in users:
        candidates = model.movie_ids[~np.isin(model.movie_ids, list(seen.get(user, set())))]
        started = time.perf_counter()
        scores = model.predict(np.full(len(candidates), user), candidates)
        order = np.lexsort((candidates, -scores))[:k]
        top = candidates[order]
        latencies.append((time.perf_counter() - started) * 1000)
        hits = np.array([movie in relevant[user] for movie in top], dtype=float)
        precision.append(hits.sum() / k)
        recall.append(hits.sum() / len(relevant[user]))
        discount = 1 / np.log2(np.arange(len(top)) + 2)
        ideal = (1 / np.log2(np.arange(min(k, len(relevant[user]))) + 2)).sum()
        ndcg.append(float((hits * discount).sum() / ideal))
        recommended.update(map(int, top))
        candidate_counts.append(len(candidates))
    report = {
        f"precision_at_{k}": float(np.mean(precision)) if precision else None,
        f"recall_at_{k}": float(np.mean(recall)) if recall else None,
        f"ndcg_at_{k}": float(np.mean(ndcg)) if ndcg else None,
        "ranking_users": len(users),
        "ranking_eligible_users": len(eligible),
        "ranking_candidate_policy": "all_trained_items_excluding_history",
        "ranking_mean_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0,
        "catalog_coverage": len(recommended) / len(model.movie_ids),
        "topk_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "topk_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
    }
    if details:
        report["per_user"] = [
            {
                "userId": int(user),
                f"precision_at_{k}": float(precision[index]),
                f"recall_at_{k}": float(recall[index]),
                f"ndcg_at_{k}": float(ndcg[index]),
            }
            for index, user in enumerate(users)
        ]
    return report
