import io
import json

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from threadpoolctl import threadpool_limits

import movie_rating.experiment_clustering as clustering
from movie_rating.data.prepare import prepare
from movie_rating.data.temporal import user_temporal_split
from movie_rating.experiment_config import ExperimentRequest
from movie_rating.experiment_data import ExperimentData
from movie_rating.lab import LabService, native
from movie_rating.models.full_dbscan import FullDBSCAN, Neighborhoods


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=2):
        yield


def small_profiles():
    frame = pd.DataFrame(
        [(u, m, r, m) for u in range(1, 7) for m, r in [(1, 5.0), (2, 3.0), (3, 1.0)]],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.DataFrame({"movieId": [1, 2, 3, 4], "genres": ["A", "A", "B", "C"]})
    return frame, movies


@pytest.fixture
def lab_data():
    frame = pd.DataFrame(
        [(u, m, 0.5 + 0.5 * ((u + m) % 10), m) for u in range(1, 25) for m in range(1, 33)],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.DataFrame(
        {
            "movieId": range(1, 33),
            "genres": ["Action|Comedy" if m % 2 else "Drama|Romance" for m in range(1, 33)],
        }
    )
    return user_temporal_split(frame)[0], movies


def test_tastes_remove_user_rating_offset_shrink_sparse_genres_and_preserve_unseen_zero(
    monkeypatch,
):
    frame, movies = small_profiles()
    reference = clustering.taste_profiles(frame, movies, minimum=3, shrinkage=10)
    assert reference.preferences[0, 0] == pytest.approx(2 / 12)
    assert reference.preferences[0, 1] == pytest.approx(-2 / 11)
    assert np.all(reference.preferences[:, 2] == 0)
    changed = frame.copy()
    changed.loc[changed.userId == 1, "rating"] += 2
    monkeypatch.setattr(clustering, "PROFILE_BATCH", 2)
    shifted = clustering.taste_profiles(changed, movies, minimum=3, shrinkage=10)
    np.testing.assert_allclose(shifted.features, reference.features, atol=1e-7)
    assert shifted.features.dtype == np.float32


def test_coverage_precedes_silhouette_and_all_noise_is_not_qualified():
    selective = {"silhouette": 0.95, "dbi": 0.1, "clusters": 2, "coverage": 0.03}
    useful = {"silhouette": 0.25, "dbi": 1.4, "clusters": 3, "coverage": 1.0}
    one_cluster = {"silhouette": None, "dbi": None, "clusters": 1, "coverage": 1.0}
    assert clustering.candidate_key(useful, 0.8) > clustering.candidate_key(selective, 0.8)
    assert not clustering.candidate_key(selective, 0.8)[0]
    assert not clustering.candidate_key(one_cluster, 0.8)[0]
    assert not clustering.candidate_key(clustering.scores(np.zeros((10, 2)), np.full(10, -1)), 0.8)[
        0
    ]


def test_eps_uses_training_distances_and_respects_manual_override():
    x = np.random.default_rng(42).normal(size=(50, 4)).astype(np.float32)
    request = ExperimentRequest(group="clustering")
    eps = clustering.eps_candidates(x, 5, request)
    np.testing.assert_allclose(
        clustering.eps_candidates(3 * x, 5, request), 3 * np.array(eps), rtol=1e-5
    )
    request.cluster_auto_eps = False
    request.eps_values = [2.0, 1.0, 2.0]
    assert clustering.eps_candidates(x, 5, request) == [1.0, 2.0]


def test_dbscan_uses_all_training_users_and_extension_keeps_noise(monkeypatch):
    monkeypatch.setattr(clustering, "BOUNDED_DBSCAN_USERS", 20)
    x = np.random.default_rng(42).normal(size=(100, 4)).astype(np.float32)
    model = clustering.fit_cluster("dbscan", {"eps": 100.0, "min_samples": 3}, x)
    assert model.fit_users == 100
    assert np.all(model.predict(x) >= 0)
    assert model.predict(np.full((1, 4), 1000, dtype=np.float32)).tolist() == [-1]
    noise = clustering.fit_cluster("dbscan", {"eps": 0.0001, "min_samples": 3}, x)
    assert np.all(noise.predict(x) == -1)


def test_large_kmeans_uses_minibatches(monkeypatch):
    monkeypatch.setattr(clustering, "MINIBATCH_USERS", 20)
    x = np.random.default_rng(42).normal(size=(100, 4)).astype(np.float32)
    model = clustering.fit_cluster("kmeans", {"n_clusters": 3}, x)
    assert isinstance(model.model, MiniBatchKMeans)
    assert len(model.predict(x)) == 100


def test_large_run_uses_same_kmeans_variant_for_search_final_fit_and_stability(
    lab_data, monkeypatch
):
    monkeypatch.setattr(clustering, "MINIBATCH_USERS", 20)
    original = clustering.fit_cluster
    variants = []

    def tracked(*args, **kwargs):
        model = original(*args, **kwargs)
        variants.append(type(model.model))
        return model

    monkeypatch.setattr(clustering, "fit_cluster", tracked)
    splits, movies = lab_data
    clustering.clustering(
        splits,
        movies,
        ExperimentRequest(group="clustering", algorithms=["kmeans"], k_values=[2, 3]),
        lambda *_: None,
    )
    assert len(variants) == 5  # Two candidates, final fit, two stability fits.
    assert all(variant is MiniBatchKMeans for variant in variants)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("eps,min_samples", [(0.3, 4), (1.0, 5), (10.0, 3), (0.001, 1000)])
def test_bounded_memory_dbscan_matches_full_sklearn_labels(seed, eps, min_samples):
    rng = np.random.default_rng(seed)
    x = np.r_[rng.normal(0, 0.2, (50, 3)), rng.normal(3, 0.2, (50, 3)), [[20.0, 20.0, 20.0]]]
    expected = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
    actual = FullDBSCAN(eps, min_samples).fit(x)
    np.testing.assert_array_equal(actual.core_sample_indices_, expected.core_sample_indices_)
    np.testing.assert_array_equal(actual.labels_, expected.labels_)


def test_shared_border_is_assigned_to_first_core_component():
    x = np.array([[-0.4], [-0.35], [-0.3], [0.0], [1.0], [1.3], [1.35], [1.4], [0.5]])
    expected = DBSCAN(eps=0.51, min_samples=4).fit(x)
    actual = FullDBSCAN(0.51, 4).fit(x)
    assert expected.labels_[-1] == 0
    np.testing.assert_array_equal(actual.labels_, expected.labels_)


def test_full_dbscan_reports_neighbors_core_connectivity_borders_and_completion():
    x = np.r_[np.zeros((8, 2)), np.full((8, 2), 5.0), [[50.0, 50.0]]]
    messages = []
    model = FullDBSCAN(0.3, 3).fit(x, update=lambda _, message: messages.append(message))
    text = "\n".join(messages)
    for stage in ("láng giềng", "xác định core", "nối cụm core", "gán border", "hoàn tất gán nhãn"):
        assert stage in text
    assert model.labels_[-1] == -1


def test_cluster_after_old_5000_user_cutoff_is_actually_trained():
    x = np.r_[np.zeros((6000, 2)), np.full((5, 2), 100.0)].astype(np.float32)
    model = clustering.fit_cluster("dbscan", {"eps": 0.01, "min_samples": 3}, x)
    assert model.fit_users == len(x)
    assert len(model.model.labels_) == len(x)
    assert len(model.model.core_sample_indices_) == len(x)
    assert set(model.model.labels_) == {0, 1}
    assert np.all(model.model.labels_[-5:] == 1)


def test_kmeans_streaming_visits_every_training_row_on_each_pass(monkeypatch):
    monkeypatch.setattr(clustering, "MINIBATCH_USERS", 20)
    original = MiniBatchKMeans.partial_fit
    seen = []

    def tracked(self, x, *args, **kwargs):
        seen.extend(x[:, 0].tolist())
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(MiniBatchKMeans, "partial_fit", tracked)
    # The final minibatch has one row, smaller than n_clusters.
    x = np.arange(8194, dtype=np.float32).reshape(-1, 2)
    clustering.fit_cluster("kmeans", {"n_clusters": 3}, x)
    unique, counts = np.unique(seen, return_counts=True)
    np.testing.assert_array_equal(unique, x[:, 0])
    assert np.all(counts == 3)


def test_neighborhood_cache_queries_all_fit_rows_and_reuses_distances():
    x = np.random.default_rng(42).normal(size=(100, 3))
    cache = Neighborhoods(x)
    cache.prepare([3, 5])
    assert all(len(cache.distances[k]) == len(x) for k in (3, 5))
    before = cache.distances[3]
    cache.prepare([3])
    assert cache.distances[3] is before


def test_ranking_metric_hand_calculation_and_future_only_items_count_as_misses():
    recall, ndcg = clustering.ranking_scores([7, 8, 9], {7, 9, 99})
    assert recall == pytest.approx(2 / 3)
    assert ndcg == pytest.approx(1.5 / (1 + 1 / np.log2(3) + 0.5))
    assert clustering.ranking_scores([], set()) == (None, None)


def test_future_ranking_excludes_seen_items_shares_cohort_and_reports_noise_fallback(monkeypatch):
    train = pd.DataFrame(
        [
            (1, 1, 5.0),
            (1, 4, 1.0),
            (2, 3, 5.0),
            (2, 4, 1.0),
            (3, 2, 5.0),
            (3, 4, 1.0),
            (4, 2, 5.0),
            (4, 4, 1.0),
        ],
        columns=["userId", "movieId", "rating"],
    )
    future = pd.DataFrame(
        [(1, 3, 5.0), (1, 99, 5.0), (2, 1, 5.0), (3, 2, 5.0)], columns=train.columns
    )
    profiles = clustering.TasteProfiles(np.arange(1, 5), [], None, None, None, None, None, 0)
    result = clustering.evaluate_ranking(
        {"train": train, "validation": future, "test": future},
        profiles,
        {"dbscan": np.array([-1, 0, 1, 1])},
        ExperimentRequest(group="clustering", cluster_eval_users=10),
        lambda *_: None,
    )
    monkeypatch.setattr(clustering, "CLUSTER_ITEM_DENSE_BYTES", 0)
    sparse_result = clustering.evaluate_ranking(
        {"train": train, "validation": future, "test": future},
        profiles,
        {"dbscan": np.array([-1, 0, 1, 1])},
        ExperimentRequest(group="clustering", cluster_eval_users=10),
        lambda *_: None,
    )
    assert sparse_result == result
    for row in result["rows"]:
        # User 3's already-seen movie is excluded; item 99 is not silently dropped.
        assert row["evaluated_users"] == 2
        assert row["recall_at_10"] == 0.75
        if row["algorithm"] == "dbscan":
            assert row["clustered_users"] == 1
            assert row["clustered_recall_at_10"] == 1.0


@pytest.mark.parametrize("all_data", [False, True])
def test_clustering_loads_per_user_past_only_protocol(configured, all_data):
    prepare(configured)
    data = ExperimentData(configured.artifacts_dir / "data")
    splits, _, provenance = data.load(
        ExperimentRequest(group="clustering", max_users=10, use_all_data=all_data)
    )
    assert provenance["protocol"] == "per_user_temporal_70_15_15_v2"
    if all_data:
        assert provenance["split_files"]["train"].endswith("train.parquet")
        assert provenance["metadata_file"].endswith("movies.parquet")
    assert set(splits["train"].userId) == set(splits["test"].userId)
    for user in splits["train"].userId.unique():
        assert splits["train"].query("userId == @user").timestamp.max() < (
            splits["validation"].query("userId == @user").timestamp.min()
        )


def test_future_rating_changes_do_not_change_cluster_features_or_hyperparameters(lab_data):
    splits, movies = lab_data
    request = ExperimentRequest(group="clustering", algorithms=["kmeans"], k_values=[2, 3])
    first = clustering.clustering(splits, movies, request, lambda *_: None)
    assert first["training"]["ratings"] == len(splits["train"])
    assert first["rows"][0]["fit_users"] == splits["train"].userId.nunique()
    assert first["rows"][0]["fit_fraction"] == 1
    changed = {s: frame.copy() for s, frame in splits.items()}
    for name in ("validation", "test"):
        changed[name]["rating"] = 0.5
        changed[name]["movieId"] += 10000
    second = clustering.clustering(changed, movies, request, lambda *_: None)
    assert first["best_configs"] == second["best_configs"]
    for key in ("silhouette", "dbi", "coverage", "cluster_sizes"):
        assert first["rows"][0][key] == second["rows"][0][key]
    assert all(r["ndcg_at_10"] is None for r in second["recommendation"]["rows"])
    assert json.dumps(native(second), allow_nan=False)


def test_no_coverage_qualified_dbscan_is_reported_without_a_false_winner(lab_data):
    splits, movies = lab_data
    result = clustering.clustering(
        splits,
        movies,
        ExperimentRequest(
            group="clustering",
            algorithms=["dbscan"],
            cluster_auto_eps=False,
            eps_values=[0.1],
            min_samples_values=[19],
        ),
        lambda *_: None,
    )
    assert not result["rows"][0]["qualified"]
    assert "Chưa có mô hình" in result["selection"]


def test_clustering_exports_include_future_ranking_and_search(configured, monkeypatch):
    service = LabService(configured)
    result = {
        "clustering_protocol": "past_taste_coverage_future_ranking_v1",
        "rows": [
            {
                "algorithm": "kmeans",
                "name": "K-Means",
                "split": "train_history",
                "coverage": 1.0,
                "cluster_sizes": {0: 10},
            }
        ],
        "recommendation": {
            "rows": [
                {"algorithm": "popularity", "split": "test", "recall_at_10": 0.2, "ndcg_at_10": 0.3}
            ]
        },
        "search": [{"algorithm": "kmeans", "config": {"n_clusters": 2}, "qualified": True}],
    }
    monkeypatch.setattr(service, "get", lambda _: {"result": result})
    assert "future_recommendation" in service.csv_export("example")
    book = load_workbook(io.BytesIO(service.excel_export("example")))
    assert {"Metrics", "Future ranking", "Cluster search", "Protocol"} <= set(book.sheetnames)
    assert book["Future ranking"]["A2"].value == "popularity"
    service.executor.shutdown(wait=True)
