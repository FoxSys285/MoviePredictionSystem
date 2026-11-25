import pytest


def test_standard_presets_are_backend_owned_and_algorithm_specific():
    from movie_rating.experiment_config import (
        ALGORITHMS,
        PRESET_IDS,
        PRESET_REDUCTIONS,
        ExperimentRequest,
        preset_catalog,
    )

    catalog = preset_catalog()
    assert set(catalog) == set(PRESET_IDS)
    assert set(catalog) == {"rating", "classification", "clustering", "association", "reinforcement"}
    for group, preset_id in PRESET_IDS.items():
        request = ExperimentRequest(group=group, preset_id=preset_id)
        preset = catalog[group]
        assert preset["id"] == request.preset_id
        assert preset["tuning"] is False
        assert set(preset["algorithms"]) == set(ALGORITHMS[group])
        for algorithm in ALGORITHMS[group]:
            assert preset["algorithms"][algorithm]["reduction"] == PRESET_REDUCTIONS[group][algorithm]

    assert PRESET_REDUCTIONS["rating"] == {
        "linear": {"method": "pca", "components": 8},
        "xgboost": None,
        "svd": None,
        "ncf": None,
    }
    assert PRESET_REDUCTIONS["classification"] == {
        "logistic": {"method": "pca", "components": 8},
        "random_forest": None,
    }
    assert PRESET_REDUCTIONS["reinforcement"] == {"dqn": None, "random": None, "popular": None}


def test_standard_preset_rejects_payload_configuration_that_disagrees():
    from pydantic import ValidationError

    from movie_rating.experiment_config import ExperimentRequest

    with pytest.raises(ValidationError, match="cấu hình cố định"):
        ExperimentRequest(
            group="rating", preset_id="rating.standard.v1", components=[2, 4, 8]
        )


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        (
            "rating",
            {
                ("linear",): {"method": "pca", "components": 8},
                ("xgboost", "svd", "ncf"): None,
            },
        ),
        (
            "classification",
            {
                ("logistic",): {"method": "pca", "components": 8},
                ("random_forest",): None,
            },
        ),
    ],
)
@pytest.mark.parametrize("choice", [None, "pca", "truncated_svd", "off"])
def test_supervised_preset_dispatches_only_the_declared_reduction(group, expected, choice, monkeypatch):
    from movie_rating import experiment_supervised
    from movie_rating.experiment_config import ExperimentRequest

    calls = []

    def fake_run(_splits, _movies, request, _update, reduction_variant=None, **_kwargs):
        algorithms = tuple(request.algorithms)
        calls.append((algorithms, reduction_variant))
        metric = "f1" if group == "classification" else "rmse"
        rows = []
        for algorithm in algorithms:
            for split, value in (("validation", 0.6 if metric == "f1" else 1.0), ("test", 0.5 if metric == "f1" else 1.1)):
                rows.append({"algorithm": algorithm, "name": algorithm, "split": split, metric: value})
        return {"rows": rows, "cv": {}, "best_configs": {}, "charts": [], "tuning": [], "notes": []}

    monkeypatch.setattr(experiment_supervised, "_supervised_single", fake_run)
    request = ExperimentRequest(group=group, preset_id=f"{group}.standard.v1", reduction_choice=choice)
    result = experiment_supervised.supervised({}, None, request, lambda *_: None)

    if choice == "off":
        expected = {tuple(request.algorithms): None}
    elif choice:
        expected = {algorithms: ({**plan, "method": choice} if plan else None)
                    for algorithms, plan in expected.items()}
    assert dict(calls) == expected
    assert result["selected_variant_id"]
    assert {row["variant_id"] for row in result["rows"] if row["split"] == "test"} == {
        f"{algorithm}_{(choice or 'pca') + '_8' if choice != 'off' and algorithm in ('linear', 'logistic') else 'raw'}"
        for algorithm in request.algorithms
    }


@pytest.mark.parametrize("choice", [None, "pca", "truncated_svd"])
def test_clustering_preset_dispatches_raw_comparator_and_single_pca(choice, monkeypatch):
    from movie_rating import experiment_clustering
    from movie_rating.experiment_config import ExperimentRequest

    calls = []

    def fake_single(_splits, _movies, request, _update, reduction_variant=None):
        calls.append((tuple(request.algorithms), reduction_variant))
        return {
            "rows": [{"algorithm": "kmeans", "name": "K-Means", "split": "train_history",
                      "fit_seconds": 0.1, "silhouette": 0.2, "clusters": 8}],
            "search": [], "clusters": [{"algorithm": "kmeans", "qualified": True}],
            "charts": [], "best_configs": {"kmeans": {"n_clusters": 8}}, "notes": [],
            "recommendation": {"rows": [
                {"algorithm": "kmeans", "split": "validation", "ndcg_at_10": 0.6},
                {"algorithm": "kmeans", "split": "test", "ndcg_at_10": 0.5},
                {"algorithm": "popularity", "split": "validation", "ndcg_at_10": 0.4},
                {"algorithm": "popularity", "split": "test", "ndcg_at_10": 0.4},
            ], "candidate_movies": 20},
            "pipeline_timings": {key: 0.1 for key in (
                "feature_creation_seconds", "reducer_fit_seconds", "transform_seconds",
                "model_training_seconds", "evaluation_seconds", "total_pipeline_seconds",
            )},
            "clustering_protocol": "test", "cluster_min_coverage": 0.8,
            "profile_kind": "test", "profile_excluded_users": 0,
            "training": {"split": "train"},
        }

    monkeypatch.setattr(experiment_clustering, "_clustering_single", fake_single)
    request = ExperimentRequest(
        group="clustering", preset_id="clustering.standard.v1", algorithms=["kmeans"],
        reduction_choice=choice,
    )
    result = experiment_clustering.clustering({}, None, request, lambda *_: None)

    assert calls == [(("kmeans",), {"method": choice or "pca", "components": 8})]
    assert result["selected_variant_id"] == f"kmeans_{choice or 'pca'}_8"
    assert result["pipeline_timings"]["total_pipeline_seconds"] >= 0


def test_association_and_reinforcement_presets_dispatch_declared_raw_and_pca_paths(monkeypatch):
    from movie_rating import experiment_reinforcement, experiment_unsupervised
    from movie_rating.experiment_config import ExperimentRequest

    association_descriptors = []

    def fake_global(_splits, _movies, _request, _update):
        return {
            "rows": [{"algorithm": "apriori", "name": "Apriori toàn cục", "split": "train",
                      "fit_seconds": 0.2, "rule_count": 3}],
            "charts": [], "rules": [], "association_data": {
                "candidate_movies": 20, "train_users": 30, "train_ratings": 100,
            },
            "best_configs": {"apriori": {}}, "notes": [],
        }

    def fake_grouped(_splits, _movies, _request, _update, _global, descriptor):
        association_descriptors.append(descriptor)
        variant_id = f"apriori_grouped_{descriptor['variant_id']}_k8"
        recommendation = [
            {"algorithm": "apriori_grouped", "variant_id": variant_id,
             "split": split, "ndcg_at_10": 0.7 if descriptor["method"] is None else 0.6,
             "group_coverage": 0.5, "recommendation_coverage": 0.8, "evaluated_users": 20,
             "sampled_users": 30, "candidate_movies": 20}
            for split in ("validation", "test")
        ]
        global_recommendation = [
            {"algorithm": "apriori_global", "variant_id": "apriori_global",
             "split": split, "ndcg_at_10": 0.4, "evaluated_users": 20,
             "sampled_users": 30, "candidate_movies": 20}
            for split in ("validation", "test")
        ]
        return ([{
            "variant_id": variant_id, "descriptor": descriptor, "k": 8,
            "recommendation": recommendation, "global_recommendation": global_recommendation,
            "groups": [], "rules": [], "fit_metadata": {
                "method": descriptor["method"], "components": descriptor["components"],
                "feature_creation_seconds": 0.1, "reducer_fit_seconds": 0.1,
                "transform_seconds": 0.1, "model_training_seconds": 0.2,
                "evaluation_seconds": 0.1, "total_pipeline_seconds": 0.6,
            },
            "validation_ndcg": recommendation[0]["ndcg_at_10"],
            "validation_coverage": 0.5, "config": {"n_clusters": 8},
            "rule_summary": {"rules": 0, "support_in_group_mean": None,
                             "support_global_mean": None, "fallback_groups": 8,
                             "empty_transactions": 0},
            "model_bytes": 10, "fit_seconds": 0.2, "reduction_fit_seconds": 0.1,
        }], [], None, 20, type("Profiles", (), {"users": list(range(30)), "excluded_users": 0})())

    monkeypatch.setattr(experiment_unsupervised, "_association_global", fake_global)
    monkeypatch.setattr(experiment_unsupervised, "_association_grouped", fake_grouped)
    association_request = ExperimentRequest(
        group="association", preset_id="association.standard.v1"
    )
    association_result = experiment_unsupervised.association(
        {"train": None}, None, association_request, lambda *_: None
    )
    assert [item["variant_id"] for item in association_descriptors] == ["pca_8"]
    assert association_request.k_values == [8]
    assert association_result["reduction"]["transaction_reduction"] is False
    assert association_result["selected_variant_id"] == "apriori_grouped_pca_8_k8"

    rl_calls = []

    def fake_rl(_splits, _movies, request, _update, reduction_variant=None):
        rl_calls.append((tuple(request.algorithms), reduction_variant, request.reduction_mode))
        return {
            "rows": [
                {"algorithm": policy, "variant_id": policy, "split": "validation",
                 "average_reward": reward}
                for policy, reward in (("dqn", 0.4), ("random", 0.6), ("popular", 0.5))
            ],
            "best_configs": {}, "charts": [], "notes": [],
            "paired_test_uncertainty": {"reference_policy": "popular", "policies": {}},
            "rl_protocol": {"validation_episode_ids_sha256": "v", "test_episode_ids_sha256": "t",
                            "train_steps_actual": 10},
        }

    monkeypatch.setattr(experiment_reinforcement, "_reinforcement_single", fake_rl)
    rl_request = ExperimentRequest(group="reinforcement", preset_id="reinforcement.standard.v1")
    rl_result = experiment_reinforcement.reinforcement({}, None, rl_request, lambda *_: None)
    assert rl_calls == [(tuple(rl_request.algorithms), None, "off")]
    assert rl_result["selected_variant_id"] == "random"

@pytest.mark.parametrize("group", ["rating", "classification", "clustering", "association", "reinforcement"])
@pytest.mark.parametrize("choice", ["off", "pca", "truncated_svd"])
def test_preset_reduction_choice_survives_serialization(group, choice):
    from movie_rating.experiment_config import ExperimentRequest, preset_reduction_plan

    request = ExperimentRequest(group=group, preset_id=f"{group}.standard.v1", reduction_choice=choice)
    restored = ExperimentRequest.model_validate(request.model_dump())
    assert restored.reduction_choice == choice
    assert restored.reduction_mode == ("off" if choice == "off" else "enabled")
    assert restored.components == [8]
    if choice != "off":
        assert restored.reduction_methods == [choice]
    for algorithm in restored.algorithms:
        plan = preset_reduction_plan(group, algorithm, choice)
        if choice == "off":
            assert plan is None
        elif plan:
            assert plan == {"method": choice, "components": 8}


def test_reduction_choice_validation():
    from pydantic import ValidationError
    from movie_rating.experiment_config import ExperimentRequest

    with pytest.raises(ValidationError):
        ExperimentRequest(group="rating", preset_id="rating.standard.v1", reduction_choice="invalid")
    with pytest.raises(ValidationError, match="preset_id"):
        ExperimentRequest(group="rating", reduction_choice="pca")
