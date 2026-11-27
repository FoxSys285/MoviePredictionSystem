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
    assert set(catalog) == {"rating"}
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
def test_standard_preset_rejects_payload_configuration_that_disagrees():
    from pydantic import ValidationError

    from movie_rating.experiment_config import ExperimentRequest

    with pytest.raises(ValidationError, match="cấu hình cố định"):
        ExperimentRequest(
            group="rating", preset_id="rating.standard.v1", components=[2, 4, 8]
        )
