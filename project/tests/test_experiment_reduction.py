import pickle

import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from movie_rating.experiment_reduction import FeatureReducer, ReductionSpec, reduction_variants


@pytest.mark.parametrize("method", ["pca", "truncated_svd"])
def test_reducer_fits_all_train_rows_in_batches_and_roundtrips(method):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(40, 5)).astype(np.float32)
    reducer = FeatureReducer(method, 5, batch_rows=7).fit(
        x, feature_schema=["genre_a", "genre_b", "genre_c", "genre_d", "genre_e"],
        fit_scope="train/fold-1",
    )
    transformed = reducer.transform(x)
    restored = reducer.inverse_transform(transformed)
    assert transformed.shape == (40, 5)
    assert reducer.fit_rows == len(x)
    assert reducer.fit_scope == "train/fold-1"
    np.testing.assert_allclose(restored, x, atol=2e-5)
    meta = reducer.metadata()
    assert meta["input_dimensions"] == 5 and meta["output_dimensions"] == 5
    assert 0 <= meta["explained_variance_ratio_train"] <= 1.00001
    clone = pickle.loads(pickle.dumps(reducer))
    np.testing.assert_allclose(clone.transform(x), transformed)
    assert clone.fingerprint == reducer.fingerprint


def test_reducer_accepts_sparse_input_without_dense_user_item_allocation():
    x = sparse.csr_matrix(np.asarray([[1, 0, 2], [0, 1, 1], [1, 1, 0], [2, 0, 1]], dtype=np.float32))
    reducer = FeatureReducer("truncated_svd", 2, batch_rows=2).fit(x, ["a", "b", "c"], "train")
    assert reducer.transform(x).shape == (4, 2)
    assert reducer.metadata()["fit_rows"] == 4


def test_truncated_svd_explained_variance_matches_projected_variance_with_nonzero_means():
    rng = np.random.default_rng(314)
    shared = rng.normal(size=(120, 2))
    x = np.column_stack(
        [
            13.0 + 1.8 * shared[:, 0] + 0.2 * shared[:, 1],
            -7.0 + 0.3 * shared[:, 0] + 1.1 * shared[:, 1],
            4.0 - 0.5 * shared[:, 0] + 0.4 * shared[:, 1],
            rng.normal(scale=0.25, size=len(shared)),
        ]
    )
    reducer = FeatureReducer("truncated_svd", 2, scale=False, batch_rows=19).fit(x)
    reference = TruncatedSVD(n_components=2, algorithm="arpack", random_state=42).fit(x)

    np.testing.assert_allclose(
        reducer.explained_variance_, reference.explained_variance_, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        reducer.explained_variance_ratio_, reference.explained_variance_ratio_,
        rtol=1e-5, atol=1e-6,
    )
    assert np.all(np.abs(np.mean(x, axis=0)[:3]) > 1.0)
    second_moment_ratio = np.square(reducer.transform(x)).mean(axis=0).sum() / np.var(x, axis=0).sum()
    assert not np.isclose(reducer.explained_variance_ratio_.sum(), second_moment_ratio, rtol=0.05)
    assert reducer.metadata()["centers_input"] is False


def test_reducer_rejects_constant_rank_deficient_and_bad_shapes():
    with pytest.raises(ValueError, match="không có phương sai"):
        FeatureReducer("pca", 1).fit(np.ones((8, 3), dtype=np.float32))
    rank_one = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=np.float32)
    with pytest.raises(ValueError, match="Feature rank=1"):
        FeatureReducer("pca", 2).fit(rank_one)
    fitted = FeatureReducer("pca", 1).fit(rank_one)
    with pytest.raises(ValueError, match="Input có shape"):
        fitted.transform(np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="components"):
        FeatureReducer("pca", 3).fit(rank_one)


def test_fit_scope_is_train_only_and_heldout_transform_does_not_refit():
    train = np.asarray([[0, 1], [1, 0], [1, 1], [2, 0]], dtype=np.float32)
    heldout = np.asarray([[100, -100], [-50, 50]], dtype=np.float32)
    reducer = FeatureReducer("pca", 1).fit(train, ["preference", "frequency"], "train-only")
    before = reducer.metadata()
    reducer.transform(heldout)
    after = reducer.metadata()
    assert before == after
    assert after["fit_rows"] == len(train)
    assert after["fit_scope"] == "train-only"


def test_reduction_specs_return_stable_raw_and_reduced_variant_ids():
    raw = reduction_variants(ReductionSpec(mode="off", components=(1, 2)))
    compare = reduction_variants(
        ReductionSpec(mode="compare", methods=("pca", "truncated_svd"), components=(2, 4))
    )
    enabled = reduction_variants(
        ReductionSpec(mode="enabled", methods=("pca",), components=(2,))
    )
    assert raw == [{"variant_id": "raw", "method": None, "components": None}]
    assert [row["variant_id"] for row in compare] == [
        "raw", "pca_2", "pca_4", "truncated_svd_2", "truncated_svd_4"
    ]
    assert [row["variant_id"] for row in enabled] == ["pca_2"]
