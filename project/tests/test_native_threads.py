def test_native_threadpool_limit_handles_project_paths_and_matches_selected_count():
    import xgboost

    from movie_rating.experiment_config import ModelParameters
    from movie_rating.native_threads import limit_native_threadpools

    assert xgboost.__version__
    assert ModelParameters().xgb_n_jobs == 4
    with limit_native_threadpools(4, include_xgboost=True) as pools:
        blas_pools = [pool for pool in pools if pool["user_api"] == "blas"]
        assert blas_pools
        assert all(pool["num_threads"] == 4 for pool in blas_pools)
