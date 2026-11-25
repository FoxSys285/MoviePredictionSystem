# XGBoost CPU thread benchmark

Measured on 2026-11-21 with the project `.venv`: Intel Core i5-12500H (12 cores / 16 logical
processors), 15.71 GiB RAM, NVIDIA RTX 3050 Laptop GPU (not used by XGBoost), XGBoost 3.4.1, and
the prepared MovieLens data on this machine.

Both benchmark invocations reused the same frozen per-user temporal sample and Train split:
4,904 users, 778,443 ratings, 544,920 Train ratings, seed 42, protocol
`per_user_temporal_70_15_15_v2`, sample SHA-256
`924e6a94f9b16b9282f4cc56a5b6e240b89edd4be099b814eff4dee5fcefeff8`. Each fit used 150 trees,
depth 6, learning rate 0.05, and matching `n_jobs` / `threadpool_limits`. The table reports the
median of two measured runs per thread count. Times are seconds; RSS is MiB.

| CPU threads | Feature/cache | Tree fit | Total `fit()` | Peak process RSS |
|---:|---:|---:|---:|---:|
| 2 | 7.223 | 4.274 | 11.498 | 979.8 |
| 4 | 6.167 | 3.811 | **9.978** | 992.6 |
| 8 | 6.058 | 4.373 | 10.431 | 980.5 |

Four threads had the lowest median total time: 13.2% below 2 threads and 4.3% below 8 threads.
Eight threads reduced feature/cache time slightly, but its tree fit was slower than both 2 and 4
threads. Peak RSS varied by about 33 MiB across the medians. The Experiment Studio default is
therefore **4 threads** on this machine; the selector still allows 2 and 8.

The feature/cache interval starts at `model.fit()` and ends immediately before the XGBoost tree
training call (including feature statistics and dense/external-memory cache preparation). Tree
fit ends at the model's completion event; total time covers the complete `model.fit()` call. A
background sampler recorded process RSS every 25 ms. `threadpoolctl` reported the BLAS/OpenMP
pools at the fit boundary at the requested limits; XGBoost also received the matching explicit
`n_jobs` value. The CUDA device was not part of these measurements.

Each thread count was measured once in each invocation. Raw logs are in the two new, uniquely
named artifact directories:

- `artifacts/benchmarks/xgboost-threads-20261121T031327-ea9d38/results.json`
- `artifacts/benchmarks/xgboost-threads-20261121T031514-d028b0/results.json`

This is a representative sampled-user benchmark, not a full-data benchmark: it uses complete
histories for 4,904 sampled prepared users, not every MovieLens 25M rating. Results may change
with a different data build, background load, power/thermal state, Windows version, or XGBoost
release. The default should be remeasured if those conditions change materially.

## Verification after the thread setting became configurable

I repeated the same 5,000-user benchmark after connecting the selected `n_jobs` value to the
worker's native threadpool limit. The sample hash, Train split, seed, estimator parameters, and
measurement method were unchanged. The first pass ran 2/4/8 threads; the second ran 8/4/2 to
reduce ordering bias. This table reports the median of those two passes per thread count.

| CPU threads | Feature/cache (s) | Tree fit (s) | Total `fit()` (s) | Peak process RSS (MiB) |
|---:|---:|---:|---:|---:|
| 2 | 6.345 | 4.092 | 10.436 | 983.3 |
| 4 | 5.716 | 3.575 | **9.291** | 982.2 |
| 8 | 6.337 | 4.322 | 10.657 | 985.3 |

Four threads remained fastest on this sample: 11.0% lower total time than 2 threads and 12.8%
lower than 8 threads. `n_jobs` and the configurable `threadpool_limits` value were equal for each
trial. BLAS/OpenMP inventory was recorded at fit start; an additional OpenMP runtime reported one
thread, so the inventory is included in the raw records rather than claiming every loaded runtime
used the selected limit.

The additional raw measurements are preserved here:

- `artifacts/benchmarks/xgboost-threads-20261121T044944-8ac181/results.json`
- `artifacts/benchmarks/xgboost-threads-20261121T045049-4f824c/results.json`

This remains a sample benchmark (4,904 users and 544,920 Train ratings), not a full MovieLens
25M run. It supports 4 as the default for this machine and sample; it does not guarantee that 4
threads will win for a different dataset, power mode, thermal state, or software build.
