# Experiment Studio: regression and run logs

The experiment workspace currently contains rating regression and its run history. The production movie application, saved recommendation models, user history, and account features remain separate from this workspace.

## Regression workflow

The API accepts only the `rating` group. Its regressors are Linear Regression, XGBoost, SVD, and Neural Collaborative Filtering. The sidebar lists all four algorithms directly.

The data loader uses per-user chronological Train, Validation, and Test splits. GridSearchCV with five folds evaluates configurations on Train; the standard preset checks one fixed configuration, while explicitly configured runs can compare parameters. Validation is used to compare runs, and Test is held for final reporting. The Global Mean baseline predicts each rating with the Train mean. RMSE and MAE are reported for every regressor and the baseline.

PCA is available as a Linear Regression feature option. Other regression algorithms use their native representations.

## Logs and reports

Each run is stored under `artifacts/experiment-lab/<run_id>/`. `events.jsonl` records progress, training messages, warnings, and errors. The log API reads entries by byte cursor and supports a complete log download. Results can be exported as CSV, XLSX, or JSON.

The main implementation is in `src/movie_rating/experiment_config.py`, `experiment_data.py`, `experiment_estimators.py`, `experiment_supervised.py`, `experiment_pipeline.py`, `experiment_logs.py`, and `lab.py`. The React page is `frontend/src/pages/Experiments.tsx`.
