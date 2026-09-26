# TV1 data audit

Generated (UTC): 2026-09-26T15:31:27.413224+00:00

This audit checks split metadata and validation results. It does not use test labels for model selection.

- [x] **train: schema, rows and unique row_id:** rows=17,500,066, missing=0, invalid_ratings=0
- [x] **validation: schema, rows and unique row_id:** rows=3,750,014, missing=0, invalid_ratings=0
- [x] **test: schema, rows and unique row_id:** rows=3,750,015, missing=0, invalid_ratings=0
- [x] **Disjoint chronological ranges:** train_max=1421944973, validation_min=1421944985, validation_max=1490320202, test_min=1490320205
- [x] **Complete clean dataset:** split_rows=25,000,095, clean_rows=25,000,095
- [x] **All ratings ordered by time:** timestamp_inversions=0
- [x] **Train-only user/movie counts:** users_sum=17,500,066, movies_sum=17,500,066
- [x] **Five expanding-time folds:** folds=5
- [x] **Shared warm validation subset:** rows=337,481, mismatched=0
- [x] **Shared warm IDs for all CV folds:** warm_cv_rows=1,436,044, folds=5
- [x] **mean validation predictions match split:** rows=3,750,014, distinct_row_ids=3,750,014, mismatched=0
- [x] **Mean validation metrics reproducible:** RMSE=1.09159074, MAE=0.84872281

Result: **PASS**
