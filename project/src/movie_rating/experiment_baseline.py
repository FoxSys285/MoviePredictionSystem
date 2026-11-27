"""Global Mean baselines, fitted exclusively on the full Train split."""

import numpy as np


def global_mean_baseline(splits):
    """Predict every held-out rating with the same Train mean and score it.

    The mean is weighted by rating rows, not by users or movies. Validation
    and Test ratings are used only to compute their respective MAE and RMSE.
    """
    train = np.asarray(splits["train"]["rating"], dtype=np.float64)
    if train.size == 0 or not np.isfinite(train).all():
        raise ValueError("Global Mean requires non-empty, finite Train ratings.")
    prediction = float(train.mean())
    rows = []
    for split in ("validation", "test"):
        ratings = np.asarray(splits[split]["rating"], dtype=np.float64)
        if ratings.size == 0 or not np.isfinite(ratings).all():
            raise ValueError(f"Global Mean requires non-empty, finite {split} ratings.")
        errors = ratings - prediction
        rows.append({"split": split, "ratings": len(ratings),
                     "mae": float(np.abs(errors).mean()),
                     "rmse": float(np.sqrt(np.mean(np.square(errors))))})
    return {"name": "Global Mean", "fit_scope": "train",
            "train_ratings": len(train), "prediction": prediction, "rows": rows}


def baseline_from_provenance(provenance, artifacts_directory):
    """Recover a saved full-data run's baseline without retraining any model.

    Require the original split files and their recorded fingerprint. Stream
    only rating columns so even a 25M-rating run needs bounded working memory.
    """
    from pathlib import Path

    import pyarrow.parquet as pq

    from movie_rating.experiment_data import parquet_fingerprint

    names = ("train", "validation", "test")
    files = provenance.get("split_files")
    expected_hash = provenance.get("sample_sha256")
    if not files or not expected_hash or any(name not in files for name in names):
        return None
    root = Path(artifacts_directory).resolve()
    paths = {name: Path(files[name]).resolve() for name in names}
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise ValueError("Baseline split files must be inside the artifacts directory.")
    if parquet_fingerprint([paths[name] for name in names]) != expected_hash:
        raise ValueError("Saved experiment split fingerprint no longer matches the data.")

    def batches(name):
        for batch in pq.ParquetFile(paths[name]).iter_batches(
            batch_size=250_000, columns=["rating"],
        ):
            ratings = np.asarray(batch.column(0).to_numpy(), dtype=np.float64)
            if not np.isfinite(ratings).all():
                raise ValueError(f"Global Mean requires finite {name} ratings.")
            yield ratings

    count, total = 0, 0.0
    for ratings in batches("train"):
        count += len(ratings)
        total += float(ratings.sum())
    if not count:
        raise ValueError("Global Mean requires non-empty Train ratings.")
    prediction = total / count
    rows = []
    for name in ("validation", "test"):
        size, absolute, squared = 0, 0.0, 0.0
        for ratings in batches(name):
            errors = ratings - prediction
            size += len(ratings)
            absolute += float(np.abs(errors).sum())
            squared += float(np.square(errors).sum())
        if not size:
            raise ValueError(f"Global Mean requires non-empty {name} ratings.")
        scores = {"mae": absolute / size, "rmse": float(np.sqrt(squared / size))}
        rows.append({"split": name, "ratings": size, **scores})
    return {"name": "Global Mean", "fit_scope": "train", "train_ratings": count,
            "prediction": prediction, "rows": rows}
