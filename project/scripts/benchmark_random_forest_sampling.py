"""Compare full and capped per-tree bootstrap on a frozen temporal user cohort."""

import argparse
import gc
import json
import sys
import time
from datetime import UTC, datetime
from uuid import uuid4

from movie_rating.experiment_config import ExperimentRequest, ModelParameters
from movie_rating.experiment_data import ExperimentData
from movie_rating.experiment_supervised import (
    candidates,
    classification_predict,
    classification_scores,
    feature_fit,
)
from movie_rating.settings import load_settings


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=5000)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=250_000)
    args = parser.parse_args()
    if not 10 <= args.users <= 5000 or not 10 <= args.trees <= 500 or args.max_samples < 1:
        parser.error("users must be 10..5000, trees 10..500, max-samples positive.")
    settings = load_settings()
    request = ExperimentRequest(
        group="classification", algorithms=["random_forest"],
        use_all_data=False, max_users=args.users,
        parameters=ModelParameters(forest_estimators=args.trees),
    )
    splits, movies, provenance = ExperimentData(settings.artifacts_dir / "data").load(request)
    rows = []
    for cap in (None, args.max_samples):
        config = {**candidates("random_forest", request)[0], "max_samples": cap}
        print(json.dumps(config), flush=True)
        started = time.perf_counter()
        builder, model, _ = feature_fit(
            splits["train"], [], movies, "random_forest", config,
            report=lambda message: print(message, flush=True),
        )
        fit_total_seconds = time.perf_counter() - started
        probability = classification_predict(builder, model, splits["validation"])
        rows.append({
            "config": config,
            "effective_bootstrap_draws": model.estimator.max_samples or len(splits["train"]),
            "fit_total_seconds": fit_total_seconds,
            "validation": classification_scores(
                splits["validation"].rating >= 3.5, probability, threshold=0.5,
            ),
        })
        print(json.dumps(rows[-1]), flush=True)
        del builder, model, probability
        gc.collect()
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Paired cohort benchmark; validation threshold 0.5; no CV or final refit.",
        "provenance": provenance,
        "train_rows": len(splits["train"]),
        "validation_rows": len(splits["validation"]),
        "results": rows,
        "speedup": rows[0]["fit_total_seconds"] / rows[1]["fit_total_seconds"],
    }
    destination = settings.artifacts_dir / "benchmarks" / (
        f"random-forest-sampling-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:6]}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
