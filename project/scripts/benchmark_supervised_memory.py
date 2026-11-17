"""Run a supervised or clustering job through the web UI's isolation/RAM guard."""

import argparse
import json
import sys
import time

from movie_rating.experiment_config import ExperimentRequest
from movie_rating.lab import LabService
from movie_rating.settings import load_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group", choices=["rating", "classification", "clustering"], default="rating"
    )
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--max-users", type=int, default=150)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--memory-budget-mb", type=int, default=4096)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    settings = load_settings()
    service = LabService(settings)
    request = ExperimentRequest(
        group=args.group,
        algorithms=args.algorithms,
        use_all_data=args.full_data,
        max_users=args.max_users,
        tune=args.tune,
        epochs=args.epochs,
        memory_budget_mb=args.memory_budget_mb,
    )
    job = service.start(request)
    print(
        json.dumps({"run_id": job["id"], "request": request.model_dump()}, ensure_ascii=False),
        flush=True,
    )
    message = None
    try:
        while True:
            current = service.get(job["id"])
            if current["message"] != message:
                message = current["message"]
                print(f"{current['progress']}% {message}", flush=True)
            if current["status"] in ("completed", "failed", "cancelled") and service.active is None:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        service.cancel(job["id"])
        service.executor.shutdown(wait=True)
        raise
    service.executor.shutdown(wait=True)
    result = current.get("result", {})
    print(
        json.dumps(
            {
                "status": current["status"],
                "run_id": job["id"],
                "provenance": result.get("provenance"),
                "runtime": result.get("runtime"),
                "rows": result.get("rows"),
                "recommendation": result.get("recommendation"),
                "selection": result.get("selection"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if current["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
