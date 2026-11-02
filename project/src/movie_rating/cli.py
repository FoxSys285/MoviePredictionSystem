import argparse
import json
import sys

from movie_rating.data.prepare import prepare
from movie_rating.services.catalog import CatalogService
from movie_rating.settings import load_settings
from movie_rating.training import evaluate_test, train


def main():
    parser = argparse.ArgumentParser(description="MovieLens 25M experiment pipeline")
    parser.add_argument("--config", help="Path to a TOML config; default: configs/default.toml")
    parser.add_argument(
        "command",
        choices=["analyze", "prepare", "train", "evaluate", "analyze-experiments",
                 "stability", "serve"],
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    try:
        if args.command == "serve":
            import uvicorn

            from movie_rating.api.app import create_app

            uvicorn.run(create_app(settings), host="127.0.0.1", port=8000)
            return
        if args.command == "analyze":
            report = CatalogService(settings).load().summary
        elif args.command == "analyze-experiments":
            from movie_rating.experiment_analysis import analyze_experiments

            report = analyze_experiments(settings)
        elif args.command == "stability":
            from movie_rating.stability import run_stability

            report = run_stability(settings)
        else:
            actions = {"prepare": prepare, "train": train, "evaluate": evaluate_test}
            report = actions[args.command](settings)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
