"""Dispatch and presentation schema for rating regression experiments."""

from movie_rating.lab import CATALOG, chart, series


def execute(splits, movies, request, update, progress_store=None):
    from movie_rating.experiment_supervised import supervised

    if request.group != "rating":
        raise ValueError("The experiment workspace currently supports regression only.")
    result = supervised(splits, movies, request, update, progress_store=progress_store)
    metrics = next(g["metrics"] for g in CATALOG["groups"] if g["id"] == request.group)
    comparisons = []
    for metric in metrics:
        key = metric["key"]
        def identity(row):
            return row.get("variant_id") or row["algorithm"]

        algorithms = list(dict.fromkeys((identity(r), r["name"]) for r in result["rows"]))
        lookup = {(identity(r), r["split"]): r for r in result["rows"]}
        splits_used = list(dict.fromkeys(r["split"] for r in result["rows"]))
        comparisons.append(
            chart(
                len(comparisons),
                metric["name"],
                "bar",
                "Thuật toán / cấu hình",
                metric["name"],
                [
                    series(
                        s.title(),
                        [name for _, name in algorithms],
                        [lookup.get((aid, s), {}).get(key) for aid, _ in algorithms],
                    )
                    for s in splits_used
                ],
                "comparison",
                metric_key=key,
            )
        )
    result["charts"] = comparisons + result.get("charts", [])
    for index, figure in enumerate(result["charts"]):
        figure["index"] = index
    result["schema_version"] = 4
    return result
