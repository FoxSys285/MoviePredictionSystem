"""Paths, configuration and DuckDB setup shared by the TV1 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or PROJECT_ROOT / "configs" / "default.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def project_path(relative: str) -> Path:
    """Resolve a configured path relative to the project directory."""
    return (PROJECT_ROOT / relative).resolve()


def sql_literal(value: str | Path) -> str:
    """Quote a path/string for DuckDB SQL. Never use shell interpolation."""
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def connection(config: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    processed = project_path(config["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    temp = processed / ".duckdb_tmp"
    temp.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit={sql_literal(config.get('duckdb_memory_limit', '2GB'))}")
    con.execute(f"SET threads={int(config.get('duckdb_threads', 2))}")
    con.execute(f"SET temp_directory={sql_literal(temp)}")
    return con


def write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rating_csv_sql(path: Path) -> str:
    return (
        f"read_csv({sql_literal(path)}, header=true, "
        "types={'userId':'INTEGER','movieId':'INTEGER',"
        "'rating':'DOUBLE','timestamp':'BIGINT'})"
    )


def parquet_sql(path: Path) -> str:
    return f"read_parquet({sql_literal(path)})"
