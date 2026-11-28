"""Start experiment workers and read progress, incremental logs, and results."""

import json
import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from movie_rating.api.accounts import check_origin
from movie_rating.experiments.jobs import ExperimentJobs, read_events


class StartExperiment(BaseModel):
    variant: Literal["baseline", "improved"]
    algorithms: list[Literal["linear", "xgboost", "svd", "ncf"]] = Field(min_length=1, max_length=4)
    max_users: int = Field(1000, ge=10, le=5000)
    seed: int = Field(42, ge=0, le=2**32 - 1)
    refit_85: bool = False

    @field_validator("algorithms")
    @classmethod
    def unique_algorithms(cls, values):
        if len(values) != len(set(values)):
            raise ValueError("Mỗi thuật toán chỉ được chọn một lần.")
        return values


def experiment_router(settings):
    router = APIRouter(prefix="/api/v1/experiments", tags=["Experiments"])
    root = settings.artifacts_dir / "experiments"
    jobs = ExperimentJobs(settings)

    def folder_for(run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
            raise HTTPException(422, "Mã thực nghiệm không hợp lệ.")
        folder = root / run_id
        if not (folder / "job.json").is_file():
            raise HTTPException(404, "Không tìm thấy lượt thực nghiệm.")
        return folder

    @router.get("/runs")
    def list_runs():
        return jobs.list_jobs()

    @router.post("/runs", status_code=202)
    def start_run(body: StartExperiment, request: Request):
        check_origin(request)
        try:
            return jobs.start(**body.model_dump())
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        except OSError as error:
            raise HTTPException(503, "Không khởi động được tiến trình thực nghiệm.") from error

    @router.get("/runs/{run_id}")
    def get_run(run_id: str):
        folder = folder_for(run_id)
        job = jobs.read_job(run_id)
        result = folder / "result.json"
        return {"job": job, "result": json.loads(result.read_text(encoding="utf-8"))
                if job["status"] == "completed" and result.is_file() else None}

    @router.get("/runs/{run_id}/result")
    def download_result(run_id: str):
        folder = folder_for(run_id)
        job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
        if job["status"] != "completed" or not (folder / "result.json").is_file():
            raise HTTPException(409, "Lượt thực nghiệm chưa hoàn tất.")
        return FileResponse(folder / "result.json", media_type="application/json",
                            filename=f"experiment-{run_id}.json")

    @router.get("/runs/{run_id}/logs")
    def logs(run_id: str, cursor: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
        return read_events(folder_for(run_id), cursor=cursor, limit=limit)

    @router.get("/runs/{run_id}/logs/download")
    def download_logs(run_id: str):
        path = folder_for(run_id) / "events.jsonl"
        if not path.is_file():
            raise HTTPException(404, "Lượt thực nghiệm chưa có log.")
        return FileResponse(path, media_type="application/x-ndjson",
                            filename=f"experiment-{run_id}.jsonl")

    return router
