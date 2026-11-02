from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles


class FrontendFiles(StaticFiles):
    async def get_response(self, path, scope):
        normalized = path.replace("\\", "/").lstrip("/")
        try:
            response = await super().get_response(path, scope)
        except HTTPException as error:
            if (
                error.status_code != 404
                or normalized == "api"
                or normalized.startswith("api/")
                or Path(path).suffix
            ):
                raise
            response = await super().get_response("index.html", scope)
        # Every document navigation must discover the current hashed JS/CSS bundle.
        # Apply this to SPA deep links and conditional 304 responses as well.
        if not Path(normalized).suffix or normalized.endswith(".html"):
            response.headers["Cache-Control"] = "no-store"
        return response


def mount_frontend(app: FastAPI, directory: Path):
    if (directory / "index.html").is_file():
        app.mount("/", FrontendFiles(directory=directory, html=True), name="frontend")
