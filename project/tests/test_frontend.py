from fastapi import FastAPI
from fastapi.testclient import TestClient

from movie_rating.api.frontend import mount_frontend


def test_frontend_deep_links_and_missing_resources(tmp_path):
    (tmp_path / "index.html").write_text("<html>MOVIE</html>", encoding="utf-8")
    app = FastAPI()
    mount_frontend(app, tmp_path)
    with TestClient(app) as client:
        assert client.get("/gallery").status_code == 200
        assert client.get("/movies/79132").status_code == 200
        assert client.get("/experiments").status_code == 200
        assert "MOVIE" in client.get("/login").text
        assert client.get("/api/v1/nonexistent").status_code == 404
        assert client.get("/assets/nonexistent.webp").status_code == 404


def test_spa_documents_do_not_cache_an_old_frontend_bundle(tmp_path):
    entry = tmp_path / "index.html"
    entry.write_text('<script src="/assets/old.js"></script>', encoding="utf-8")
    app = FastAPI()
    mount_frontend(app, tmp_path)
    with TestClient(app) as client:
        for route in ("/", "/index.html", "/contact", "/genres", "/watch-history", "/experiments"):
            response = client.get(route)
            assert response.headers["cache-control"] == "no-store"
            conditional = client.get(route, headers={"If-None-Match": response.headers["etag"]})
            assert conditional.status_code == 304
            assert conditional.headers["cache-control"] == "no-store"
        old_etag = client.get("/contact").headers["etag"]
        entry.write_text('<script src="/assets/current-build.js"></script>', encoding="utf-8")
        updated = client.get("/contact", headers={"If-None-Match": old_etag})
        assert updated.status_code == 200
        assert "/assets/current-build.js" in updated.text


def test_experiment_documents_are_downloadable_and_allowlisted(configured):
    from movie_rating.api.app import create_app

    with TestClient(create_app(configured)) as client:
        for document, filename in (("protocol", "experimental-protocol.md"),
                                   ("challenges", "experimental-challenges.md"),
                                   ("figures-stability", "experiment-figures-and-stability.md"),
                                   ("results", "experimental-results.md")):
            response = client.get(f"/api/v1/experiments/documents/{document}")
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/markdown")
            assert filename in response.headers["content-disposition"]
            assert response.text.startswith("# ")
        assert client.get("/api/v1/experiments/documents/other").status_code == 422


def test_health_identifies_the_newly_launched_server(configured, monkeypatch):
    import os

    from movie_rating.api.app import create_app

    monkeypatch.setenv("MOVIE_SERVER_INSTANCE", "launcher-regression-check")
    with TestClient(create_app(configured)) as client:
        response = client.get("/api/v1/health").json()
        assert response["status"] == "ok"
        assert response["server_instance"] == "launcher-regression-check"
        assert response["process_id"] == os.getpid()
