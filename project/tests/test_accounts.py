import sqlite3

from fastapi.testclient import TestClient

from movie_rating.api.app import create_app
from movie_rating.data.prepare import prepare


def test_account_lifecycle(configured):
    with TestClient(create_app(configured)) as client:
        assert client.get("/api/v1/auth/me").status_code == 401
        response = client.post(
            "/api/v1/auth/register",
            json={
                "name": "Người xem phim",
                "email": "viewer@example.com",
                "password": "test-password-123",
            },
        )
        assert response.status_code == 201
        assert response.json()["name"] == "Người xem phim"
        assert "password" not in response.text
        assert "HttpOnly" in response.headers["set-cookie"]
        assert client.get("/api/v1/auth/me").json()["email"] == "viewer@example.com"
        assert client.post("/api/v1/auth/logout").status_code == 200
        assert client.get("/api/v1/auth/me").status_code == 401
        assert (
            client.post(
                "/api/v1/auth/login",
                json={
                    "email": "viewer@example.com",
                    "password": "incorrect",
                },
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/v1/auth/login",
                json={
                    "email": "VIEWER@example.com",
                    "password": "test-password-123",
                },
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/v1/auth/register",
                json={
                    "name": "Another User",
                    "email": "viewer@example.com",
                    "password": "test-password-123",
                },
            ).status_code
            == 409
        )
    with sqlite3.connect(configured.artifacts_dir / "application.sqlite3") as db:
        stored = db.execute("SELECT password_hash, salt FROM users").fetchone()
        assert "test-password-123" not in stored
        assert len(stored[0]) == 64


def test_reject_invalid_account_and_cross_origin_request(configured):
    with TestClient(create_app(configured)) as client:
        assert (
            client.post(
                "/api/v1/auth/register",
                json={
                    "name": "User",
                    "email": "not-an-email",
                    "password": "short",
                },
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/auth/register",
                headers={"Origin": "https://other.example"},
                json={"name": "User", "email": "u@example.com", "password": "test-password-123"},
            ).status_code
            == 403
        )


def test_contact_message_saved(configured):
    with TestClient(create_app(configured)) as client:
        response = client.post(
            "/api/v1/contact",
            json={
                "name": "Người xem",
                "email": "viewer@example.com",
                "subject": "Góp ý",
                "message": "Giao diện rất phù hợp với đồ án phim.",
            },
        )
        assert response.status_code == 201
        with sqlite3.connect(configured.artifacts_dir / "application.sqlite3") as db:
            assert db.execute("SELECT message FROM messages").fetchone()[0].startswith("Giao diện")


def test_catalog_filters_and_missing_rating(configured):
    prepare(configured)
    with TestClient(create_app(configured)) as client:
        result = client.get("/api/v1/movies", params={"genre": "Drama", "sort": "rating"}).json()
        assert result["total"] == 12
        assert "mean_rating" in result["items"][0]
        assert client.get("/api/v1/movies", params={"genre": "Action"}).json()["total"] == 0
        assert client.get("/api/v1/movies", params={"year": "2020"}).json()["total"] == 0
        assert client.get("/api/v1/movies", params={"min_rating": 4.9}).json()["total"] == 0
