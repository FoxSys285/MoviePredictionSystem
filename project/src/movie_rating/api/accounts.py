import os

from fastapi import APIRouter, Cookie, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from movie_rating.services.accounts import SESSION_SECONDS, AccountStore
from movie_rating.services.demo_profiles import DemoProfiles
from movie_rating.settings import Settings

COOKIE = "movie_session"


def check_origin(request: Request):
    origin = request.headers.get("origin")
    allowed = {
        "http://127.0.0.1:5173", "http://localhost:5173",
        "http://127.0.0.1:8000", "http://localhost:8000",
    }
    configured = os.getenv("MOVIE_FRONTEND_ORIGIN")
    if configured:
        allowed.add(configured.rstrip("/"))
    if origin and origin.rstrip("/") not in allowed:
        raise HTTPException(403, "Nguồn yêu cầu không được phép.")


def set_session_cookie(response: Response, token: str):
    response.delete_cookie(COOKIE, path="/api/v1/auth")
    response.set_cookie(
        COOKIE, token, max_age=SESSION_SECONDS, httponly=True, samesite="lax",
        secure=os.getenv("MOVIE_SECURE_COOKIES") == "1", path="/api/v1",
    )


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str):
        value = value.strip().lower()
        if "@" not in value or "." not in value.rsplit("@", 1)[-1]:
            raise ValueError("Email không hợp lệ.")
        return value


class RegisterRequest(LoginRequest):
    name: str = Field(min_length=2, max_length=100)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str):
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Vui lòng nhập họ và tên.")
        return value


class ContactRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=3, max_length=254)
    subject: str = Field(min_length=2, max_length=120)
    message: str = Field(min_length=10, max_length=5000)

    @field_validator("name", "subject", "message")
    @classmethod
    def validate_text(cls, value: str):
        if not value.strip():
            raise ValueError("Nội dung không được để trống.")
        return value.strip()

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str):
        return LoginRequest.validate_email(value)


def account_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/v1")
    store = AccountStore(settings.artifacts_dir / "application.sqlite3")
    demos = DemoProfiles(settings, store)

    @router.get("/auth/demo-accounts")
    def demo_accounts():
        try:
            return demos.accounts()
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(503, "Chưa tải được tài khoản demo từ dataset.") from error

    def session(response: Response, user: dict):
        set_session_cookie(response, store.new_session(user["id"]))
        return user

    @router.post("/auth/register", status_code=201)
    def register(body: RegisterRequest, request: Request, response: Response):
        check_origin(request)
        try:
            user = store.register(body.name, body.email, body.password)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return session(response, user)

    @router.post("/auth/login")
    def login(body: LoginRequest, request: Request, response: Response):
        check_origin(request)
        user = store.authenticate(body.email, body.password)
        if not user:
            raise HTTPException(401, "Email hoặc mật khẩu chưa chính xác.")
        return session(response, user)

    @router.get("/auth/me")
    def me(response: Response, movie_session: str | None = Cookie(default=None)):
        user = store.user_for_session(movie_session)
        if not user:
            raise HTTPException(401, "Bạn chưa đăng nhập.")
        # Migrate existing auth-only cookies so rating and history endpoints receive them.
        set_session_cookie(response, movie_session)
        return user

    @router.post("/auth/logout")
    def logout(
        request: Request, response: Response, movie_session: str | None = Cookie(default=None)
    ):
        check_origin(request)
        store.logout(movie_session)
        response.delete_cookie(COOKIE, path="/api/v1/auth")
        response.delete_cookie(COOKIE, path="/api/v1")
        return {"ok": True}

    @router.post("/contact", status_code=201)
    def contact(body: ContactRequest, request: Request):
        check_origin(request)
        message_id = store.save_message(body.name, body.email, body.subject, body.message)
        return {"id": message_id, "message": "Tin nhắn của bạn đã được lưu thành công."}

    return router
