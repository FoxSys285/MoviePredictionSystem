"""Local application accounts, opaque sessions and contact message storage."""

import hashlib
import hmac
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SESSION_SECONDS = 7 * 24 * 3600


class AccountStore:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=15) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL, created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL,
                    subject TEXT NOT NULL, message TEXT NOT NULL, created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_ratings (
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    movie_id INTEGER NOT NULL,
                    rating REAL NOT NULL CHECK(rating BETWEEN 0.5 AND 5
                        AND rating * 2 = CAST(rating * 2 AS INTEGER)),
                    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                    PRIMARY KEY(user_id, movie_id)
                );
                CREATE INDEX IF NOT EXISTS ratings_by_user_time
                    ON user_ratings(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS dataset_profiles (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id),
                    source_key TEXT NOT NULL, dataset_user_id INTEGER NOT NULL,
                    imported_rating_count INTEGER NOT NULL,
                    profile_modified INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(source_key, dataset_user_id)
                );
            """)
            yield connection

    @staticmethod
    def password_hash(password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600_000).hex()

    def register(self, name: str, email: str, password: str) -> dict:
        if email.lower().endswith("@demo.movie.local"):
            raise ValueError("Tên miền này dành riêng cho tài khoản demo MovieLens.")
        salt = secrets.token_hex(16)
        password_hash = self.password_hash(password, salt)
        try:
            with self.connect() as db:
                cursor = db.execute(
                    "INSERT INTO users(name,email,password_hash,salt,created_at) VALUES(?,?,?,?,?)",
                    (name, email.lower(), password_hash, salt, int(time.time())),
                )
                return {"id": cursor.lastrowid, "name": name, "email": email.lower(),
                        "profile_type": "personal", "dataset_user_id": None,
                        "imported_rating_count": 0, "profile_modified": False}
        except sqlite3.IntegrityError as error:
            raise ValueError("Email này đã được đăng ký. Vui lòng đăng nhập.") from error

    def authenticate(self, email: str, password: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()
        # Equal-cost hashing also applies to unknown accounts.
        salt = row["salt"] if row else "0" * 32
        calculated = self.password_hash(password, salt)
        if row and hmac.compare_digest(calculated, row["password_hash"]):
            return self.public_user(row["id"])
        return None

    def public_user(self, user_id: int) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT u.id,u.name,u.email,p.dataset_user_id,p.imported_rating_count,"
                "p.profile_modified FROM users u LEFT JOIN dataset_profiles p ON p.user_id=u.id "
                "WHERE u.id=?", (user_id,),
            ).fetchone()
        if not row:
            return None
        user = dict(row)
        user.update(profile_type="dataset" if user["dataset_user_id"] is not None else "personal",
                    imported_rating_count=user["imported_rating_count"] or 0,
                    profile_modified=bool(user["profile_modified"]))
        return user

    def new_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE expires_at<=?", (int(time.time()),))
            db.execute(
                "INSERT INTO sessions VALUES(?,?,?)",
                (digest, user_id, int(time.time()) + SESSION_SECONDS),
            )
        return token

    def user_for_session(self, token: str | None) -> dict | None:
        if not token:
            return None
        with self.connect() as db:
            row = db.execute(
                "SELECT u.id,u.name,u.email FROM users u JOIN sessions s ON u.id=s.user_id "
                "WHERE s.token_hash=? AND s.expires_at>?",
                (hashlib.sha256(token.encode()).hexdigest(), int(time.time())),
            ).fetchone()
        return self.public_user(row["id"]) if row else None

    def logout(self, token: str | None):
        if token:
            with self.connect() as db:
                db.execute(
                    "DELETE FROM sessions WHERE token_hash=?",
                    (hashlib.sha256(token.encode()).hexdigest(),),
                )

    def save_message(self, name: str, email: str, subject: str, message: str) -> int:
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO messages(name,email,subject,message,created_at) VALUES(?,?,?,?,?)",
                (name, email, subject, message, int(time.time())),
            )
            return int(cursor.lastrowid)

    def ratings_for_user(self, user_id: int) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT movie_id, rating, created_at, updated_at FROM user_ratings "
                "WHERE user_id=? ORDER BY updated_at DESC, movie_id DESC", (user_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def dataset_profile(self, user_id: int) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM dataset_profiles WHERE user_id=?", (user_id,)).fetchone()
        return dict(row) if row else None

    def save_rating(self, user_id: int, movie_id: int, rating: float) -> dict:
        timestamp = time.time_ns() // 1_000_000
        with self.connect() as db:
            db.execute(
                "INSERT INTO user_ratings(user_id,movie_id,rating,created_at,updated_at) "
                "VALUES(?,?,?,?,?) ON CONFLICT(user_id,movie_id) DO UPDATE SET "
                "rating=excluded.rating, updated_at=excluded.updated_at",
                (user_id, movie_id, rating, timestamp, timestamp),
            )
            db.execute("UPDATE dataset_profiles SET profile_modified=1 WHERE user_id=?", (user_id,))
        return {"movie_id": movie_id, "rating": rating, "updated_at": timestamp}

    def delete_rating(self, user_id: int, movie_id: int):
        with self.connect() as db:
            result = db.execute("DELETE FROM user_ratings WHERE user_id=? AND movie_id=?", (user_id, movie_id))
            if result.rowcount:
                db.execute("UPDATE dataset_profiles SET profile_modified=1 WHERE user_id=?", (user_id,))
