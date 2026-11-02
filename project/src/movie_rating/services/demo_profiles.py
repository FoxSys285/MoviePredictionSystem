"""Explicit, synthetic logins for five anonymous dataset profiles; import once per source."""

import hashlib
import json
import secrets
import threading
import time

import pandas as pd

from movie_rating.services.accounts import AccountStore
from movie_rating.settings import Settings

DEMO_PASSWORD = "MovieLens@2026"


def dataset_source_key(settings: Settings):
    files = [settings.dataset_dir / name for name in ("ratings.csv", "movies.csv")]
    identity = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


class DemoProfiles:
    def __init__(self, settings: Settings, store: AccountStore):
        self.settings, self.store = settings, store
        self.lock = threading.Lock()

    def _source_key(self):
        return dataset_source_key(self.settings)

    def _existing(self, source_key):
        with self.store.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT u.name,u.email,p.dataset_user_id,p.imported_rating_count "
                "FROM dataset_profiles p JOIN users u ON u.id=p.user_id "
                "WHERE p.source_key=? ORDER BY p.dataset_user_id", (source_key,),
            )]

    def accounts(self):
        # Lazy, bounded-memory scan. No import on ordinary registration or startup.
        with self.lock:
            source_key = self._source_key()
            rows = self._existing(source_key)
            if not rows:
                profiles = self._read_profiles()
                prepared = []
                for dataset_id, ratings in profiles.items():
                    salt = secrets.token_hex(16)
                    prepared.append((dataset_id, ratings, salt,
                                     self.store.password_hash(DEMO_PASSWORD, salt)))
                if self._source_key() != source_key:
                    raise ValueError("Dataset vừa thay đổi. Vui lòng tải lại danh sách demo.")
                with self.store.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    for dataset_id, ratings, salt, password_hash in prepared:
                        if db.execute("SELECT 1 FROM dataset_profiles WHERE source_key=? "
                                      "AND dataset_user_id=?", (source_key, dataset_id)).fetchone():
                            continue
                        email = f"movielens{dataset_id}.{source_key[:8]}@demo.movie.local"
                        cursor = db.execute(
                            "INSERT INTO users(name,email,password_hash,salt,created_at) VALUES(?,?,?,?,?)",
                            (f"MovieLens · User {dataset_id}", email, password_hash, salt, int(time.time())),
                        )
                        user_id = cursor.lastrowid
                        db.execute("INSERT INTO dataset_profiles(user_id,source_key,dataset_user_id,"
                                   "imported_rating_count) VALUES(?,?,?,?)",
                                   (user_id, source_key, dataset_id, len(ratings)))
                        db.executemany("INSERT INTO user_ratings VALUES(?,?,?,?,?)", [
                            (user_id, int(row.movieId), float(row.rating), int(row.timestamp) * 1000,
                             int(row.timestamp) * 1000) for row in ratings.itertuples(index=False)
                        ])
                rows = self._existing(source_key)
        return {"accounts": [{**row, "password": DEMO_PASSWORD} for row in rows],
                "source": "MovieLens", "selection": "lowest_five_user_ids",
                "notice": "Tài khoản demo do ứng dụng tạo; lịch sử đánh giá lấy từ dataset. "
                          "Mật khẩu này không phải mật khẩu gốc của người dùng MovieLens."}

    def _read_profiles(self):
        movie_ids = set(pd.read_csv(self.settings.dataset_dir / "movies.csv", usecols=["movieId"]).movieId)
        selected = {}
        with pd.read_csv(self.settings.dataset_dir / "ratings.csv",
                         usecols=["userId", "movieId", "rating", "timestamp"],
                         chunksize=self.settings.data.chunksize) as chunks:
            for chunk in chunks:
                valid = (chunk.userId.notna() & (chunk.userId > 0) & (chunk.userId % 1 == 0)
                         & chunk.movieId.isin(movie_ids) & chunk.rating.between(0.5, 5)
                         & (chunk.rating * 2 % 1 == 0) & chunk.timestamp.notna()
                         & (chunk.timestamp >= 0) & (chunk.timestamp % 1 == 0))
                chunk = chunk.loc[valid]
                # This remains correct even when input rows are not sorted by user.
                ids = sorted(set(selected) | set(chunk.userId.unique()))[:5]
                selected = {int(uid): selected.get(int(uid), []) for uid in ids}
                for uid, frame in chunk.loc[chunk.userId.isin(ids)].groupby("userId"):
                    selected[int(uid)].append(frame)
        if not selected:
            raise ValueError("Dataset chưa có lịch sử đánh giá hợp lệ để tạo tài khoản demo.")
        return {uid: pd.concat(frames).sort_values(["timestamp", "movieId"], kind="stable")
                .drop_duplicates("movieId", keep="last") for uid, frames in sorted(selected.items())}
