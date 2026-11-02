"""Prepare the same demo profiles shown on Login, without running the web server."""

import json

from movie_rating.services.accounts import AccountStore
from movie_rating.services.demo_profiles import DemoProfiles
from movie_rating.settings import load_settings


if __name__ == "__main__":
    settings = load_settings()
    store = AccountStore(settings.artifacts_dir / "application.sqlite3")
    result = DemoProfiles(settings, store).accounts()
    print(json.dumps({"accounts": result["accounts"], "notice": result["notice"]},
                     ensure_ascii=False, indent=2))
