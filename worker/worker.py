import os
import sys

# Add API code to sys.path BEFORE importing core/db
CANDIDATES = [
    os.path.abspath("/app/services_api"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "services", "api")),
]
for p in CANDIDATES:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from redis import Redis
from rq import Worker
from core.config import settings
from db.db import init_db

if __name__ == "__main__":
    init_db()
    r = Redis.from_url(settings.redis_url)
    w = Worker(["ingest"], connection=r)
    w.work()
