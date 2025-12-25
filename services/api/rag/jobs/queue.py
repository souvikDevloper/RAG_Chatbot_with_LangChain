from rq import Queue
from redis import Redis
from core.config import settings

def get_queue() -> Queue:
    r = Redis.from_url(settings.redis_url)
    return Queue("ingest", connection=r, default_timeout=60 * 30)
