from core.config import settings
from rag.jobs.queue import get_queue
from rag.jobs.tasks import ingest_document_sync

def enqueue_ingest(job_id: str, document_id: str, collection_id: str) -> None:
    if settings.use_queue:
        q = get_queue()
        q.enqueue("rag.jobs.tasks.ingest_document", job_id, document_id, collection_id)
    else:
        ingest_document_sync(job_id, document_id, collection_id)
