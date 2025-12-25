import json
from db.db import get_conn
from rag.util.time import now_iso
from rag.ingest.pipeline import ingest_path
from core.models import ProviderConfig, KeysConfig
from rag.runtime.context import set_runtime

def _update_job(job_id: str, status: str, progress: float | None = None, error: str | None = None):
    conn = get_conn()
    try:
        if progress is None:
            conn.execute(
                "UPDATE ingest_jobs SET status=?, error=?, updated_at=? WHERE id=?",
                (status, error, now_iso(), job_id),
            )
        else:
            conn.execute(
                "UPDATE ingest_jobs SET status=?, progress=?, error=?, updated_at=? WHERE id=?",
                (status, float(progress), error, now_iso(), job_id),
            )
        conn.commit()
    finally:
        conn.close()

def ingest_document(job_id: str, document_id: str, collection_id: str):
    ingest_document_sync(job_id, document_id, collection_id)

def ingest_document_sync(job_id: str, document_id: str, collection_id: str):
    _update_job(job_id, "running", progress=0.01, error=None)

    conn = get_conn()
    try:
        doc = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        job = conn.execute("SELECT * FROM ingest_jobs WHERE id=?", (job_id,)).fetchone()
        if not doc or not job:
            _update_job(job_id, "failed", error="document/job not found")
            return
        storage_path = doc["storage_path"]
        filename = doc["filename"]

        provider_json = job["provider_json"] or "{}"
        keys_json = job["keys_json"] or "{}"
    finally:
        conn.close()

    try:
        provider = ProviderConfig(**json.loads(provider_json))
        keys = KeysConfig(embed_api_key=json.loads(keys_json).get("embed_api_key"))
        set_runtime(provider, keys)

        ingest_path(
            collection_id=collection_id,
            document_id=document_id,
            filename=filename,
            path=storage_path,
            on_progress=lambda p: _update_job(job_id, "running", progress=p),
        )
        conn = get_conn()
        try:
            conn.execute("UPDATE documents SET status=? WHERE id=?", ("ready", document_id))
            conn.commit()
        finally:
            conn.close()
        _update_job(job_id, "complete", progress=1.0, error=None)
    except Exception as e:
        conn = get_conn()
        try:
            conn.execute("UPDATE documents SET status=? WHERE id=?", ("failed", document_id))
            conn.commit()
        finally:
            conn.close()
        _update_job(job_id, "failed", error=str(e))
