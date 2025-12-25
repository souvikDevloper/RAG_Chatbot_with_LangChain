from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.config import settings
from core.utils import sha256_file
from db.db import get_conn
from rag.util.ids import new_id
from rag.util.time import now_iso
from rag.jobs.enqueue import enqueue_ingest

from pathlib import Path
import shutil
import json

router = APIRouter()

@router.post("/collections/{collection_id}/documents", response_model=dict)
def upload_document(
    collection_id: str,
    file: UploadFile = File(...),
    embed_api_key: str = Form(...),
    embed_base_url: str = Form("https://api.openai.com/v1"),
    embed_model: str = Form("text-embedding-3-small"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    if not embed_api_key:
        raise HTTPException(status_code=400, detail="embed_api_key is required for ingestion (BYOK)")

    doc_id = new_id()
    job_id = new_id()

    dest_dir = settings.blob_path / collection_id / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    digest = sha256_file(dest_path)
    ts = now_iso()

    provider_json = json.dumps({
        "embed_base_url": embed_base_url,
        "embed_model": embed_model,
    })
    keys_json = json.dumps({
        "embed_api_key": embed_api_key,
    })

    conn = get_conn()
    try:
        c = conn.execute("SELECT id FROM collections WHERE id=?", (collection_id,)).fetchone()
        if not c:
            raise HTTPException(status_code=404, detail="collection not found")

        conn.execute(
            "INSERT INTO documents (id, collection_id, filename, storage_path, sha256, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, collection_id, file.filename, str(dest_path), digest, "queued", ts),
        )
        conn.execute(
            "INSERT INTO ingest_jobs (id, document_id, status, progress, error, provider_json, keys_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, doc_id, "queued", 0.0, None, provider_json, keys_json, ts, ts),
        )
        conn.commit()
    finally:
        conn.close()

    enqueue_ingest(job_id=job_id, document_id=doc_id, collection_id=collection_id)
    return {"document_id": doc_id, "job_id": job_id}

@router.get("/collections/{collection_id}/documents", response_model=list[dict])
def list_documents(collection_id: str):
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, collection_id, filename, status, created_at FROM documents WHERE collection_id=? ORDER BY created_at DESC",
            (collection_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
