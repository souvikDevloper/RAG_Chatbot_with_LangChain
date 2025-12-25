from fastapi import APIRouter, HTTPException
from core.models import CollectionCreate, CollectionOut
from db.db import get_conn
from rag.util.ids import new_id
from rag.util.time import now_iso

router = APIRouter()

@router.post("", response_model=CollectionOut)
def create_collection(req: CollectionCreate):
    cid = new_id()
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO collections (id, name, created_at) VALUES (?, ?, ?)",
            (cid, req.name, now_iso()),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": cid, "name": req.name, "created_at": now_iso()}

@router.get("", response_model=list[CollectionOut])
def list_collections():
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM collections ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

@router.get("/{collection_id}", response_model=CollectionOut)
def get_collection(collection_id: str):
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM collections WHERE id=?", (collection_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="collection not found")
        return dict(row)
    finally:
        conn.close()
